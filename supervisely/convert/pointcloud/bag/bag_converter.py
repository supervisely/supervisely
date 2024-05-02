from pathlib import Path
from typing import List

from supervisely import ProjectMeta, batched, generate_free_name, is_development, logger
from supervisely.api.api import Api
from supervisely.convert.base_converter import AvailablePointcloudConverters
from supervisely.convert.pointcloud.bag.bag_helper import pc2_to_pcd
from supervisely.convert.pointcloud.pointcloud_converter import PointcloudConverter
from supervisely.io.fs import (
    get_file_ext,
    get_file_name,
    list_files_recursively,
    silent_remove,
)


class BagConverter(PointcloudConverter):
    class Item(PointcloudConverter.Item):
        def __init__(
            self,
            item_path,
            ann_data: str = None,
            related_images: list = None,
            custom_data: dict = None,
        ):
            super().__init__(item_path, ann_data, related_images, custom_data)
            self._topic = None
            self._type = "point_cloud"

        @property
        def topic(self):
            return self._topic

        @topic.setter
        def topic(self, topic: str):
            self._topic = topic

    def __init__(self, input_data: str, labeling_interface: str):
        self._input_data: str = input_data
        self._items: List[PointcloudConverter.Item] = []
        self._meta: ProjectMeta = None
        self._labeling_interface: str = labeling_interface
        self._total_msg_count = 0
        self._is_pcd_episode = False

    def __str__(self) -> str:
        return AvailablePointcloudConverters.BAG

    @property
    def key_file_ext(self) -> str:
        return ".bag"

    def validate_format(self) -> bool:
        import rosbag # pylint: disable=import-error

        def _filter_fn(file_path):
            return get_file_ext(file_path).lower() == self.key_file_ext

        bag_files = list_files_recursively(self._input_data, filter_fn=_filter_fn)

        if len(bag_files) == 0:
            return False

        self._items = []
        for bag_file in bag_files:
            with rosbag.Bag(bag_file) as bag:
                cloud_msg_cnt = 0
                types, bag_msg_cnt = [], []
                topics_info = bag.get_type_and_topic_info()[1]

                for i in range(0, len(topics_info.values())):
                    types.append(list(topics_info.values())[i][0])
                    bag_msg_cnt.append(list(topics_info.values())[i][1])

                topics = zip(topics_info, types, bag_msg_cnt)
                for topic, topic_type, msg_count in topics:
                    if topic_type == "sensor_msgs/PointCloud2":
                        cloud_msg_cnt += msg_count

                        if self._is_pcd_episode:
                            item = self.Item(item_path=bag_file, frame_number=cloud_msg_cnt) # pylint: disable=unexpected-keyword-arg
                        else:
                            item = self.Item(item_path=bag_file)
                        item.topic = topic
                        self._items.append(item)
                self._total_msg_count += cloud_msg_cnt

        return self.items_count > 0

    def convert(self, item: Item, log_progress=True):
        import rosbag # pylint: disable=import-error
        import sensor_msgs.point_cloud2 as pc2 # pylint: disable=import-error

        paths = []
        index = 0
        bag_path = Path(item.path)
        topic = item.topic

        with rosbag.Bag(item.path) as bag:
            msg_count = bag.get_message_count(topic_filters=item.topic)
            if log_progress:
                convert_progress, convert_progress_cb = self.get_progress(
                    msg_count,
                    f"Convert {topic} topic from {bag_path.name} to pointclouds...",
                )
            else:
                convert_progress_cb = None

            for _, msg, _ in bag.read_messages(
                topics=[item.topic]
            ):  # read_messages return generator
                p_ = []
                gen = pc2.read_points(msg, skip_nans=True, field_names=("x", "y", "z"))

                for p in gen:
                    p_.append(p)

                topic = topic.replace("/", "_")  # replace / with _ to avoid path issues
                pcd_path = bag_path.parent / bag_path.stem / topic / f"{index:06d}.pcd"
                if not pcd_path.parent.exists():
                    pcd_path.parent.mkdir(parents=True, exist_ok=True)
                pc2_to_pcd(p_, pcd_path.as_posix())
                index = index + 1
                paths.append(pcd_path)
                if log_progress:
                    convert_progress_cb(1)

        if log_progress and is_development():
            convert_progress.close()
        return paths

    def upload_dataset(self, api: Api, dataset_id: int, batch_size: int = 1, log_progress=True):
        self._upload_dataset(
            api, dataset_id, batch_size, log_progress, is_episodes=self._is_pcd_episode
        )

    def _upload_dataset(
        self,
        api: Api,
        dataset_id: int,
        batch_size: int = 1,
        log_progress=True,
        is_episodes=False,
    ):
        """Converts and uploads bag files to Supervisely dataset."""

        multiple_items = self.items_count > 1
        if multiple_items:
            logger.info(f"Found {self.items_count} topics in the input data.")
            logger.info(f"Will create dataset in parent dataset for each topic.")

        datasets = []
        if multiple_items:
            project_info = api.project.get_info_by_id(dataset_id)
            for item in self._items:
                ds_name = get_file_name(item.path)
                ds = api.dataset.create(
                    project_info.id, ds_name, change_name_if_conflict=True, parent_id=dataset_id
                )
                datasets.append(ds)

        if log_progress:
            progress, progress_cb = self.get_progress(self._total_msg_count, "Uploading...")
        else:
            progress_cb = None

        for idx, item in enumerate(self._items):
            current_dataset_id = dataset_id if not multiple_items else datasets[idx].id
            pcd_paths = self.convert(item, log_progress=log_progress)

            existing_names = set([pcd.name for pcd in api.pointcloud.get_list(current_dataset_id)])

            for batch in batched(pcd_paths, batch_size=batch_size):
                pcd_names = []
                pcd_paths = []
                for pcd_path in batch:
                    pcd_name = generate_free_name(
                        existing_names, pcd_path.name, with_ext=True, extend_used_names=True
                    )
                    pcd_names.append(pcd_name)
                    pcd_paths.append(pcd_path)

                upload_fn = (
                    api.pointcloud_episode.upload_paths
                    if is_episodes
                    else api.pointcloud.upload_paths
                )
                upload_fn(current_dataset_id, pcd_names, pcd_paths)

                if log_progress:
                    progress_cb(len(batch))

            logger.info(f"Dataset ID:{current_dataset_id} has been successfully uploaded.")
            for pcd_path in pcd_paths:
                silent_remove(pcd_path)

        if log_progress:
            if is_development():
                progress.close()

import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Union

from supervisely import (
    Api,
    ObjClass,
    PointcloudAnnotation,
    PointcloudEpisodeAnnotation,
    PointcloudEpisodeFrame,
    PointcloudEpisodeObject,
    PointcloudFigure,
    ProjectMeta,
    generate_free_name,
    is_development,
    logger,
)
from supervisely.convert.base_converter import AvailablePointcloudConverters
from supervisely.convert.pointcloud.bag.bag_helper import process_pc2_msg, process_vector3_msg
from supervisely.convert.pointcloud.pointcloud_converter import PointcloudConverter
from supervisely.geometry.cuboid_3d import Cuboid3d
from supervisely.io.fs import (
    get_file_ext,
    get_file_name,
    list_files_recursively,
    silent_remove,
)
from supervisely.project.project_settings import LabelingInterface


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

    def __init__(
            self,
            input_data: str,
            labeling_interface: Optional[Union[LabelingInterface, str]],
            upload_as_links: bool,
            remote_files_map: Optional[Dict[str, str]] = None,
    ):
        super().__init__(input_data, labeling_interface, upload_as_links, remote_files_map)

        self._total_msg_count = 0
        self._is_pcd_episode = False

    def __str__(self) -> str:
        return AvailablePointcloudConverters.BAG

    @property
    def key_file_ext(self) -> str:
        return ".bag"

    def validate_format(self) -> bool:
        import rosbag  # pylint: disable=import-error

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
                pcd_topic = None
                ann_topic = None
                for topic, topic_type, msg_count in topics:
                    if topic_type == "sensor_msgs/PointCloud2":
                        if "SlyAnnotations" in topic:
                            logger.warn(
                                f"{topic} topic: only [geometry_msgs/Vector3Stamped] supported for Supervisely annotations"
                            )
                            # ann_topic = topic
                            continue
                        cloud_msg_cnt += msg_count
                        pcd_topic = topic
                    elif topic_type == "geometry_msgs/Vector3Stamped":
                        if "SlyAnnotations" in topic:
                            ann_topic = topic
                            continue

                if pcd_topic is not None:
                    if self._is_pcd_episode:
                        item = self.Item( # pylint: disable=unexpected-keyword-arg
                            item_path=bag_file, frame_number=cloud_msg_cnt
                        )  
                    else:
                        item = self.Item(item_path=bag_file)
                    item.topic = pcd_topic
                    item.ann_data = ann_topic
                    self._items.append(item)
                self._total_msg_count += cloud_msg_cnt

        return self.items_count > 0

    def convert(self, item: Item, meta: ProjectMeta):
        import rosbag  # pylint: disable=import-error

        bag_path = Path(item.path)
        topic = item.topic
        ann_topic = item.ann_data

        with rosbag.Bag(item.path) as bag:
            msg_count = bag.get_message_count(topic_filters=item.topic)
            if ann_topic is not None:
                ann_topics_info = bag.get_type_and_topic_info(topic_filters=[item.ann_data])[1]
                ann_topic_type = list(ann_topics_info.values())[0][0]
            progress, progress_cb = self.get_progress(
                msg_count, f"Convert {topic} topic from {bag_path.name} to pcd"
            )

            time_to_data = defaultdict(dict)

            for _, msg, rostime in bag.read_messages(topics=[item.topic]):
                process_pc2_msg(time_to_data, msg, rostime, bag_path, topic, meta, is_ann=False)

                progress_cb(1)
            if is_development():
                progress.close()

            # get annotations
            if ann_topic is not None:
                msg_count = bag.get_message_count(topic_filters=item.ann_data)
                progress, progress_cb = self.get_progress(
                    msg_count, f"Convert {item.ann_data} topic to JSON annotations"
                )

                time_to_vectors = defaultdict(list)
                for _, msg, rostime in bag.read_messages(topics=[item.ann_data]):
                    if ann_topic_type == "geometry_msgs/Vector3Stamped":
                        frame_id = msg.header.frame_id
                        if re.match(r"\d+\.\d+", frame_id):
                            time_to_vectors[frame_id].append(msg)
                    elif ann_topic_type == "sensor_msgs/PointCloud2":
                        logger.warn(
                            f"{item.ann_data} topic: only [geometry_msgs/Vector3Stamped] supported for Supervisely annotations"
                        )
                        # process_pc2_msg(time_to_data, msg, rostime, bag_path, ann_topic, meta, is_ann=True)
                        progress_cb(1)
                if len(time_to_vectors) > 0:
                    process_vector3_msg(time_to_data, time_to_vectors, bag_path, meta, ann_topic, progress_cb)

                if is_development():
                    progress.close()

        return time_to_data

    def upload_dataset(self, api: Api, dataset_id: int, batch_size: int = 1, log_progress=True):
        self._upload_dataset(api, dataset_id, log_progress, is_episodes=self._is_pcd_episode)

    def _upload_dataset(
        self,
        api: Api,
        dataset_id: int,
        log_progress=True,
        is_episodes=False,
    ):
        """
        Converts and uploads bag files to Supervisely dataset.
        Note: This method is used by both the BagConverter and the BagEpisodeConverter.
        """
        obj_cls = ObjClass("object", Cuboid3d)
        self._meta = ProjectMeta(obj_classes=[obj_cls])
        meta, _, _ = self.merge_metas_with_conflicts(api, dataset_id)

        multiple_items = self.items_count > 1
        datasets = []
        dataset_info = api.dataset.get_info_by_id(dataset_id)

        if multiple_items:
            logger.info(
                f"Found {self.items_count} topics in the input data."
                "Will create dataset in parent dataset for each topic."
            )
            nested_datasets = api.dataset.get_list(dataset_info.project_id, parent_id=dataset_id)
            existing_ds_names = set([ds.name for ds in nested_datasets])
            for item in self._items:
                ds_name = generate_free_name(existing_ds_names, get_file_name(item.path))
                ds = api.dataset.create(
                    dataset_info.project_id,
                    ds_name,
                    change_name_if_conflict=True,
                    parent_id=dataset_id,
                )
                existing_ds_names.add(ds.name)
                datasets.append(ds)

        if log_progress:
            progress, progress_cb = self.get_progress(self._total_msg_count, "Uploading...")
        else:
            progress_cb = None

        for idx, item in enumerate(self._items):
            current_dataset = dataset_info if not multiple_items else datasets[idx]
            current_dataset_id = current_dataset.id
            time_to_data = self.convert(item, meta)

            existing_names = set([pcd.name for pcd in api.pointcloud.get_list(current_dataset_id)])

            ann_episode = PointcloudEpisodeAnnotation()
            frame_to_pointcloud_ids = {}
            for idx, (time, data) in enumerate(time_to_data.items()):
                pcd_path = data["pcd"].as_posix()
                ann_path = data["ann"].as_posix() if data["ann"] is not None else None
                pcd_meta = data["meta"]

                pcd_name = generate_free_name(
                    existing_names, f"{time}.pcd", with_ext=True, extend_used_names=True
                )
                if is_episodes:
                    pcd_meta["frame"] = idx
                upload_fn = (
                    api.pointcloud_episode.upload_path
                    if is_episodes
                    else api.pointcloud.upload_path
                )
                info = upload_fn(current_dataset_id, pcd_name, pcd_path, pcd_meta)
                pcd_id = info.id
                frame_to_pointcloud_ids[idx] = pcd_id

                if ann_path is not None:
                    ann = PointcloudAnnotation.load_json_file(ann_path, meta)
                    if is_episodes:
                        objects = ann_episode.objects
                        figures = []
                        for fig in ann.figures:
                            obj_cls = meta.get_obj_class(fig.parent_object.obj_class.name)
                            if obj_cls is not None:
                                obj = PointcloudEpisodeObject(obj_cls)
                                objects = objects.add(obj)
                                figure = PointcloudFigure(obj, fig.geometry, frame_index=idx)
                                figures.append(figure)
                        frames = ann_episode.frames
                        frames = frames.add(PointcloudEpisodeFrame(idx, figures))
                        ann_episode = ann_episode.clone(objects=objects, frames=frames)
                    else:
                        api.pointcloud.annotation.append(pcd_id, ann)

                silent_remove(pcd_path)
                if ann_path is not None:
                    silent_remove(ann_path)
                if log_progress:
                    progress_cb(1)

            if is_episodes:
                ann_episode = ann_episode.clone(frames_count=len(time_to_data))
                api.pointcloud_episode.annotation.append(
                    current_dataset_id, ann_episode, frame_to_pointcloud_ids
                )

            logger.info(f"Dataset ID:{current_dataset_id} has been successfully uploaded.")

        if log_progress:
            if is_development():
                progress.close()

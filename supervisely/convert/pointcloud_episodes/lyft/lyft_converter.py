from typing import Dict, Optional, Union

from supervisely.convert.base_converter import AvailablePointcloudEpisodesConverters
from supervisely.convert.pointcloud.lyft.lyft_converter import LyftConverter
from supervisely.convert.pointcloud_episodes.pointcloud_episodes_converter import (
    PointcloudEpisodeConverter,
)
from supervisely.pointcloud_annotation.pointcloud_episode_annotation import (
    PointcloudEpisodeAnnotation,
)
from supervisely.project.project_settings import LabelingInterface

from pathlib import Path
from typing import Dict, Optional

from supervisely import (
    Api,
    ObjClass,
    PointcloudEpisodeAnnotation,
    PointcloudEpisodeFrame,
    PointcloudEpisodeObject,
    PointcloudFigure,
    ProjectMeta,
    logger,
    is_development,
)
from supervisely.io import fs
from supervisely.io.fs import get_file_name
from supervisely.convert.pointcloud.lyft import lyft_helper
from supervisely.api.api import ApiField
from datetime import datetime
from supervisely.geometry.cuboid_3d import Cuboid3d


class LyftEpisodesConverter(LyftConverter, PointcloudEpisodeConverter):
    """Converter for LYFT pointcloud episodes format."""

    def __init__(
        self,
        input_data: str,
        labeling_interface: Optional[Union[LabelingInterface, str]],
        upload_as_links: bool,
        remote_files_map: Optional[Dict[str, str]] = None,
    ):
        super().__init__(input_data, labeling_interface, upload_as_links, remote_files_map)

        self._type = "point_cloud_episode"
        self._is_pcd_episode = True

    def __str__(self) -> str:
        return AvailablePointcloudEpisodesConverters.LYFT

    def upload_dataset(self, api: Api, dataset_id: int, batch_size: int = 1, log_progress=True):
        unique_names = {name for item in self._items for name in item.ann_data["names"]}
        self._meta = ProjectMeta([ObjClass(name, Cuboid3d) for name in unique_names])
        meta, renamed_classes, _ = self.merge_metas_with_conflicts(api, dataset_id)

        scene_names = set([item._scene_name for item in self._items])

        dataset_info = api.dataset.get_info_by_id(dataset_id)
        scene_name_to_dataset = {}
        frame_to_pointcloud_ids = {}
        ann_episode = PointcloudEpisodeAnnotation()

        multiple_scenes = len(scene_names) > 1
        if multiple_scenes:
            logger.info(
                f"Found {len(scene_names)} scenes ({self.items_count} pointclouds) in the input data."
            )
            # * Create a nested dataset for each scene
            for name in scene_names:
                ds = api.dataset.create(
                    dataset_info.project_id,
                    name,
                    change_name_if_conflict=True,
                    parent_id=dataset_id,
                )
                scene_name_to_dataset[name] = ds
        else:
            scene_name_to_dataset[scene_names[0]] = dataset_info

        if log_progress:
            progress, progress_cb = self.get_progress(self.items_count, "Converting pointclouds...")
        else:
            progress_cb = None

        scene = self._items[0]._scene_name
        for idx, item in enumerate(self._items):
            if item._scene_name != scene:
                ann_episode = PointcloudEpisodeAnnotation()
            # * Get the current dataset for the scene
            current_dataset = scene_name_to_dataset.get(item._scene_name, None)
            if current_dataset is None:
                raise RuntimeError("Dataset not found for scene name: {}".format(item._scene_name))
            current_dataset_id = current_dataset.id

            # * Convert timestamp to ISO format
            iso_time = datetime.utcfromtimestamp(item.ann_data["timestamp"] / 1e6).isoformat() + "Z"
            item.ann_data["timestamp"] = iso_time

            # * Convert pointcloud from ".bin" to ".pcd"
            pcd_path = str(Path(item.path).with_suffix(".pcd"))
            if fs.file_exists(pcd_path):
                logger.warning(f"Overwriting file with path: {pcd_path}")
            lyft_helper.convert_bin_to_pcd(item.path, pcd_path)

            # * Upload pointcloud
            pcd_meta = {}
            pcd_meta["frame"] = idx

            pcd_name = get_file_name(pcd_path)
            info = api.pointcloud_episode.upload_path(
                current_dataset_id, pcd_name, pcd_path, pcd_meta
            )
            pcd_id = info.id
            frame_to_pointcloud_ids[idx] = pcd_id

            # * Convert annotation and upload
            ann = self.to_supervisely(item, meta, renamed_classes)
            objects = ann_episode.objects
            figures = []
            for (
                fig
            ) in (
                ann.figures
            ):  # todo figures must be extended from previous episodes, not as new objects
                obj_cls = meta.get_obj_class(fig.parent_object.obj_class.name)
                if obj_cls is not None:
                    obj = PointcloudEpisodeObject(obj_cls)
                    objects = objects.add(obj)
                    figure = PointcloudFigure(obj, fig.geometry, frame_index=idx)
                    figures.append(figure)
            frames = ann_episode.frames
            frames = frames.add(PointcloudEpisodeFrame(idx, figures))
            ann_episode = ann_episode.clone(objects=objects, frames=frames)

            # * Upload related images
            image_jsons = []
            camera_names = []
            for img_path, rimage_info in lyft_helper.generate_rimage_infos(
                item._related_images, item.ann_data
            ):
                img = api.pointcloud.upload_related_image(img_path)
                image_jsons.append(
                    {
                        ApiField.ENTITY_ID: pcd_id,
                        ApiField.NAME: rimage_info[ApiField.NAME],
                        ApiField.HASH: img,
                        ApiField.META: rimage_info[ApiField.META],
                    }
                )
                camera_names.append(rimage_info[ApiField.META]["deviceId"])
            if len(image_jsons) > 0:
                api.pointcloud.add_related_images(image_jsons, camera_names)

            sample_cnt = None
            ann_episode = ann_episode.clone(frames_count=sample_cnt)
            api.pointcloud_episode.annotation.append(
                current_dataset_id, ann_episode, frame_to_pointcloud_ids
            )

            # * Clean up
            fs.silent_remove(pcd_path)
            if log_progress:
                progress_cb(1)

        logger.info(f"Dataset ID:{current_dataset_id} has been successfully uploaded.")

        if log_progress:
            if is_development():
                progress.close()

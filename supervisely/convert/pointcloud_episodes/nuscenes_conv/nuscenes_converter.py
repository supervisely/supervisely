from os import path as osp
from pathlib import Path
from typing import Dict, Optional

import supervisely.convert.pointcloud_episodes.nuscenes_conv.nuscenes_helper as helpers
import supervisely.io.fs as fs
from supervisely._utils import is_development
from supervisely.annotation.obj_class import ObjClass
from supervisely.annotation.tag_meta import TagMeta, TagValueType
from supervisely.api.api import Api, ApiField
from supervisely.convert.base_converter import AvailablePointcloudConverters
from supervisely.convert.pointcloud_episodes.pointcloud_episodes_converter import (
    PointcloudEpisodeConverter,
)
from supervisely.geometry.cuboid_3d import Cuboid3d
from supervisely.pointcloud_annotation.pointcloud_episode_annotation import (
    PointcloudEpisodeAnnotation,
)
from supervisely.pointcloud_annotation.pointcloud_episode_frame import (
    PointcloudEpisodeFrame,
)
from supervisely.pointcloud_annotation.pointcloud_episode_frame_collection import (
    PointcloudEpisodeFrameCollection,
)
from supervisely.pointcloud_annotation.pointcloud_episode_object import (
    PointcloudEpisodeObject,
)
from supervisely.pointcloud_annotation.pointcloud_episode_object_collection import (
    PointcloudEpisodeObjectCollection,
)
from supervisely.pointcloud_annotation.pointcloud_episode_tag_collection import (
    PointcloudEpisodeTagCollection,
)
from supervisely.pointcloud_annotation.pointcloud_figure import PointcloudFigure
from supervisely.project.project_meta import ProjectMeta
from supervisely.sly_logger import logger
from supervisely.tiny_timer import TinyTimer


class NuscenesEpisodesConverter(PointcloudEpisodeConverter):
    """Converter for NuScenes pointcloud episodes format."""

    def __init__(
        self,
        input_data: str,
        labeling_interface: str,
        upload_as_links: bool,
        remote_files_map: Optional[Dict[str, str]] = None,
    ):
        super().__init__(input_data, labeling_interface, upload_as_links, remote_files_map)
        self._nuscenes = None

    def __str__(self) -> str:
        return AvailablePointcloudConverters.NUSCENES

    def validate_format(self) -> bool:
        try:
            from nuscenes import NuScenes
        except ImportError:
            logger.warning("Please, run 'pip install nuscenes-devkit' to import NuScenes data.")
            return False

        def filter_fn(path):
            return all([(Path(path) / name).exists() for name in ["maps", "samples"]])

        input_path = next((d for d in fs.dirs_filter(self._input_data, filter_fn)), None)
        if input_path is None:
            return False

        sample_dir = input_path + "/samples/"
        if any([not fs.dir_exists(f"{sample_dir}/{d}") for d in helpers.DIR_NAMES]):
            return False

        fil_fn = lambda p: all(fs.file_exists(f"{p}/{name}.json") for name in helpers.TABLE_NAMES)
        ann_dir = next((d for d in fs.dirs_filter(input_path, fil_fn)), None)
        if ann_dir is None:
            return False

        version = osp.basename(ann_dir)
        try:
            t = TinyTimer()
            nuscenes = NuScenes(version=version, dataroot=input_path, verbose=False)
            self._nuscenes: NuScenes = nuscenes
            logger.debug(f"NuScenes initialization took {t.get_sec():.3f} sec")
        except Exception as e:
            logger.debug(f"Failed to initialize NuScenes: {e}")
            return False

        return True

    def to_supervisely(
        self,
        scene_samples,
        meta: ProjectMeta,
        renamed_classes: dict = {},
        renamed_tags: dict = {},
    ) -> PointcloudEpisodeAnnotation:
        token_to_obj = {}
        frames = []
        tags = []
        for sample_i, sample in enumerate(scene_samples):
            figures = []
            for obj in sample.anns:
                instance_token = obj.instance_token
                class_name = obj.category
                parent_obj_token = obj.parent_token
                parent_object = None
                if parent_obj_token == "":
                    # * Create a new object
                    obj_class_name = renamed_classes.get(class_name, class_name)
                    obj_class = meta.get_obj_class(obj_class_name)
                    obj_tags = None  # ! TODO: fix tags
                    pcd_ep_obj = PointcloudEpisodeObject(obj_class, obj_tags)
                    # * Assign the object to the starting token
                    token_to_obj[instance_token] = pcd_ep_obj
                    parent_object = pcd_ep_obj
                else:
                    # * -> Figure has a parent object, get it
                    token_to_obj[instance_token] = token_to_obj[parent_obj_token]
                    parent_object = token_to_obj[parent_obj_token]
                geom = obj.to_supervisely()
                pcd_figure = PointcloudFigure(parent_object, geom, sample_i)
                figures.append(pcd_figure)
            frame = PointcloudEpisodeFrame(sample_i, figures)
            frames.append(frame)
        tag_collection = PointcloudEpisodeTagCollection(tags) if len(tags) > 0 else None
        return PointcloudEpisodeAnnotation(
            len(frames),
            PointcloudEpisodeObjectCollection(list(set(token_to_obj.values()))),
            PointcloudEpisodeFrameCollection(frames),
            tag_collection,
        )

    def upload_dataset(self, api: Api, dataset_id: int, batch_size: int = 1, log_progress=True):
        nuscenes = self._nuscenes

        tag_metas = [TagMeta(attr["name"], TagValueType.NONE) for attr in nuscenes.attribute]
        obj_classes = []
        for category in nuscenes.category:
            color = nuscenes.colormap[category["name"]]
            description = category["description"]
            if len(description) > 255:
                # * Trim description to fit into 255 characters limit
                sentences = description.split(".")
                trimmed_description = ""
                for sentence in sentences:
                    if len(trimmed_description) + len(sentence) + 1 > 255:
                        break
                    trimmed_description += sentence + "."
                description = trimmed_description.strip()
            obj_classes.append(ObjClass(category["name"], Cuboid3d, color, description=description))

        self._meta = ProjectMeta(obj_classes, tag_metas)
        meta, renamed_classes, renamed_tags = self.merge_metas_with_conflicts(api, dataset_id)

        dataset_info = api.dataset.get_info_by_id(dataset_id)
        scene_name_to_dataset = {}

        scene_names = [scene["name"] for scene in nuscenes.scene]
        scene_cnt = len(scene_names)
        total_sample_cnt = sum([scene["nbr_samples"] for scene in nuscenes.scene])

        multiple_scenes = len(scene_names) > 1
        if multiple_scenes:
            logger.info(f"Found {scene_cnt} scenes ({total_sample_cnt} samples) in the input data.")
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
            progress, progress_cb = self.get_progress(
                total_sample_cnt, "Converting episode scenes..."
            )
        else:
            progress_cb = None

        for scene in nuscenes.scene:
            current_dataset_id = scene_name_to_dataset[scene["name"]].id

            log = nuscenes.get("log", scene["log_token"])
            sample_token = scene["first_sample_token"]

            # * Extract scene's samples
            scene_samples = []
            for i in range(scene["nbr_samples"]):
                sample = nuscenes.get("sample", sample_token)
                lidar_path, boxes, _ = nuscenes.get_sample_data(sample["data"]["LIDAR_TOP"])
                if not osp.exists(lidar_path):
                    logger.warning(f'Scene "{scene["name"]}" has no LIDAR data.')
                    continue

                timestamp = sample["timestamp"]
                anns = []
                for box, name, inst_token in helpers.Sample.generate_boxes(nuscenes, boxes):
                    current_instance_token = inst_token["token"]
                    parent_token = inst_token["prev"]

                    # get category, attributes and visibility
                    ann = nuscenes.get("sample_annotation", current_instance_token)
                    category = ann["category_name"]
                    attributes = [
                        nuscenes.get("attribute", attr)["name"] for attr in ann["attribute_tokens"]
                    ]
                    visibility = nuscenes.get("visibility", ann["visibility_token"])["level"]

                    anns.append(
                        helpers.AnnotationObject(
                            name,
                            box,
                            current_instance_token,
                            parent_token,
                            category,
                            attributes,
                            visibility,
                        )
                    )

                # get camera data
                sample_data = nuscenes.get("sample_data", sample["data"]["LIDAR_TOP"])
                cal_sensor = nuscenes.get(
                    "calibrated_sensor", sample_data["calibrated_sensor_token"]
                )
                ego_pose = nuscenes.get("ego_pose", sample_data["ego_pose_token"])

                camera_data = [
                    helpers.CamData(nuscenes, sensor, token, cal_sensor, ego_pose)
                    for sensor, token in sample["data"].items()
                    if sensor.startswith("CAM")
                ]
                scene_samples.append(helpers.Sample(timestamp, lidar_path, anns, camera_data))
                sample_token = sample["next"]

            # * Convert and upload pointclouds
            frame_to_pointcloud_ids = {}
            for idx, sample in enumerate(scene_samples):
                pcd_path = sample.convert_lidar_to_supervisely()

                pcd_name = fs.get_file_name(pcd_path)
                pcd_meta = {
                    "frame": idx,
                    "vehicle": log["vehicle"],
                    "date": log["date_captured"],
                    "location": log["location"],
                    "description": scene["description"],
                }
                info = api.pointcloud_episode.upload_path(
                    current_dataset_id, pcd_name, pcd_path, pcd_meta
                )
                fs.silent_remove(pcd_path)

                pcd_id = info.id
                frame_to_pointcloud_ids[idx] = pcd_id

                # * Upload related images
                image_jsons = []
                camera_names = []
                for img_path, rimage_info in [
                    data.get_info(sample.timestamp) for data in sample.cam_data
                ]:
                    img = api.pointcloud_episode.upload_related_image(img_path)
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
                    api.pointcloud_episode.add_related_images(image_jsons, camera_names)

                if log_progress:
                    progress_cb(1)

            # * Convert and upload annotations
            pcd_ann = self.to_supervisely(scene_samples, meta, renamed_classes, renamed_tags)
            try:
                api.pointcloud_episode.annotation.append(
                    current_dataset_id, pcd_ann, frame_to_pointcloud_ids
                )
                logger.info(f"Dataset ID:{current_dataset_id} has been successfully uploaded.")
            except Exception as e:
                error_msg = getattr(getattr(e, "response", e), "text", str(e))
                logger.warning(
                    f"Failed to upload annotation for scene: {scene['name']}. Message: {error_msg}"
                )

        if log_progress:
            if is_development():
                progress.close()

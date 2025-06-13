import os
from pathlib import Path
from typing import Optional, List
from supervisely import PointcloudEpisodeAnnotation, ProjectMeta, is_development, logger, ObjClass, ObjClassCollection
from supervisely.geometry.cuboid_3d import Cuboid3d
from supervisely.api.api import Api, ApiField
from supervisely.convert.base_converter import AvailablePointcloudEpisodesConverters
from supervisely.convert.pointcloud_episodes.kitti_360.kitti_360_helper import *
from supervisely.convert.pointcloud_episodes.pointcloud_episodes_converter import PointcloudEpisodeConverter
from supervisely.io.fs import (
    file_exists,
    get_file_name,
    get_file_name_with_ext,
    list_files_recursively,
    silent_remove,
)
from supervisely.pointcloud_annotation.pointcloud_episode_frame_collection import PointcloudEpisodeFrameCollection
from supervisely.pointcloud_annotation.pointcloud_episode_object_collection import PointcloudEpisodeObjectCollection
from supervisely.pointcloud_annotation.pointcloud_episode_object import PointcloudEpisodeObject
from supervisely.pointcloud_annotation.pointcloud_episode_frame import PointcloudEpisodeFrame
from supervisely.pointcloud_annotation.pointcloud_figure import PointcloudFigure

class KITTI360Converter(PointcloudEpisodeConverter):

    class Item:

        def __init__(
            self,
            scene_name: str,
            frame_paths: List[str],
            ann_data: Annotation3D,
            poses_path: str,
            related_images: Optional[tuple] = None,
            custom_data: Optional[dict] = None,
        ):
            self._scene_name = scene_name
            self._frame_paths = frame_paths
            self._ann_data = ann_data
            self._poses_path = poses_path
            self._related_images = related_images or []

            self._type = "point_cloud_episode"
            self._custom_data = custom_data if custom_data is not None else {}

    def __init__(self, *args, **kwargs):
        self._calib_path = None
        super().__init__(*args, **kwargs)

    def __str__(self) -> str:
        return AvailablePointcloudEpisodesConverters.KITTI360

    @property
    def key_file_ext(self) -> str:
        return ".bin"

    def validate_format(self) -> bool:
        try:
            import kitti360scripts
        except ImportError:
            logger.warn("Please run 'pip install kitti360Scripts' to import KITTI-360 data.")
            return False

        self._items = []
        subdirs = os.listdir(self._input_data)
        if len(subdirs) == 1:
            self._input_data = os.path.join(self._input_data, subdirs[0])

        # * Get calibration path
        calib_dir = next(iter([(Path(path).parent).as_posix() for path in list_files_recursively(self._input_data, [".txt"], None, True) if Path(path).stem.startswith("calib")]), None)
        if calib_dir is None:
            return False
        self._calib_path = calib_dir

        # * Get pointcloud files paths
        velodyne_files = list_files_recursively(self._input_data, [".bin"], None, True)
        if len(velodyne_files) == 0:
            return False

        # * Get annotation files paths and related images
        boxes_ann_files = list_files_recursively(self._input_data, [".xml"], None, True)
        if len(boxes_ann_files) == 0:
            return False
        rimage_files = list_files_recursively(self._input_data, [".png"], None, True)

        kitti_anns = []
        for ann_file in boxes_ann_files:
            key_name = Path(ann_file).stem

            # * Get pointcloud files
            frame_paths = []
            for path in velodyne_files:
                if key_name in Path(path).parts:
                    frame_paths.append(path)
            if len(frame_paths) == 0:
                logger.warn("No frames found for name: %s", key_name)
                continue

            # * Get related images
            rimages = []
            for rimage in rimage_files:
                path = Path(rimage)
                if key_name in path.parts:
                    cam_name = path.parts[-3]
                    rimages.append((cam_name, rimage))

            # * Get poses
            poses_filter = (
                lambda x: x.endswith("cam0_to_world.txt") and key_name in Path(x).parts
            )
            poses_path = next(
                path
                for path in list_files_recursively(self._input_data, [".txt"], None, True)
                if poses_filter(path)
            )
            if poses_path is None:
                logger.warn("No poses found for name: %s", key_name)
                continue

            # * Parse annotation
            ann = Annotation3D(ann_file)
            kitti_anns.append(ann)

            self._items.append(
                self.Item(key_name, frame_paths, ann, poses_path, rimages)
            )

        # * Get object class names for meta
        obj_class_names = set()
        for ann in kitti_anns:
            for obj in ann.get_objects():
                obj_class_names.add(obj.name)
        obj_classes = [ObjClass(obj_class, Cuboid3d) for obj_class in obj_class_names]
        self._meta = ProjectMeta(obj_classes=ObjClassCollection(obj_classes))
        return self.items_count > 0

    def to_supervisely(
        self,
        item,
        meta: ProjectMeta,
        renamed_classes: dict = {},
        renamed_tags: dict = {},
        static_transformations: StaticTransformations = None,
    ) -> PointcloudEpisodeAnnotation:
        static_transformations.set_cam2world(item._poses_path)

        frame_cnt = len(item._frame_paths)
        objs, frames = [], []

        frame_idx_to_figures = {idx: [] for idx in range(frame_cnt)}
        for obj in item._ann_data.get_objects():
            pcd_obj = PointcloudEpisodeObject(meta.get_obj_class(obj.name))
            objs.append(pcd_obj)

            for idx in range(frame_cnt):
                if obj.start_frame <= idx <= obj.end_frame:
                    tr_matrix = static_transformations.world_to_velo_transformation(obj, idx)
                    geom = convert_kitti_cuboid_to_supervisely_geometry(tr_matrix)
                    frame_idx_to_figures[idx].append(PointcloudFigure(pcd_obj, geom, idx))
        for idx, figures in frame_idx_to_figures.items():
            frame = PointcloudEpisodeFrame(idx, figures)
            frames.append(frame)
        obj_collection = PointcloudEpisodeObjectCollection(objs)
        frame_collection = PointcloudEpisodeFrameCollection(frames)
        return PointcloudEpisodeAnnotation(
            frame_cnt, objects=obj_collection, frames=frame_collection
        )

    def upload_dataset(self, api: Api, dataset_id: int, batch_size: int = 1, log_progress=True):
        meta, renamed_classes, renamed_tags = self.merge_metas_with_conflicts(api, dataset_id)

        dataset_info = api.dataset.get_info_by_id(dataset_id)
        if log_progress:
            progress, progress_cb = self.get_progress(sum([len(item._frame_paths) for item in self._items]), "Converting pointcloud episodes...")
        else:
            progress_cb = None
        static_transformations = StaticTransformations(self._calib_path)
        scene_ds = dataset_info
        multiple_items = self.items_count > 1
        for item in self._items:
            scene_ds = api.dataset.create(dataset_info.project_id, item._scene_name, parent_id=dataset_id) if multiple_items else dataset_info
            frame_to_pcd_ids = {}
            for idx, frame_path in enumerate(item._frame_paths):
                # * Convert pointcloud from ".bin" to ".pcd"
                pcd_path = str(Path(frame_path).with_suffix(".pcd"))
                if file_exists(pcd_path):
                    logger.warning(f"Overwriting file with path: {pcd_path}")
                convert_bin_to_pcd(frame_path, pcd_path)

                # * Upload pointcloud
                pcd_name = get_file_name_with_ext(pcd_path)
                info = api.pointcloud_episode.upload_path(scene_ds.id, pcd_name, pcd_path, {"frame": idx})
                pcd_id = info.id
                frame_to_pcd_ids[idx] = pcd_id

                # * Clean up
                silent_remove(pcd_path)

                if log_progress:
                    progress_cb(1)

            # * Upload photocontext
            rimage_jsons = []
            cam_names = []
            hashes = api.pointcloud_episode.upload_related_images(
                [rimage_path for _, rimage_path in item._related_images]
            )
            for (cam_name, rimage_path), img, pcd_id in zip(
                item._related_images, hashes, list(frame_to_pcd_ids.values())
            ):
                cam_num = int(cam_name[-1])
                rimage_info = convert_calib_to_image_meta(
                    get_file_name(rimage_path), static_transformations, cam_num
                )
                image_json = {
                    ApiField.ENTITY_ID: pcd_id,
                    ApiField.NAME: cam_name,
                    ApiField.HASH: img,
                    ApiField.META: rimage_info[ApiField.META],
                }
                rimage_jsons.append(image_json)
                cam_names.append(cam_name)
            if rimage_jsons:
                api.pointcloud_episode.add_related_images(rimage_jsons, cam_names)

            # * Convert annotation and upload
            try:
                ann = self.to_supervisely(
                    item, meta, renamed_classes, renamed_tags, static_transformations
                )
                api.pointcloud_episode.annotation.append(scene_ds.id, ann, frame_to_pcd_ids)
            except Exception as e:
                logger.error(
                    f"Failed to upload annotation for scene: {scene_ds.name}. Error: {repr(e)}",
                    stack_info=False,
                )
                continue

            logger.info(f"Dataset ID:{scene_ds.id} has been successfully uploaded.")

        if log_progress:
            if is_development():
                progress.close()

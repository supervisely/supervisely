import imghdr
import os

import supervisely.convert.pointcloud_episodes.sly.sly_pointcloud_episodes_helper as sly_episodes_helper
from supervisely import PointcloudEpisodeAnnotation, ProjectMeta, logger
from supervisely.convert.base_converter import AvailablePointcloudEpisodesConverters
from supervisely.convert.pointcloud_episodes.pointcloud_episodes_converter import (
    PointcloudEpisodeConverter,
)
from supervisely.io.fs import JUNK_FILES, get_file_ext, get_file_name_with_ext
from supervisely.io.json import load_json_file
from supervisely.pointcloud.pointcloud import validate_ext as validate_pcd_ext


class SLYPointcloudEpisodesConverter(PointcloudEpisodeConverter):

    def __str__(self) -> str:
        return AvailablePointcloudEpisodesConverters.SLY

    @property
    def ann_ext(self) -> str:
        return ".json"

    @property
    def key_file_ext(self) -> str:
        return ".json"

    def generate_meta_from_annotation(self, ann_path: str, meta: ProjectMeta) -> ProjectMeta:
        meta = sly_episodes_helper.get_meta_from_annotation(ann_path, meta)
        return meta

    def validate_ann_file(self, ann_path: str, meta: ProjectMeta) -> bool:
        try:
            ann_json = load_json_file(ann_path)
            if "annotation" in ann_json:
                ann_json = ann_json["annotation"]
            ann = PointcloudEpisodeAnnotation.from_json(ann_json, meta)
            self._annotation = ann
            return True
        except Exception as e:
            return False

    def validate_key_file(self, key_file_path: str) -> bool:
        try:
            self._meta = ProjectMeta.from_json(load_json_file(key_file_path))
            return True
        except Exception:
            return False

    def validate_format(self) -> bool:
        sly_ann_detected = False
        ann_path = None
        pcd_dict = {}
        frames_pcd_map = None
        rimg_dict, rimg_json_dict = {}, {}
        for root, _, files in os.walk(self._input_data):
            dir_name = os.path.basename(root)
            for file in files:
                full_path = os.path.join(root, file)
                if file == "key_id_map.json":
                    continue
                if file == "meta.json":
                    is_valid = self.validate_key_file(full_path)
                    if is_valid:
                        continue
                if file == "annotation.json":
                    ann_path = full_path
                    continue
                if file == "frame_pointcloud_map.json":
                    frames_pcd_map = load_json_file(full_path)
                    continue

                ext = get_file_ext(full_path)
                recognized_ext = imghdr.what(full_path)
                if file in JUNK_FILES:
                    continue
                elif ext == self.ann_ext:
                    rimg_json_dict[file] = full_path
                elif recognized_ext:
                    if ext.lower() == ".pcd":
                        logger.warning(
                            f"File '{file}' has been recognized as '.{recognized_ext}' format. Skipping."
                        )
                        continue
                    if dir_name not in rimg_dict:
                        rimg_dict[dir_name] = []
                    rimg_dict[dir_name].append(full_path)
                else:
                    try:
                        validate_pcd_ext(ext)
                        pcd_dict[file] = full_path
                    except:
                        continue

        if self._meta is not None:
            meta = self._meta
            sly_ann_detected = True
        else:
            meta = ProjectMeta()
        if ann_path is not None:
            if self._meta is None:
                meta = self.generate_meta_from_annotation(ann_path, meta)
            is_valid = self.validate_ann_file(ann_path, meta)
            if is_valid:
                sly_ann_detected = True

        if not sly_ann_detected:
            return False

        self._items = []
        updated_frames_pcd_map = {}
        if frames_pcd_map:
            list_of_pcd_names = list(frames_pcd_map.values())
        else:
            list_of_pcd_names = sorted(pcd_dict.keys())

        for i, pcd_name in enumerate(list_of_pcd_names):
            if pcd_name in pcd_dict:
                updated_frames_pcd_map[i] = pcd_name
                item = self.Item(pcd_dict[pcd_name], i)
                rimg_dir_name = pcd_name.replace(".pcd", "_pcd")
                rimgs = rimg_dict.get(rimg_dir_name, [])
                for rimg_path in rimgs:
                    rimg_ann_name = f"{get_file_name_with_ext(rimg_path)}.json"
                    if rimg_ann_name in rimg_json_dict:
                        rimg_ann_path = rimg_json_dict[rimg_ann_name]
                        rimg_fig_name = f"{get_file_name_with_ext(rimg_path)}.figures.json"
                        rimg_fig_path = rimg_json_dict.get(rimg_fig_name, None)
                        if rimg_fig_path is not None and not os.path.exists(rimg_fig_path):
                            rimg_fig_path = None
                        item.set_related_images((rimg_path, rimg_ann_path, rimg_fig_path))
                self._items.append(item)
            else:
                logger.warning(f"Pointcloud file {pcd_name} not found. Skipping frame.")
                continue
        self._frame_pointcloud_map = updated_frames_pcd_map
        self._frame_count = len(self._frame_pointcloud_map)

        self._meta = meta
        return sly_ann_detected

    def to_supervisely(
        self,
        item: PointcloudEpisodeConverter.Item,
        meta: ProjectMeta = None,
        renamed_classes: dict = None,
        renamed_tags: dict = None,
    ) -> PointcloudEpisodeAnnotation:
        """Convert to Supervisely format."""
        if self._annotation is not None:
            if renamed_classes or renamed_tags:
                ann_json = self._annotation.to_json()
                ann_json = sly_episodes_helper.rename_in_json(
                    ann_json, renamed_classes, renamed_tags
                )
                self._annotation = PointcloudEpisodeAnnotation.from_json(ann_json, meta)
            return self._annotation
        else:
            return item.create_empty_annotation()

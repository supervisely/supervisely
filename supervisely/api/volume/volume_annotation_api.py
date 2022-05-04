# coding: utf-8

import json
from supervisely.api.module_api import ApiField
from supervisely.video_annotation.key_id_map import KeyIdMap
from supervisely.volume_annotation.volume_annotation import VolumeAnnotation

from supervisely.api.entity_annotation.entity_annotation_api import EntityAnnotationAPI
from supervisely.io.json import load_json_file


class VolumeAnnotationAPI(EntityAnnotationAPI):
    _method_download_bulk = "volumes.annotations.bulk.info"
    _entity_ids_str = ApiField.VOLUME_IDS

    def download(self, volume_id):
        """
        :param video_id: int
        :return: video annotation to given id in json format
        """
        volume_info = self._api.volume.get_info_by_id(volume_id)
        return self._download(volume_info.dataset_id, volume_id)

    def append(self, volume_id, ann: VolumeAnnotation, key_id_map: KeyIdMap = None):
        info = self._api.volume.get_info_by_id(volume_id)
        self._append(
            self._api.volume.tag,
            self._api.volume.object,
            self._api.volume.figure,
            info.project_id,
            info.dataset_id,
            volume_id,
            ann.tags,
            ann.objects,
            ann.figures,
            key_id_map,
        )

    def upload_paths(
        self,
        volume_ids,
        ann_paths,
        project_meta,
        interpolation_dirs=None,
        progress_cb=None,
    ):
        if interpolation_dirs is None:
            interpolation_dirs = [None] * len(ann_paths)

        key_id_map = KeyIdMap()
        for volume_id, ann_path, interpolation_dir in zip(
            volume_ids, ann_paths, interpolation_dirs
        ):
            ann_json = load_json_file(ann_path)
            ann = VolumeAnnotation.from_json(ann_json, project_meta)
            self.append(volume_id, ann, key_id_map)

            # create empty figures for meshes
            self._api.volume.figure.append_bulk(
                volume_id, ann.spatial_figures, key_id_map
            )
            # upload existing interpolations or create on the fly and and add them to empty mesh figures
            self._api.volume.figure.upload_stl_meshes(
                volume_id, ann.spatial_figures, key_id_map, interpolation_dir
            )
            if progress_cb is not None:
                progress_cb(1)

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
        # build interpolations for objects on server
        # self._api.volume.figure.interpolate(volume_id, ann.spatial_figures, key_id_map)

    def upload_paths(self, volume_ids, ann_paths, project_meta, progress_cb=None):
        key_id_map = KeyIdMap()
        for volume_id, ann_path in zip(volume_ids, ann_paths):
            ann_json = load_json_file(ann_path)
            ann = VolumeAnnotation.from_json(ann_json, project_meta)

            # ignore existing key_id_map because the new objects will be created
            self.append(volume_id, ann, key_id_map)

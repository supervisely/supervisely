# coding: utf-8

from supervisely_lib.api.module_api import ApiField, RemoveableBulkModuleApi
from supervisely_lib.api.volume.volume_annotation_api import VolumeAnnotationApi
from supervisely_lib.io.fs import ensure_base_path
from supervisely_lib.volume.volume import validate_format

class VolumeApi(RemoveableBulkModuleApi):
    def __init__(self, api):
        super().__init__(api)
        self.annotation = VolumeAnnotationApi(api)
        # TODO: add objects, figures, tags api

    @staticmethod
    def info_sequence():
        # TODO: what is processingPath?
        return [ApiField.ID,
                ApiField.NAME,
                ApiField.DESCRIPTION,
                ApiField.TAGS,
                ApiField.TEAM_ID,
                ApiField.WORKSPACE_ID,
                ApiField.PROJECT_ID,
                ApiField.DATASET_ID,
                ApiField.LINK,
                ApiField.PATH_ORIGINAL,
                ApiField.PREVIEW,
                ApiField.META,
                ApiField.FILE_META,
                ApiField.FIGURES_COUNT,
                ApiField.ANN_OBJECTS_COUNT,
                ApiField.CREATED_AT,
                ApiField.UPDATED_AT
                ]

    @staticmethod
    def info_tuple_name():
        return 'VolumeInfo'

    def _convert_json_info(self, info: dict, skip_missing=True):
        return super(VolumeApi, self)._convert_json_info(info, skip_missing=skip_missing)

    def get_info_by_id(self, id):
        for info in self.tmp_infos:  # TODO: remove after creating info method
            if info.id == id:
                return info

    def get_list(self, dataset_id, filters=None):
        infos = self.get_list_all_pages('volumes.list', {ApiField.DATASET_ID: dataset_id,
                                                        ApiField.FILTER: filters or []})
        self.tmp_infos = infos  # TODO: remove after creating info method
        return infos

    def download_path(self, id, path, progress_cb=None):
        validate_format(path)
        ensure_base_path(path)
        response = self._download(id, is_stream=True)

        with open(path, 'wb') as fd:
            mb1 = 1024 * 1024
            for chunk in response.iter_content(chunk_size=mb1):
                fd.write(chunk)

                if progress_cb is not None:
                    progress_cb(len(chunk))

    def _download(self, id, is_stream=False):
        # TODO: videos.download -> volumes.download
        response = self._api.post('videos.download', {ApiField.ID: id}, stream=is_stream)
        return response


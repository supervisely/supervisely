# coding: utf-8

from supervisely_lib.api.module_api import ApiField, RemoveableBulkModuleApi

from supervisely_lib.io.fs import ensure_base_path
from supervisely_lib._utils import batched

from supervisely_lib.api.pointcloud.pointcloud_annotation_api import PointcloudAnnotationAPI
from supervisely_lib.api.pointcloud.pointcloud_object_api import PointcloudObjectApi
from supervisely_lib.api.pointcloud.pointcloud_figure_api import PointcloudFigureApi
from supervisely_lib.api.pointcloud.pointcloud_tag_api import PointcloudTagApi


class PointcloudApi(RemoveableBulkModuleApi):
    def __init__(self, api):
        super().__init__(api)
        self.annotation = PointcloudAnnotationAPI(api)
        self.object = PointcloudObjectApi(api)
        self.figure = PointcloudFigureApi(api)
        self.tag = PointcloudTagApi(api)

    @staticmethod
    def info_sequence():
        return [ApiField.ID,
                ApiField.DESCRIPTION,
                ApiField.NAME,
                ApiField.TEAM_ID,
                ApiField.WORKSPACE_ID,
                ApiField.PROJECT_ID,
                ApiField.DATASET_ID,
                ApiField.LINK,
                ApiField.HASH,
                ApiField.PATH_ORIGINAL,
                #ApiField.PREVIEW,
                ApiField.CLOUD_MIME,
                ApiField.FIGURES_COUNT,
                ApiField.ANN_OBJECTS_COUNT,
                ApiField.TAGS,
                ApiField.CREATED_AT,
                ApiField.UPDATED_AT]

    @staticmethod
    def info_tuple_name():
        return 'PointCloudInfo'

    def _convert_json_info(self, info: dict, skip_missing=True):
        return super(PointcloudApi, self)._convert_json_info(info, skip_missing=skip_missing)

    def get_list(self, dataset_id, filters=None):
        return self.get_list_all_pages('point-clouds.list',  {ApiField.DATASET_ID: dataset_id, ApiField.FILTER: filters or []})

    def get_info_by_id(self, id):
        return self._get_info_by_id(id, 'point-clouds.info')

    def _download(self, id, is_stream=False):
        response = self._api.post('point-clouds.download', {ApiField.ID: id}, stream=is_stream)
        return response

    def download_path(self, id, path):
        response = self._download(id, is_stream=True)
        ensure_base_path(path)
        with open(path, 'wb') as fd:
            for chunk in response.iter_content(chunk_size=1024*1024):
                fd.write(chunk)

    def get_list_related_images(self, id):
        dataset_id = self.get_info_by_id(id).dataset_id
        filters = [{"field": ApiField.ENTITY_ID, "operator": "=", "value": id}]
        return self.get_list_all_pages('point-clouds.images.list',
                                       {ApiField.DATASET_ID: dataset_id, ApiField.FILTER: filters},
                                       convert_json_info_cb=lambda x : x)

    def download_related_image(self, id, path):
        response = self._api.post('point-clouds.images.download', {ApiField.ID: id}, stream=True)
        ensure_base_path(path)
        with open(path, 'wb') as fd:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                fd.write(chunk)
        return response

    #@TODO: copypaste from video_api
    def upload_hash(self, dataset_id, name, hash, meta=None):
        meta = {} if meta is None else meta
        return self.upload_hashes(dataset_id, [name], [hash], [meta])[0]

    # @TODO: copypaste from video_api
    def upload_hashes(self, dataset_id, names, hashes, metas=None, progress_cb=None):
        return self._upload_bulk_add(lambda item: (ApiField.HASH, item), dataset_id, names, hashes, metas, progress_cb)

    # @TODO: copypaste from video_api
    def _upload_bulk_add(self, func_item_to_kv, dataset_id, names, items, metas=None, progress_cb=None):
        if metas is None:
            metas = [{}] * len(items)

        results = []
        if len(names) == 0:
            return results
        if len(names) != len(items):
            raise RuntimeError("Can not match \"names\" and \"items\" lists, len(names) != len(items)")

        for batch in batched(list(zip(names, items, metas))):
            images = []
            for name, item, meta in batch:
                item_tuple = func_item_to_kv(item)
                images.append({ApiField.NAME: name,
                               item_tuple[0]: item_tuple[1],
                               ApiField.META: meta if meta is not None else {}})
            response = self._api.post('point-clouds.bulk.add', {ApiField.DATASET_ID: dataset_id,
                                                                ApiField.POINTCLOUDS: images})
            if progress_cb is not None:
                progress_cb(len(images))

            results = [self._convert_json_info(item) for item in response.json()]
            name_to_res = {img_info.name: img_info for img_info in results}
            ordered_results = [name_to_res[name] for name in names]

            return ordered_results

    def add_related_images(self, images_json):
        response = self._api.post('point-clouds.images.add', {ApiField.IMAGES: images_json})
        return response.json()
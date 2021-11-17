# coding: utf-8

from supervisely_lib.api.module_api import ApiField, RemoveableBulkModuleApi
from supervisely_lib.api.volume.volume_annotation_api import VolumeAnnotationApi
from supervisely_lib.io.fs import ensure_base_path, get_file_hash
from supervisely_lib.volume.volume import validate_format
from supervisely_lib._utils import batched
from requests_toolbelt import MultipartEncoder
from collections import defaultdict
from supervisely_lib.api.volume.volume_figure_api import VolumeFigureApi
from supervisely_lib.api.volume.volume_object_api import VolumeObjectApi
from supervisely_lib.api.volume.volume_tag_api import VolumeTagApi

class VolumeApi(RemoveableBulkModuleApi):
    def __init__(self, api):
        super().__init__(api)
        self.annotation = VolumeAnnotationApi(api)
        self.object = VolumeObjectApi(api)
        self.figure = VolumeFigureApi(api)
        self.tag = VolumeTagApi(api)

    @staticmethod
    def info_sequence():
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
        return self._get_info_by_id(id, 'volumes.info')

    def get_list(self, dataset_id, filters=None):
        infos = self.get_list_all_pages('volumes.list', {ApiField.DATASET_ID: dataset_id,
                                                         ApiField.FILTER: filters or []})
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
        response = self._api.post('volumes.download', {ApiField.ID: id}, stream=is_stream)
        return response

    def upload_hash(self, dataset_id, name, hash, meta):
        return self.upload_hashes(dataset_id, [name], [hash], [meta])[0]

    def upload_hashes(self, dataset_id, names, hashes, metas, progress_cb=None):
        return self._upload_bulk_add(lambda item: (ApiField.HASH, item), dataset_id, names, hashes, metas, progress_cb)

    def _upload_bulk_add(self, func_item_to_kv, dataset_id, names, items, metas, progress_cb=None):
        results = []
        if len(names) == 0:
            return results
        if len(names) != len(items):
            raise RuntimeError("Can not match \"names\" and \"items\" lists, len(names) != len(items)")

        for batch in batched(list(zip(names, items, metas))):
            volumes = []
            for name, hash, meta in batch:
                item_tuple = func_item_to_kv(hash)
                volumes.append({ApiField.NAME: name,
                                item_tuple[0]: item_tuple[1],
                                ApiField.META: meta})

            response = self._api.post('volumes.bulk.add', {ApiField.DATASET_ID: dataset_id, ApiField.VOLUMES: volumes})
            if progress_cb is not None:
                progress_cb(len(volumes))

            results = [self._convert_json_info(item) for item in response.json()]
            name_to_res = {vol_info.name: vol_info for vol_info in results}
            ordered_results = [name_to_res[name] for name in names]

            return ordered_results

    def upload_path(self, dataset_id, name, path, meta):
        return self.upload_paths(dataset_id, [name], [path], metas=[meta])[0]

    def upload_paths(self, dataset_id, names, paths, metas, progress_cb=None):
        def path_to_bytes_stream(path):
            return open(path, 'rb')

        hashes = self._upload_data_bulk(path_to_bytes_stream, get_file_hash, paths, progress_cb)
        return self.upload_hashes(dataset_id, names, hashes, metas)

    def check_existing_hashes(self, hashes):
        results = []
        if len(hashes) == 0:
            return results
        for hashes_batch in batched(hashes, batch_size=900):
            response = self._api.post('import-storage.internal.hashes.list', {ApiField.HASHES: hashes_batch})
            results.extend(response.json())
        return results

    def _upload_data_bulk(self, func_item_to_byte_stream, func_item_hash, items, progress_cb):
        hashes = []
        if len(items) == 0:
            return hashes

        hash_to_items = defaultdict(list)

        for idx, item in enumerate(items):
            item_hash = func_item_hash(item)
            hashes.append(item_hash)
            hash_to_items[item_hash].append(item)

        unique_hashes = set(hashes)
        remote_hashes = self.check_existing_hashes(list(unique_hashes))
        new_hashes = unique_hashes - set(remote_hashes)

        if progress_cb is not None:
            progress_cb(len(remote_hashes))

        items_to_upload = []
        for hash in new_hashes:
            items_to_upload.extend(hash_to_items[hash])

        for batch in batched(items_to_upload):
            content_dict = {}
            for idx, item in enumerate(batch):
                content_dict["{}-file".format(idx)] = (str(idx), func_item_to_byte_stream(item), 'nrrd/*')
            encoder = MultipartEncoder(fields=content_dict)
            self._api.post('import-storage.bulk.upload', encoder)
            if progress_cb is not None:
                progress_cb(len(batch))

        return hashes


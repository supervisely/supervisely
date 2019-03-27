# coding: utf-8

from collections import namedtuple
import numpy as np

from supervisely_lib.api.module_api import ApiField, ModuleApi
from supervisely_lib._utils import camel_to_snake
from supervisely_lib.imaging import image
from supervisely_lib.io.fs import ensure_base_path


class ImageApi(ModuleApi):
    _info_sequence = [ApiField.ID,
                      ApiField.NAME,
                      ApiField.LINK,
                      ApiField.HASH,
                      ApiField.MIME,
                      ApiField.EXT,
                      ApiField.SIZE,
                      ApiField.WIDTH,
                      ApiField.HEIGHT,
                      ApiField.LABELS_COUNT,
                      ApiField.DATASET_ID,
                      ApiField.CREATED_AT,
                      ApiField.UPDATED_AT]

    Info = namedtuple('ImageInfo', [camel_to_snake(name) for name in _info_sequence])

    def get_list(self, dataset_id, filters=None):
        return self.get_list_all_pages('images.list',  {ApiField.DATASET_ID: dataset_id, ApiField.FILTER: filters or []})

    def get_info_by_id(self, id):
        return self._get_info_by_id(id, 'images.info')

    def download(self, id):
        response = self.api.post('images.download', {ApiField.ID: id})
        return response

    def download_np(self, id):
        response = self.download(id)
        image_bytes = np.asarray(bytearray(response.content), dtype="uint8")
        img = image.read_bytes(image_bytes)
        return img

    def download_to_file(self, id, path):
        response = self.download(id)
        ensure_base_path(path)
        with open(path, 'wb') as fd:
            for chunk in response.iter_content(chunk_size=1024*1024):
                fd.write(chunk)

    def download_batch(self, ids, paths, progress_cb=None):
        for id, path in zip(ids, paths):
            self.download_to_file(id, path)
            if progress_cb is not None:
                progress_cb()

    def check_existing_hashes(self, hashes):
        BATCH_SIZE = 900
        #@TODO: method is copypaste of agent_rpc. move to sly.utils
        def batched(seq, batch_size):
            for i in range(0, len(seq), batch_size):
                yield seq[i:i + batch_size]

        results = []
        for hashes_batch in batched(hashes, BATCH_SIZE):
            response = self.api.post('images.internal.hashes.list', hashes_batch)
            results.extend(response.json())
        return results

    def check_image_uploaded(self, hash):
        response = self.api.post('images.internal.hashes.list', [hash])
        results = response.json()
        if len(results) == 0:
            return False
        else:
            return True

    def upload_np(self, img_np, ext='.png'):
        data = image.write_bytes(img_np, ext)
        return self.upload(data)

    def upload_path(self, img_path):
        data = open(img_path, 'rb').read()
        return self.upload(data)

    def upload(self, data):
        response = self.api.post('images.upload', data)
        return response.json()[ApiField.HASH]

    def upload_link(self, link):
        response = self.api.post('images.remote.upsert', {ApiField.LINK: link})
        return response.json()[ApiField.ID]

    def upload_batch(self, img_paths, progress_cb=None):
        results = []
        for path in img_paths:
            results.append(self.upload_path(path))
            if progress_cb is not None:
                progress_cb()
        return results

    def add(self, dataset_id, name, hash):
        response = self.api.post('images.add', {ApiField.DATASET_ID: dataset_id, ApiField.HASH: hash, ApiField.NAME: name})
        return self._convert_json_info(response.json())

    def add_link(self, dataset_id, name, link):
        response = self.api.post('images.add', {ApiField.DATASET_ID: dataset_id, ApiField.LINK: link, ApiField.NAME: name})
        return self._convert_json_info(response.json())

    def add_batch(self, dataset_id, names, hashes, progress_cb=None):
        # @TODO: add special method to api
        ids = []
        for name, hash in zip(names, hashes):
            ids.append(self.add(dataset_id, name, hash).id)
            if progress_cb is not None:
                progress_cb()
        return ids

    def _convert_json_info(self, info: dict):
        if info is None:
            return None
        field_values = []
        for field_name in self.__class__._info_sequence:
            if field_name == ApiField.EXT:
                continue
            field_values.append(info[field_name])
            if field_name == ApiField.MIME:
                field_values.append(info[field_name].split('/')[1])
        return self.__class__.Info._make(field_values)

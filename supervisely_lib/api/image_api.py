# coding: utf-8

from collections import namedtuple
import numpy as np
import io
from collections import defaultdict

from supervisely_lib.api.module_api import ApiField, ModuleApi
from supervisely_lib._utils import camel_to_snake
from supervisely_lib.imaging import image as sly_image
from supervisely_lib.io.fs import ensure_base_path, get_file_hash, get_file_ext
from supervisely_lib._utils import batched
from requests_toolbelt import MultipartDecoder, MultipartEncoder
import re


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

    # @TODO: reimplement to new method images.bulk.info
    def get_info_by_id_batch(self, ids):
        results = []
        if len(ids) == 0:
            return results
        dataset_id = self.get_info_by_id(ids[0]).dataset_id
        for batch in batched(ids):
            filters = [{"field": ApiField.ID, "operator": "in", "value": batch}]
            results.extend(self.get_list_all_pages('images.list', {ApiField.DATASET_ID: dataset_id,
                                                                   ApiField.FILTER: filters}))
        return results

    def _download(self, id):
        response = self.api.post('images.download', {ApiField.ID: id})
        return response

    def download_np(self, id):
        response = self._download(id)
        img = sly_image.read_bytes(response.content)
        return img

    def download_path(self, id, path):
        response = self._download(id)
        ensure_base_path(path)
        with open(path, 'wb') as fd:
            for chunk in response.iter_content(chunk_size=1024*1024):
                fd.write(chunk)

    def _download_batch(self, dataset_id, ids):
        for batch_ids in batched(ids):
            response = self.api.post('images.bulk.download',
                                     {ApiField.DATASET_ID: dataset_id, ApiField.IMAGE_IDS: batch_ids})
            decoder = MultipartDecoder.from_response(response)
            for idx, part in enumerate(decoder.parts):
                img_id = int(re.findall('name="(.*)"', part.headers[b'Content-Disposition'].decode('utf-8'))[0])
                yield img_id, part

    def download_paths(self, dataset_id, ids, paths, progress_cb=None):
        if len(ids) == 0:
            return
        if len(ids) != len(paths):
            raise RuntimeError("Can not match \"ids\" and \"paths\" lists, len(ids) != len(paths)")

        id_to_path = {id: path for id, path in zip(ids, paths)}
        #debug_ids = []
        for img_id, resp_part in self._download_batch(dataset_id, ids):
            #debug_ids.append(img_id)
            with open(id_to_path[img_id], 'wb') as w:
                w.write(resp_part.content)
            if progress_cb is not None:
                progress_cb(1)
        #if ids != debug_ids:
        #    raise RuntimeError("images.bulk.download: imageIds order is broken")

    def download_nps(self, dataset_id, ids, progress_cb=None):
        images = []
        if len(ids) == 0:
            return images

        id_to_img = {}
        for img_id, resp_part in self._download_batch(dataset_id, ids):
            id_to_img[img_id] = sly_image.read_bytes(resp_part.content)
            if progress_cb is not None:
                progress_cb(1)

        images = [id_to_img[id] for id in ids]
        return images

    def check_existing_hashes(self, hashes):
        results = []
        if len(hashes) == 0:
            return results
        for hashes_batch in batched(hashes, batch_size=900):
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

        # upload only new images to supervisely server
        items_to_upload = []
        for hash in new_hashes:
            items_to_upload.extend(hash_to_items[hash])

        for batch in batched(items_to_upload):
            content_dict = {}
            for idx, item in enumerate(batch):
                content_dict["{}-file".format(idx)] = (str(idx), func_item_to_byte_stream(item), 'image/*')
            encoder = MultipartEncoder(fields=content_dict)
            self.api.post('images.bulk.upload', encoder)
            if progress_cb is not None:
                progress_cb(len(batch))

        return hashes

    def upload_path(self, dataset_id, name, path):
        return self.upload_paths(dataset_id, [name], [path])[0]

    def upload_paths(self, dataset_id, names, paths, progress_cb=None):
        def path_to_bytes_stream(path):
            return open(path, 'rb')

        hashes = self._upload_data_bulk(path_to_bytes_stream, get_file_hash, paths, progress_cb)
        return self.upload_hashes(dataset_id, names, hashes)

    def upload_np(self, dataset_id, name, img):
        return self.upload_nps(dataset_id, [name], [img])[0]

    def upload_nps(self, dataset_id, names, imgs, progress_cb=None):
        def img_to_bytes_stream(item):
            img, name = item[0], item[1]
            img_bytes = sly_image.write_bytes(img, get_file_ext(name))
            return io.BytesIO(img_bytes)

        def img_to_hash(item):
            img, name = item[0], item[1]
            return sly_image.get_hash(img, get_file_ext(name))

        img_name_list = list(zip(imgs, names))
        hashes = self._upload_data_bulk(img_to_bytes_stream, img_to_hash, img_name_list, progress_cb)
        return self.upload_hashes(dataset_id, names, hashes)

    def upload_link(self, dataset_id, name, link):
        return self.upload_links(dataset_id, [name], [link])[0]

    def upload_links(self, dataset_id, names, links, progress_cb=None):
        return self._upload_bulk_add(lambda item: (ApiField.LINK, item), dataset_id, names, links, progress_cb)

    def upload_hash(self, dataset_id, name, hash):
        return self.upload_hashes(dataset_id, [name], [hash])[0]

    def upload_hashes(self, dataset_id, names, hashes, progress_cb=None):
        return self._upload_bulk_add(lambda item: (ApiField.HASH, item), dataset_id, names, hashes, progress_cb)

    def upload_id(self, dataset_id, name, id):
        return self.upload_ids(dataset_id, [name], [id])[0]

    def upload_ids(self, dataset_id, names, ids, progress_cb=None):
        # all ids have to be from single dataset
        infos = self.get_info_by_id_batch(ids)
        hashes = [info.hash for info in infos]
        return self.upload_hashes(dataset_id, names, hashes, progress_cb)

    def _upload_bulk_add(self, func_item_to_kv, dataset_id, names, items, progress_cb=None):
        results = []

        if len(names) == 0:
            return results
        if len(names) != len(items):
            raise RuntimeError("Can not match \"names\" and \"items\" lists, len(names) != len(items)")

        for batch in batched(list(zip(names, items))):
            images = []
            for name, item in batch:
                item_tuple = func_item_to_kv(item)
                #@TODO: 'title' -> ApiField.NAME
                images.append({'title': name, item_tuple[0]: item_tuple[1]})
            response = self.api.post('images.bulk.add', {ApiField.DATASET_ID: dataset_id, ApiField.IMAGES: images})
            if progress_cb is not None:
                progress_cb(len(images))
            results.extend([self._convert_json_info(info_json) for info_json in response.json()])

        name_to_res = {img_info.name: img_info for img_info in results}
        ordered_results = [name_to_res[name] for name in names]

        return ordered_results

    #@TODO: reimplement
    def _convert_json_info(self, info: dict):
        if info is None:
            return None
        temp_ext = None
        field_values = []
        for field_name in self.__class__._info_sequence:
            if field_name == ApiField.EXT:
                continue
            field_values.append(info[field_name])
            if field_name == ApiField.MIME:
                temp_ext = info[field_name].split('/')[1]
                field_values.append(temp_ext)
        for idx, field_name in enumerate(self.__class__._info_sequence):
            if field_name == ApiField.NAME:
                cur_ext = get_file_ext(field_values[idx])
                if not cur_ext:
                    field_values[idx] = "{}.{}".format(field_values[idx], temp_ext)
                    break

                cur_ext = cur_ext.replace(".", "").lower()
                if temp_ext == 'jpeg' and cur_ext in ['jpg', 'jpeg']:
                    break

                if temp_ext not in field_values[idx]:
                    field_values[idx] = "{}.{}".format(field_values[idx], temp_ext)
                break
        return self.__class__.Info._make(field_values)


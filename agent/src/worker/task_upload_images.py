# coding: utf-8

import cv2
import supervisely_lib as sly
import supervisely_lib.worker_proto.worker_api_pb2 as api_proto

from worker.task_sly import TaskSly
from worker.agent_utils import create_img_meta_str

#@TODO: legacy functionality - remove in future
class TaskUploadImages(TaskSly):
    def task_main_func(self):
        if self.info.get('images', None) is None:
            raise ValueError('TASK_PROJECT_INFO_EMPTY')

        cnt_skipped_images = 0

        to_upload_paths = []
        to_upload_infos = []
        for img_info in self.info['images']:
            hash_ = img_info['hash']
            ext = img_info['ext']
            st_path = self.data_mgr.storage.images.check_storage_object(hash_, ext)
            if st_path is None:
                self.logger.warning('Image not found in local storage.', extra={'hash': hash_, 'ext': ext})
                cnt_skipped_images += 1
                continue

            img = cv2.imread(st_path)
            height, width = img.shape[:2]
            img_sizeb = sly.fs.get_file_size(st_path)

            img_meta_str = create_img_meta_str(img_sizeb, width=width, height=height)
            proto_img_info = api_proto.Image(hash=hash_, ext=ext, meta=img_meta_str)

            to_upload_paths.append(st_path)
            to_upload_infos.append(proto_img_info)

        self.data_mgr.upload_images_to_remote(to_upload_paths, to_upload_infos)

        self.logger.info("CNT_SKIPPED_IMAGES", extra={'cnt_skipped': cnt_skipped_images})

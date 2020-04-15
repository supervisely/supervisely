# coding: utf-8

from supervisely_lib.api.module_api import ApiField, ModuleApi
from supervisely_lib.io.fs import ensure_base_path
from supervisely_lib.imaging.image import read_bytes


class VideoFrameAPI(ModuleApi):
    def _download(self, video_id, frame_index):
        response = self._api.post('videos.download-frame', {ApiField.VIDEO_ID: video_id, ApiField.FRAME: frame_index})
        return response

    def download_np(self, video_id, frame_index):
        response = self._download(video_id, frame_index)
        img = read_bytes(response.content, input_is_bgr=False)
        return img

    def download_path(self, video_id, frame_index, path):
        response = self._download(video_id, frame_index)
        ensure_base_path(path)
        with open(path, 'wb') as fd:
            for chunk in response.iter_content(chunk_size=1024*1024):
                fd.write(chunk)

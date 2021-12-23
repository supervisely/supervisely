# coding: utf-8
from __future__ import annotations
from supervisely_lib.api.module_api import ApiField, ModuleApi
from supervisely_lib.io.fs import ensure_base_path
from supervisely_lib.imaging.image import read_bytes
import numpy as np

class VideoFrameAPI(ModuleApi):
    """
    :class:`Frame<supervisely_lib.video_annotation.frame.Frame>` for a single video. :class:`VideoFrameAPI<VideoFrameAPI>` object is immutable.
    """
    def _download(self, video_id, frame_index):
        '''
        :param video_id: int
        :param frame_index: int
        :return: Response class object containing frame data with given index from given video id
        '''
        response = self._api.post('videos.download-frame', {ApiField.VIDEO_ID: video_id, ApiField.FRAME: frame_index})
        return response

    def download_np(self, video_id: int, frame_index: int) -> np.ndarray:
        """
        Download Image for frame with given index from given video ID in numpy format(RGB).

        :param video_id: Video ID in Supervisely.
        :type video_id: int
        :param frame_index: Index of frame to download.
        :type frame_index: int
        :return: Image in RGB numpy matrix format
        :rtype: :class:`np.ndarray`
        :Usage example:

         .. code-block:: python

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            video_id = 198703211
            frame_idx = 5
            image_np = api.video.frame.download_np(video_id, frame_idx)
        """
        response = self._download(video_id, frame_index)
        img = read_bytes(response.content)
        return img

    def download_path(self, video_id: int, frame_index: int, path: str) -> None:
        """
        Downloads Image on the given path for frame with given index from given Video ID.

        :param video_id: Video ID in Supervisely.
        :type video_id: int
        :param frame_index: Index of frame to download.
        :type frame_index: int
        :param path: Local save path for Image.
        :type path: str
        :return: None
        :rtype: :class:`NoneType`
        :Usage example:

         .. code-block:: python

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            video_id = 198703211
            frame_idx = 5
            save_path = '/home/admin/Downloads/img/result.png'
            api.video.frame.download_path(video_id, frame_idx, save_path)
        """
        response = self._download(video_id, frame_index)
        ensure_base_path(path)
        with open(path, 'wb') as fd:
            for chunk in response.iter_content(chunk_size=1024*1024):
                fd.write(chunk)

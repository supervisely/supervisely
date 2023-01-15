# coding: utf-8
from __future__ import annotations
from typing import List, Tuple, NamedTuple, Dict, Optional, Callable
from requests import Response
import datetime
import os
import json
import urllib.parse
from functools import partial
from requests_toolbelt import MultipartEncoder, MultipartEncoderMonitor
from numerize.numerize import numerize

from supervisely.api.module_api import ApiField, RemoveableBulkModuleApi
from supervisely.api.video.video_annotation_api import VideoAnnotationAPI
from supervisely.api.video.video_object_api import VideoObjectApi
from supervisely.api.video.video_figure_api import VideoFigureApi
from supervisely.api.video.video_frame_api import VideoFrameAPI
from supervisely.api.video.video_tag_api import VideoTagApi
from supervisely.sly_logger import logger
from supervisely.io.fs import get_file_ext, get_file_hash, get_file_size
import supervisely.io.fs as sly_fs

from supervisely.io.fs import ensure_base_path
from supervisely._utils import batched, is_development, abs_url, rand_str
from supervisely.video.video import (
    get_info,
    get_video_streams,
    gen_video_stream_name,
    validate_ext,
)

from supervisely.task.progress import Progress


class VideoInfo(NamedTuple):
    id: int
    name: str
    hash: str
    link: str
    team_id: int
    workspace_id: int
    project_id: int
    dataset_id: int
    path_original: str
    frames_to_timecodes: list
    frames_count: int
    frame_width: int
    frame_height: int
    created_at: str
    updated_at: str
    tags: list
    file_meta: dict
    custom_data: dict
    processing_path: str

    @property
    def duration(self) -> float:
        # in seconds
        ndigits = 0
        return round(self.file_meta.get("duration"), ndigits=ndigits)

    @property
    def duration_hms(self) -> str:
        return str(datetime.timedelta(seconds=self.duration))

    @property
    def frames_count_compact(self) -> str:
        return numerize(self.frames_count)

    @property
    def image_preview_url(self):
        res = f"/previews/q/ext:jpeg/resize:fill:300:0:0/q:70/plain/image-converter/videoframe/33p/{self.processing_path}?videoStreamIndex=0"
        if is_development():
            res = abs_url(res)
        return res


class VideoApi(RemoveableBulkModuleApi):
    """
    API for working with :class:`Video<supervisely.video.video>`. :class:`VideoApi<VideoApi>` object is immutable.

    :param api: API connection to the server.
    :type api: Api
    :Usage example:

     .. code-block:: python

        import supervisely as sly

        # You can connect to API directly
        address = 'https://app.supervise.ly/'
        token = 'Your Supervisely API Token'
        api = sly.Api(address, token)

        # Or you can use API from environment
        os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
        os.environ['API_TOKEN'] = 'Your Supervisely API Token'
        api = sly.Api.from_env()

        dataset_id = 466654
        ds = api.video.get_list(dataset_id)
    """

    def __init__(self, api):
        super().__init__(api)
        self.annotation = VideoAnnotationAPI(api)
        self.object = VideoObjectApi(api)
        self.frame = VideoFrameAPI(api)
        self.figure = VideoFigureApi(api)
        self.tag = VideoTagApi(api)

    @staticmethod
    def info_sequence():
        """
        NamedTuple VideoInfo information about Video.

        :Example:

         .. code-block:: python

            VideoInfo(id=198702499,
                      name='Videos_dataset_cars_cars.mp4',
                      hash='ehYHLNFWmMNuF2fPUgnC/g/tkIIEjNIOhdbNLQXkE8Y=',
                      team_id=16087,
                      workspace_id=23821,
                      project_id=124974,
                      dataset_id=466639,
                      path_original='/h5un6l2bnaz1vj8a9qgms4-public/videos/w/7/i4/GZYoCs...9F3kyVJ7.mp4',
                      frames_to_timecodes=[0, 0.033367, 0.066733, 0.1001,...,10.777433, 10.8108, 10.844167],
                      frames_count=326,
                      frame_width=3840,
                      frame_height=2160,
                      created_at='2021-03-23T13:14:25.536Z',
                      updated_at='2021-03-23T13:16:43.300Z')
        """
        return [
            ApiField.ID,
            ApiField.NAME,
            ApiField.HASH,
            ApiField.LINK,
            ApiField.TEAM_ID,
            ApiField.WORKSPACE_ID,
            ApiField.PROJECT_ID,
            ApiField.DATASET_ID,
            ApiField.PATH_ORIGINAL,
            ApiField.FRAMES_TO_TIMECODES,
            ApiField.FRAMES_COUNT,
            ApiField.FRAME_WIDTH,
            ApiField.FRAME_HEIGHT,
            ApiField.CREATED_AT,
            ApiField.UPDATED_AT,
            ApiField.TAGS,
            ApiField.FILE_META,
            ApiField.CUSTOM_DATA,
            ApiField.PROCESSING_PATH,
        ]

    @staticmethod
    def info_tuple_name():
        return "VideoInfo"

    def url(self, dataset_id: int, video_id: int, video_frame: Optional[int] = 0) -> str:
        """
        :param dataset_id: int
        :param video_id: int
        :param video_frame: int
        :return: url of given video id
        """
        result = urllib.parse.urljoin(
            self._api.server_address,
            f"app/videos/?"
            f"datasetId={dataset_id}&"
            f"videoFrame={video_frame}&"
            f"videoId={video_id}",
        )
        return result

    def _convert_json_info(self, info: dict, skip_missing=True):
        res = super(VideoApi, self)._convert_json_info(info, skip_missing=skip_missing)
        # processing_path = info.get("processingPath", "")
        d = res._asdict()
        # d["processing_path"] = processing_path
        return VideoInfo(**d)

    def get_list(
        self, dataset_id: int, filters: Optional[List[Dict[str, str]]] = None
    ) -> List[VideoInfo]:
        """
        Get list of information about all video annotations for a given dataset.

        :param dataset_id: Dataset ID in Supervisely.
        :type dataset_id: int
        :param filters: List of parameters to sort output Annotations.
        :type filters: List[dict], optional
        :return: Information about VideoAnnotations. See :class:`info_sequence<info_sequence>`
        :rtype: :class:`List[VideoInfo]`

        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()
            dataset_id = 466639
            ann_infos = api.video.get_list(dataset_id)
            print(ann_infos)
            # Output: [
            #     [
            #         198702499,
            #         "Videos_dataset_cars_cars.mp4",
            #         "ehYHLNFWmMNuF2fPUgnC/g/tkIIEjNIOhdbNLQXkE8Y=",
            #         16087,
            #         23821,
            #         124974,
            #         466639,
            #         "/h5un6l2bnaz1vj8a9qgms4-public/videos/w/7/i4/GZYoCsd7CcTsU3mfN...F3kyVJ7.mp4",
            #         [
            #             0,
            #             0.033367,
            #             0.066733,
            #               ...
            #             10.8108,
            #             10.844167
            #         ],
            #         326,
            #         3840,
            #         2160,
            #         "2021-03-23T13:14:25.536Z",
            #         "2021-03-23T13:16:43.300Z"
            #     ],
            #     [
            #         198702500,
            #         "Videos_dataset_animals_sea_lion.mp4",
            #         "30/TQ1BcIOn1ykAI4RFgRO/6psRtr3lq3HF6NPmr4uQ=",
            #         16087,
            #         23821,
            #         124974,
            #         466639,
            #         "/h5un6l2bnaz1vj8a9qgms4-public/videos/l/k/iX/CaPbL6...5dE.mp4",
            #         [
            #             0,
            #             0.041708,
            #             0.083417,
            #               ...
            #             10.135125,
            #             10.176833
            #         ],
            #         245,
            #         1920,
            #         1080,
            #         "2021-03-23T13:14:25.536Z",
            #         "2021-03-23T13:16:43.300Z"
            #     ]
            # ]

            ann_infos_filter = api.video.get_list(466639, filters=[{'field': 'id', 'operator': '=', 'value': '198702499'}])
            print(ann_infos_filter)
            # Output: [
            #     [
            #         198702499,
            #         "Videos_dataset_cars_cars.mp4",
            #         "ehYHLNFWmMNuF2fPUgnC/g/tkIIEjNIOhdbNLQXkE8Y=",
            #         16087,
            #         23821,
            #         124974,
            #         466639,
            #         "/h5un6l2bnaz1vj8a9qgms4-public/videos/w/7/i4/GZYoCsd7CcTsU...kyVJ7.mp4",
            #         [
            #             0,
            #             0.033367,
            #             0.066733,
            #              ...
            #             10.8108,
            #             10.844167
            #         ],
            #         326,
            #         3840,
            #         2160,
            #         "2021-03-23T13:14:25.536Z",
            #         "2021-03-23T13:16:43.300Z"
            #     ]
            # ]
        """
        return self.get_list_all_pages(
            "videos.list", {ApiField.DATASET_ID: dataset_id, ApiField.FILTER: filters or []}
        )

    def get_info_by_id(self, id: int, raise_error: Optional[bool] = False) -> VideoInfo:
        """
        Get Video information by ID.

        :param id: Video ID in Supervisely.
        :type id: int
        :return: Information about Video. See :class:`info_sequence<info_sequence>`
        :rtype: :class:`VideoInfo`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            video_id = 198702499

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            video_info = api.video.get_info_by_id(video_id)
            print(video_info)
            # Output: [
            #     198702499,
            #     "Videos_dataset_cars_cars.mp4",
            #     "ehYHLNFWmMNuF2fPUgnC/g/tkIIEjNIOhdbNLQXkE8Y=",
            #     16087,
            #     23821,
            #     124974,
            #     466639,
            #     "/h5un6l2bnaz1vj8a9qgms4-public/videos/w/7/i4/GZYoCsd7CcT...3kyVJ7.mp4",
            #     [
            #         0,
            #         0.033367,
            #         0.066733,
            #         0.1001,
            #           ...
            #         10.8108,
            #         10.844167
            #     ],
            #     326,
            #     3840,
            #     2160,
            #     "2021-03-23T13:14:25.536Z",
            #     "2021-03-23T13:16:43.300Z"
            # ]
        """
        info = self._get_info_by_id(id, "videos.info")
        if info is None and raise_error is True:
            raise KeyError(f"Video with id={id} not found in your account")
        return info

    def get_json_info_by_id(self, id: int, raise_error: Optional[bool] = False) -> Dict:
        data = None
        response = self._get_response_by_id(id, "videos.info", id_field=ApiField.ID)
        if response is None:
            if raise_error is True:
                raise KeyError(f"Video with id={id} not found in your account")
            return None
        data = response.json()
        return data

    def get_destination_ids(self, id: int) -> Tuple[int, int]:
        """
        Get project ID and dataset ID for given Video ID.

        :param id: Video ID in Supervisely.
        :type id: int
        :return: Project ID and dataset ID
        :rtype: :class:`Tuple[int, int]`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            video_id = 198702499

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            project_id, dataset_id = api.video.get_destination_ids(video_id) # 124974 466639
        """
        dataset_id = self._api.video.get_info_by_id(id).dataset_id
        project_id = self._api.dataset.get_info_by_id(dataset_id).project_id
        return project_id, dataset_id

    def upload_hash(
        self, dataset_id: int, name: str, hash: str, stream_index: Optional[int] = None
    ) -> VideoInfo:
        """
        Upload Video from given hash to Dataset.

        :param dataset_id: Dataset ID in Supervisely.
        :type dataset_id: int
        :param name: Video name.
        :type name: str
        :param hash: Video hash.
        :type hash: str
        :param stream_index: Index of video stream.
        :type stream_index: int, optional
        :return: Information about Video. See :class:`info_sequence<info_sequence>`
        :rtype: :class:`VideoInfo`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            dst_dataset_id = 466654
            src_video_id = 186580617
            video_info = api.video.get_info_by_id(src_video_id)
            hash = video_info.hash
            # It is necessary to upload video with the same extention as in src dataset
            name = video_info.name
            new_video_info = api.video.upload_hash(dst_dataset_id, name, hash)
            print(new_video_info)
            # Output: [
            #     198723201,
            #     "MOT16-03.mp4",
            #     "ob69way77JS/qKdKHfOI/d0oZNdabAlG4m+c7YHtGnM=",
            #     null,
            #     null,
            #     null,
            #     466654,
            #     "videos/P/H/Pd/rOz7a5usR...9Z91YT.mp4",
            #     [
            #         0,
            #         0.066667,
            #         0.133333,
            #           ...
            #         99.866667,
            #         99.933333
            #     ],
            #     1500,
            #     960,
            #     540,
            #     "2021-03-23T15:51:40.595Z",
            #     "2021-03-23T15:51:40.595Z"
            # ]
        """
        meta = {}
        if stream_index is not None and type(stream_index) is int:
            meta = {"videoStreamIndex": stream_index}
        return self.upload_hashes(dataset_id, [name], [hash], [meta])[0]

    def upload_hashes(
        self,
        dataset_id: int,
        names: List[str],
        hashes: List[str],
        metas: Optional[List[Dict]] = None,
        progress_cb: Optional[Callable] = None,
    ) -> List[VideoInfo]:
        """
        Upload Videos from given hashes to Dataset.

        :param dataset_id: Dataset ID in Supervisely.
        :type dataset_id: int
        :param names: Videos names.
        :type names: List[str]
        :param hashes: Videos hashes.
        :type hashes: List[str]
        :param metas: Videos metadata.
        :type metas: List[dict], optional
        :param progress_cb: Function for tracking download progress.
        :type progress_cb: Progress, optional
        :return: List with information about Videos. See :class:`info_sequence<info_sequence>`
        :rtype: :class:`List[VideoInfo]`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            src_dataset_id = 466639
            dst_dataset_id = 468620

            hashes = []
            names = []
            metas = []
            video_infos = api.video.get_list(src_dataset_id)
            # Create lists of hashes, videos names and meta information for each video
            for video_info in video_infos:
                hashes.append(video_info.hash)
                # It is necessary to upload videos with the same names(extentions) as in src dataset
                names.append(video_info.name)
                metas.append({video_info.name: video_info.frame_height})

            progress = sly.Progress("Videos upload: ", len(hashes))
            new_videos_info = api.video.upload_hashes(dst_dataset_id, names, hashes, metas, progress.iters_done_report)
            # Output:
            # {"message": "progress", "event_type": "EventType.PROGRESS", "subtask": "Videos upload: ", "current": 0, "total": 2, "timestamp": "2021-03-24T10:18:57.111Z", "level": "info"}
            # {"message": "progress", "event_type": "EventType.PROGRESS", "subtask": "Videos upload: ", "current": 2, "total": 2, "timestamp": "2021-03-24T10:18:57.304Z", "level": "info"}
        """
        results = self._upload_bulk_add(
            lambda item: (ApiField.HASH, item), dataset_id, names, hashes, metas, progress_cb
        )
        return results

    def _upload_bulk_add(
        self, func_item_to_kv, dataset_id, names, items, metas=None, progress_cb=None
    ):
        if metas is None:
            metas = [{}] * len(items)

        results = []
        if len(names) == 0:
            return results
        if len(names) != len(items):
            raise RuntimeError('Can not match "names" and "items" lists, len(names) != len(items)')

        for name in names:
            validate_ext(os.path.splitext(name)[1])

        for batch in batched(list(zip(names, items, metas))):
            images = []
            for name, item, meta in batch:
                item_tuple = func_item_to_kv(item)
                images.append(
                    {
                        "title": name,
                        item_tuple[0]: item_tuple[1],
                        ApiField.META: meta if meta is not None else {},
                    }
                )
            response = self._api.post(
                "videos.bulk.add", {ApiField.DATASET_ID: dataset_id, ApiField.VIDEOS: images}
            )
            if progress_cb is not None:
                progress_cb(len(images))

            results = [self._convert_json_info(item) for item in response.json()]
            name_to_res = {img_info.name: img_info for img_info in results}
            ordered_results = [name_to_res[name] for name in names]

            return ordered_results

    def _download(self, id, is_stream=False):
        """
        :param id: int
        :param is_stream: bool
        :return: Response object containing video with given id
        """
        response = self._api.post("videos.download", {ApiField.ID: id}, stream=is_stream)
        return response

    def download_path(self, id: int, path: str, progress_cb: Optional[Callable] = None) -> None:
        """
        Downloads Video from Dataset to local path by ID.

        :param id: Video ID in Supervisely.
        :type id: int
        :param path: Local save path for Video.
        :type path: str
        :param progress_cb: Function to check progress.
        :type progress_cb: Function, optional
        :return: None
        :rtype: :class:`NoneType`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            video_info = api.video.get_info_by_id(770918)
            save_path = os.path.join("/home/admin/work/projects/videos/", video_info.name)

            api.video.download_path(770918, save_path)
        """
        response = self._download(id, is_stream=True)
        ensure_base_path(path)

        with open(path, "wb") as fd:
            mb1 = 1024 * 1024
            for chunk in response.iter_content(chunk_size=mb1):
                fd.write(chunk)

                if progress_cb is not None:
                    progress_cb(len(chunk))

    def download_range_by_id(
        self, id: int, frame_start: int, frame_end: int, is_stream: Optional[bool] = True
    ) -> Response:
        """
        Downloads Video with given ID between given start and end frames.

        :param id: Video ID in Supervisely.
        :type id: int
        :param frame_start: Start frame for video download.
        :type frame_start: int
        :param frame_end: End frame for video download.
        :type frame_end: int
        :param is_stream: Use stream for video download or not.
        :type is_stream: bool, optional
        :return: Response object
        :rtype: :class:`Response`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            video_id = 198835945
            start_fr = 5
            end_fr= 35
            response = api.video.download_range_by_id(video_id, start_fr, end_fr)
        """
        raise NotImplementedError("Method is not supported")
        # path_original = self.get_info_by_id(id).path_original
        # return self.download_range_by_path(path_original, frame_start, frame_end, is_stream)

    def download_range_by_path(
        self,
        path_original: str,
        frame_start: int,
        frame_end: int,
        is_stream: Optional[bool] = False,
    ) -> Response:
        """
        Downloads Video with given path in Supervisely between given start and end frames.

        :param path_original: Path to Video in Supervisely.
        :type path_original: str
        :param frame_start: Start frame for video download.
        :type frame_start: int
        :param frame_end: End frame for video download.
        :type frame_end: int
        :param is_stream: Use stream for video download or not.
        :type is_stream: bool, optional
        :return: Response object
        :rtype: :class:`Response`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            video_id = 198835945
            start_fr = 5
            end_fr= 35
            video_info = api.video.get_info_by_id(video_id)
            path_sl = video_info.path_original
            response = api.video.download_range_by_path(path_sl, start_fr, end_fr)
        """
        raise NotImplementedError("Method is not supported")
        # response = self._api.get(
        #     method="image-converter/transcode" + path_original,
        #     params={"startFrame": frame_start, "endFrame": frame_end, "transmux": True},
        #     stream=is_stream,
        #     use_public_api=False,
        # )
        # return response

    def download_save_range(
        self, video_id: int, frame_start: int, frame_end: int, save_path: str
    ) -> str:
        """
        Download video with given ID in Supervisely between given start and end frames on the given path.

        :param video_id: Video ID in Supervisely.
        :type video_id: int
        :param frame_start: Start frame for video download.
        :type frame_start: int
        :param frame_end: End frame for video download.
        :type frame_end: int
        :param save_path: Path to save video.
        :type save_path: str
        :return: Full path to saved video
        :rtype: :class:`str`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()
            video_id = 198835945
            start_fr = 5
            end_fr= 35
            video_info = api.video.get_info_by_id(video_id)
            name = video_info.name
            save_path = os.path.join('/home/admin/work/projects/videos', name)
            result = api.video.download_save_range(video_id, start_fr, end_fr, save_path)
            print(result)
            # Output: /home/admin/work/projects/videos/MOT16-03.mp4
        """
        raise NotImplementedError("Method is not supported")
        # response = self.download_range_by_id(video_id, frame_start, frame_end)
        # with open(save_path, "wb") as fd:
        #     for chunk in response.iter_content(chunk_size=128):
        #         fd.write(chunk)
        # return save_path

    def notify_progress(
        self,
        track_id: int,
        video_id: int,
        frame_start: int,
        frame_end: int,
        current: int,
        total: int,
    ):
        """
        :param track_id: int
        :param video_id: int
        :param frame_start: int
        :param frame_end: int
        :param current: int
        :param total: int
        :return: str
        """
        response = self._api.post(
            "videos.notify-annotation-tool",
            {
                "type": "videos:fetch-figures-in-range",
                "data": {
                    ApiField.TRACK_ID: track_id,
                    ApiField.VIDEO_ID: video_id,
                    ApiField.FRAME_RANGE: [frame_start, frame_end],
                    ApiField.PROGRESS: {ApiField.CURRENT: current, ApiField.TOTAL: total},
                },
            },
        )
        return response.json()[ApiField.STOPPED]

    def notify_tracking_error(self, track_id: int, error: str, message: str):
        """
        :param track_id: int
        :param error: str
        :param message: str
        :return: #TODO nothing to return
        """
        response = self._api.post(
            "videos.notify-annotation-tool",
            {
                "type": "videos:tracking-error",
                "data": {
                    ApiField.TRACK_ID: track_id,
                    ApiField.ERROR: {ApiField.MESSAGE: "{}: {}".format(error, message)},
                },
            },
        )

    # def upload(self):
    #     #"/videos.bulk.upload"
    #     pass
    #
    # def upload_path(self, dataset_id, name, path, meta=None):
    #     metas = None if meta is None else [meta]
    #     return self.upload_paths(dataset_id, [name], [path], metas=metas)[0]

    # @TODO: copypaste from image_api
    def check_existing_hashes(self, hashes: List[str]) -> List[str]:
        """
        Checks existing hashes for Videos.

        :param hashes: List of hashes.
        :type hashes: List[str]
        :return: List of existing hashes
        :rtype: :class:`List[str]`
        :Usage example: Checkout detailed example `here <https://app.supervise.ly/explore/notebooks/guide-10-check-existing-images-and-upload-only-the-new-ones-1545/overview>`_ (you must be logged into your Supervisely account)

         .. code-block:: python

            import supervisely as sly

            # Helpful method when your uploading was interrupted
            # You can check what videos has been successfully uploaded by their hashes and what not
            # And continue uploading the rest of the videos from that point

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            # Find project
            project = api.project.get_info_by_id(WORKSPACE_ID, PROJECT_ID)

            # Get paths of all videos in a directory
            videos_paths = sly.fs.list_files('videos_to_upload')

            #Calculate hashes for all videos paths
            hash_to_video = {}
            videos_hashes = []

            for idx, item in enumerate(videos_paths):
                item_hash = sly.fs.get_file_hash(item)
                videos_hashes.append(item_hash)
                hash_to_video[item_hash] = item

            # Get hashes that are already on server
            remote_hashes = api.video.check_existing_hashes(videos_hashes)
            already_uploaded_videos = {hh: hash_to_video[hh] for hh in remote_hashes}
        """
        results = []
        if len(hashes) == 0:
            return results
        for hashes_batch in batched(hashes, batch_size=900):
            response = self._api.post("images.internal.hashes.list", hashes_batch)
            results.extend(response.json())
        return results

    def upload_paths(
        self,
        dataset_id: int,
        names: List[str],
        paths: List[str],
        progress_cb: Optional[Callable] = None,
        metas: Optional[List[Dict]] = None,
        infos=None,
        item_progress=None,
    ) -> List[VideoInfo]:
        """
        Uploads Videos with given names from given local paths to Dataset.

        :param dataset_id: Dataset ID in Supervisely.
        :type dataset_id: int
        :param names: List of Videos names.
        :type names: List[str]
        :param paths: List of local Videos paths.
        :type paths: List[str]
        :param progress_cb: Function for tracking download progress.
        :type progress_cb: Progress, optional
        :param metas: Videos metadata.
        :type metas: List[dict], optional
        :raises: :class:`RuntimeError` if len(names) != len(paths)
        :return: List with information about Videos. See :class:`info_sequence<info_sequence>`
        :rtype: :class:`List[VideoInfo]`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            video_names = ["7777.mp4", "8888.mp4", "9999.mp4"]
            video_paths = ["/home/admin/Downloads/video/770918.mp4", "/home/admin/Downloads/video/770919.mp4", "/home/admin/Downloads/video/770920.mp4"]

            video_infos = api.video.upload_path(dataset_id, names=video_names, paths=video_paths)
        """

        def path_to_bytes_stream(path):
            return open(path, "rb")

        update_headers = False
        if infos is not None:
            if len(infos) != len(names):
                raise ValueError("Infos have to be None or provided for all videos")
            update_headers = True

        if update_headers:
            self._api.add_header("x-skip-processing", "true")

        video_info_results = []
        hashes = [get_file_hash(x) for x in paths]

        self._upload_data_bulk(
            path_to_bytes_stream,
            zip(paths, hashes),
            progress_cb=progress_cb,
            item_progress=item_progress,
        )
        if update_headers:
            self.upsert_infos(hashes, infos)
            self._api.pop_header("x-skip-processing")
        metas = self._api.import_storage.get_meta_by_hashes(hashes)
        metas2 = [meta["meta"] for meta in metas]

        for name, hash, meta in zip(names, hashes, metas2):
            try:
                all_streams = meta["streams"]
                video_streams = get_video_streams(all_streams)
                for stream_info in video_streams:
                    stream_index = stream_info["index"]

                    # TODO: check is community
                    # if instance_type == sly.COMMUNITY:
                    #     if _check_video_requires_processing(file_info, stream_info) is True:
                    #         warn_video_requires_processing(file_name)
                    #         continue

                    item_name = name
                    info = self._api.video.get_info_by_name(dataset_id, item_name)
                    if info is not None:
                        item_name = gen_video_stream_name(name, stream_index)
                    res = self.upload_hash(dataset_id, item_name, hash, stream_index)
                    video_info_results.append(res)
            except Exception as e:
                logger.warning(
                    "File skipped {!r}: error occurred during processing {!r}".format(name, str(e))
                )
        return video_info_results

    def upload_path(
        self,
        dataset_id: int,
        name: str,
        path: str,
        meta: Dict = None,
        item_progress: Optional[Progress] = None,
    ) -> VideoInfo:
        progress_cb = item_progress
        p = None
        if item_progress is not None and type(item_progress) is bool:
            p = Progress(f"Uploading {name}", total_cnt=get_file_size(path), is_size=True)
            progress_cb = p.iters_done_report

        results = self.upload_paths(
            dataset_id=dataset_id,
            names=[name],
            paths=[path],
            progress_cb=None,
            metas=[meta],
            infos=None,
            item_progress=progress_cb,
        )
        if type(item_progress) is bool:
            p.set_current_value(value=p.total, report=True)

        return results[0]

    def _upload_uniq_videos_single_req(
        self, func_item_to_byte_stream, hashes_items_to_upload, progress_cb=None
    ):
        content_dict = {}
        for idx, (_, item) in enumerate(hashes_items_to_upload):
            content_dict["{}-file".format(idx)] = (
                str(idx),
                func_item_to_byte_stream(item),
                "video/*",
            )
        encoder = MultipartEncoder(fields=content_dict)

        if progress_cb is not None:

            def _callback(monitor, progress):
                progress(monitor.bytes_read)

            callback = partial(_callback, progress=progress_cb)
            monitor = MultipartEncoderMonitor(encoder, callback)
            resp = self._api.post("videos.bulk.upload", monitor)
        else:
            resp = self._api.post("videos.bulk.upload", encoder)

        resp_list = json.loads(resp.text)
        remote_hashes = [d["hash"] for d in resp_list if "hash" in d]
        if len(remote_hashes) != len(hashes_items_to_upload):
            problem_items = [
                (hsh, item, resp["errors"])
                for (hsh, item), resp in zip(hashes_items_to_upload, resp_list)
                if resp.get("errors")
            ]
            logger.warn(
                "Not all images were uploaded within request.",
                extra={
                    "total_cnt": len(hashes_items_to_upload),
                    "ok_cnt": len(remote_hashes),
                    "items": problem_items,
                },
            )
        return remote_hashes

    def _upload_data_bulk(
        self,
        func_item_to_byte_stream,
        items_hashes,
        retry_cnt=3,
        progress_cb=None,
        item_progress=None,
    ):
        hash_to_items = {i_hash: item for item, i_hash in items_hashes}

        unique_hashes = set(hash_to_items.keys())
        remote_hashes = set(
            self.check_existing_hashes(list(unique_hashes))
        )  # existing -- from server
        if progress_cb:
            progress_cb(len(remote_hashes))
        # pending_hashes = unique_hashes #- remote_hashes #@TODO: only fo debug!
        pending_hashes = unique_hashes - remote_hashes

        for retry_idx in range(retry_cnt):
            # single attempt to upload all data which is not uploaded yet
            for hashes in batched(list(pending_hashes)):
                pending_hashes_items = [(h, hash_to_items[h]) for h in hashes]
                hashes_rcv = self._upload_uniq_videos_single_req(
                    func_item_to_byte_stream, pending_hashes_items, item_progress
                )
                pending_hashes -= set(hashes_rcv)
                if set(hashes_rcv) - set(hashes):
                    logger.warn(
                        "Hash inconsistency in images bulk upload.",
                        extra={"sent": hashes, "received": hashes_rcv},
                    )
                if progress_cb:
                    progress_cb(len(hashes_rcv))

            if not pending_hashes:
                return

            logger.warn(
                "Unable to upload videos (data).",
                extra={
                    "retry_idx": retry_idx,
                    "items": [(h, hash_to_items[h]) for h in pending_hashes],
                },
            )
            # now retry it for the case if it is a shadow server/connection error

        raise RuntimeError(
            "Unable to upload videos (data). "
            "Please check if videos are in supported format and if ones aren't corrupted."
        )

    # @TODO: add case to documentation with detailed explanation
    def upsert_info(self, hash: str, info: Dict) -> Dict:
        return self.upsert_infos([hash], [info])

    def upsert_infos(
        self, hashes: List[str], infos: List[Dict], links: Optional[List[str]] = None
    ) -> Dict:
        payload = []
        if links is None:
            links = [None] * len(hashes)
        for h, l, info in zip(hashes, links, infos):
            item = {ApiField.HASH: h, ApiField.META: info}
            if l is not None:
                item[ApiField.LINK] = l
            payload.append(item)

        resp = self._api.post("videos.bulk.upsert_file_meta", payload)
        return resp.json()

    def upload_links(
        self,
        dataset_id: int,
        names: List[str],
        hashes: List[str],
        links: List[str],
        infos: List[Dict],
        metas: Optional[List[Dict]] = None,
    ) -> List[VideoInfo]:
        if infos is not None:
            self.upsert_infos(hashes, infos, links)
        return self._upload_bulk_add(
            lambda item: (ApiField.LINK, item), dataset_id, names, links, metas
        )

    def update_custom_data(self, id: int, data: dict):
        resp = self._api.post(
            "videos.custom-data.set", {ApiField.ID: id, ApiField.CUSTOM_DATA: data}
        )
        return resp.json()

    def upload_link(
        self,
        dataset_id: int,
        link: str,
        name: Optional[str] = None,
        meta: Optional[List[Dict]] = None,
    ):
        if name is None:
            name = rand_str(10) + get_file_ext(link)
        local_path = os.path.join(os.getcwd(), name)

        try:
            sly_fs.download(link, local_path)
            video_info = get_info(local_path)
            h = get_file_hash(local_path)
            sly_fs.silent_remove(local_path)
        except Exception as e:
            sly_fs.silent_remove(local_path)
            raise e
        name = self.get_free_name(dataset_id, name)
        return self.upload_links(
            dataset_id, names=[name], hashes=[h], links=[link], infos=[video_info], metas=[meta]
        )

    def add_existing(
        self,
        dataset_id: int,
        video_info: VideoInfo,
        name: str,
    ) -> VideoInfo:
        if video_info.link is not None:
            return self.upload_links(
                dataset_id,
                names=[name],
                hashes=[video_info.hash],
                links=[video_info.link],
                infos=None,
            )[0]
        else:
            return self.upload_hash(dataset_id, name, video_info.hash)

    def _remove_batch_api_method_name(self):
        """ """
        return "images.bulk.remove"

    def _remove_batch_field_name(self):
        """ """
        return ApiField.IMAGE_IDS

    def remove_batch(
        self,
        ids: List[int],
        progress_cb: Optional[Callable] = None,
        batch_size: Optional[int] = 50,
    ):
        """
        Remove videos from supervisely by ids.

        :param ids: List of Videos IDs in Supervisely.
        :type ids: List[int]
        :param progress_cb: Function for tracking progress of removing.
        :type progress_cb: Progress, optional
        :return: :class:`None<None>`
        :rtype: :class:`NoneType<NoneType>`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            video_ids = [2389126, 2389127]
            api.video.remove_batch(video_ids)
        """
        super(VideoApi, self).remove_batch(ids, progress_cb=progress_cb, batch_size=batch_size)

    def remove(self, video_id: int):
        """
        Remove video from supervisely by id.

        :param video_id: Videos ID in Supervisely.
        :type video_id: int
        :return: :class:`None<None>`
        :rtype: :class:`NoneType<NoneType>`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            video_id = 2389126
            api.video.remove(video_id)
        """
        super(VideoApi, self).remove(video_id)

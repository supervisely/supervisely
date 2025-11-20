# coding: utf-8
from __future__ import annotations

import asyncio
import datetime
import json
import os
import re
import urllib.parse
from functools import partial
from typing import (
    AsyncGenerator,
    Callable,
    Dict,
    Iterator,
    List,
    Literal,
    NamedTuple,
    Optional,
    Tuple,
    Union,
)

import aiofiles
from numerize.numerize import numerize
from requests import Response
from requests_toolbelt import (
    MultipartDecoder,
    MultipartEncoder,
    MultipartEncoderMonitor,
)
from tqdm import tqdm

import supervisely.io.fs as sly_fs
from supervisely._utils import (
    abs_url,
    batched,
    generate_free_name,
    is_development,
    rand_str,
)
from supervisely.api.module_api import ApiField, RemoveableBulkModuleApi
from supervisely.api.video.video_annotation_api import VideoAnnotationAPI
from supervisely.api.video.video_figure_api import VideoFigureApi
from supervisely.api.video.video_frame_api import VideoFrameAPI
from supervisely.api.video.video_object_api import VideoObjectApi
from supervisely.api.video.video_tag_api import VideoTagApi
from supervisely.io.fs import (
    ensure_base_path,
    get_file_ext,
    get_file_hash,
    get_file_hash_async,
    get_file_hash_chunked,
    get_file_name_with_ext,
    get_file_size,
    list_files,
    list_files_recursively,
)
from supervisely.sly_logger import logger
from supervisely.task.progress import Progress
from supervisely.video.video import (
    gen_video_stream_name,
    get_info,
    get_video_streams,
    is_valid_format,
    validate_ext,
)


class VideoInfo(NamedTuple):
    """
    Object with :class:`Video<supervisely.video.video>` parameters from Supervisely.

    :Example:

     .. code-block:: python

        VideoInfo(
            id=19371139,
            name='Videos_dataset_animals_sea_lion.mp4'
            hash='30/TQ1BcIOn1AI4RFgRO/6psRtr3lqNPmr4uQ=',
            link=None,
            team_id=435,
            workspace_id=684,
            project_id=17208,
            dataset_id=55846,
            path_original='/h5un6l2bnaz1vj8a9qgms4-public/videos/Z/d/HD/lfgipl...NXrg5vz.mp4',
            frames_to_timecodes=[],
            frames_count=245,
            frame_width=1920,
            frame_height=1080,
            created_at='2023-02-07T19:35:01.808Z',
            updated_at='2023-02-07T19:35:01.808Z',
            tags=[],
            file_meta={
                'codecName': 'h264',
                'codecType': 'video',
                'duration': 10.218542,
                'formatName': 'mov,mp4,m4a,3gp,3g2,mj2',
                'framesCount': 245,
                'framesToTimecodes': [],
                'height': 1080,
                'index': 0,
                'mime': 'video/mp4',
                'rotation': 0,
                'size': '6795452',
                'startTime': 0,
                'streams': [],
                'width': 1920
            },
            custom_data={},
            processing_path='1/194'
        )
    """

    #: :class:`int`: Video ID in Supervisely.
    id: int

    #: :class:`str`: Video filename.
    name: str

    #: :class:`str`: Video hash obtained by base64(sha256(file_content)).
    #: Use hash for files that are expected to be stored at Supervisely or your deployed agent.
    hash: str

    #: :class:`str`: Link to video.
    link: str

    #: :class:`int`: :class:`TeamApi<supervisely.api.team_api.TeamApi>` ID in Supervisely.
    team_id: int

    #: :class:`int`: :class:`WorkspaceApi<supervisely.api.workspace_api.WorkspaceApi>` ID in Supervisely.
    workspace_id: int

    #: :class:`int`: :class:`Project<supervisely.project.project.Project>` ID in Supervisely.
    project_id: int

    #: :class:`int`: :class:`Dataset<supervisely.project.project.Dataset>` ID in Supervisely.
    dataset_id: int

    #: :class:`str`: Relative storage URL to video. e.g.
    #: "/h5un6l2bnaz1vms4-public/videos/Z/d/HD/lfgipl...NXrg5vz.mp4".
    path_original: str

    #: :class: `list`: A list of timecodes in the format "SS.nnn" corresponding to each frame.
    frames_to_timecodes: list

    #: :class: `int`: Number of frames in the video
    frames_count: int

    #: :class:`int`: Video frames width in pixels.
    frame_width: int

    #: :class:`int`: Video frames height in pixels.
    frame_height: int

    #: :class:`str`: Video creation time. e.g. "2019-02-22T14:59:53.381Z".
    created_at: str

    #: :class:`str`: Time of last video update. e.g. "2019-02-22T14:59:53.381Z".
    updated_at: str

    #: :class:`list`: Video :class:`VideoTag<supervisely.video_annotation.video_tag.VideoTag>` list.
    #: e.g. "[{'entityId': 19371139, 'tagId': 377141, 'id': 12241539, 'labelerLogin': 'admin',
    #: 'createdAt': '2023-02-07T19:35:01.808Z', 'updatedAt': '2023-02-07T19:35:01.808Z',
    #: 'frameRange': [244, 244]}, {...}]".
    tags: list

    #: :class:`dict`: A dictionary containing metadata about the video file.
    file_meta: dict

    #: :class:`dict`: Video object meta information.
    meta: dict

    #: :class:`dict`: A dictionary containing custom data associated with the video.
    custom_data: dict

    #: :class:`str`: Path to the video file on the server.
    processing_path: str

    @property
    def duration(self) -> float:
        """
        Duration of the video in seconds.

        :return: Duration of the video in seconds.
        :rtype: :class:`float`
        """

        ndigits = 0
        return round(self.file_meta.get("duration"), ndigits=ndigits)

    @property
    def duration_hms(self) -> str:
        """
        Duration of the video in "HH:MM:SS.nnn" format.

        :return: Duration of the video in "HH:MM:SS.nnn" format.
        :rtype: :class:`str`
        """

        return str(datetime.timedelta(seconds=self.duration))

    @property
    def frames_count_compact(self) -> str:
        """
        String representation of the number of frames in the video. Used for converting large numbers into readable strings.

        :return: Number of frames in the video represented in string format.
        :rtype: :class:`str`
        """

        return numerize(self.frames_count)

    @property
    def image_preview_url(self) -> str:
        """
        URL to an image preview of the video.

        :return: URL to an image preview of the video.
        :rtype: :class:`str`
        """

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

        import os
        from dotenv import load_dotenv

        import supervisely as sly

        # Load secrets and create API object from .env file (recommended)
        # Learn more here: https://developer.supervisely.com/getting-started/basics-of-authentication
        if sly.is_development():
            load_dotenv(os.path.expanduser("~/supervisely.env"))
        api = sly.Api.from_env()

        # Pass values into the API constructor (optional, not recommended)
        # api = sly.Api(server_address="https://app.supervisely.com", token="4r47N...xaTatb")

        video_id = 19371139
        video_info = api.video.get_info_by_id(video_id) # api usage example
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
        Get list of all :class:`VideoInfo<VideoInfo>` field names.

        :return: List of :class:`VideoInfo<VideoInfo>` field names.`
        :rtype: :class:`list`
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
            ApiField.META,
            ApiField.CUSTOM_DATA,
            ApiField.PROCESSING_PATH,
        ]

    @staticmethod
    def info_tuple_name():
        """
        Get string name of :class:`VideoInfo<VideoInfo>` NamedTuple.

        :return: NamedTuple name.
        :rtype: :class:`str`
        """

        return "VideoInfo"

    def url(self, dataset_id: int, video_id: int, video_frame: Optional[int] = 0) -> str:
        """
        Get url of the video by dataset ID and video ID

        :param dataset_id: :class:`Dataset<supervisely.project.project.Dataset>` ID in which the Video is located.
        :type dataset_id: :class:`int`
        :param video_id: Video ID in Supervisely.
        :type video_id: :class:`int`
        :param video_frame: Video frame index.
        :type video_frame: :class:`int`, optional
        :return: Url of the video by dataset_id and video_id.
        :rtype: :class:`str`
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
        """Private method. Convert video information from json to VideoInfo<VideoInfo>"""

        res = super(VideoApi, self)._convert_json_info(info, skip_missing=skip_missing)
        # processing_path = info.get("processingPath", "")
        d = res._asdict()
        # d["processing_path"] = processing_path
        return VideoInfo(**d)

    def get_list(
        self,
        dataset_id: int,
        filters: Optional[List[Dict[str, str]]] = None,
        raw_video_meta: Optional[bool] = False,
        fields: Optional[List[str]] = None,
        force_metadata_for_links: Optional[bool] = False,
    ) -> List[VideoInfo]:
        """
        Get list of information about all videos for a given dataset ID.

        :param dataset_id: :class:`Dataset<supervisely.project.project.Dataset>` ID in Supervisely.
        :type dataset_id: int
        :param filters: List of parameters to sort output Videos. See: https://api.docs.supervisely.com/#tag/Videos/paths/~1videos.list/get
        :type filters: List[Dict[str, str]], optional
        :param raw_video_meta: Get normalized metadata from server if False.
        :type raw_video_meta: bool
        :param fields: List of fields to return.
        :type fields: List[str], optional
        :param force_metadata_for_links: Specify whether to force retrieving video metadata from the server.
        :type force_metadata_for_links: Optional[bool]
        :return: List of information about videos in given dataset.
        :rtype: :class:`List[VideoInfo]`

        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            dataset_id = 55846

            video_infos = api.video.get_list(dataset_id)
            print(video_infos)
            # Output: [VideoInfo(...), VideoInfo(...)]

            filtered_video_infos = api.video.get_list(dataset_id, filters=[{'field': 'id', 'operator': '=', 'value': '19371139'}])
            print(filtered_video_infos)
            # Output: [VideoInfo(id=19371139, ...)]
        """
        data = {
            ApiField.DATASET_ID: dataset_id,
            ApiField.FILTER: filters or [],
            ApiField.RAW_VIDEO_META: raw_video_meta,
            ApiField.FORCE_METADATA_FOR_LINKS: force_metadata_for_links,
        }
        if fields is not None:
            data[ApiField.FIELDS] = fields
        return self.get_list_all_pages("videos.list", data)

    def get_list_generator(
        self,
        dataset_id: int,
        filters: Optional[List[Dict[str, str]]] = None,
        sort: Optional[str] = "id",
        sort_order: Optional[str] = "asc",
        limit: Optional[int] = None,
        raw_video_meta: Optional[bool] = False,
        batch_size: Optional[int] = None,
        force_metadata_for_links: Optional[bool] = False,
    ) -> Iterator[List[VideoInfo]]:
        data = {
            ApiField.DATASET_ID: dataset_id,
            ApiField.FILTER: filters or [],
            ApiField.SORT: sort,
            ApiField.SORT_ORDER: sort_order,
            ApiField.RAW_VIDEO_META: raw_video_meta,
            ApiField.PAGINATION_MODE: ApiField.TOKEN,
            ApiField.FORCE_METADATA_FOR_LINKS: force_metadata_for_links,
        }
        if batch_size is not None:
            data[ApiField.PER_PAGE] = batch_size
        else:
            # use default value on instance (20k)
            # #tag/Images/paths/~1images.list/get
            pass
        return self.get_list_all_pages_generator(
            "videos.list",
            data,
            limit=limit,
            return_first_response=False,
        )

    def get_info_by_id(
        self,
        id: int,
        raise_error: Optional[bool] = False,
        force_metadata_for_links=True,
    ) -> VideoInfo:
        """
        Get Video information by ID in VideoInfo<VideoInfo> format.

        :param id: Video ID in Supervisely.
        :type id: int
        :param raise_error: Return an error if the video info was not received.
        :type raise_error: bool
        :param force_metadata_for_links: Specify whether to force retrieving video metadata from the server.
        :type force_metadata_for_links: bool
        :return: Information about Video. See :class:`info_sequence<info_sequence>`
        :rtype: :class:`VideoInfo`

        :Usage example:

         .. code-block:: python

            import supervisely as sly


            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            video_id = 198702499
            video_info = api.video.get_info_by_id(video_id)
            print(video_info)
            # Output:
            # VideoInfo(
            #     id=198702499,
            #     name='Videos_dataset_animals_sea_lion.mp4'
            #     hash='30/TQ1BcIOn1AI4RFgRO/6psRtr3lqNPmr4uQ=',
            #     link=None,
            #     team_id=435,
            #     workspace_id=684,
            #     project_id=17208,
            #     dataset_id=55846,
            #     path_original='/h5s-public/videos/Z/d/HD/lfgNXrg5vz.mp4',
            #     frames_to_timecodes=[],
            #     frames_count=245,
            #     frame_width=1920,
            #     frame_height=1080,
            #     created_at='2023-02-07T19:35:01.808Z',
            #     updated_at='2023-02-07T19:35:01.808Z',
            #     tags=[],
            #     file_meta={
            #         'codecName': 'h264',
            #         'codecType': 'video',
            #         'duration': 10.218542,
            #         'formatName': 'mov,mp4,m4a,3gp,3g2,mj2',
            #         'framesCount': 245,
            #         'framesToTimecodes': [],
            #         'height': 1080,
            #         'index': 0,
            #         'mime': 'video/mp4',
            #         'rotation': 0,
            #         'size': '6795452',
            #         'startTime': 0,
            #         'streams': [],
            #         'width': 1920
            #     },
            #     custom_data={},
            #     processing_path='1/194'
            # )
        """

        info = self._get_info_by_id(
            id,
            "videos.info",
            fields={ApiField.FORCE_METADATA_FOR_LINKS: force_metadata_for_links},
        )
        if info is None and raise_error is True:
            raise KeyError(f"Video with id={id} not found in your account")
        return info

    def get_info_by_id_batch(
        self,
        ids: List[int],
        progress_cb: Optional[Union[tqdm, Callable]] = None,
        force_metadata_for_links: Optional[bool] = False,
    ) -> List[VideoInfo]:
        """
        Get Video information by ID.

        :param ids: List of Video IDs in Supervisely, they must belong to the same dataset.
        :type ids: List[int]
        :param progress_cb: Function for tracking download progress.
        :type progress_cb: Optional[Union[tqdm, Callable]]
        :param force_metadata_for_links: Specify whether to force retrieving video metadata from the server.
        :type force_metadata_for_links: bool
        :return: List of information about Videos. See :class:`info_sequence<info_sequence>`.
        :rtype: List[VideoInfo]
        :Usage example:

         .. code-block:: python

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            video_ids = [376728, 376729, 376730, 376731, 376732, 376733]

            video_infos = api.video.get_info_by_id_batch(video_ids)
        """
        results = []
        if len(ids) == 0:
            return results

        fields = [
            "id",
            "title",
            "description",
            "createdAt",
            "updatedAt",
            "dataId",
            "remoteDataId",
            "meta",
            "pathOriginal",
            "hash",
            "groupId",
            "projectId",
            "datasetId",
            "createdBy",
            "customData",
        ]

        if force_metadata_for_links is True:
            fields.append("videoMeta")

        dataset_id = self.get_info_by_id(ids[0]).dataset_id
        for batch in batched(ids):
            filters = [{"field": ApiField.ID, "operator": "in", "value": batch}]
            results.extend(
                self.get_list_all_pages(
                    "videos.list",
                    {
                        ApiField.DATASET_ID: dataset_id,
                        ApiField.FILTER: filters,
                        ApiField.FIELDS: fields,
                        ApiField.FORCE_METADATA_FOR_LINKS: force_metadata_for_links,
                    },
                )
            )
            if progress_cb is not None:
                progress_cb(len(batch))
        temp_map = {info.id: info for info in results}
        ordered_results = [temp_map[id] for id in ids]
        return ordered_results

    def get_json_info_by_id(
        self,
        id: int,
        raise_error: Optional[bool] = False,
        force_metadata_for_links: Optional[bool] = True,
    ) -> Dict:
        """
        Get Video information by ID in json format.

        :param id: Video ID in Supervisely.
        :type id: int
        :param raise_error: Return an error if the video info was not received.
        :type raise_error: bool
        :param force_metadata_for_links: Specify whether to force retrieving video metadata from the server.
        :type force_metadata_for_links: bool
        :return: Information about Video. See :class:`info_sequence<info_sequence>`
        :rtype: dict

        :Usage example:

         .. code-block:: python

            import supervisely as sly


            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            video_id = 19371139
            video_info = api.video.get_info_by_id(video_id)
            print(video_info)
            # Output:
            # {
            #     'createdAt': '2023-02-07T19:35:01.808Z',
            #     'customData': {},
            #     'datasetId': 55846,
            #     'description': '',
            #     'fileMeta': {
            #         'codecName': 'h264',
            #         'codecType': 'video',
            #         'duration': 10.218542,
            #         'formatName': 'mov,mp4,m4a,3gp,3g2,mj2',
            #         'framesCount': 245,
            #         'framesToTimecodes': [],
            #         'height': 1080,
            #         'index': 0,
            #         'mime': 'video/mp4',
            #         'rotation': 0,
            #         'size': '6795452',
            #         'startTime': 0,
            #         'streams': [],
            #         'width': 1920
            #     },
            #     'fullStorageUrl': 'https://app.supervisely.com/h..i35vz.mp4',
            #     'hash': '30/TQ1BcIOn1ykA2psRtr3lq3HF6NPmr4uQ=',
            #     'id': 19371139,
            #     'link': None,
            #     'meta': {'videoStreamIndex': 0},
            #     'name': 'Videos_dataset_animals_sea_lion.mp4',
            #     'pathOriginal': '/h5u1vqgms4-public/videos/Z/d/HD/lfgiplg5vz.mp4',
            #     'processingPath': '1/194',
            #     'projectId': 17208,
            #     'tags': [
            #         {
            #             'createdAt': '2023-02-07T19:35:01.808Z',
            #             'entityId': 19371139,
            #             'frameRange': [244, 244],
            #             'id': 12241539,
            #             'labelerLogin': 'admin',
            #             'tagId': 377141,
            #             'updatedAt': '2023-02-07T19:35:01.808Z'
            #         }
            #     ],
            #     'teamId': 435,
            #     'updatedAt': '2023-02-07T19:35:01.808Z',
            #     'workspaceId': 684
            # }
        """

        data = None
        response = self._get_response_by_id(
            id,
            "videos.info",
            id_field=ApiField.ID,
            fields={ApiField.FORCE_METADATA_FOR_LINKS: force_metadata_for_links},
        )
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


            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            video_id = 198702499
            project_id, dataset_id = api.video.get_destination_ids(video_id)

            print(project_id, dataset_id)
            # Output: 17208 55846
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
        :param name: Video name with extension.
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

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            dst_dataset_id = 55846
            src_video_id = 186580617
            video_info = api.video.get_info_by_id(src_video_id)
            hash = video_info.hash
            # It is necessary to upload video with the same extention as in src dataset
            name = video_info.name
            new_video_info = api.video.upload_hash(dst_dataset_id, name, hash)
            print(new_video_info)
            # Output:
            # VideoInfo(
            #     id=19371139,
            #     name='Videos_dataset_animals_sea_lion.mp4'
            #     hash='30/TQ1BcIOn1AI4RFgRO/6psRtr3lqNPmr4uQ=',
            #     link=None,
            #     team_id=435,
            #     workspace_id=684,
            #     project_id=17208,
            #     dataset_id=55846,
            #     path_original='/h5s-public/videos/Z/d/HD/lfgNXrg5vz.mp4',
            #     frames_to_timecodes=[],
            #     frames_count=245,
            #     frame_width=1920,
            #     frame_height=1080,
            #     created_at='2023-02-07T19:35:01.808Z',
            #     updated_at='2023-02-07T19:35:01.808Z',
            #     tags=[],
            #     file_meta={
            #         'codecName': 'h264',
            #         'codecType': 'video',
            #         'duration': 10.218542,
            #         'formatName': 'mov,mp4,m4a,3gp,3g2,mj2',
            #         'framesCount': 245,
            #         'framesToTimecodes': [],
            #         'height': 1080,
            #         'index': 0,
            #         'mime': 'video/mp4',
            #         'rotation': 0,
            #         'size': '6795452',
            #         'startTime': 0,
            #         'streams': [],
            #         'width': 1920
            #     },
            #     custom_data={},
            #     processing_path='1/194'
            # )
        """

        if hash is None:
            raise ValueError(
                "Video hash is None, it may occur when the video was uploaded as link. "
                "Please, use upload_id() method instead."
            )

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
        progress_cb: Optional[Union[tqdm, Callable]] = None,
    ) -> List[VideoInfo]:
        """
        Upload Videos from given hashes to Dataset.

        :param dataset_id: Dataset ID in Supervisely.
        :type dataset_id: int
        :param names: Videos names with extension.
        :type names: List[str]
        :param hashes: Videos hashes.
        :type hashes: List[str]
        :param metas: Videos metadata.
        :type metas: List[dict], optional
        :param progress_cb: Function for tracking download progress.
        :type progress_cb: tqdm or callable, optional
        :return: List with information about Videos. See :class:`info_sequence<info_sequence>`
        :rtype: :class:`List[VideoInfo]`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
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

        no_hash_names = []
        for name, hash in zip(names, hashes):
            if hash is None:
                no_hash_names.append(name)

        if len(no_hash_names) > 0:
            raise ValueError(
                f"Video hashes are None for the following videos: {no_hash_names}. "
                "It may occur when the videos were uploaded as links. "
                "Please, use upload_ids() method instead."
            )

        results = self._upload_bulk_add(
            lambda item: (ApiField.HASH, item),
            dataset_id,
            names,
            hashes,
            metas,
            progress_cb,
        )
        return results

    def upload_id(
        self, dataset_id: int, name: str, id: int, meta: Optional[Dict] = None
    ) -> VideoInfo:
        """
        Uploads video from given id to Dataset.

        :param dataset_id: Destination dataset ID.
        :type dataset_id: int
        :param name: Video name with extension.
        :type name: str
        :param id: Source video ID in Supervisely.
        :type id: int
        :param meta: Video metadata.
        :type meta: Optional[Dict]
        :return: Information about Video. See :class:`info_sequence<info_sequence>`
        :rtype: VideoInfo
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            src_video_id = 186580617
            dst_dataset_id = 468620

            new_video_info = api.video.upload_id(dst_dataset_id, 'new_video_name.mp4', src_video_id)
        """
        metas = None if meta is None else [meta]
        return self.upload_ids(dataset_id, [name], [id], metas=metas)[0]

    def upload_ids(
        self,
        dataset_id: int,
        names: List[str],
        ids: List[int],
        metas: Optional[List[Dict]] = None,
        progress_cb: Optional[Union[tqdm, Callable]] = None,
        infos: Optional[List[VideoInfo]] = None,
    ) -> List[VideoInfo]:
        """
        Uploads videos from given ids to Dataset.

        :param dataset_id: Destination dataset ID.
        :type dataset_id: int
        :param names: Videos names with extension.
        :type names: List[str]
        :param ids: Source videos IDs in Supervisely.
        :type ids: List[int]
        :param metas: Videos metadata.
        :type metas: Optional[List[Dict]]
        :param progress_cb: Function for tracking download progress.
        :type progress_cb: Optional[Union[tqdm, Callable]]
        :param infos: Videos information.
        :type infos: Optional[List[VideoInfo]]
        :return: List with information about Videos. See :class:`info_sequence<info_sequence>`
        :rtype: List[VideoInfo]
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            src_dataset_id = 466639
            dst_dataset_id = 468620

            ids = []
            names = []
            metas = []

            video_infos = api.video.get_list(src_dataset_id)
            # Create lists of ids, videos names and meta information for each video
            for video_info in video_infos:
                ids.append(video_info.id)
                # It is necessary to upload videos with the same names(extentions) as in src dataset
                names.append(video_info.name)
                metas.append({video_info.name: video_info.frame_height})

            progress = sly.Progress("Videos upload: ", len(ids))
            new_videos_info = api.video.upload_ids(dst_dataset_id, names, ids, metas, progress.iters_done_report)

        """
        if metas is None:
            metas = [{}] * len(names)

        if infos is None:
            infos = self.get_info_by_id_batch(ids, progress_cb=progress_cb)

        links, links_names, links_order, links_metas = [], [], [], []
        hashes, hashes_names, hashes_order, hashes_metas = [], [], [], []
        for idx, (name, info, meta) in enumerate(zip(names, infos, metas)):
            if info.link is not None:
                links.append(info.link)
                links_names.append(name)
                links_order.append(idx)
                links_metas.append(meta)
            else:
                hashes.append(info.hash)
                hashes_names.append(name)
                hashes_order.append(idx)
                hashes_metas.append(meta)

        result = [None] * len(names)

        if len(links) > 0:
            res_infos_links = self.upload_links(
                dataset_id,
                links,
                links_names,
                metas=links_metas,
                skip_download=True,
                force_metadata_for_links=False,
            )

            for info, pos in zip(res_infos_links, links_order):
                result[pos] = info

        if len(hashes) > 0:
            res_infos_hashes = self.upload_hashes(
                dataset_id,
                hashes_names,
                hashes,
                metas=hashes_metas,
                progress_cb=progress_cb,
            )

            for info, pos in zip(res_infos_hashes, hashes_order):
                result[pos] = info

        return result

    def copy_batch(
        self,
        dst_dataset_id: int,
        ids: List[int],
        change_name_if_conflict: Optional[bool] = False,
        with_annotations: Optional[bool] = False,
        progress_cb: Optional[Union[tqdm, Callable]] = None,
    ) -> List[VideoInfo]:
        """
        Copies Videos with given IDs to Dataset.

        :param dst_dataset_id: Destination Dataset ID in Supervisely.
        :type dst_dataset_id: int
        :param ids: Videos IDs in Supervisely.
        :type ids: List[int]
        :param change_name_if_conflict: If True adds suffix to the end of Image name when Dataset already contains an Image with identical name, If False and images with the identical names already exist in Dataset raises error.
        :type change_name_if_conflict: bool, optional
        :param with_annotations: If True Image will be copied to Dataset with annotations, otherwise only Images without annotations.
        :type with_annotations: bool, optional
        :param progress_cb: Function for tracking the progress of copying.
        :type progress_cb: tqdm or callable, optional
        :raises: :class:`TypeError` if type of ids is not list
        :raises: :class:`ValueError` if videos ids are from the destination Dataset
        :return: List with information about Videos. See :class:`info_sequence<info_sequence>`
        :rtype: :class:`List[VideoInfo]`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            dataset_id = 1780

            video_infos = api.video.get_list(dataset_id)

            video_ids = [video_info.id for video_info in video_infos]

            destination_dataset_id = 2574
            destination_video_infos = api.video.copy_batch(destination_dataset_id, video_ids, with_annotations=True)
        """
        if type(ids) is not list:
            raise TypeError(
                "ids parameter has type {!r}. but has to be of type {!r}".format(type(ids), list)
            )

        if len(ids) == 0:
            return

        ids_info = self.get_info_by_id_batch(ids, force_metadata_for_links=False)
        if len(set(vid_info.dataset_id for vid_info in ids_info)) > 1:
            raise ValueError("Videos ids have to be from the same dataset")

        existing_videos = self.get_list(dst_dataset_id, force_metadata_for_links=False)
        existing_names = {video.name for video in existing_videos}

        if change_name_if_conflict:
            new_names = [
                generate_free_name(existing_names, info.name, with_ext=True, extend_used_names=True)
                for info in ids_info
            ]
        else:
            new_names = [info.name for info in ids_info]
            names_intersection = existing_names.intersection(set(new_names))
            if len(names_intersection) != 0:
                raise ValueError(
                    "Videos with the same names already exist in destination dataset. "
                    'Please, use argument "change_name_if_conflict=True" to automatically resolve '
                    "names intersection"
                )

        new_videos = self.upload_ids(dst_dataset_id, new_names, ids, progress_cb=progress_cb)
        new_ids = [new_video.id for new_video in new_videos]

        if with_annotations:
            src_project_id = self._api.dataset.get_info_by_id(ids_info[0].dataset_id).project_id
            dst_project_id = self._api.dataset.get_info_by_id(dst_dataset_id).project_id
            self._api.project.merge_metas(src_project_id, dst_project_id)
            self._api.video.annotation.copy_batch(ids, new_ids)

        return new_videos

    def _upload_bulk_add(
        self,
        func_item_to_kv,
        dataset_id,
        names,
        items,
        metas=None,
        progress_cb=None,
        force_metadata_for_links=True,
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
                "videos.bulk.add",
                {
                    ApiField.DATASET_ID: dataset_id,
                    ApiField.VIDEOS: images,
                    ApiField.FORCE_METADATA_FOR_LINKS: force_metadata_for_links,
                },
            )
            if progress_cb is not None:
                progress_cb(len(images))

            results = [self._convert_json_info(item) for item in response.json()]
            name_to_res = {img_info.name: img_info for img_info in results}
            ordered_results = [name_to_res[name] for name in names]

            return ordered_results

    def _download(self, id, is_stream=False):
        """
        Private method. Download video with given ID

        :param id: int
        :param is_stream: bool
        :return: Response object containing video with given id
        """

        response = self._api.post("videos.download", {ApiField.ID: id}, stream=is_stream)
        return response

    def download_path(
        self, id: int, path: str, progress_cb: Optional[Union[tqdm, Callable]] = None
    ) -> None:
        """
        Downloads Video from Dataset to local path by ID.

        :param id: Video ID in Supervisely.
        :type id: int
        :param path: Local save path for Video.
        :type path: str
        :param progress_cb: Function to check progress.
        :type progress_cb: tqdm or callable, optional
        :return: None
        :rtype: :class:`NoneType`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
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

    def download_frames(
        self, video_id: int, frames: List[int], paths: List[str], progress_cb=None
    ) -> None:
        endpoint = "videos.bulk.download-frame"
        response: Response = self._api.get(
            endpoint,
            params={},
            data={ApiField.VIDEO_ID: video_id, ApiField.FRAMES: frames},
            stream=True,
        )
        response.raise_for_status()

        files = {frame_n: None for frame_n in frames}
        file_paths = {frame_n: path for frame_n, path in zip(frames, paths)}

        try:
            decoder = MultipartDecoder.from_response(response)
            for part in decoder.parts:
                content_utf8 = part.headers[b"Content-Disposition"].decode("utf-8")
                # Find name="1245" preceded by a whitespace, semicolon or beginning of line.
                # The regex has 2 capture group: one for the prefix and one for the actual name value.
                frame_n = int(re.findall(r'(^|[\s;])name="(\d*)"', content_utf8)[0][1])
                if files[frame_n] is None:
                    file_path = file_paths[frame_n]
                    files[frame_n] = open(file_path, "wb")
                    if progress_cb is not None:
                        progress_cb(1)
                f = files[frame_n]
                f.write(part.content)

        finally:
            for f in files.values():
                if f is not None:
                    f.close()

    def download_range_by_id(
        self,
        id: int,
        frame_start: int,
        frame_end: int,
        is_stream: Optional[bool] = True,
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

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
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

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
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

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
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
        Send message to the Annotation Tool and return info if tracking was stopped

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
                    ApiField.PROGRESS: {
                        ApiField.CURRENT: current,
                        ApiField.TOTAL: total,
                    },
                },
            },
        )
        return response.json()[ApiField.STOPPED]

    def notify_tracking_error(self, track_id: int, error: str, message: str):
        """
        Send message to the Annotation Tool

        :param track_id: int
        :param error: str
        :param message: str
        :return: None
        """

        self._api.post(
            "videos.notify-annotation-tool",
            {
                "type": "videos:tracking-error",
                "data": {
                    ApiField.TRACK_ID: track_id,
                    ApiField.ERROR: {ApiField.MESSAGE: "{}: {}".format(error, message)},
                },
            },
        )

    def notify_tracking_warning(self, track_id: int, video_id: int, message: str):
        self._api.post(
            "videos.notify-annotation-tool",
            data={
                "type": "videos:tracking-warning",
                "data": {
                    ApiField.VIDEO_ID: str(video_id),
                    ApiField.TRACK_ID: str(track_id),
                    ApiField.MESSAGE: message,
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
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            # Helpful method when your uploading was interrupted
            # You can check what videos has been successfully uploaded by their hashes and what not
            # And continue uploading the rest of the videos from that point

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
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
        progress_cb: Optional[Union[tqdm, Callable]] = None,
        metas: Optional[List[Dict]] = None,
        infos=None,
        item_progress=None,
    ) -> List[VideoInfo]:
        """
        Uploads Videos with given names from given local paths to Dataset.

        :param dataset_id: Dataset ID in Supervisely.
        :type dataset_id: int
        :param names: List of Videos names with extension.
        :type names: List[str]
        :param paths: List of local Videos paths.
        :type paths: List[str]
        :param progress_cb: Function for tracking download progress.
        :type progress_cb: tqdm or callable, optional
        :param metas: Videos metadata.
        :type metas: List[dict], optional
        :param infos:
        :type infos:
        :param item_progress:
        :type item_progress:
        :return: List with information about Videos. See :class:`info_sequence<info_sequence>`
        :rtype: :class:`List[VideoInfo]`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            dataset_id=55846
            video_names = ["7777.mp4", "8888.mp4", "9999.mp4"]
            video_paths = ["/home/admin/Downloads/video/770918.mp4", "/home/admin/Downloads/video/770919.mp4", "/home/admin/Downloads/video/770920.mp4"]

            video_infos = api.video.upload_paths(
                dataset_id=dataset_id,
                names=video_names,
                paths=video_paths,
            )
        """

        for name, path in zip(names, paths):
            file_ext = get_file_ext(path)
            validate_ext(file_ext)
            name_ext = os.path.splitext(name)[1]
            if name_ext != file_ext:
                raise ValueError(
                    f"The name extension '{name_ext}' does not match the file extension '{file_ext}'"
                )

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
        hashes = [get_file_hash_chunked(x) for x in paths]

        self._upload_data_bulk(
            path_to_bytes_stream,
            zip(paths, hashes),
            progress_cb=progress_cb,
            item_progress=item_progress,
        )
        if update_headers:
            self.upsert_infos(hashes, infos)
            self._api.pop_header("x-skip-processing")

        unique_hashes = list(set(hashes))
        unique_metas = self._api.import_storage.get_meta_by_hashes(unique_hashes)

        hash_meta_dict = {}
        for hash_value, meta in zip(unique_hashes, unique_metas):
            hash_meta_dict[hash_value] = meta

        metas = [hash_meta_dict[hash_value] for hash_value in hashes]

        metas2 = [meta["meta"] for meta in metas]

        names = self.get_free_names(dataset_id, names)

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

                    # item_name = name
                    # info = self._api.video.get_info_by_name(dataset_id, item_name)
                    # if info is not None:
                    #     item_name = gen_video_stream_name(name, stream_index)
                    res = self.upload_hash(dataset_id, name, hash, stream_index)
                    video_info_results.append(res)
            except Exception as e:
                from supervisely.io.exception_handlers import (
                    ErrorHandler,
                    handle_exception,
                )

                msg = f"File skipped {name}: error occurred during processing: "
                handled_exc = handle_exception(e)
                if handled_exc is not None:
                    if isinstance(handled_exc, ErrorHandler.API.PaymentRequired):
                        raise e  # re-raise original exception (will be handled in the UI)
                    else:
                        msg += handled_exc.get_message_for_exception()
                else:
                    msg += repr(e)
                logger.warning(msg)
        return video_info_results

    def upload_path(
        self,
        dataset_id: int,
        name: str,
        path: str,
        meta: Dict = None,
        item_progress: Optional[Progress] = None,
    ) -> VideoInfo:
        """
        Uploads Video with given name from given local path to Dataset.

        :param dataset_id: Dataset ID in Supervisely.
        :type dataset_id: int
        :param name: Video name with extension.
        :type name: str
        :param path: Local video path.
        :type path: str
        :param meta: Video metadata.
        :type meta: dict, optional
        :param item_progress:
        :type item_progress:
        :return: List with information about Videos. See :class:`info_sequence<info_sequence>`
        :rtype: :class:`VideoInfo`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            dataset_id=55846
            video_name = "7777.mp4"
            video_path = "/home/admin/Downloads/video/770918.mp4"

            video_infos = api.video.upload_path(
                dataset_id=dataset_id,
                name=video_name,
                path=video_path,
            )
        """
        progress_cb = item_progress
        p = None
        if item_progress is not None and type(item_progress) is bool:
            p = Progress(f"Uploading {name}", total_cnt=get_file_size(path), is_size=True)
            # progress_cb = p.iters_done_report
            progress_cb = p.set_current_value

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
        """Private method. Used to upload multiple unique videos in a single HTTP request."""

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

            if isinstance(progress_cb, tqdm):
                callback = partial(_callback, progress=progress_cb.update)
            else:
                callback = partial(_callback, progress=progress_cb)
            monitor = MultipartEncoderMonitor(encoder, callback)
            resp = self._api.post("videos.bulk.upload", monitor)
        else:
            resp = self._api.post("videos.bulk.upload", encoder)

        # close all opened files
        for value in content_dict.values():
            from io import BufferedReader

            if isinstance(value[1], BufferedReader):
                value[1].close()

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
        """Private method. Used for batch uploading of multiple unique videos."""

        # count all items to adjust progress_cb and create hash to item mapping with unique hashes
        items_count_total = 0
        hash_to_items = {}
        for item, i_hash in items_hashes:
            hash_to_items[i_hash] = item
            items_count_total += 1

        unique_hashes = set(hash_to_items.keys())
        remote_hashes = set(
            self.check_existing_hashes(list(unique_hashes))
        )  # existing -- from server
        if progress_cb is not None:
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
                if progress_cb is not None:
                    progress_cb(len(hashes_rcv))

            if not pending_hashes:
                if progress_cb is not None:
                    progress_cb(items_count_total - len(unique_hashes))
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
        """
        Update Video file metadata

        :param hash: Video hash.
        :type hash: str
        :param info: Uploading info.
        :type info: dict
        :return: Return updating result
        :rtype: dict
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            video_id = 19388386
            video_info = api.video.get_info_by_id(video_id)
            video_hash = video_info.hash

            res = api.video.upsert_info(video_hash, {"field": "value"})
            print(res)

            # Output: {'success': True}
        """

        return self.upsert_infos([hash], [info])

    def upsert_infos(
        self, hashes: List[str], infos: List[Dict], links: Optional[List[str]] = None
    ) -> Dict:
        """
        Update Video files metadata

        :param hashes: Video hash.
        :type hashes: str
        :param infos: Uploading info.
        :type infos: dict
        :return: Return updating result
        :rtype: dict
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()


            dataset_id = 56443
            video_ids = [19388386, 19388387, 19388388]
            video_infos = api.video.get_list(
                dataset_id=dataset_id,
                filters=[{'field': 'id', 'operator': 'in', 'value': video_ids}]
            )
            video_hashes = [video_info.hash for video_info in video_infos]
            new_infos = [{"field1": "value1"}, {"field2": "value2"}, {"field3": "value3"}]

            res = api.video.upsert_infos(video_hashes, new_infos)
            print(res)

            # Output: {'success': True}
        """

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
        links: List[str],
        names: List[str],
        infos: List[Dict] = None,
        hashes: List[str] = None,
        metas: Optional[List[Dict]] = None,
        skip_download: Optional[bool] = False,
        progress_cb: Optional[Union[tqdm, Callable]] = None,
        force_metadata_for_links: Optional[bool] = True,
    ) -> List[VideoInfo]:
        """
        Upload Videos from given links to Dataset.

        :param dataset_id: Dataset ID in Supervisely.
        :type dataset_id: int
        :param links: Videos links.
        :type links: List[str]
        :param names: Videos names with extension.
        :type names: List[str]
        :param infos: Videos infos.
        :type infos: List[dict]
        :param hashes: Videos hashes.
        :type hashes: List[str]
        :param metas: Videos metadatas.
        :type metas: List[dict], optional
        :param skip_download: Skip download videos to local storage.
        :type skip_download: Optional[bool]
        :param progress_cb: Function for tracking the progress of copying.
        :type progress_cb: tqdm or callable, optional
        :param force_metadata_for_links: Specify whether to force retrieving videos metadata from the server after upload
        :type force_metadata_for_links: Optional[bool]
        :return: List with information about Videos. See :class:`info_sequence<info_sequence>`
        :rtype: :class:`List[VideoInfo]`
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            from supervisely.video.video import get_info

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            dataset_id = 55847
            links = [
                "https://videos...7477606_main.mp4",
                "https://videos...040243048_main.mp4",
                "https://videos...065451525_main.mp4"
            ]
            names = ["cars.mp4", "animals.mp4", "traffic.mp4"]
            video_infos = api.video.upload_links(dataset_id, links, names)
            print(video_infos)

            # Output: [
                VideoInfo(id=19593405, ...),
                VideoInfo(id=19593406, ...),
                VideoInfo(id=19593407, ...)
            ]
        """

        # This was deprecated, do not uncomment, otherwise it will break the code.
        # if infos is not None and hashes is not None and not skip_download:
        #     self.upsert_infos(hashes, infos, links)
        return self._upload_bulk_add(
            lambda item: (ApiField.LINK, item),
            dataset_id,
            names,
            links,
            metas,
            progress_cb=progress_cb,
            force_metadata_for_links=force_metadata_for_links,
        )

    def update_custom_data(self, id: int, data: dict):
        """
        Upload custom data info in VideoInfo object.

        :param video_id: Videos ID in Supervisely.
        :type video_id: int
        :param metas: Metadata dict with custom values.
            Note: Do not recommend changing metas as it affects displaying
            data in label tools. In case changing the metadata is necessary,
            make sure to include an "streams" field with its value in the request body.
        :type metas: dict
        :return: Return updating result
        :rtype: dict
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            from supervisely.video.video import get_info

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            video_id = 19402023

            api.video.update_custom_data(video_id, {"field": "value"})

            video_info = api.video.get_info_by_id(video_id)
            print(video_info.custom_data)

            # Output: {'field': 'value'}
        """

        resp = self._api.post(
            "videos.custom-data.set", {ApiField.ID: id, ApiField.CUSTOM_DATA: data}
        )
        return resp.json()

    def upload_link(
        self,
        dataset_id: int,
        link: str,
        name: Optional[str] = None,
        info: Optional[Dict] = None,
        hash: Optional[str] = None,
        meta: Optional[List[Dict]] = None,
        skip_download: Optional[bool] = False,
        force_metadata_for_links: Optional[bool] = True,
    ):
        """
        Upload Video from given link to Dataset.

        :param dataset_id: Dataset ID in Supervisely.
        :type dataset_id: int
        :param link: Video link.
        :type link: str
        :param name: Video name with extension.
        :type name: str, optional
        :param info: Video info.
        :type info: dict, optional
        :param hash: Video hash.
        :type hash: str, optional
        :param meta: Video metadata.
        :type meta: List[Dict], optional
        :param skip_download: Skip download video to local storage.
        :type skip_download: Optional[bool]
        :param force_metadata_for_links: Specify whether to force retrieving video metadata from the server after upload
        :type force_metadata_for_links: Optional[bool]
        :return: List with information about Video. See :class:`info_sequence<info_sequence>`
        :rtype: :class:`List[VideoInfo]`

         .. code-block:: python

            import supervisely as sly
            from supervisely.video.video import get_info

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            dataset_id = 55847
            link = "https://video...040243048_main.mp4"
            name = "cars.mp4"

            info = api.video.upload_link(dataset_id, link, name)
            print(info)

            # Output: [
            #     VideoInfo(
            #         id=19371139,
            #         name='cars.mp4'
            #         hash='30/TQ1BcIOn1AI4RFgRO/6psRtr3lqNPmr4uQ=',
            #         link=None,
            #         team_id=435,
            #         workspace_id=684,
            #         project_id=17208,
            #         dataset_id=55847,
            #         path_original='/h5ung-public/videos/Z/d/HD/lfgipl...NXrg5vz.mp4',
            #         frames_to_timecodes=[],
            #         frames_count=245,
            #         frame_width=1920,
            #         frame_height=1080,
            #         created_at='2023-02-07T19:35:01.808Z',
            #         updated_at='2023-02-07T19:35:01.808Z',
            #         tags=[],
            #         file_meta={
            #             'codecName': 'h264',
            #             'codecType': 'video',
            #             'duration': 10.218542,
            #             'formatName': 'mov,mp4,m4a,3gp,3g2,mj2',
            #             'framesCount': 245,
            #             'framesToTimecodes': [],
            #             'height': 1080,
            #             'index': 0,
            #             'mime': 'video/mp4',
            #             'rotation': 0,
            #             'size': '6795452',
            #             'startTime': 0,
            #             'streams': [],
            #             'width': 1920
            #         },
            #         custom_data={},
            #         processing_path='1/194'
            #     )
            # ]
        """

        if name is None:
            name = rand_str(10) + get_file_ext(link)

        if not skip_download:
            local_path = os.path.join(os.getcwd(), name)
            try:
                sly_fs.download(link, local_path)
                video_info = get_info(local_path)
                h = get_file_hash(local_path)
                sly_fs.silent_remove(local_path)
            except Exception as e:
                sly_fs.silent_remove(local_path)
                raise e
        else:
            video_info = info
            h = hash
        name = self.get_free_name(dataset_id, name)
        links = self.upload_links(
            dataset_id,
            links=[link],
            names=[name],
            infos=[video_info],
            hashes=[h],
            metas=[meta],
            skip_download=skip_download,
            force_metadata_for_links=force_metadata_for_links,
        )
        if len(links) != 1:
            raise RuntimeError(
                (
                    f"API response: '{links}' (len > 1). "
                    "Validation error. Only one item is allowed. "
                    "Please, contact technical support."
                )
            )
        return links[0]

    def add_existing(
        self,
        dataset_id: int,
        video_info: VideoInfo,
        name: str,
    ) -> VideoInfo:
        """
        Add existing video from source Dataset to destination Dataset.

        :param dataset_id: Destination Dataset ID in Supervisely.
        :type dataset_id: int
        :param video_info: Information about the video.
        :type video_info: VideoInfo
        :param name: Video name.
        :type name: str
        :return: Information about Video. See :class:`info_sequence<info_sequence>`
        :rtype: :class:`VideoInfo`

         .. code-block:: python

            import supervisely as sly
            from supervisely.video.video import get_info

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            dataset_id = 55846
            video_id = 19371139

            video_info = api.video.get_info_by_id(video_id)

            new_info = api.video.add_existing(dataset_id, video_info, "sea lion.mp4")
            print(new_info)

            # Output:
            # VideoInfo(
            #     id=19371140,
            #     name='sea lion.mp4'
            #     hash='30/TQ1BcIOn1AI4RFgRO/6psRtr3lqNPmr4uQ=',
            #     link=None,
            #     team_id=435,
            #     workspace_id=684,
            #     project_id=17208,
            #     dataset_id=55846,
            #     path_original='/h5ung-public/videos/Z/d/HD/lfgipl...NXrg5vz.mp4',
            #     frames_to_timecodes=[],
            #     frames_count=245,
            #     frame_width=1920,
            #     frame_height=1080,
            #     created_at='2023-02-07T19:35:01.808Z',
            #     updated_at='2023-02-07T19:35:01.808Z',
            #     tags=[],
            #     file_meta={
            #         'codecName': 'h264',
            #         'codecType': 'video',
            #         'duration': 10.218542,
            #         'formatName': 'mov,mp4,m4a,3gp,3g2,mj2',
            #         'framesCount': 245,
            #         'framesToTimecodes': [],
            #         'height': 1080,
            #         'index': 0,
            #         'mime': 'video/mp4',
            #         'rotation': 0,
            #         'size': '6795452',
            #         'startTime': 0,
            #         'streams': [],
            #         'width': 1920
            #     },
            #     custom_data={},
            #     processing_path='1/194'
            # )
        """

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
        """Private method. Returns API method name used for batch removal of videos."""

        return "images.bulk.remove"

    def _remove_batch_field_name(self):
        """Private method. Returns constant string that represents API field name that contains the IDs of the images."""

        return ApiField.IMAGE_IDS

    def remove_batch(
        self,
        ids: List[int],
        progress_cb: Optional[Union[tqdm, Callable]] = None,
        batch_size: Optional[int] = 50,
    ):
        """
        Remove videos from supervisely by IDs.
        All video IDs must belong to the same dataset.
        Therefore, it is necessary to sort IDs before calling this method.

        :param ids: List of Videos IDs in Supervisely.
        :type ids: List[int]
        :param progress_cb: Function for tracking progress of removing.
        :type progress_cb: tqdm or callable, optional
        :return: :class:`None<None>`
        :rtype: :class:`NoneType<NoneType>`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
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

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            video_id = 2389126
            api.video.remove(video_id)
        """

        super(VideoApi, self).remove(video_id)

    def get_free_names(self, dataset_id: int, names: List[str]) -> List[str]:
        """
        Returns list of free names for given dataset.

        :param dataset_id: Dataset ID in Supervisely.
        :type dataset_id: int
        :param names: List of names to check.
        :type names: List[str]
        :return: List of free names.
        :rtype: List[str]
        """

        videos_in_dataset = self.get_list(dataset_id, force_metadata_for_links=False)
        used_names = {video_info.name for video_info in videos_in_dataset}
        new_names = [
            generate_free_name(used_names, name, with_ext=True, extend_used_names=True)
            for name in names
        ]
        return new_names

    def raise_name_intersections_if_exist(
        self, dataset_id: int, names: List[str], message: str = None
    ):
        """
        Raises error if videos with given names already exist in dataset.
        Default error message:
        "Videos with the following names already exist in dataset [ID={dataset_id}]: {name_intersections}.
        Please, rename videos and try again or set change_name_if_conflict=True to rename automatically on upload."

        :param dataset_id: Dataset ID in Supervisely.
        :type dataset_id: int
        :param names: List of names to check.
        :type names: List[str]
        :param message: Error message.
        :type message: str, optional
        :return: None
        :rtype: None
        """
        videos_in_dataset = self.get_list(dataset_id, force_metadata_for_links=False)
        used_names = {video_info.name for video_info in videos_in_dataset}
        name_intersections = used_names.intersection(set(names))
        if message is None:
            message = f"Videos with the following names already exist in dataset [ID={dataset_id}]: {name_intersections}. Please, rename videos and try again or set change_name_if_conflict=True to rename automatically on upload."
        if len(name_intersections) > 0:
            raise ValueError(f"{message}")

    def upload_dir(
        self,
        dataset_id: int,
        dir_path: str,
        recursive: Optional[bool] = True,
        change_name_if_conflict: Optional[bool] = True,
        progress_cb: Optional[Union[tqdm, Callable]] = None,
    ) -> List[VideoInfo]:
        """
        Uploads all videos with supported extensions from given directory to Supervisely.
        Optionally, uploads videos from subdirectories of given directory.

        :param dataset_id: Dataset ID in Supervisely.
        :type dataset_id: int
        :param dir_path: Path to directory with videos.
        :type dir_path: str
        :param recursive: If True, will upload videos from subdirectories of given directory recursively. Otherwise, will upload videos only from given directory.
        :type recursive: bool, optional
        :param change_name_if_conflict: If True adds suffix to the end of Video name when Dataset already contains an Video with identical name, If False and videos with the identical names already exist in Dataset raises error.
        :type change_name_if_conflict: bool, optional
        :param progress_cb: Function for tracking upload progress.
        :type progress_cb: Optional[Union[tqdm, Callable]]
        :return: List of uploaded videos infos
        :rtype: List[VideoInfo]
        """

        if recursive:
            paths = list_files_recursively(dir_path, filter_fn=is_valid_format)
        else:
            paths = list_files(dir_path, filter_fn=is_valid_format)

        names = [get_file_name_with_ext(path) for path in paths]

        if change_name_if_conflict is False:
            self.raise_name_intersections_if_exist(dataset_id, names)

        video_infos = self.upload_paths(dataset_id, names, paths, progress_cb=progress_cb)
        return video_infos

    def upload_dirs(
        self,
        dataset_id: int,
        dir_paths: List[str],
        recursive: Optional[bool] = True,
        change_name_if_conflict: Optional[bool] = True,
        progress_cb: Optional[Union[tqdm, Callable]] = None,
    ) -> List[VideoInfo]:
        """
        Uploads all videos with supported extensions from given directories to Supervisely.
        Optionally, uploads videos from subdirectories of given directories.

        :param dataset_id: Dataset ID in Supervisely.
        :type dataset_id: int
        :param dir_paths: List of paths to directories with videos.
        :type dir_paths: List[str]
        :param recursive: If True, will upload videos from subdirectories of given directories recursively. Otherwise, will upload videos only from given directories.
        :type recursive: bool, optional
        :param change_name_if_conflict: If True adds suffix to the end of Video name when Dataset already contains an Video with identical name, If False and videos with the identical names already exist in Dataset raises error.
        :type change_name_if_conflict: bool, optional
        :param progress_cb: Function for tracking upload progress.
        :type progress_cb: Optional[Union[tqdm, Callable]]
        :return: List of uploaded videos infos
        :rtype: List[VideoInfo]
        """

        video_infos = []
        for dir_path in dir_paths:
            video_infos.extend(
                self.upload_dir(
                    dataset_id, dir_path, recursive, change_name_if_conflict, progress_cb
                )
            )
        return video_infos

    def set_remote(self, videos: List[int], links: List[str]):
        """
        Updates the source of existing videos by setting new remote links.
        This method is used when a video was initially uploaded as a file or added via a link,
        but later it was decided to change its location (e.g., moved to another storage or re-uploaded elsewhere).
        By updating the link, the video source can be redirected to the new location.

        :param videos: List of video ids.
        :type videos: List[int]
        :param links: List of new remote links.
        :type links: List[str]
        :return: json-encoded content of a response.

        :Usage example:

            .. code-block:: python

                import supervisely as sly

                api = sly.Api.from_env()

                videos = [123, 124, 125]
                links = [
                    "s3://bucket/f1champ/ds1/lap_1.mp4",
                    "s3://bucket/f1champ/ds1/lap_2.mp4",
                    "s3://bucket/f1champ/ds1/lap_3.mp4",
                ]
                result = api.video.set_remote(videos, links)
        """

        if len(videos) == 0:
            raise ValueError("List of videos can not be empty.")

        if len(videos) != len(links):
            raise ValueError("Length of 'videos' and 'links' should be equal.")

        videos_list = []
        for vid, lnk in zip(videos, links):
            videos_list.append({ApiField.ID: vid, ApiField.LINK: lnk})

        data = {ApiField.VIDEOS: videos_list, ApiField.CLEAR_LOCAL_DATA_SOURCE: True}
        r = self._api.post("videos.update.links", data)
        return r.json()

    async def _download_async(
        self,
        id: int,
        is_stream: bool = False,
        range_start: Optional[int] = None,
        range_end: Optional[int] = None,
        headers: Optional[dict] = None,
        chunk_size: int = 1024 * 1024,
    ) -> AsyncGenerator:
        """
        Download Video with given ID asynchronously.

        :param id: Video ID in Supervisely.
        :type id: int
        :param is_stream: If True, returns stream of bytes, otherwise returns response object.
        :type is_stream: bool, optional
        :param range_start: Start byte of range for partial download.
        :type range_start: int, optional
        :param range_end: End byte of range for partial download.
        :type range_end: int, optional
        :param headers: Headers for request.
        :type headers: dict, optional
        :param chunk_size: Size of chunk for partial download. Default is 1MB.
        :type chunk_size: int, optional
        :return: Stream of bytes or response object.
        :rtype: AsyncGenerator
        """
        api_method_name = "videos.download"

        json_body = {ApiField.ID: id}

        if is_stream:
            async for chunk, hhash in self._api.stream_async(
                api_method_name,
                "POST",
                json_body,
                headers=headers,
                range_start=range_start,
                range_end=range_end,
                chunk_size=chunk_size,
            ):
                yield chunk, hhash
        else:
            response = await self._api.post_async(api_method_name, json_body, headers=headers)
            yield response

    async def download_path_async(
        self,
        id: int,
        path: str,
        semaphore: Optional[asyncio.Semaphore] = None,
        range_start: Optional[int] = None,
        range_end: Optional[int] = None,
        headers: Optional[dict] = None,
        chunk_size: int = 1024 * 1024,
        check_hash: bool = True,
        progress_cb: Optional[Union[tqdm, Callable]] = None,
        progress_cb_type: Literal["number", "size"] = "number",
    ) -> None:
        """
        Downloads Video with given ID to local path.

        :param id: Video ID in Supervisely.
        :type id: int
        :param path: Local save path for Video.
        :type path: str
        :param semaphore: Semaphore for limiting the number of simultaneous downloads.
        :type semaphore: :class:`asyncio.Semaphore`, optional
        :param range_start: Start byte of range for partial download.
        :type range_start: int, optional
        :param range_end: End byte of range for partial download.
        :type range_end: int, optional
        :param headers: Headers for request.
        :type headers: dict, optional
        :param chunk_size: Size of chunk for partial download. Default is 1MB.
        :type chunk_size: int, optional
        :param check_hash: If True, checks hash of downloaded file.
                        Check is not supported for partial downloads.
                        When range is set, hash check is disabled.
        :type check_hash: bool, optional
        :param progress_cb: Function for tracking download progress.
        :type progress_cb: Optional[Union[tqdm, Callable]]
        :param progress_cb_type: Type of progress callback. Can be "number" or "size". Default is "number".
        :type progress_cb_type: Literal["number", "size"], optional
        :return: None
        :rtype: :class:`NoneType`
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            import asyncio

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            video_info = api.video.get_info_by_id(770918)
            save_path = os.path.join("/path/to/save/", video_info.name)

            semaphore = asyncio.Semaphore(100)
            loop = sly.utils.get_or_create_event_loop()
            loop.run_until_complete(
                    api.video.download_path_async(video_info.id, save_path, semaphore)
                )
        """
        if range_start is not None or range_end is not None:
            check_hash = False  # hash check is not supported for partial downloads
            headers = headers or {}
            headers["Range"] = f"bytes={range_start or ''}-{range_end or ''}"
            logger.debug(f"Image ID: {id}. Setting Range header: {headers['Range']}")

        writing_method = "ab" if range_start not in [0, None] else "wb"

        ensure_base_path(path)
        hash_to_check = None
        if semaphore is None:
            semaphore = self._api.get_default_semaphore()
        async with semaphore:
            async with aiofiles.open(path, writing_method) as fd:
                async for chunk, hhash in self._download_async(
                    id,
                    is_stream=True,
                    range_start=range_start,
                    range_end=range_end,
                    headers=headers,
                    chunk_size=chunk_size,
                ):
                    await fd.write(chunk)
                    hash_to_check = hhash
                    if progress_cb is not None and progress_cb_type == "size":
                        progress_cb(len(chunk))
            if check_hash:
                if hash_to_check is not None:
                    downloaded_file_hash = await get_file_hash_async(path)
                    if hash_to_check != downloaded_file_hash:
                        raise RuntimeError(
                            f"Downloaded hash of video with ID:{id} does not match the expected hash: {downloaded_file_hash} != {hash_to_check}"
                        )
            if progress_cb is not None and progress_cb_type == "number":
                progress_cb(1)

    async def download_paths_async(
        self,
        ids: List[int],
        paths: List[str],
        semaphore: Optional[asyncio.Semaphore] = None,
        headers: Optional[dict] = None,
        chunk_size: int = 1024 * 1024,
        check_hash: bool = True,
        progress_cb: Optional[Union[tqdm, Callable]] = None,
        progress_cb_type: Literal["number", "size"] = "number",
    ) -> None:
        """
        Download Videos with given IDs and saves them to given local paths asynchronously.

        :param ids: List of Video IDs in Supervisely.
        :type ids: :class:`List[int]`
        :param paths: Local save paths for Videos.
        :type paths: :class:`List[str]`
        :param semaphore: Semaphore
        :type semaphore: :class:`asyncio.Semaphore`, optional
        :param headers: Headers for request.
        :type headers: dict, optional
        :param chunk_size: Size of chunk for partial download. Default is 1MB.
        :type chunk_size: int, optional
        :param check_hash: If True, checks hash of downloaded files.
        :type check_hash: bool, optional
        :param progress_cb: Function for tracking download progress.
        :type progress_cb: Optional[Union[tqdm, Callable]]
        :param progress_cb_type: Type of progress callback. Can be "number" or "size". Default is "number".
        :type progress_cb_type: Literal["number", "size"], optional
        :raises: :class:`ValueError` if len(ids) != len(paths)
        :return: None
        :rtype: :class:`NoneType`

        :Usage example:

         .. code-block:: python

            import supervisely as sly
            import asyncio

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            ids = [770914, 770915]
            paths = ["/path/to/save/video1.mp4", "/path/to/save/video2.mp4"]
            loop = sly.utils.get_or_create_event_loop()
            loop.run_until_complete(api.video.download_paths_async(ids, paths))
        """
        if len(ids) == 0:
            return
        if len(ids) != len(paths):
            raise ValueError('Can not match "ids" and "paths" lists, len(ids) != len(paths)')
        if semaphore is None:
            semaphore = self._api.get_default_semaphore()
        tasks = []
        for img_id, img_path in zip(ids, paths):
            task = self.download_path_async(
                img_id,
                img_path,
                semaphore=semaphore,
                headers=headers,
                chunk_size=chunk_size,
                check_hash=check_hash,
                progress_cb=progress_cb,
                progress_cb_type=progress_cb_type,
            )

            tasks.append(task)
        await asyncio.gather(*tasks)

    def rename(
        self,
        id: int,
        name: str,
    ) -> VideoInfo:
        """Renames Video with given ID to a new name.

        :param id: Video ID in Supervisely.
        :type id: int
        :param name: New Video name.
        :type name: str
        :return: Information about updated Video.
        :rtype: :class:`VideoInfo`

        :Usage example:

        .. code-block:: python

            import supervisely as sly

            api = sly.Api.from_env()
            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'

            video_id = 123456
            new_video_name = "VID_3333_new.mp4"

            api.video.rename(id=video_id, name=new_video_name)
        """

        data = {
            ApiField.ID: id,
            ApiField.NAME: name,
        }

        response = self._api.post("images.editInfo", data)
        return self._convert_json_info(response.json())

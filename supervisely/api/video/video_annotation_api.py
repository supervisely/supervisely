# coding: utf-8
from __future__ import annotations

import json
from typing import Callable, Dict, List, Optional, Union

from tqdm import tqdm

from supervisely._utils import batched
from supervisely.api.entity_annotation.entity_annotation_api import EntityAnnotationAPI
from supervisely.api.module_api import ApiField
from supervisely.io.json import load_json_file
from supervisely.project.project_meta import ProjectMeta
from supervisely.task.progress import Progress
from supervisely.video_annotation.key_id_map import KeyIdMap
from supervisely.video_annotation.video_annotation import VideoAnnotation


class VideoAnnotationAPI(EntityAnnotationAPI):
    """
    :class:`VideoAnnotation<supervisely.video_annotation.video_annotation.VideoAnnotation>` for a single video. :class:`VideoAnnotationAPI<VideoAnnotationAPI>` object is immutable.

    :param api: API connection to the server.
    :type api: Api
    :Usage example:

     .. code-block:: python

        import os
        from dotenv import load_dotenv

        import supervisely as sly

        # Load secrets and create API object from .env file (recommended)
        # Learn more here: https://developer.supervisely.com/getting-started/basics-of-authentication
        if sly.is_development():
            load_dotenv(os.path.expanduser("~/supervisely.env"))
        api = sly.Api.from_env()

        # Pass values into the API constructor (optional, not recommended)
        # api = sly.Api(server_address="https://app.supervise.ly", token="4r47N...xaTatb")

        video_id = 186648102
        ann_info = api.video.annotation.download(video_id)
    """

    _method_download_bulk = "videos.annotations.bulk.info"
    _entity_ids_str = ApiField.VIDEO_IDS

    def download(self, video_id: int) -> Dict:
        """
        Download information about VideoAnnotation by video ID from API.

        :param video_id: Video ID in Supervisely.
        :type video_id: int
        :return: Information about VideoAnnotation in json format
        :rtype: :class:`dict`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            video_id = 198702499
            ann_info = api.video.annotation.download(video_id)
            print(ann_info)
            # Output: {
            #     "videoId": 198702499,
            #     "videoName": "Videos_dataset_cars_cars.mp4",
            #     "createdAt": "2021-03-23T13:14:25.536Z",
            #     "updatedAt": "2021-03-23T13:16:43.300Z",
            #     "description": "",
            #     "tags": [],
            #     "objects": [],
            #     "size": {
            #         "height": 2160,
            #         "width": 3840
            #     },
            #     "framesCount": 326,
            #     "frames": []
            # }
        """

        video_info = self._api.video.get_info_by_id(video_id)
        return self._download(video_info.dataset_id, video_id)

    def append(
        self,
        video_id: int,
        ann: VideoAnnotation,
        key_id_map: Optional[KeyIdMap] = None,
        progress_cb: Optional[Union[tqdm, Callable]] = None,
    ) -> None:
        """
        Loads an VideoAnnotation to a given video ID in the API.

        :param video_id: Video ID in Supervisely.
        :type video_id: int
        :param ann: VideoAnnotation object.
        :type ann: VideoAnnotation
        :param key_id_map: KeyIdMap object.
        :type key_id_map: KeyIdMap, optional
        :param progress: Progress.
        :type progress: Optional[Union[tqdm, Callable]]
        :return: None
        :rtype: :class:`NoneType`

        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            video_id = 198704259
            api.video.annotation.append(video_id, video_ann)
        """

        info = self._api.video.get_info_by_id(video_id)
        self._append(
            self._api.video.tag,
            self._api.video.object,
            self._api.video.figure,
            info.project_id,
            info.dataset_id,
            video_id,
            ann.tags,
            ann.objects,
            ann.figures,
            key_id_map,
            progress_cb,
        )

    def upload_paths(
        self,
        video_ids: List[int],
        ann_paths: List[str],
        project_meta: ProjectMeta,
        progress_cb: Optional[Union[tqdm, Callable]] = None,
    ) -> None:
        """
        Loads an VideoAnnotations from a given paths to a given videos IDs in the API. Videos IDs must be from one dataset.

        :param video_ids: Videos IDs in Supervisely.
        :type video_ids: List[int]
        :param ann_paths: Paths to annotations on local machine.
        :type ann_paths: List[str]
        :param project_meta: Input :class:`ProjectMeta<supervisely.project.project_meta.ProjectMeta>` for VideoAnnotations.
        :type project_meta: ProjectMeta
        :param progress_cb: Function for tracking download progress.
        :type progress_cb: tqdm or callable, optional
        :return: None
        :rtype: :class:`NoneType`

        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            video_ids = [121236918, 121236919]
            ann_paths = ['/home/admin/work/supervisely/example/ann1.json', '/home/admin/work/supervisely/example/ann2.json']
            api.video.annotation.upload_paths(video_ids, ann_paths, meta)
        """
        # video_ids from the same dataset

        for video_id, ann_path in zip(video_ids, ann_paths):
            ann_json = load_json_file(ann_path)
            ann = VideoAnnotation.from_json(ann_json, project_meta)

            # ignore existing key_id_map because the new objects will be created
            self.append(video_id, ann)
            if progress_cb is not None:
                progress_cb(1)

    def copy_batch(
        self,
        src_video_ids: List[int],
        dst_video_ids: List[int],
        progress_cb: Optional[Union[tqdm, Callable]] = None,
    ) -> None:
        """
        Copy annotations from one images IDs to another in API.

        :param src_video_ids: Images IDs in Supervisely.
        :type src_video_ids: List[int]
        :param dst_video_ids: Unique IDs of images in API.
        :type dst_video_ids: List[int]
        :param progress_cb: Function for tracking download progress.
        :type progress_cb: tqdm or callable, optional
        :raises: :class:`RuntimeError`, if len(src_video_ids) != len(dst_video_ids)
        :return: None
        :rtype: :class:`NoneType`

        :Usage example:

         .. code-block:: python

            import supervisely as sly
            from tqdm import tqdm

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            src_ids = [121236918, 121236919]
            dst_ids = [547837053, 547837054]
            p = tqdm(desc="Annotations copy: ", total=len(src_ids))

            copy_anns = api.annotation.copy_batch(src_ids, dst_ids, progress_cb=p)
            # Output:
            # {"message": "progress", "event_type": "EventType.PROGRESS", "subtask": "Annotations copy: ", "current": 0, "total": 2, "timestamp": "2021-03-16T15:24:31.286Z", "level": "info"}
            # {"message": "progress", "event_type": "EventType.PROGRESS", "subtask": "Annotations copy: ", "current": 2, "total": 2, "timestamp": "2021-03-16T15:24:31.288Z", "level": "info"}
        """
        if len(src_video_ids) != len(dst_video_ids):
            raise RuntimeError(
                'Can not match "src_video_ids" and "dst_video_ids" lists, '
                "len(src_video_ids) != len(dst_video_ids)"
            )
        if len(src_video_ids) == 0:
            return

        src_dataset_id = self._api.video.get_info_by_id(src_video_ids[0]).dataset_id
        dst_dataset_id = self._api.video.get_info_by_id(dst_video_ids[0]).dataset_id
        dst_dataset_info = self._api.dataset.get_info_by_id(dst_dataset_id)
        dst_project_meta = ProjectMeta.from_json(
            self._api.project.get_meta(dst_dataset_info.project_id)
        )
        for src_ids_batch, dst_ids_batch in batched(list(zip(src_video_ids, dst_video_ids))):
            ann_jsons = self.download_bulk(src_dataset_id, src_ids_batch)
            for dst_id, ann_json in zip(dst_ids_batch, ann_jsons):
                try:
                    ann = VideoAnnotation.from_json(ann_json, dst_project_meta)
                except Exception as e:
                    raise RuntimeError("Failed to validate Annotation") from e
                self.append(dst_id, ann)
                if progress_cb is not None:
                    progress_cb(1)

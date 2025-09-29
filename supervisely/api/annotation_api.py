# coding: utf-8
"""download/upload/edit :class:`Annotation<supervisely.annotation.annotation.Annotation>`"""

# docs
from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from copy import deepcopy
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Literal,
    NamedTuple,
    Optional,
    Union,
)
from uuid import uuid4

from tqdm import tqdm

from supervisely._utils import batched, run_coroutine
from supervisely.annotation.annotation import Annotation, AnnotationJsonFields
from supervisely.annotation.label import Label, LabelJsonFields
from supervisely.annotation.tag import Tag
from supervisely.annotation.tag_meta import TagMeta, TagValueType
from supervisely.api.module_api import ApiField, ModuleApi
from supervisely.geometry.alpha_mask import AlphaMask
from supervisely.geometry.constants import BITMAP
from supervisely.project.project_meta import ProjectMeta
from supervisely.project.project_type import _LABEL_GROUP_TAG_NAME
from supervisely.sly_logger import logger


class AnnotationInfo(NamedTuple):
    """
    AnnotationInfo
    """

    image_id: int
    image_name: str
    annotation: dict
    created_at: str
    updated_at: str
    dataset_id: int = None

    def to_json(self) -> Dict[str, Any]:
        """
        Convert AnnotationInfo to JSON format.

        :return: AnnotationInfo in JSON format.
        :rtype: :class:`Dict[str, Any]`
        """
        return {
            ApiField.IMAGE_ID: self.image_id,
            ApiField.IMAGE_NAME: self.image_name,
            ApiField.ANNOTATION: self.annotation,
            ApiField.CREATED_AT: self.created_at,
            ApiField.UPDATED_AT: self.updated_at,
            ApiField.DATASET_ID: self.dataset_id,
        }


class AnnotationApi(ModuleApi):
    """
    Annotation for a single image. :class:`AnnotationApi<AnnotationApi>` object is immutable.

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
        # api = sly.Api(server_address="https://app.supervisely.com", token="4r47N...xaTatb")

        dataset_id = 254737
        ann_infos = api.annotation.get_list(dataset_id)
    """

    @staticmethod
    def info_sequence():
        """
        NamedTuple AnnotationInfo information about Annotation.

        :Example:

         .. code-block:: python

            AnnotationInfo(image_id=121236919,
                           image_name='IMG_1836',
                           annotation={'description': '', 'tags': [], 'size': {'height': 800, 'width': 1067}, 'objects': []},
                           created_at='2019-12-19T12:06:59.435Z',
                           updated_at='2021-02-06T11:07:26.080Z')
        """
        return [
            ApiField.IMAGE_ID,
            ApiField.IMAGE_NAME,
            ApiField.ANNOTATION,
            ApiField.CREATED_AT,
            ApiField.UPDATED_AT,
            ApiField.DATASET_ID,
        ]

    @staticmethod
    def info_tuple_name():
        """
        NamedTuple name - **AnnotationInfo**.
        """
        return "AnnotationInfo"

    def get_list(
        self,
        dataset_id: int,
        filters: Optional[List[Dict[str, str]]] = None,
        progress_cb: Optional[Union[tqdm, Callable]] = None,
        force_metadata_for_links: Optional[bool] = True,
    ) -> List[AnnotationInfo]:
        """
        Get list of information about all annotations for a given dataset.

        :param dataset_id: Dataset ID in Supervisely.
        :type dataset_id: int
        :param filters: List of parameters to sort output Annotations.
        :type filters: List[dict], optional
        :param progress_cb: Function for tracking download progress.
        :type progress_cb: tqdm or callable, optional
        :return: Information about Annotations. See :class:`info_sequence<info_sequence>`
        :rtype: :class:`List[AnnotationInfo]`

        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            dataset_id = 254737
            ann_infos = api.annotation.get_list(dataset_id)
            print(json.dumps(ann_infos[0], indent=4))
            # Output: [
            #     121236918,
            #     "IMG_0748.jpeg",
            #     {
            #         "description": "",
            #         "tags": [],
            #         "size": {
            #             "height": 800,
            #             "width": 1067
            #         },
            #         "objects": []
            #     },
            #     "2019-12-19T12:06:59.435Z",
            #     "2021-02-06T11:07:26.080Z"
            # ]

            ann_infos_filter = api.annotation.get_list(dataset_id, filters={ 'field': 'name', 'operator': '=', 'value': 'IMG_1836' })
            print(json.dumps(ann_infos_filter, indent=4))
            # Output: [
            #     121236919,
            #     "IMG_1836",
            #     {
            #         "description": "",
            #         "tags": [],
            #         "size": {
            #             "height": 800,
            #             "width": 1067
            #         },
            #         "objects": []
            #     },
            #     "2019-12-19T12:06:59.435Z",
            #     "2021-02-06T11:07:26.080Z"
            # ]
        """
        return self.get_list_all_pages(
            "annotations.list",
            {
                ApiField.DATASET_ID: dataset_id,
                ApiField.FILTER: filters or [],
                ApiField.FORCE_METADATA_FOR_LINKS: force_metadata_for_links,
            },
            progress_cb,
        )

    def get_list_generator(
        self,
        dataset_id: int,
        filters: Optional[List[Dict[str, str]]] = None,
        progress_cb: Optional[Union[tqdm, Callable]] = None,
        batch_size: Optional[int] = 50,
        force_metadata_for_links: Optional[bool] = True,
    ) -> List[AnnotationInfo]:
        """
        Get list of information about all annotations for a given dataset.

        :param dataset_id: Dataset ID in Supervisely.
        :type dataset_id: int
        :param filters: List of parameters to sort output Annotations.
        :type filters: List[dict], optional
        :param progress_cb: Function for tracking download progress.
        :type progress_cb: tqdm or callable, optional
        :return: Information about Annotations. See :class:`info_sequence<info_sequence>`
        :rtype: :class:`List[AnnotationInfo]`

        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            dataset_id = 254737
            ann_infos = api.annotation.get_list(dataset_id)
            print(json.dumps(ann_infos[0], indent=4))
            # Output: [
            #     121236918,
            #     "IMG_0748.jpeg",
            #     {
            #         "description": "",
            #         "tags": [],
            #         "size": {
            #             "height": 800,
            #             "width": 1067
            #         },
            #         "objects": []
            #     },
            #     "2019-12-19T12:06:59.435Z",
            #     "2021-02-06T11:07:26.080Z"
            # ]

            ann_infos_filter = api.annotation.get_list(dataset_id, filters={ 'field': 'name', 'operator': '=', 'value': 'IMG_1836' })
            print(json.dumps(ann_infos_filter, indent=4))
            # Output: [
            #     121236919,
            #     "IMG_1836",
            #     {
            #         "description": "",
            #         "tags": [],
            #         "size": {
            #             "height": 800,
            #             "width": 1067
            #         },
            #         "objects": []
            #     },
            #     "2019-12-19T12:06:59.435Z",
            #     "2021-02-06T11:07:26.080Z"
            # ]
        """
        data = {
            ApiField.DATASET_ID: dataset_id,
            ApiField.FILTER: filters or [],
            ApiField.FORCE_METADATA_FOR_LINKS: force_metadata_for_links,
            ApiField.PAGINATION_MODE: ApiField.TOKEN,
        }
        if batch_size is not None:
            data[ApiField.PER_PAGE] = batch_size
        else:
            # use default value on instance (learn in API documentation)
            # 20k for instance
            # 50 by default in SDK
            pass

        return self.get_list_all_pages_generator("annotations.list", data, progress_cb)

    def download(
        self,
        image_id: int,
        with_custom_data: Optional[bool] = False,
        force_metadata_for_links: Optional[bool] = True,
    ) -> AnnotationInfo:
        """
        Download AnnotationInfo by image ID from API.

        :param image_id: Image ID in Supervisely.
        :type image_id: int
        :param with_custom_data: Include custom data in the response.
        :type with_custom_data: bool, optional
        :param force_metadata_for_links: Force metadata for links.
        :type force_metadata_for_links: bool, optional

        :return: Information about Annotation. See :class:`info_sequence<info_sequence>`
        :rtype: :class:`AnnotationInfo`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            image_id = 121236918
            ann_info = api.annotation.download(image_id)
            print(json.dumps(ann_info, indent=4))
            # Output: [
            #     121236918,
            #     "IMG_0748.jpeg",
            #     {
            #         "description": "",
            #         "tags": [],
            #         "size": {
            #             "height": 800,
            #             "width": 1067
            #         },
            #         "objects": []
            #     },
            #     "2019-12-19T12:06:59.435Z",
            #     "2021-02-06T11:07:26.080Z"
            # ]
        """
        response = self._api.post(
            "annotations.info",
            {
                ApiField.IMAGE_ID: image_id,
                ApiField.WITH_CUSTOM_DATA: with_custom_data,
                ApiField.FORCE_METADATA_FOR_LINKS: force_metadata_for_links,
                ApiField.INTEGER_COORDS: False,
            },
        )
        result = response.json()
        # Convert annotation to pixel coordinate system
        result[ApiField.ANNOTATION] = Annotation._to_pixel_coordinate_system_json(
            result[ApiField.ANNOTATION]
        )
        # check if there are any AlphaMask geometries in the batch
        additonal_geometries = defaultdict(int)
        labels = result[ApiField.ANNOTATION][AnnotationJsonFields.LABELS]
        for idx, label in enumerate(labels):
            if label[LabelJsonFields.GEOMETRY_TYPE] == AlphaMask.geometry_name():
                figure_id = label[LabelJsonFields.ID]
                additonal_geometries[figure_id] = idx

        # if so, download them separately and update the annotation
        if len(additonal_geometries) > 0:
            figure_ids = list(additonal_geometries.keys())
            figures = self._api.image.figure.download_geometries_batch(figure_ids)
            for figure_id, geometry in zip(figure_ids, figures):
                label_idx = additonal_geometries[figure_id]
                labels[label_idx].update({BITMAP: geometry})
        ann_info = self._convert_json_info(result)

        return ann_info

    def download_json(
        self,
        image_id: int,
        with_custom_data: Optional[bool] = False,
        force_metadata_for_links: Optional[bool] = True,
    ) -> Dict[str, Union[str, int, list, dict]]:
        """
        Download Annotation in json format by image ID from API.

        :param image_id: Image ID in Supervisely.
        :type image_id: int
        :param with_custom_data: Include custom data in the response.
        :type with_custom_data: bool, optional
        :param force_metadata_for_links: Force metadata for links.
        :type force_metadata_for_links: bool, optional

        :return: Annotation in json format
        :rtype: :class:`dict`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            image_id = 121236918
            ann_json = api.annotation.download_json(image_id)
            print(ann_json)
            # Output: {
            #         "description": "",
            #         "tags": [],
            #         "size": {
            #             "height": 800,
            #             "width": 1067
            #         },
            #         "objects": []
            #     }
        """
        return self.download(
            image_id=image_id,
            with_custom_data=with_custom_data,
            force_metadata_for_links=force_metadata_for_links,
        ).annotation

    def download_batch(
        self,
        dataset_id: int,
        image_ids: List[int],
        progress_cb: Optional[Union[tqdm, Callable]] = None,
        with_custom_data: Optional[bool] = False,
        force_metadata_for_links: Optional[bool] = True,
    ) -> List[AnnotationInfo]:
        """
        Get list of AnnotationInfos for given dataset ID from API.

        :param dataset_id: Dataset ID in Supervisely.
        :type dataset_id: int
        :param image_ids: List of integers.
        :type image_ids: List[int]
        :param progress_cb: Function for tracking download progress.
        :type progress_cb: tqdm
        :param with_custom_data: Include custom data in the response.
        :type with_custom_data: bool, optional
        :param force_metadata_for_links: Force metadata for links.
        :type force_metadata_for_links: bool, optional

        :return: Information about Annotations. See :class:`info_sequence<info_sequence>`
        :rtype: :class:`List[AnnotationInfo]`

        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            dataset_id = 254737
            image_ids = [121236918, 121236919]
            p = tqdm(desc="Annotations downloaded: ", total=len(image_ids))

            ann_infos = api.annotation.download_batch(dataset_id, image_ids, progress_cb=p)
            # Output:
            # {"message": "progress", "event_type": "EventType.PROGRESS", "subtask": "Annotations downloaded: ", "current": 0, "total": 2, "timestamp": "2021-03-16T15:20:06.168Z", "level": "info"}
            # {"message": "progress", "event_type": "EventType.PROGRESS", "subtask": "Annotations downloaded: ", "current": 2, "total": 2, "timestamp": "2021-03-16T15:20:06.510Z", "level": "info"}

            Optimizing the download process by using the context to avoid redundant API calls.:
            # 1. Download the project meta
            project_id = api.dataset.get_info_by_id(dataset_id).project_id
            project_meta = api.project.get_meta(project_id)

            # 2. Use the context to avoid redundant API calls
            dataset_id = 254737
            image_ids = [121236918, 121236919]
            with sly.ApiContext(api, dataset_id=dataset_id, project_id=project_id, project_meta=project_meta):
                ann_infos = api.annotation.download_batch(dataset_id, image_ids)
        """
        # use context to avoid redundant API calls
        context = self._api.optimization_context
        context_dataset_id = context.get("dataset_id")
        project_meta = context.get("project_meta")
        project_id = context.get("project_id")
        if dataset_id != context_dataset_id:
            context["dataset_id"] = dataset_id
            project_id, project_meta = None, None

        if not isinstance(project_meta, ProjectMeta):
            if project_id is None:
                project_id = self._api.dataset.get_info_by_id(dataset_id).project_id
                context["project_id"] = project_id
            project_meta = ProjectMeta.from_json(self._api.project.get_meta(project_id))
            context["project_meta"] = project_meta

        need_download_alpha_masks = False
        for obj_cls in project_meta.obj_classes:
            if obj_cls.geometry_type == AlphaMask:
                need_download_alpha_masks = True
                break

        id_to_ann = {}
        for batch in batched(image_ids):
            post_data = {
                ApiField.DATASET_ID: dataset_id,
                ApiField.IMAGE_IDS: batch,
                ApiField.WITH_CUSTOM_DATA: with_custom_data,
                ApiField.FORCE_METADATA_FOR_LINKS: force_metadata_for_links,
                ApiField.INTEGER_COORDS: False,
            }
            results = self._api.post("annotations.bulk.info", data=post_data).json()
            if need_download_alpha_masks is True:
                additonal_geometries = defaultdict(tuple)
                for ann_idx, ann_dict in enumerate(results):
                    # check if there are any AlphaMask geometries in the batch
                    for label_idx, label in enumerate(
                        ann_dict[ApiField.ANNOTATION][AnnotationJsonFields.LABELS]
                    ):
                        if label[LabelJsonFields.GEOMETRY_TYPE] == AlphaMask.geometry_name():
                            figure_id = label[LabelJsonFields.ID]
                            additonal_geometries[figure_id] = (ann_idx, label_idx)

                # if there are any AlphaMask geometries, download them separately and update the annotation
                if len(additonal_geometries) > 0:
                    figure_ids = list(additonal_geometries.keys())
                    figures = self._api.image.figure.download_geometries_batch(figure_ids)
                    for figure_id, geometry in zip(figure_ids, figures):
                        ann_idx, label_idx = additonal_geometries[figure_id]
                        results[ann_idx][ApiField.ANNOTATION][AnnotationJsonFields.LABELS][
                            label_idx
                        ].update({BITMAP: geometry})

            for ann_dict in results:
                # Convert annotation to pixel coordinate system
                ann_dict[ApiField.ANNOTATION] = Annotation._to_pixel_coordinate_system_json(
                    ann_dict[ApiField.ANNOTATION]
                )
                ann_info = self._convert_json_info(ann_dict)
                id_to_ann[ann_info.image_id] = ann_info

            if progress_cb is not None:
                progress_cb(len(batch))
        ordered_results = [id_to_ann[image_id] for image_id in image_ids]
        return ordered_results

    def download_json_batch(
        self,
        dataset_id: int,
        image_ids: List[int],
        progress_cb: Optional[Union[tqdm, Callable]] = None,
        force_metadata_for_links: Optional[bool] = True,
    ) -> List[Dict]:
        """
        Get list of AnnotationInfos for given dataset ID from API.

        :param dataset_id: Dataset ID in Supervisely.
        :type dataset_id: int
        :param image_ids: List of integers.
        :type image_ids: List[int]
        :param progress_cb: Function for tracking download progress.
        :type progress_cb: tqdm
        :param force_metadata_for_links: Force metadata for links.
        :type force_metadata_for_links: bool, optional

        :return: Information about Annotations. See :class:`info_sequence<info_sequence>`
        :rtype: :class:`List[Dict]`

        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            dataset_id = 254737
            image_ids = [121236918, 121236919]
            p = tqdm(desc="Annotations downloaded: ", total=len(image_ids))

            anns_jsons = api.annotation.download_json_batch(dataset_id, image_ids, progress_cb=p)
            # Output:
            # {"message": "progress", "event_type": "EventType.PROGRESS", "subtask": "Annotations downloaded: ", "current": 0, "total": 2, "timestamp": "2021-03-16T15:20:06.168Z", "level": "info"}
            # {"message": "progress", "event_type": "EventType.PROGRESS", "subtask": "Annotations downloaded: ", "current": 2, "total": 2, "timestamp": "2021-03-16T15:20:06.510Z", "level": "info"}
        """
        results = self.download_batch(
            dataset_id=dataset_id,
            image_ids=image_ids,
            progress_cb=progress_cb,
            force_metadata_for_links=force_metadata_for_links,
        )
        return [ann_info.annotation for ann_info in results]

    def upload_path(
        self,
        img_id: int,
        ann_path: str,
        skip_bounds_validation: Optional[bool] = False,
    ) -> None:
        """
        Loads an annotation from a given path to a given image ID in the API.

        :param img_id: Image ID in Supervisely.
        :type img_id: int
        :param ann_path: Path to annotation on host.
        :type ann_path: str
        :return: None
        :rtype: :class:`NoneType`

        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            image_id = 121236918
            ann_path = '/home/admin/work/supervisely/example/ann.json'
            upl_path = api.annotation.upload_path(image_id, ann_path)
        """
        self.upload_paths([img_id], [ann_path], skip_bounds_validation=skip_bounds_validation)

    def upload_paths(
        self,
        img_ids: List[int],
        ann_paths: List[str],
        progress_cb: Optional[Union[tqdm, Callable]] = None,
        skip_bounds_validation: Optional[bool] = False,
    ) -> None:
        """
        Loads an annotations from a given paths to a given images IDs in the API. Images IDs must be from one dataset.

        :param img_ids: Images IDs in Supervisely.
        :type img_ids: List[int]
        :param ann_paths: Paths to annotations on local machine.
        :type ann_paths: List[str]
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

            img_ids = [121236918, 121236919]
            ann_pathes = ['/home/admin/work/supervisely/example/ann1.json', '/home/admin/work/supervisely/example/ann2.json']
            upl_paths = api.annotation.upload_paths(img_ids, ann_pathes)

            # Optimizing the upload process by using the context to avoid redundant API calls.
            # Usefull when uploading a large number of annotations in one dataset.
            # 1. Download the project meta
            dataset_id = 254737
            project_id = api.dataset.get_info_by_id(dataset_id).project_id
            project_meta = api.project.get_meta(project_id)

            # 2. Use the context to avoid redundant API calls
            with sly.ApiContext(api, dataset_id=dataset_id, project_id=project_id, project_meta=project_meta):
                api.annotation.upload_paths(img_ids, ann_pathes)
        """

        def read_json(ann_path):
            with open(ann_path) as json_file:
                return json.load(json_file)

        self._upload_batch(
            read_json,
            img_ids,
            ann_paths,
            progress_cb,
            skip_bounds_validation=skip_bounds_validation,
        )

    def upload_json(
        self,
        img_id: int,
        ann_json: Dict,
        skip_bounds_validation: Optional[bool] = False,
    ) -> None:
        """
        Loads an annotation from dict to a given image ID in the API.

        :param img_id: Image ID in Supervisely.
        :type img_id: int
        :param ann_json: Annotation in JSON format.
        :type ann_json: dict
        :return: None
        :rtype: :class:`NoneType`

        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            image_id = 121236918
            upl_json = api.annotation.upload_json(image_id, ann_json)
        """
        self.upload_jsons([img_id], [ann_json], skip_bounds_validation=skip_bounds_validation)

    def upload_jsons(
        self,
        img_ids: List[int],
        ann_jsons: List[Dict],
        progress_cb: Optional[Union[tqdm, Callable]] = None,
        skip_bounds_validation: Optional[bool] = False,
    ) -> None:
        """
        Loads an annotations from dicts to a given images IDs in the API. Images IDs must be from one dataset.

        :param img_ids: Image ID in Supervisely.
        :type img_ids: List[int]
        :param ann_jsons: Annotation in JSON format.
        :type ann_jsons: List[dict]
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

            img_ids = [121236918, 121236919]
            api.annotation.upload_jsons(img_ids, ann_jsons)

            # Optimizing the upload process by using the context to avoid redundant API calls.
            # Usefull when uploading a large number of annotations in one dataset.
            # 1. Download the project meta
            dataset_id = 254737
            project_id = api.dataset.get_info_by_id(dataset_id).project_id
            project_meta = api.project.get_meta(project_id)

            # 2. Use the context to avoid redundant API calls
            with sly.ApiContext(api, dataset_id=dataset_id, project_id=project_id, project_meta=project_meta):
                api.annotation.upload_jsons(img_ids, ann_jsons)
        """
        self._upload_batch(
            lambda x: x,
            img_ids,
            ann_jsons,
            progress_cb,
            skip_bounds_validation=skip_bounds_validation,
        )

    def upload_ann(
        self,
        img_id: int,
        ann: Annotation,
        skip_bounds_validation: Optional[bool] = False,
    ) -> None:
        """
        Loads an :class:`Annotation<supervisely.annotation.annotation.Annotation>` to a given image ID in the API.

        :param img_id: Image ID in Supervisely.
        :type img_id: int
        :param ann: Annotation object.
        :type ann: Annotation
        :return: None
        :rtype: :class:`NoneType`

        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            image_id = 121236918
            upl_ann = api.annotation.upload_ann(image_id, ann)
        """
        self.upload_anns([img_id], [ann], skip_bounds_validation=skip_bounds_validation)

    def upload_anns(
        self,
        img_ids: List[int],
        anns: List[Annotation],
        progress_cb: Optional[Union[tqdm, Callable]] = None,
        skip_bounds_validation: Optional[bool] = False,
    ) -> None:
        """
        Loads an :class:`Annotations<supervisely.annotation.annotation.Annotation>` to a given images IDs in the API. Images IDs must be from one dataset.

        :param img_ids: Image ID in Supervisely.
        :type img_ids: List[int]
        :param anns: List of Annotation objects.
        :type anns: List[Annotation]
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

            img_ids = [121236918, 121236919]
            upl_anns = api.annotation.upload_anns(img_ids, [ann1, ann2])

            # Optimizing the upload process by using the context to avoid redundant API calls.
            # Usefull when uploading a large number of annotations in one dataset.
            # 1. Download the project meta
            dataset_id = 254737
            project_id = api.dataset.get_info_by_id(dataset_id).project_id
            project_meta = api.project.get_meta(project_id)

            # 2. Use the context to avoid redundant API calls
            with sly.ApiContext(api, dataset_id=dataset_id, project_id=project_id, project_meta=project_meta):
                api.annotation.upload_anns(img_ids, [ann1, ann2])
        """
        # img_ids from the same dataset
        self._upload_batch(
            Annotation.to_json,
            img_ids,
            anns,
            progress_cb,
            skip_bounds_validation=skip_bounds_validation,
        )

    def _upload_batch(
        self,
        func_ann_to_json: Callable,
        img_ids: List[int],
        anns: List[Union[Dict, Annotation, str]],
        progress_cb=None,
        skip_bounds_validation: Optional[bool] = False,
    ):
        """
        General method for uploading annotations to instance.

        Method is used in: upload_paths, upload_jsons, upload_anns

        :param func_ann_to_json: Function to convert annotation to json or read annotation from file.
        :type func_ann_to_json: callable
        :param img_ids: List of image IDs in Supervisely to which annotations will be uploaded.
        :type img_ids: List[int]
        :param anns: List of annotations. Can be json, Annotation object or path to annotation file.
        :type anns: List[Union[Dict, Annotation, str]]
        :param progress_cb: Function for tracking download progress.
        :type progress_cb: tqdm or callable, optional
        :param skip_bounds_validation: Skip bounds validation.
        :type skip_bounds_validation: bool, optional
        :return: None
        :rtype: :class:`NoneType`
        """
        # img_ids from the same dataset
        if len(img_ids) == 0:
            return
        if len(img_ids) != len(anns):
            raise RuntimeError(
                f'Lists "img_ids" and "anns" have different lengths: {len(img_ids)} != {len(anns)}.'
            )

        # use context to avoid redundant API calls
        dataset_id = self._api.image.get_info_by_id(
            img_ids[0], force_metadata_for_links=False
        ).dataset_id
        context = self._api.optimization_context
        context_dataset_id = context.get("dataset_id")
        project_id = context.get("project_id")
        project_meta = context.get("project_meta")
        if dataset_id != context_dataset_id:
            context["dataset_id"] = dataset_id
            project_id, project_meta = None, None

        if not isinstance(project_meta, ProjectMeta):
            if project_id is None:
                project_id = self._api.dataset.get_info_by_id(dataset_id).project_id
                context["project_id"] = project_id
            project_meta = ProjectMeta.from_json(self._api.project.get_meta(project_id))
            context["project_meta"] = project_meta

        need_upload_alpha_masks = False
        for obj_cls in project_meta.obj_classes:
            if obj_cls.geometry_type == AlphaMask:
                need_upload_alpha_masks = True
                break

        for batch in batched(list(zip(img_ids, anns))):
            data = []
            if need_upload_alpha_masks:
                special_figures = []
                special_geometries = []
                # check if there are any AlphaMask geometries in the batch
                for img_id, ann in batch:
                    ann_json = func_ann_to_json(ann)
                    ann_json = deepcopy(ann_json)  # Avoid changing the original data

                    ann_json = Annotation._to_subpixel_coordinate_system_json(ann_json)
                    filtered_labels = []
                    if AnnotationJsonFields.LABELS not in ann_json:
                        raise RuntimeError(
                            f"Annotation JSON does not contain '{AnnotationJsonFields.LABELS}' field"
                        )
                    for label_json in ann_json[AnnotationJsonFields.LABELS]:
                        for key in [
                            LabelJsonFields.GEOMETRY_TYPE,
                            LabelJsonFields.OBJ_CLASS_NAME,
                        ]:
                            if key not in label_json:
                                raise RuntimeError(f"Label JSON does not contain '{key}' field")
                        if label_json[LabelJsonFields.GEOMETRY_TYPE] == AlphaMask.geometry_name():
                            label_json.update({ApiField.ENTITY_ID: img_id})

                            obj_cls_name = label_json.get(LabelJsonFields.OBJ_CLASS_NAME)
                            obj_cls = project_meta.get_obj_class(obj_cls_name)
                            if obj_cls is None:
                                raise RuntimeError(
                                    f"Object class '{obj_cls_name}' not found in project meta"
                                )
                            # update obj class id in label json
                            label_json[LabelJsonFields.OBJ_CLASS_ID] = obj_cls.sly_id

                            geometry = label_json.pop(
                                BITMAP
                            )  # remove alpha mask geometry from label json
                            special_geometries.append(geometry)
                            special_figures.append(label_json)
                        else:
                            filtered_labels.append(label_json)
                    if len(filtered_labels) != len(ann_json[AnnotationJsonFields.LABELS]):
                        ann_json[AnnotationJsonFields.LABELS] = filtered_labels
                    data.append({ApiField.IMAGE_ID: img_id, ApiField.ANNOTATION: ann_json})
            else:
                for img_id, ann in batch:
                    ann_json = func_ann_to_json(ann)
                    ann_json = deepcopy(ann_json)  # Avoid changing the original data
                    ann_json = Annotation._to_subpixel_coordinate_system_json(ann_json)
                    data.append({ApiField.IMAGE_ID: img_id, ApiField.ANNOTATION: ann_json})

            self._api.post(
                "annotations.bulk.add",
                data={
                    ApiField.DATASET_ID: dataset_id,
                    ApiField.ANNOTATIONS: data,
                    ApiField.SKIP_BOUNDS_VALIDATION: skip_bounds_validation,
                },
            )
            if need_upload_alpha_masks:
                if len(special_figures) > 0:
                    # 1. create figures
                    json_body = {
                        ApiField.DATASET_ID: dataset_id,
                        ApiField.FIGURES: special_figures,
                        ApiField.SKIP_BOUNDS_VALIDATION: skip_bounds_validation,
                    }
                    resp = self._api.post("figures.bulk.add", json_body)
                    added_fig_ids = [resp_obj[ApiField.ID] for resp_obj in resp.json()]

                    # 2. upload alpha mask geometries
                    self._api.image.figure.upload_geometries_batch(
                        added_fig_ids, special_geometries
                    )

            if progress_cb is not None:
                progress_cb(len(batch))

    def get_info_by_id(self, id):
        """
        get_info_by_id
        """
        raise NotImplementedError("Method is not supported")

    def get_info_by_name(self, parent_id, name):
        """
        get_info_by_name
        """
        raise NotImplementedError("Method is not supported")

    def exists(self, parent_id, name):
        """
        exists
        """
        raise NotImplementedError("Method is not supported")

    def get_free_name(self, parent_id, name):
        """
        get_free_name
        """
        raise NotImplementedError("Method is not supported")

    def _add_sort_param(self, data):
        """
        _add_sort_param
        """
        return data

    def copy_batch(
        self,
        src_image_ids: List[int],
        dst_image_ids: List[int],
        progress_cb: Optional[Union[tqdm, Callable]] = None,
        force_metadata_for_links: Optional[bool] = True,
        skip_bounds_validation: Optional[bool] = False,
    ) -> None:
        """
        Copy annotations from one images IDs to another in API.

        :param src_image_ids: Images IDs in Supervisely.
        :type src_image_ids: List[int]
        :param dst_image_ids: Unique IDs of images in API.
        :type dst_image_ids: List[int]
        :param progress_cb: Function for tracking download progress.
        :type progress_cb: tqdm or callable, optional
        :raises: :class:`RuntimeError`, if len(src_image_ids) != len(dst_image_ids)
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
        if len(src_image_ids) != len(dst_image_ids):
            raise RuntimeError(
                'Can not match "src_image_ids" and "dst_image_ids" lists, '
                "len(src_image_ids) != len(dst_image_ids)"
            )
        if len(src_image_ids) == 0:
            return

        src_dataset_id = self._api.image.get_info_by_id(src_image_ids[0]).dataset_id
        for cur_batch in batched(list(zip(src_image_ids, dst_image_ids))):
            src_ids_batch, dst_ids_batch = zip(*cur_batch)
            ann_infos = self.download_batch(
                src_dataset_id,
                src_ids_batch,
                force_metadata_for_links=force_metadata_for_links,
            )
            ann_jsons = [ann_info.annotation for ann_info in ann_infos]
            self.upload_jsons(
                dst_ids_batch, ann_jsons, skip_bounds_validation=skip_bounds_validation
            )
            if progress_cb is not None:
                progress_cb(len(src_ids_batch))

    def copy(
        self,
        src_image_id: int,
        dst_image_id: int,
        force_metadata_for_links: Optional[bool] = True,
        skip_bounds_validation: Optional[bool] = False,
    ) -> None:
        """
        Copy annotation from one image ID to another image ID in API.

        :param src_image_id: Image ID in Supervisely.
        :type src_image_id: int
        :param dst_image_id: Image ID in Supervisely.
        :type dst_image_id: int
        :return: None
        :rtype: :class:`NoneType`

        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            src_id = 121236918
            dst_id = 547837053
            api.annotation.copy(src_id, dst_id)
        """
        self.copy_batch(
            [src_image_id],
            [dst_image_id],
            force_metadata_for_links=force_metadata_for_links,
            skip_bounds_validation=skip_bounds_validation,
        )

    def copy_batch_by_ids(
        self,
        src_image_ids: List[int],
        dst_image_ids: List[int],
        batch_size: Optional[int] = 50,
        save_source_date: Optional[bool] = True,
    ) -> None:
        """
        Copy annotations from one images IDs to another images IDs in API.

        :param src_image_ids: Images IDs in Supervisely.
        :type src_image_ids: List[int]
        :param dst_image_ids: Images IDs in Supervisely.
        :type dst_image_ids: List[int]
        :return: None
        :rtype: :class:`NoneType`
        :raises: :class:`RuntimeError` if len(src_image_ids) != len(dst_image_ids)

        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            src_ids = [121236918, 121236919]
            dst_ids = [547837053, 547837054]
            api.annotation.copy_batch_by_ids(src_ids, dst_ids)
        """
        if len(src_image_ids) != len(dst_image_ids):
            raise RuntimeError(
                'Can not match "src_image_ids" and "dst_image_ids" lists, '
                "len(src_image_ids) != len(dst_image_ids)"
            )
        if len(src_image_ids) == 0:
            return
        for cur_batch in batched(list(zip(src_image_ids, dst_image_ids)), batch_size=batch_size):
            src_ids_batch, dst_ids_batch = zip(*cur_batch)
            self._api.post(
                "annotations.bulk.copy",
                data={
                    "srcImageIds": src_ids_batch,
                    "destImageIds": dst_ids_batch,
                    "preserveSourceDate": save_source_date,
                },
            )

    def _convert_json_info(self, info: dict, skip_missing=True) -> AnnotationInfo:
        """
        _convert_json_info
        """
        res = super()._convert_json_info(info, skip_missing=skip_missing)
        return AnnotationInfo(**res._asdict())

    def append_labels(
        self,
        image_id: int,
        labels: List[Label],
        skip_bounds_validation: Optional[bool] = False,
    ) -> None:
        """
        Append labels to image with given ID in API.

        :param image_id: Image ID to append labels.
        :type image_id: int
        :param labels: List of labels to append.
        :type labels: List[Label]
        :return: None
        :rtype: :class:`NoneType`
        """
        if len(labels) == 0:
            return

        alpha_mask_geometry = AlphaMask._impl_json_class_name()
        payload = []
        special_geometries = {}
        for idx, label in enumerate(labels):
            _label_json = label.to_json()
            if isinstance(label.geometry, AlphaMask):
                _label_json.pop(alpha_mask_geometry)  # remove alpha mask geometry from label json
                special_geometries[idx] = label.geometry.to_json()[BITMAP]
            else:
                _label_json["geometry"] = label.geometry.to_json()
            if "classId" not in _label_json:
                raise KeyError("Update project meta from server to get class id")
            payload.append(_label_json)

        added_ids = []
        for batch_jsons in batched(payload, batch_size=100):
            resp = self._api.post(
                "figures.bulk.add",
                {
                    ApiField.ENTITY_ID: image_id,
                    ApiField.FIGURES: batch_jsons,
                    ApiField.SKIP_BOUNDS_VALIDATION: skip_bounds_validation,
                },
            )
            for resp_obj in resp.json():
                figure_id = resp_obj[ApiField.ID]
                added_ids.append(figure_id)

        # upload alpha mask geometries
        if len(special_geometries) > 0:
            fidure_ids = [added_ids[idx] for idx in special_geometries.keys()]
            self._api.image.figure.upload_geometries_batch(fidure_ids, special_geometries.values())

    def get_label_by_id(
        self, label_id: int, project_meta: ProjectMeta, with_tags: Optional[bool] = True
    ) -> Label:
        """Returns Supervisely Label object by it's ID.

        :param label_id: ID of the label to get
        :type label_id: int
        :param project_meta: Supervisely ProjectMeta object
        :type project_meta: ProjectMeta
        :param with_tags: If True, tags will be added to the Label object
        :type with_tags: bool, optional
        :return: Supervisely Label object
        :rtype: Label
        :Usage example:

         .. code-block:: python

            import os
            from dotenv import load_dotenv

            import supervisely as sly

            # Load secrets and create API object from .env file (recommended)
            # Learn more here: https://developer.supervisely.com/getting-started/basics-of-authentication
            load_dotenv(os.path.expanduser("~/supervisely.env"))
            api = sly.Api.from_env()

            label_id = 121236918

            project_id = 254737
            project_meta = sly.ProjectMeta.from_json(api.project.get_meta(project_id))

            label = api.annotation.get_label_by_id(label_id, project_meta)
        """
        resp = self._api.get(
            "figures.info", {ApiField.ID: label_id, ApiField.DECOMPRESS_BITMAP: False}
        )
        geometry = resp.json()

        class_id = geometry.get("classId")
        geometry["classTitle"] = project_meta.get_obj_class_by_id(class_id).name
        geometry.update(geometry.get("geometry"))
        geometry["tags"] = self._get_label_tags(label_id) if with_tags else []

        return Label.from_json(geometry, project_meta)

    def _get_label_tags(self, label_id: int) -> List[Dict[str, Any]]:
        """Returns tags of the label with given ID in JSON format.

        :param label_id: ID of the label to get tags
        :type label_id: int
        :return: list of tags in JSON format
        :rtype: List[Dict[str, Any]]
        :Usage example:

         .. code-block:: python

            import os
            from dotenv import load_dotenv

            import supervisely as sly

            # Load secrets and create API object from .env file (recommended)
            # Learn more here: https://developer.supervisely.com/getting-started/basics-of-authentication
            load_dotenv(os.path.expanduser("~/supervisely.env"))
            api = sly.Api.from_env()

            label_id = 121236918
            tags_json = api.annotation.get_label_tags(label_id)
        """
        return self._api.get("figures.tags.list", {ApiField.ID: label_id}).json()

    def update_label(self, label_id: int, label: Label) -> None:
        """Updates label with given ID in Supervisely with new Label object.
        NOTE: This method only updates label's geometry and tags, not class title, etc.

        :param label_id: ID of the label to update
        :type label_id: int
        :param label: Supervisely Label object
        :type label: Label
        :Usage example:

         .. code-block:: python

            import os
            from dotenv import load_dotenv

            import supervisely as sly

            # Load secrets and create API object from .env file (recommended)
            # Learn more here: https://developer.supervisely.com/getting-started/basics-of-authentication
            load_dotenv(os.path.expanduser("~/supervisely.env"))
            api = sly.Api.from_env()

            new_label: sly.Label
            label_id = 121236918

            api.annotation.update_label(label_id, new_label)
        """
        payload = {
            ApiField.ID: label_id,
            ApiField.TAGS: [tag.to_json() for tag in label.tags],
            ApiField.GEOMETRY: label.geometry.to_json(),
            ApiField.NN_CREATED: label._nn_created,
            ApiField.NN_UPDATED: label._nn_updated,
        }
        self._api.post("figures.editInfo", payload)

    def update_label_priority(self, label_id: int, priority: int) -> None:
        """Updates label's priority with given ID in Supervisely.
        Priority increases with the number: a higher number indicates a higher priority.
        The higher priority means that the label will be displayed on top of the others.
        The lower priority means that the label will be displayed below the others.

        :param label_id: ID of the label to update
        :type label_id: int
        :param priority: New priority of the label
        :type priority: int

        :Usage example:

            .. code-block:: python

            import os
            from dotenv import load_dotenv

            import supervisely as sly

            # Load secrets and create API object from .env file (recommended)
            # Learn more here: https://developer.supervisely.com/getting-started/basics-of-authentication
            load_dotenv(os.path.expanduser("~/supervisely.env"))

            api = sly.Api.from_env()

            label_ids = [123, 456, 789]
            priorities = [1, 2, 3]

            for label_id, priority in zip(label_ids, priorities):
                api.annotation.update_label_priority(label_id, priority)

            # The label with ID 789 will be displayed on top of the others.
            # The label with ID 123 will be displayed below the others.

        """
        self._api.post(
            "figures.priority.update",
            {
                ApiField.ID: label_id,
                ApiField.PRIORITY: priority,
            },
        )

    async def download_async(
        self,
        image_id: int,
        semaphore: Optional[asyncio.Semaphore] = None,
        with_custom_data: Optional[bool] = False,
        force_metadata_for_links: Optional[bool] = True,
        progress_cb: Optional[Union[tqdm, Callable]] = None,
        progress_cb_type: Literal["number", "size"] = "number",
    ) -> AnnotationInfo:
        """
        Download AnnotationInfo by image ID from API.

        :param image_id: Image ID in Supervisely.
        :type image_id: int
        :param semaphore: Semaphore for limiting the number of simultaneous downloads.
        :type semaphore: asyncio.Semaphore, optional
        :param with_custom_data: Include custom data in the response.
        :type with_custom_data: bool, optional
        :param force_metadata_for_links: Force metadata for links.
        :type force_metadata_for_links: bool, optional
        :param progress_cb: Function for tracking download progress.
        :type progress_cb: tqdm or callable, optional
        :param progress_cb_type: Type of progress callback. Can be "number" or "size". Default is "number".
        :type progress_cb_type: str, optional
        :return: Information about Annotation. See :class:`info_sequence<info_sequence>`
        :rtype: :class:`AnnotationInfo`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            image_id = 121236918
            loop = sly.utils.get_or_create_event_loop()
            ann_info = loop.run_until_complete(api.annotation.download_async(image_id))
        """
        if semaphore is None:
            semaphore = self._api.get_default_semaphore()
        async with semaphore:
            response = await self._api.post_async(
                "annotations.info",
                {
                    ApiField.IMAGE_ID: image_id,
                    ApiField.WITH_CUSTOM_DATA: with_custom_data,
                    ApiField.FORCE_METADATA_FOR_LINKS: force_metadata_for_links,
                    ApiField.INTEGER_COORDS: False,
                },
            )
            if progress_cb is not None and progress_cb_type == "size":
                progress_cb(len(response.content))

            result = response.json()
        # Convert annotation to pixel coordinate system
        result[ApiField.ANNOTATION] = Annotation._to_pixel_coordinate_system_json(
            result[ApiField.ANNOTATION]
        )
        # check if there are any AlphaMask geometries in the batch
        additonal_geometries = defaultdict(int)
        labels = result[ApiField.ANNOTATION][AnnotationJsonFields.LABELS]
        for idx, label in enumerate(labels):
            if label[LabelJsonFields.GEOMETRY_TYPE] == AlphaMask.geometry_name():
                figure_id = label[LabelJsonFields.ID]
                additonal_geometries[figure_id] = idx

        # if so, download them separately and update the annotation
        if len(additonal_geometries) > 0:
            figure_ids = list(additonal_geometries.keys())
            figures = await self._api.image.figure.download_geometries_batch_async(
                figure_ids,
                (progress_cb if progress_cb is not None and progress_cb_type == "size" else None),
                semaphore=semaphore,
            )
            for figure_id, geometry in zip(figure_ids, figures):
                label_idx = additonal_geometries[figure_id]
                labels[label_idx].update({BITMAP: geometry})
        ann_info = self._convert_json_info(result)
        if progress_cb is not None and progress_cb_type == "number":
            progress_cb(1)
        return ann_info

    async def download_batch_async(
        self,
        dataset_id: int,
        image_ids: List[int],
        semaphore: Optional[asyncio.Semaphore] = None,
        with_custom_data: Optional[bool] = False,
        force_metadata_for_links: Optional[bool] = True,
        progress_cb: Optional[Union[tqdm, Callable]] = None,
        progress_cb_type: Literal["number", "size"] = "number",
    ) -> List[AnnotationInfo]:
        """
        Get list of AnnotationInfos for given dataset ID from API.

        :param dataset_id: Dataset ID in Supervisely.
        :type dataset_id: int
        :param image_ids: List of integers.
        :type image_ids: List[int]
        :param semaphore: Semaphore for limiting the number of simultaneous downloads.
        :type semaphore: asyncio.Semaphore, optional
        :param with_custom_data: Include custom data in the response.
        :type with_custom_data: bool, optional
        :param force_metadata_for_links: Force metadata for links.
        :type force_metadata_for_links: bool, optional
        :param progress_cb: Function for tracking download progress. Total should be equal to len(image_ids) or None.
        :type progress_cb: tqdm or callable, optional
        :param progress_cb_type: Type of progress callback. Can be "number" or "size". Default is "number".
        :type progress_cb_type: str, optional
        :return: Information about Annotations. See :class:`info_sequence<info_sequence>`
        :rtype: :class:`List[AnnotationInfo]`

        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            dataset_id = 254737
            image_ids = [121236918, 121236919]
            pbar = tqdm(desc="Download annotations", total=len(image_ids))

            loop = sly.utils.get_or_create_event_loop()
            ann_infos = loop.run_until_complete(
                                api.annotation.download_batch_async(dataset_id, image_ids, progress_cb=pbar)
                            )
        """

        # use context to avoid redundant API calls
        context = self._api.optimization_context
        context_dataset_id = context.get("dataset_id")
        project_meta = context.get("project_meta")
        project_id = context.get("project_id")
        if dataset_id != context_dataset_id:
            context["dataset_id"] = dataset_id
            project_id, project_meta = None, None

        if not isinstance(project_meta, ProjectMeta):
            if project_id is None:
                project_id = self._api.dataset.get_info_by_id(dataset_id).project_id
                context["project_id"] = project_id
            project_meta = ProjectMeta.from_json(self._api.project.get_meta(project_id))
            context["project_meta"] = project_meta

        if semaphore is None:
            semaphore = self._api.get_default_semaphore()
        tasks = []
        for image in image_ids:
            task = self.download_async(
                image_id=image,
                semaphore=semaphore,
                with_custom_data=with_custom_data,
                force_metadata_for_links=force_metadata_for_links,
                progress_cb=progress_cb,
                progress_cb_type=progress_cb_type,
            )
            tasks.append(task)
        ann_infos = await asyncio.gather(*tasks)
        return ann_infos

    async def download_bulk_async(
        self,
        dataset_id: int,
        image_ids: List[int],
        progress_cb: Optional[Union[tqdm, Callable]] = None,
        with_custom_data: Optional[bool] = False,
        force_metadata_for_links: Optional[bool] = True,
        semaphore: Optional[asyncio.Semaphore] = None,
    ) -> List[AnnotationInfo]:
        """
        Get list of AnnotationInfos for given dataset ID from API.
        This method is optimized for downloading a large number of small size annotations with a single API call.

        :param dataset_id: Dataset ID in Supervisely.
        :type dataset_id: int
        :param image_ids: List of integers.
        :type image_ids: List[int]
        :param progress_cb: Function for tracking download progress.
        :type progress_cb: tqdm
        :param with_custom_data: Include custom data in the response.
        :type with_custom_data: bool, optional
        :param force_metadata_for_links: Force metadata for links.
        :type force_metadata_for_links: bool, optional
        :param semaphore: Semaphore for limiting the number of simultaneous downloads.
        :type semaphore: asyncio.Semaphore, optional
        :return: Information about Annotations. See :class:`info_sequence<info_sequence>`
        :rtype: :class:`List[AnnotationInfo]`

        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            dataset_id = 254737
            image_ids = [121236918, 121236919]
            p = tqdm(desc="Annotations downloaded: ", total=len(image_ids))

            ann_infos = await api.annotation.download_bulk_async(dataset_id, image_ids, progress_cb=p)

            Optimizing the download process by using the context to avoid redundant API calls.:
            # 1. Download the project meta
            project_id = api.dataset.get_info_by_id(dataset_id).project_id
            project_meta = api.project.get_meta(project_id)

            # 2. Use the context to avoid redundant API calls
            dataset_id = 254737
            image_ids = [121236918, 121236919]
            with sly.ApiContext(api, dataset_id=dataset_id, project_id=project_id, project_meta=project_meta):
                ann_infos = await api.annotation.download_bulk_async(dataset_id, image_ids)
        """
        if semaphore is None:
            semaphore = self._api.get_default_semaphore()

        # use context to avoid redundant API calls
        context = self._api.optimization_context
        context_dataset_id = context.get("dataset_id")
        project_meta = context.get("project_meta")
        project_id = context.get("project_id")
        if dataset_id != context_dataset_id:
            context["dataset_id"] = dataset_id
            project_id, project_meta = None, None

        if not isinstance(project_meta, ProjectMeta):
            if project_id is None:
                project_id = self._api.dataset.get_info_by_id(dataset_id).project_id
                context["project_id"] = project_id
            project_meta = ProjectMeta.from_json(self._api.project.get_meta(project_id))
            context["project_meta"] = project_meta

        need_download_alpha_masks = False
        for obj_cls in project_meta.obj_classes:
            if obj_cls.geometry_type == AlphaMask:
                need_download_alpha_masks = True
                break

        id_to_ann = {}
        for batch in batched(image_ids):
            json_data = {
                ApiField.DATASET_ID: dataset_id,
                ApiField.IMAGE_IDS: batch,
                ApiField.WITH_CUSTOM_DATA: with_custom_data,
                ApiField.FORCE_METADATA_FOR_LINKS: force_metadata_for_links,
                ApiField.INTEGER_COORDS: False,
            }
            async with semaphore:
                results = await self._api.post_async("annotations.bulk.info", json=json_data)
                results = results.json()
            if need_download_alpha_masks is True:
                additonal_geometries = defaultdict(tuple)
                for ann_idx, ann_dict in enumerate(results):
                    # check if there are any AlphaMask geometries in the batch
                    for label_idx, label in enumerate(
                        ann_dict[ApiField.ANNOTATION][AnnotationJsonFields.LABELS]
                    ):
                        if label[LabelJsonFields.GEOMETRY_TYPE] == AlphaMask.geometry_name():
                            figure_id = label[LabelJsonFields.ID]
                            additonal_geometries[figure_id] = (ann_idx, label_idx)

                # if there are any AlphaMask geometries, download them separately and update the annotation
                if len(additonal_geometries) > 0:
                    figure_ids = list(additonal_geometries.keys())
                    figures = await self._api.image.figure.download_geometries_batch_async(
                        figure_ids, semaphore=semaphore
                    )
                    for figure_id, geometry in zip(figure_ids, figures):
                        ann_idx, label_idx = additonal_geometries[figure_id]
                        results[ann_idx][ApiField.ANNOTATION][AnnotationJsonFields.LABELS][
                            label_idx
                        ].update({BITMAP: geometry})

            for ann_dict in results:
                # Convert annotation to pixel coordinate system
                ann_dict[ApiField.ANNOTATION] = Annotation._to_pixel_coordinate_system_json(
                    ann_dict[ApiField.ANNOTATION]
                )
                ann_info = self._convert_json_info(ann_dict)
                id_to_ann[ann_info.image_id] = ann_info

            if progress_cb is not None:
                progress_cb(len(batch))
        ordered_results = [id_to_ann[image_id] for image_id in image_ids]
        return ordered_results

    def append_labels_group(
        self,
        dataset_id: int,
        image_ids: List[int],
        labels: List[Label],
        project_meta: Optional[ProjectMeta] = None,
        group_name: Optional[str] = None,
    ) -> None:
        """
        Append group of labels to corresponding multiview images.
        This method will automatically add a tech tag to the labels to group them together.
        Please note that grouped labels is supported only in images project with multiview setup.

        :param dataset_id: Dataset ID in Supervisely.
        :type dataset_id: int
        :param image_ids: List of Images IDs in Supervisely.
        :type image_ids: List[int]
        :param labels: List of Labels in Supervisely.
        :type labels: List[Label]
        :param project_meta: Project meta. If not provided, will try to get it from the server.
        :type project_meta: ProjectMeta, optional
        :param group_name: Group name. Labels will be assigned by tag with this value.
        :type group_name: str, optional
        :return: :class:`None<None>`
        :rtype: :class:`NoneType<NoneType>`
        :raises ValueError: if number of images and labels are not the same

        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            dataset_id = 123456
            paths = ['path/to/audi_01.png', 'path/to/audi_02.png']
            images_group_name = 'audi'

            image_infos = api.image.upload_multiview_images(dataset_id, images_group_name, paths)

            image_ids = [info.id for info in image_infos]
            labels = [label1, label2]
            labels_group_name = 'left_wheel'

            # upload group of labels to corresponding multiview images
            api.annotation.append_labels_group(image_ids, labels, labels_group_name)
        """

        if len(image_ids) != len(labels):
            raise ValueError(
                "Number of images and labels must be the same."
                "If specific image does not have label, pass None instead."
            )

        if group_name is None:
            group_name = str(uuid4().hex)

        if project_meta is None:
            logger.warning(
                "Project meta is not provided. Will try to get it from the server. "
                "It is recommended to provide project meta to avoid extra API requests."
            )
            dataset_info = self._api.dataset.get_info_by_id(dataset_id)
            if dataset_info is None:
                raise ValueError(f"Dataset with ID {dataset_id} not found")

            project_id = dataset_info.project_id
            project_meta = ProjectMeta.from_json(self._api.project.get_meta(project_id))

        tag_meta = TagMeta(_LABEL_GROUP_TAG_NAME, TagValueType.ANY_STRING)
        labels = [l.add_tag(Tag(tag_meta, group_name)) for l in labels if l is not None]

        anns_json = self._api.annotation.download_json_batch(
            dataset_id=dataset_id, image_ids=image_ids, force_metadata_for_links=False
        )
        anns = [Annotation.from_json(ann_json, project_meta) for ann_json in anns_json]
        updated_anns = [ann.add_label(label) for ann, label in zip(anns, labels)]

        self._api.annotation.upload_anns(image_ids, updated_anns)

    async def upload_anns_async(
        self,
        image_ids: List[int],
        anns: Union[List[Annotation], Generator],
        dataset_id: Optional[int] = None,
        log_progress: bool = True,
        semaphore: Optional[asyncio.Semaphore] = None,
    ) -> None:
        """
        Optimized method for uploading annotations to images in large batches.
        This method significantly improves performance when uploading large numbers of annotations
        by processing different components in parallel batches.

        IMPORTANT: If you pass anns as a generator, you must be sure that the generator will yield the same number of annotations
        as the number of image IDs provided.

        The method works by:
        1. Separating regular figures and alpha masks for specialized processing
        2. Batching figure creation requests to reduce API overhead
        3. Processing image-level tags, object tags, and geometries separately
        4. Using concurrent async operations to maximize throughput
        5. Processing alpha mask geometries with specialized upload method

        This approach can be faster than traditional sequential upload methods
        when dealing with large annotation batches.

        :param image_ids: List of image IDs in Supervisely.
        :type image_ids: List[int]
        :param anns: List of annotations to upload. Can be a generator or a list.
        :type anns: Union[List[Annotation], Generator]
        :param dataset_id: Dataset ID. If None, will be determined from image IDs or context.
        :type dataset_id: int, optional
        :param log_progress: Whether to log progress information.
        :type log_progress: bool, optional
        :param semaphore: Semaphore to control concurrency level. If None, a default will be used.
        :type semaphore: asyncio.Semaphore, optional
        :return: None
        :rtype: :class:`NoneType`

        :Usage example:

        .. code-block:: python

            import asyncio
            import supervisely as sly
            from tqdm import tqdm

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            # Prepare your annotations and image IDs
            image_ids = [121236918, 121236919]
            anns = [annotation1, annotation2]

            # Option 1: Using the synchronous wrapper
            api.annotation.upload_anns_fast(image_ids, anns)

            # Option 2: Using the async method directly
            upload_annotations = api.annotation.upload_anns_async(
                    image_ids,
                    anns,
                    semaphore=asyncio.Semaphore(10)  # Control concurrency
                )

            sly.run_coroutine(upload_annotations)
        """
        if len(image_ids) == 0:
            return

        if not isinstance(anns, Generator):
            if len(image_ids) != len(anns):
                raise RuntimeError(
                    'Can not match "img_ids" and "anns" lists, len(img_ids) != len(anns)'
                )

        if semaphore is None:
            semaphore = self._api.get_default_semaphore()

        def _groupby_image_tags(image_level_tags: dict, tag_meta: ProjectMeta) -> dict:
            """
            Group image tags by tag_id and tag_value for efficient batch processing
            Returns: Dict[tag_id, Dict[tag_value, List[image_ids]]]
            """
            result = defaultdict(lambda: defaultdict(list))
            for img_id, tags in image_level_tags.items():
                for tag in tags:
                    sly_id = tag_meta.get(tag.name).sly_id
                    value = tag.value
                    result[sly_id][value].append(img_id)
            return result

        def _prepare_tags(tags: List[Tag]) -> List[Dict[str, Any]]:
            """
            Prepare tags for bulk upload
            Returns: List[Dict[str, Any]]
            """
            return [
                {
                    ApiField.TAG_ID: tag_metas.get(tag.name).sly_id,
                    ApiField.FIGURE_ID: None,
                    ApiField.VALUE: tag.value,
                }
                for tag in tags
            ]

        # Handle context and dataset_id
        context = self._api.optimization_context
        context_dataset_id = context.get("dataset_id")
        project_id = context.get("project_id")
        project_meta = context.get("project_meta")

        # Determine dataset_id with proper fallback logic
        if dataset_id is None:
            dataset_id = context_dataset_id
            if dataset_id is None:
                dataset_id = self._api.image.get_info_by_id(
                    image_ids[0], force_metadata_for_links=False
                ).dataset_id
                context["dataset_id"] = dataset_id
                project_id, project_meta = None, None
        # If dataset_id was provided but differs from context (or context is None)
        elif dataset_id != context_dataset_id or context_dataset_id is None:
            context["dataset_id"] = dataset_id
            project_id, project_meta = None, None

        # Get project meta if needed
        if not isinstance(project_meta, ProjectMeta):
            if project_id is None:
                project_id = self._api.dataset.get_info_by_id(dataset_id).project_id
                context["project_id"] = project_id
            project_meta = ProjectMeta.from_json(self._api.project.get_meta(project_id))
            context["project_meta"] = project_meta

        tag_metas = project_meta.tag_metas

        # Prepare bulk data
        regular_figures = []
        regular_figures_tags = []
        alpha_mask_figures = []
        alpha_mask_geometries = []
        alpha_mask_figures_tags = []
        image_level_tags = {}  # Track image-level tags by image ID
        image_tags_count = 0

        for img_id, ann in zip(image_ids, anns):
            # Handle image-level tags
            if len(ann.img_tags) > 0:
                image_tags_count += len(ann.img_tags)
                image_level_tags[img_id] = [tag for tag in ann.img_tags]

            if len(ann.labels) == 0:
                continue

            # Process each label in the annotation
            for label in ann.labels:
                obj_cls = project_meta.get_obj_class(label.obj_class.name)
                if obj_cls is None:
                    raise RuntimeError(
                        f"Object class '{label.obj_class.name}' not found in project meta"
                    )

                figure_data = {
                    ApiField.ENTITY_ID: img_id,
                    LabelJsonFields.OBJ_CLASS_ID: obj_cls.sly_id,
                }

                if isinstance(label.geometry, AlphaMask):
                    geometry = label.geometry.to_json()[BITMAP]
                    figure_data[LabelJsonFields.GEOMETRY_TYPE] = AlphaMask.geometry_name()
                    alpha_mask_figures.append(figure_data)
                    alpha_mask_geometries.append(geometry)
                    alpha_mask_figures_tags.append(_prepare_tags(label.tags))
                else:
                    figure_data[LabelJsonFields.GEOMETRY_TYPE] = label.geometry.name()
                    figure_data[ApiField.GEOMETRY] = label.geometry.to_json()
                    regular_figures.append(figure_data)
                    regular_figures_tags.append(_prepare_tags(label.tags))

        async def create_figures_batch(figures_batch, tags_batch, progress_cb):
            """Create a batch of figures and associate their tags"""
            async with semaphore:
                response = await self._api.post_async(
                    "figures.bulk.add",
                    json={
                        ApiField.DATASET_ID: dataset_id,
                        ApiField.FIGURES: figures_batch,
                    },
                )
                figure_ids = [item[ApiField.ID] for item in response.json()]

                # Update tags with figure IDs
                for figure_id, tags in zip(figure_ids, tags_batch):
                    for tag in tags:
                        tag[ApiField.FIGURE_ID] = figure_id
                if progress_cb is not None:
                    progress_cb.update(len(figures_batch))
                return figure_ids, tags_batch

        async def add_tags_to_objects(tags_batch, progress_cb):
            """Add tags to objects in batches"""
            if not tags_batch:
                return

            async with semaphore:
                await self._api.post_async(
                    "figures.tags.bulk.add",
                    json={
                        ApiField.PROJECT_ID: project_id,
                        ApiField.TAGS: tags_batch,
                    },
                )
                if progress_cb is not None:
                    progress_cb.update(len(tags_batch))

        async def add_tags_to_images(tag_id, tag_value, image_ids_batch, progress_cb):
            """Add a tag to multiple images"""
            async with semaphore:
                await self._api.post_async(
                    "image-tags.bulk.add-to-image",
                    json={
                        ApiField.TAG_ID: tag_id,
                        ApiField.VALUE: tag_value,
                        ApiField.IDS: image_ids_batch,
                    },
                )
                if progress_cb is not None:
                    progress_cb.update(len(image_ids_batch))

        # 1. Process regular figures
        regular_figure_tasks = []
        batch_size = 1000

        if log_progress:
            f_pbar = tqdm(
                desc="Uploading figures", total=len(regular_figures) + len(alpha_mask_figures)
            )
        else:
            f_pbar = None

        for figures_batch, tags_batch in zip(
            batched(regular_figures, batch_size),
            batched(regular_figures_tags, batch_size),
        ):
            task = create_figures_batch(figures_batch, tags_batch, f_pbar)
            regular_figure_tasks.append(task)

        # 2. Process alpha mask figures
        alpha_mask_tasks = []
        for figures_batch, tags_batch in zip(
            batched(alpha_mask_figures, batch_size),
            batched(alpha_mask_figures_tags, batch_size),
        ):
            task = create_figures_batch(figures_batch, tags_batch, f_pbar)
            alpha_mask_tasks.append(task)

        # Wait for all figure creation tasks to complete
        regular_results = (
            await asyncio.gather(*regular_figure_tasks) if regular_figure_tasks else []
        )
        alpha_results = await asyncio.gather(*alpha_mask_tasks) if alpha_mask_tasks else []

        # 3. Upload alpha mask geometries
        alpha_figure_ids = []
        for figure_ids, _ in alpha_results:
            alpha_figure_ids.extend(figure_ids)

        if log_progress:
            am_pbar = tqdm(desc="Uploading alpha mask geometries", total=len(alpha_mask_geometries))
        else:
            am_pbar = None
        alpha_mask_geometry_task = self._api.image.figure.upload_geometries_batch_async(
            alpha_figure_ids,
            alpha_mask_geometries,
            semaphore=semaphore,
            progress_cb=am_pbar,
        )

        # 4. Collect all object tags
        all_object_tags = []
        for _, tags_batch in regular_results + alpha_results:
            for tags in tags_batch:
                all_object_tags.extend(tags)

        # 5. Add tags to objects in batches
        object_tag_tasks = []
        batch_size = 1000

        if log_progress:
            ot_pbar = tqdm(desc="Uploading tags to objects", total=len(all_object_tags))
        else:
            ot_pbar = None
        for tags_batch in batched(all_object_tags, batch_size):
            task = add_tags_to_objects(tags_batch, ot_pbar)
            object_tag_tasks.append(task)

        # 6. Add tags to images
        image_tag_tasks = []
        batch_size = 1000
        if log_progress:
            it_pbar = tqdm(desc="Uploading tags to images", total=image_tags_count)
        else:
            it_pbar = None
        image_tags_by_meta = _groupby_image_tags(image_level_tags, tag_metas)
        for tag_meta_id, values_dict in image_tags_by_meta.items():
            for tag_value, img_ids_for_tag in values_dict.items():
                for batch in batched(img_ids_for_tag, batch_size):
                    task = add_tags_to_images(tag_meta_id, tag_value, batch, it_pbar)
                    image_tag_tasks.append(task)

        # Execute all remaining tasks
        await asyncio.gather(alpha_mask_geometry_task, *object_tag_tasks, *image_tag_tasks)

    def upload_anns_fast(
        self,
        image_ids: List[int],
        anns: List[Annotation],
        dataset_id: Optional[int] = None,
        log_progress: bool = True,
    ) -> None:
        """
        Upload annotations to images in a dataset using optimized method.

        :param image_ids: List of image IDs in Supervisely.
        :type image_ids: List[int]
        :param anns: List of Annotation objects.
        :type anns: List[Annotation]
        :param dataset_id: Dataset ID. If None, will be determined from image IDs or context.
        :type dataset_id: int, optional
        :param log_progress: Whether to log progress information.
        :type log_progress: bool, optional
        :return: None
        :rtype: :class:`NoneType`

        :Usage example:

            .. code-block:: python

            import supervisely as sly
            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            dataset_id = 123456
            image_ids = [121236918, 121236919]
            anns = [annotation1, annotation2]
            api.annotation.upload_fast(image_ids, anns, dataset_id)

        """
        upload_coroutine = self.upload_anns_async(
            image_ids=image_ids,
            anns=anns,
            dataset_id=dataset_id,
            log_progress=log_progress,
        )
        run_coroutine(upload_coroutine)

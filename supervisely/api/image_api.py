# coding: utf-8
"""Download/upload images from/to Supervisely."""

# docs
from __future__ import annotations

import asyncio
import copy
import io
import json
import os
import pickle
import re
import tempfile
import urllib.parse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from math import ceil
from pathlib import Path
from time import sleep
from typing import (
    Any,
    AsyncGenerator,
    Callable,
    Dict,
    Generator,
    Iterator,
    List,
    Literal,
    NamedTuple,
    Optional,
    Tuple,
    Union,
)
from uuid import uuid4

import aiofiles
import numpy as np
import requests
from requests.exceptions import HTTPError
from requests_toolbelt import MultipartDecoder, MultipartEncoder
from tqdm import tqdm

from supervisely._utils import (
    batched,
    compare_dicts,
    generate_free_name,
    get_bytes_hash,
    resize_image_url,
)
from supervisely.annotation.annotation import Annotation
from supervisely.annotation.tag import Tag
from supervisely.annotation.tag_meta import TagApplicableTo, TagMeta, TagValueType
from supervisely.api.constants import DOWNLOAD_BATCH_SIZE
from supervisely.api.dataset_api import DatasetInfo
from supervisely.api.entities_collection_api import (
    AiSearchThresholdDirection,
    CollectionTypeFilter,
)
from supervisely.api.entity_annotation.figure_api import FigureApi
from supervisely.api.entity_annotation.tag_api import TagApi
from supervisely.api.file_api import FileInfo
from supervisely.api.module_api import (
    ApiField,
    RemoveableBulkModuleApi,
    _get_single_item,
)
from supervisely.imaging import image as sly_image
from supervisely.io.env import (
    add_uploaded_ids_to_env,
    app_categories,
    increment_upload_count,
)
from supervisely.io.fs import (
    OFFSETS_PKL_BATCH_SIZE,
    OFFSETS_PKL_SUFFIX,
    clean_dir,
    ensure_base_path,
    get_file_ext,
    get_file_hash,
    get_file_hash_async,
    get_file_name,
    get_file_name_with_ext,
    list_files,
    list_files_recursively,
)
from supervisely.project.project_meta import ProjectMeta
from supervisely.project.project_type import (
    _BLOB_TAG_NAME,
    _MULTISPECTRAL_TAG_NAME,
    _MULTIVIEW_TAG_NAME,
)
from supervisely.sly_logger import logger

SUPPORTED_CONFLICT_RESOLUTIONS = ["skip", "rename", "replace"]
API_DEFAULT_PER_PAGE = 500


@dataclass
class BlobImageInfo:
    """
    Object with image parameters that describes image in blob file.

    :Example:

     .. code-block:: python

        BlobImageInfo(
            name='IMG_3861.jpeg',
            offset_start=0,
            offset_end=148388,
        )
    """

    name: str
    offset_start: int
    offset_end: int

    @staticmethod
    def from_image_info(image_info: ImageInfo) -> BlobImageInfo:
        return BlobImageInfo(
            name=image_info.name,
            offset_start=image_info.offset_start,
            offset_end=image_info.offset_end,
        )

    def add_team_file_id(self, team_file_id: int):
        """
        Add file ID from Team Files to BlobImageInfo object to extend data imported from offsets file.
        This data is used to link offsets with blob file that is already uploaded to Supervisely storage.
        """
        setattr(self, "team_file_id", team_file_id)
        return self

    def to_dict(self, team_file_id: int = None) -> Dict:
        """
        Create dictionary from BlobImageInfo object that can be used for request to Supervisely API.
        """
        return {
            ApiField.TITLE: self.name,
            ApiField.TEAM_FILE_ID: team_file_id or getattr(self, "team_file_id", None),
            ApiField.SOURCE_BLOB: {
                ApiField.OFFSET_START: self.offset_start,
                ApiField.OFFSET_END: self.offset_end,
            },
        }

    @staticmethod
    def from_dict(offset_dict: Dict, return_team_file_id: bool = False) -> BlobImageInfo:
        """
        Create BlobImageInfo object from dictionary that is returned by Supervisely API.

        :param offset_dict: Dictionary with image offsets.
        :type offset_dict: Dict
        :param return_team_file_id: If True, return team file ID.
                                    Default is False to make size of the object smaller for pickling.
        :type return_team_file_id: bool
        :return: BlobImageInfo object.
        :rtype: BlobImageInfo
        """
        blob_info = BlobImageInfo(
            name=offset_dict[ApiField.TITLE],
            offset_start=offset_dict[ApiField.SOURCE_BLOB][ApiField.OFFSET_START],
            offset_end=offset_dict[ApiField.SOURCE_BLOB][ApiField.OFFSET_END],
        )
        if return_team_file_id:
            blob_info.add_team_file_id(offset_dict[ApiField.TEAM_FILE_ID])
        return blob_info

    @property
    def offsets_dict(self) -> Dict:
        return {
            ApiField.OFFSET_START: self.offset_start,
            ApiField.OFFSET_END: self.offset_end,
        }

    @staticmethod
    def load_from_pickle_generator(
        file_path: str, batch_size: int = OFFSETS_PKL_BATCH_SIZE
    ) -> Generator[List["BlobImageInfo"], None, None]:
        """
        Load BlobImageInfo objects from a pickle file in batches of specified size.
        The file should contain a list of BlobImageInfo objects.

        :param file_path: Path to the pickle file containing BlobImageInfo objects.
        :type file_path: str
        :param batch_size: Size of each batch. Default is 10000.
        :type batch_size: int
        :return: Generator yielding batches of BlobImageInfo objects.
        :rtype: Generator[List[BlobImageInfo], None, None]
        """
        try:
            current_batch = []

            with open(file_path, "rb") as f:
                while True:
                    try:
                        # Load one pickle object at a time
                        data = pickle.load(f)

                        if isinstance(data, list):
                            # More efficient way to process lists
                            remaining_items = data
                            while remaining_items:
                                # Calculate how many more items we need to fill the current batch
                                items_needed = batch_size - len(current_batch)

                                if items_needed > 0:
                                    # Take only what we need from the remaining items
                                    current_batch.extend(remaining_items[:items_needed])
                                    remaining_items = remaining_items[items_needed:]
                                else:
                                    # current_batch is already full or overflowing, don't add more items
                                    # and proceed directly to yielding the batch
                                    pass

                                # If we have a full batch, yield it
                                if len(current_batch) >= batch_size:
                                    yield current_batch
                                    current_batch = []
                        else:
                            # Handle single item
                            current_batch.append(data)

                            if len(current_batch) >= batch_size:
                                yield current_batch
                                current_batch = []

                    except EOFError:
                        # End of file reached
                        break
                    except Exception as e:
                        logger.error(f"Error reading pickle data: {str(e)}")
                        break

            # Yield any remaining items in the final batch
            if current_batch:
                yield current_batch

        except Exception as e:
            logger.error(f"Failed to load BlobImageInfo objects from {file_path}: {str(e)}")
            yield []

    @staticmethod
    def dump_to_pickle(
        offsets: Union[Generator[List[BlobImageInfo]], List[BlobImageInfo]], file_path: str
    ):
        """
        Dump BlobImageInfo objects to a pickle file in batches.
        To read the data back, use the `load_from_pickle_generator` method.

        :param offsets: Generator yielding batches of BlobImageInfo objects or a list of BlobImageInfo objects.
        :type offsets: Generator[List[BlobImageInfo]] or List[BlobImageInfo]
        :param file_path: Path to the pickle file.
        :type file_path: str
        """

        try:
            if isinstance(offsets, Generator):
                with open(file_path, "ab") as f:
                    for batch in offsets:
                        pickle.dump(batch, f)
            elif isinstance(offsets, list):
                with open(file_path, "ab") as f:
                    pickle.dump(offsets, f)
            else:
                raise NotImplementedError(
                    f"Invalid type of 'offsets' parameter for 'dump_offsets' method: {type(offsets)}"
                )
        except Exception as e:
            logger.error(f"Failed to dump BlobImageInfo objects to {file_path}: {str(e)}")


class ImageInfo(NamedTuple):
    """
    Object with image parameters from Supervisely.

    :Example:

     .. code-block:: python

        ImageInfo(
            id=770915,
            name='IMG_3861.jpeg',
            link=None,
            hash='ZdpMD+ZMJx0R8BgsCzJcqM7qP4M8f1AEtoYc87xZmyQ=',
            mime='image/jpeg',
            ext='jpeg',
            size=148388,
            width=1067,
            height=800,
            labels_count=4,
            dataset_id=2532,
            created_at='2021-03-02T10:04:33.973Z',
            updated_at='2021-03-02T10:04:33.973Z',
            meta={},
            path_original='/h5un6l2bnaz1vj8a9qgms4-public/images/original/7/h/Vo/...jpg',
            full_storage_url='http://app.supervisely.com/h5un6l2bnaz1vj8a9qgms4-public/images/original/7/h/Vo/...jpg'),
            tags=[],
            created_by='admin'
            related_data_id=None,
            download_id=None,
            offset_start=None,
            offset_end=None,
        )
    """

    #: :class:`int`: Image ID in Supervisely.
    id: int

    #: :class:`str`: Image filename.
    name: str

    #: :class:`str`: Use link as ID for images that are expected to be stored at remote server.
    #: e.g. "http://your-server/image1.jpg".
    link: str

    #: :class:`str`: Image hash obtained by base64(sha256(file_content)).
    #: Use hash for files that are expected to be stored at Supervisely or your deployed agent.
    hash: str

    #: :class:`str`: Image MIME type.
    mime: str

    #: :class:`str`: Image file extension.
    ext: str

    #: :class:`int`: Image size (in bytes).
    size: int

    #: :class:`int`: Image width.
    width: int

    #: :class:`int`: Image height.
    height: int

    #: :class:`int`: Number of :class:`Labels<supervisely.annotation.label.Label>` in the Image.
    labels_count: int

    #: :class:`int`: :class:`Dataset<supervisely.project.project.Dataset>` ID in Supervisely.
    dataset_id: int

    #: :class:`str`: Image creation time. e.g. "2019-02-22T14:59:53.381Z".
    created_at: str

    #: :class:`str`: Time of last image update. e.g. "2019-02-22T14:59:53.381Z".
    updated_at: str

    #: :class:`dict`: Custom additional image info that contain image technical and/or user-generated data.
    #: To set custom sort parameter for image, you can do the follwoing:
    #: 1. With the uploading use `add_custom_sort` context manager to set the key name of meta object that will be used for custom sorting.
    #: 2. Before uploading add value to meta dict with method `update_custom_sort`
    #: 3. Before uploading add key-value pair with key `customSort` to meta dict, image info file or meta file.
    #: 4. After uploading `set_custom_sort` method to set custom sort value for image.
    #: e.g. {'my-key':'a', 'my-key: "b", "customSort": "c"}.
    meta: dict

    #: :class:`str`: Relative storage URL to image. e.g.
    #: "/h5un6l2bnaz1vj8a9qgms4-public/images/original/7/h/Vo/...jpg".
    path_original: str

    #: :class:`str`: Full storage URL to image. e.g.
    #: "http://app.supervisely.com/h5un6l2bnaz1vj8a9qgms4-public/images/original/7/h/Vo/...jpg".
    full_storage_url: str

    #: :class:`str`: Image :class:`Tags<supervisely.annotation.tag.Tag>` list.
    #: e.g. "[{'entityId': 2836466, 'tagId': 345022, 'id': 2224609, 'labelerLogin': 'admin',
    #: 'createdAt': '2021-03-05T14:15:39.923Z', 'updatedAt': '2021-03-05T14:15:39.923Z'}, {...}]".
    tags: List[Dict]

    #: :class:`str`: ID of a user who created the image.
    created_by: str

    #: :class:`int`: ID of the blob file in Supervisely storage related to the image.
    related_data_id: Optional[int] = None

    #: :class:`str`: Unique ID of the image that links it to the corresponding blob file in Supervisely storage
    #: uses for downloading source blob file.
    download_id: Optional[str] = None

    #: :class:`int`: Bytes offset of the blob file that points to the start of the image data.
    offset_start: Optional[int] = None

    #: :class:`int`: Bytes offset of the blob file that points to the end of the image data.
    offset_end: Optional[int] = None

    #: :class:`dict`: Image meta that could have the confidence level of the image in Enntities Collection of type AI Search.
    ai_search_meta: Optional[dict] = None

    #: :class:`str`: Timestamp of the last update of the embeddings for the image.
    #: This field is used to track when the embeddings were last updated.
    #: It is set to None if the embeddings have never been computed for the image.
    #: Format: "YYYY-MM-DDTHH:MM:SS.sssZ"
    embeddings_updated_at: Optional[str] = None

    #: :class:`int`: :class:`Dataset<supervisely.project.project.Project>` ID in Supervisely.
    project_id: int = None

    # DO NOT DELETE THIS COMMENT
    #! New fields must be added with default values to keep backward compatibility.

    @property
    def preview_url(self):
        """
        Get Image preview URL.

        :return: Image preview URL.
        :rtype: :class:`str`
        """
        return resize_image_url(self.full_storage_url)


class ImageApi(RemoveableBulkModuleApi):
    """
    API for working with :class:`Image<supervisely.imaging.image>`. :class:`ImageApi<ImageApi>` object is immutable.

    :param api: API connection to the server
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

        image_info = api.image.get_info_by_id(image_id) # api usage example
    """

    def __init__(self, api):
        super().__init__(api)
        self.figure = FigureApi(api)  # @TODO: rename to object like in labeling UI
        self.tag = TagApi(api)

    @staticmethod
    def info_sequence():
        """
        Get list of all :class:`ImageInfo<ImageInfo>` field names.

        :return: List of :class:`ImageInfo<ImageInfo>` field names.`
        :rtype: :class:`List[str]`
        """
        return [
            ApiField.ID,
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
            ApiField.UPDATED_AT,
            ApiField.META,
            ApiField.PATH_ORIGINAL,
            ApiField.FULL_STORAGE_URL,
            ApiField.TAGS,
            ApiField.CREATED_BY_ID[0][0],
            ApiField.RELATED_DATA_ID,
            ApiField.DOWNLOAD_ID,
            ApiField.OFFSET_START,
            ApiField.OFFSET_END,
            ApiField.AI_SEARCH_META,
            ApiField.EMBEDDINGS_UPDATED_AT,
            ApiField.PROJECT_ID,
        ]

    @staticmethod
    def info_tuple_name():
        """
        Get string name of :class:`ImageInfo<ImageInfo>` NamedTuple.

        :return: NamedTuple name.
        :rtype: :class:`str`
        """
        return "ImageInfo"

    def _add_custom_sort(self, meta: dict, name: Optional[str] = None) -> dict:
        """
        Add `customSort` key with value to meta dict based on the `sort_by` attribute of `ImageApi` object:
         - `sort_by` attribute is set by `add_custom_sort` context manager and available for the duration of the context.
         - `sort_by` attribute is used to set the key name of meta object that will "link" its value to the custom sorting.

        :param meta: Custom additional image info that contain image technical and/or user-generated data.
        :type meta: dict
        :param name: Name of the image. Used for improved debug logging.
        :type name: str, optional
        :return: Updated meta.
        :rtype: dict
        """
        sort_value = meta.get(self.sort_by, None)
        if sort_value:
            meta[ApiField.CUSTOM_SORT] = str(sort_value)
            message = f"Custom sorting applied with key '{self.sort_by}' and value '{sort_value}'."
        else:
            message = f"Custom sorting will not be applied. Key '{self.sort_by}' not found in meta."
        if name:
            message = f"Image '{name}': {message}"
        logger.debug(message)
        return meta

    @contextmanager
    def add_custom_sort(self, key: str):
        """
        Use this context manager to set the key name of meta object that will be used for custom sorting.
        This context manager allows you to set the `sort_by` attribute of ImageApi object for the duration of the context, then delete it.
        If nested functions support this functionality, each image they process will automatically receive a custom sorting parameter based on the available meta object.

        :param key: It is a key name of meta object that will be used for sorting.
        :type key: str
        """
        # pylint: disable=access-member-before-definition
        if hasattr(self, "sort_by") and self.sort_by != key:
            raise AttributeError(
                f"Attribute 'sort_by' already exists and has different value: {self.sort_by}"
            )
        # pylint: enable=access-member-before-definition
        self.sort_by = key
        self.sort_by_context_counter = getattr(self, "sort_by_context_counter", 0) + 1
        try:
            yield
        finally:
            self.sort_by_context_counter -= 1
            if self.sort_by_context_counter == 0:
                del self.sort_by
                del self.sort_by_context_counter

    def get_list_generator(
        self,
        dataset_id: int = None,
        filters: Optional[List[Dict[str, str]]] = None,
        sort: Optional[str] = "id",  #! Does not work with pagination mode 'token'
        sort_order: Optional[str] = "asc",  #! Does not work with pagination mode 'token'
        limit: Optional[int] = None,
        force_metadata_for_links: Optional[bool] = False,
        batch_size: Optional[int] = None,
        project_id: int = None,
    ) -> Iterator[List[ImageInfo]]:
        """
        Returns a generator that yields lists of images in the given :class:`Dataset<supervisely.project.project.Dataset>` or :class:`Project<supervisely.project.project.Project>`.

        :param dataset_id: :class:`Dataset<supervisely.project.project.Dataset>` ID in which the Images are located.
        :type dataset_id: :class:`int`
        :param filters: List of params to sort output Images.
        :type filters: :class:`List[Dict]`, optional
        :param sort: Field name to sort. One of {'id' (default), 'name', 'description', 'labelsCount', 'createdAt', 'updatedAt', `customSort`}
        :type sort: :class:`str`, optional
        :param sort_order: Sort order. One of {'asc' (default), 'desc'}
        :type sort_order: :class:`str`, optional
        :param limit: Max number of list elements. No limit if None (default).
        :type limit: :class:`int`, optional
        :param force_metadata_for_links: If True, updates meta for images with remote storage links when listing.
        :type force_metadata_for_links: bool, optional
        :param batch_size: Number of images to get in each request.
        :type batch_size: int, optional
        :param project_id: :class:`Project<supervisely.project.project.Project>` ID in which the Images are located.
        :type project_id: :class:`int`
        :return: Generator that yields lists of images in the given :class:`Dataset<supervisely.project.project.Dataset>` or :class:`Project<supervisely.project.project.Project>`.

        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            for images_batch in api.image.get_list_generator(dataset_id):
                print(images_batch)
        """

        self._validate_project_and_dataset_id(project_id, dataset_id)
        data = {
            ApiField.PROJECT_ID: project_id,
            ApiField.DATASET_ID: dataset_id,
            ApiField.FILTER: filters or [],
            ApiField.SORT: sort,
            ApiField.SORT_ORDER: sort_order,
            ApiField.FORCE_METADATA_FOR_LINKS: force_metadata_for_links,
            ApiField.PAGINATION_MODE: ApiField.TOKEN,
        }

        if batch_size is not None:
            data[ApiField.PER_PAGE] = batch_size
        else:
            # use default value on instance (20k)
            # #tag/Images/paths/~1images.list/get
            pass
        return self.get_list_all_pages_generator(
            "images.list",
            data,
            limit=limit,
            return_first_response=False,
        )

    def get_list(
        self,
        dataset_id: int = None,
        filters: Optional[List[Dict[str, str]]] = None,
        sort: Optional[str] = "id",
        sort_order: Optional[str] = "asc",
        limit: Optional[int] = None,
        force_metadata_for_links: Optional[bool] = True,
        return_first_response: Optional[bool] = False,
        project_id: Optional[int] = None,
        only_labelled: Optional[bool] = False,
        fields: Optional[List[str]] = None,
        recursive: Optional[bool] = False,
        entities_collection_id: Optional[int] = None,
        ai_search_collection_id: Optional[int] = None,
        ai_search_threshold: Optional[float] = None,
        ai_search_threshold_direction: AiSearchThresholdDirection = AiSearchThresholdDirection.ABOVE,
        extra_fields: Optional[List[str]] = None,
    ) -> List[ImageInfo]:
        """
        List of Images in the given :class:`Dataset<supervisely.project.project.Dataset>`.

        :param dataset_id: :class:`Dataset<supervisely.project.project.Dataset>` ID in which the Images are located.
        :type dataset_id: :class:`int`
        :param filters: List of params to sort output Images.
        :type filters: :class:`List[Dict]`, optional
        :param sort: Field name to sort. One of {'id' (default), 'name', 'description', 'labelsCount', 'createdAt', 'updatedAt', `customSort`}
        :type sort: :class:`str`, optional
        :param sort_order: Sort order. One of {'asc' (default), 'desc'}
        :type sort_order: :class:`str`, optional
        :param limit: Max number of list elements. No limit if None (default).
        :type limit: :class:`int`, optional
        :param force_metadata_for_links: If True, updates meta for images with remote storage links when listing.
        :type force_metadata_for_links: bool, optional
        :param return_first_response: If True, returns first response without waiting for all pages.
        :type return_first_response: bool, optional
        :param project_id: :class:`Project<supervisely.project.project.Project>` ID in which the Images are located.
        :type project_id: :class:`int`
        :param only_labelled: If True, returns only images with labels.
        :type only_labelled: bool, optional
        :param fields: List of fields to return. If None, returns all fields.
        :type fields: List[str], optional
        :param recursive: If True, returns all images from dataset recursively (including images in nested datasets).
        :type recursive: bool, optional
        :param entities_collection_id: :class:`EntitiesCollection` ID of `Default` type to which the images belong.
        :type entities_collection_id: int, optional
        :param ai_search_collection_id: :class:`EntitiesCollection` ID of type `AI Search` to which the images belong.
        :type ai_search_collection_id: int, optional
        :param ai_search_threshold: Confidence level to filter images in AI Search collection.
        :type ai_search_threshold: float, optional
        :param ai_search_threshold_direction: Direction of the confidence level filter. One of {'above' (default), 'below'}.
        :type ai_search_threshold_direction: str, optional
        :param extra_fields: List of extra fields to return. If None, returns no extra fields.
        :type extra_fields: List[str], optional
        :return: Objects with image information from Supervisely.
        :rtype: :class:`List[ImageInfo]<ImageInfo>`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            # Get list of Images with width = 1067
            img_infos = api.image.get_list(dataset_id, filters=[{ 'field': 'width', 'operator': '=', 'value': '1067' }])
            print(img_infos)
            # Output: [ImageInfo(id=770915,
            #                    name='IMG_3861.jpeg',
            #                    link=None,
            #                    hash='ZdpMD+ZMJx0R8BgsCzJcqM7qP4M8f1AEtoYc87xZmyQ=',
            #                    mime='image/jpeg',
            #                    ext='jpeg',
            #                    size=148388,
            #                    width=1067,
            #                    height=800,
            #                    labels_count=4,
            #                    dataset_id=2532,
            #                    created_at='2021-03-02T10:04:33.973Z',
            #                    updated_at='2021-03-02T10:04:33.973Z',
            #                    meta={},
            #                    path_original='/h5un6l2bnaz1vj8a9qgms4-public/images/original/7/h/Vo/...jpg',
            #                    full_storage_url='http://app.supervisely.com/h5un6l2bnaz1vj8a9qgms4-public/images/original/7/h/Vo/...jpg'),
            #                    tags=[],
            # ImageInfo(id=770916,
            #           name='IMG_1836.jpeg',
            #           link=None,
            #           hash='YZKQrZH5C0rBvGGA3p7hjWahz3/pV09u5m30Bz8GeYs=',
            #           mime='image/jpeg',
            #           ext='jpeg',
            #           size=140222,
            #           width=1067,
            #           height=800,
            #           labels_count=3,
            #           dataset_id=2532,
            #           created_at='2021-03-02T10:04:33.973Z',
            #           updated_at='2021-03-02T10:04:33.973Z',
            #           meta={},
            #           path_original='/h5un6l2bnaz1vj8a9qgms4-public/images/original/C/Y/Hq/...jpg',
            #           full_storage_url='http://app.supervisely.com/h5un6l2bnaz1vj8a9qgms4-public/images/original/C/Y/Hq/...jpg'),
            #           tags=[]
            # ]
        """
        self._validate_project_and_dataset_id(project_id, dataset_id)
        data = {
            ApiField.PROJECT_ID: project_id,
            ApiField.DATASET_ID: dataset_id,
            ApiField.FILTER: filters or [],
            ApiField.SORT: sort,
            ApiField.SORT_ORDER: sort_order,
            ApiField.FORCE_METADATA_FOR_LINKS: force_metadata_for_links,
            ApiField.RECURSIVE: recursive,
        }
        if only_labelled:
            data[ApiField.FILTERS] = [
                {
                    "type": "objects_class",
                    "data": {
                        "from": 1,
                        "to": 9999,
                        "include": True,
                        "classId": None,
                    },
                }
            ]
        # Handle collection filtering
        collection_info = None
        if entities_collection_id is not None and ai_search_collection_id is not None:
            raise ValueError(
                "You can use only one of entities_collection_id or ai_search_collection_id"
            )
        elif entities_collection_id is not None:
            collection_info = (CollectionTypeFilter.DEFAULT, entities_collection_id)
        elif ai_search_collection_id is not None:
            collection_info = (CollectionTypeFilter.AI_SEARCH, ai_search_collection_id)

        if collection_info is not None:
            collection_type, collection_id = collection_info
            if ApiField.FILTERS not in data:
                data[ApiField.FILTERS] = []

            collection_filter_data = {
                ApiField.COLLECTION_ID: collection_id,
                ApiField.INCLUDE: True,
            }
            if ai_search_threshold is not None:
                if collection_type != CollectionTypeFilter.AI_SEARCH:
                    raise ValueError(
                        "ai_search_threshold is only available for AI Search collection"
                    )
                collection_filter_data[ApiField.THRESHOLD] = ai_search_threshold
                collection_filter_data[ApiField.THRESHOLD_DIRECTION] = ai_search_threshold_direction
            data[ApiField.FILTERS].append(
                {
                    ApiField.TYPE: collection_type,
                    ApiField.DATA: collection_filter_data,
                }
            )

        if fields is not None:
            data[ApiField.FIELDS] = fields
        if extra_fields is not None:
            data[ApiField.EXTRA_FIELDS] = extra_fields
        return self.get_list_all_pages(
            "images.list",
            data=data,
            limit=limit,
            return_first_response=return_first_response,
        )

    def get_filtered_list(
        self,
        dataset_id: int = None,
        filters: Optional[List[Dict]] = None,
        sort: Optional[str] = "id",
        sort_order: Optional[str] = "asc",
        force_metadata_for_links: Optional[bool] = True,
        limit: Optional[int] = None,
        return_first_response: Optional[bool] = False,
        project_id: int = None,
    ) -> List[ImageInfo]:
        """
        List of filtered Images in the given :class:`Dataset<supervisely.project.project.Dataset>`.
        Differs in a more flexible filter format from the get_list() method.

        :param dataset_id: :class:`Dataset<supervisely.project.project.Dataset>` ID in which the Images are located.
        :type dataset_id: :class:`int`
        :param filters: List of params to sort output Images.
        :type filters: :class:`List[Dict]`, optional
        :param sort: Field name to sort. One of {'id' (default), 'name', 'description', 'labelsCount', 'createdAt', 'updatedAt', 'customSort'}.
        :type sort: :class:`str`, optional
        :param sort_order: Sort order. One of {'asc' (default), 'desc'}
        :type sort_order: :class:`str`, optional
        :param project_id: :class:`Project<supervisely.project.project.Project>` ID in which the Images are located.
        :type project_id: :class:`int`
        :return: Objects with image information from Supervisely.
        :rtype: :class:`List[ImageInfo]<ImageInfo>`

        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            # Get list of Images with names containing subsequence '2008'
            img_infos = api.image.get_filtered_list(dataset_id, filters=[{ 'type': 'images_filename', 'data': { 'value': '2008' } }])
        """
        self._validate_project_and_dataset_id(project_id, dataset_id)
        if filters is None or not filters:
            return self.get_list(
                dataset_id,
                sort=sort,
                sort_order=sort_order,
                limit=limit,
                force_metadata_for_links=force_metadata_for_links,
                return_first_response=return_first_response,
                project_id=project_id,
            )

        data = {
            ApiField.PROJECT_ID: project_id,
            ApiField.DATASET_ID: dataset_id,
            ApiField.FILTERS: filters,
            ApiField.SORT: sort,
            ApiField.SORT_ORDER: sort_order,
            ApiField.FORCE_METADATA_FOR_LINKS: force_metadata_for_links,
        }

        if not all(["type" in filter.keys() for filter in filters]):
            raise ValueError("'type' field not found in filter")
        if not all(["data" in filter.keys() for filter in filters]):
            raise ValueError("'data' field not found in filter")

        allowed_filter_types = [
            "images_filename",
            "images_tag",
            "objects_tag",
            "objects_class",
            "objects_annotator",
            "tagged_by_annotator",
            "issues_count",
            "job",
        ]
        if not all([filter["type"] in allowed_filter_types for filter in filters]):
            raise ValueError(f"'type' field must be one of: {allowed_filter_types}")

        return self.get_list_all_pages(
            "images.list",
            data=data,
            limit=limit,
            return_first_response=return_first_response,
        )

    def get_info_by_id(self, id: int, force_metadata_for_links=True) -> ImageInfo:
        """
        Get Image information by ID.

        :param id: Image ID in Supervisely.
        :type id: int
        :return: Object with image information from Supervisely.
        :rtype: :class:`ImageInfo<ImageInfo>`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            # You can get Image ID by listing all images in the Dataset as shown in get_list
            # Or you can open certain image in Supervisely Annotation Tool UI and get last digits of the URL
            img_info = api.image.get_info_by_id(770918)
        """
        return self._get_info_by_id(
            id,
            "images.info",
            fields={ApiField.FORCE_METADATA_FOR_LINKS: force_metadata_for_links},
        )

    def _get_info_by_filters(self, parent_id, filters, force_metadata_for_links):
        """_get_info_by_filters"""
        items = self.get_list(parent_id, filters, force_metadata_for_links=force_metadata_for_links)
        return _get_single_item(items)

    def get_info_by_name(
        self,
        dataset_id: int,
        name: str,
        force_metadata_for_links: Optional[bool] = True,
    ) -> ImageInfo:
        """Returns image info by image name from given dataset id.

        :param dataset_id: Dataset ID in Supervisely, where Image is located.
        :type dataset_id: int
        :param name: Image name in Supervisely.
        :type name: str
        :param force_metadata_for_links: If True, returns full_storage_url and path_original fields in ImageInfo.
        :type force_metadata_for_links: bool, optional
        :return: Object with image information from Supervisely.
        :rtype: :class:`ImageInfo<ImageInfo>`
        """
        return self._get_info_by_name(
            get_info_by_filters_fn=lambda module_name: self._get_info_by_filters(
                dataset_id, module_name, force_metadata_for_links
            ),
            name=name,
        )

    # @TODO: reimplement to new method images.bulk.info
    def get_info_by_id_batch(
        self,
        ids: List[int],
        progress_cb: Optional[Union[tqdm, Callable]] = None,
        force_metadata_for_links=True,
        fields: Optional[List[str]] = None,
    ) -> List[ImageInfo]:
        """
        Get Images information by ID.

        :param ids: Images IDs in Supervisely.
        :type ids: List[int]
        :param progress_cb: Function for tracking the progress.
        :type progress_cb: tqdm or callable, optional
        :return: Objects with image information from Supervisely.
        :rtype: :class:`List[ImageInfo]`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            img_ids = [376728, 376729, 376730, 376731, 376732, 376733]
            img_infos = image.get_info_by_id_batch(img_ids)
        """
        results = []
        if len(ids) == 0:
            return results
        infos_dict = {}
        ids_set = set(ids)
        while any(ids_set):
            img_id = ids_set.pop()
            image_info = self.get_info_by_id(img_id, force_metadata_for_links=False)
            if image_info is None:
                raise KeyError(
                    f"Image (id: {img_id}) is either archived, doesn't exist or you don't have enough permissions to access it"
                )
            dataset_id = image_info.dataset_id
            for batch in batched(ids):
                filters = [{"field": ApiField.ID, "operator": "in", "value": batch}]
                data = {
                    ApiField.DATASET_ID: dataset_id,
                    ApiField.FILTER: filters,
                    ApiField.FORCE_METADATA_FOR_LINKS: force_metadata_for_links,
                }
                if fields is not None:
                    data[ApiField.FIELDS] = fields
                temp_results = self.get_list_all_pages(
                    "images.list",
                    data,
                )
                results.extend(temp_results)
                if progress_cb is not None and len(temp_results) > 0:
                    progress_cb(len(temp_results))
            ids_set = ids_set - set([info.id for info in results])
            infos_dict.update({info.id: info for info in results})

        ordered_results = [infos_dict[id] for id in ids]
        return ordered_results

    def _download(self, id, is_stream=False):
        """
        :param id: int
        :param is_stream: bool
        :return: Response class object contain metadata of image with given id
        """
        response = self._api.post("images.download", {ApiField.ID: id}, stream=is_stream)
        return response

    def download_np(self, id: int, keep_alpha: Optional[bool] = False) -> np.ndarray:
        """
        Download Image with given id in numpy format.

        :param id: Image ID in Supervisely.
        :type id: int
        :param keep_alpha: If True keeps alpha mask for image, otherwise don't.
        :type keep_alpha: bool, optional
        :return: Image in RGB numpy matrix format
        :rtype: :class:`np.ndarray`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            image_np = api.image.download_np(770918)
        """
        response = self._download(id)
        img = sly_image.read_bytes(response.content, keep_alpha)
        return img

    def download_path(self, id: int, path: str) -> None:
        """
        Downloads Image from Dataset to local path by ID.

        :param id: Image ID in Supervisely.
        :type id: int
        :param path: Local save path for Image.
        :type path: str
        :return: None
        :rtype: :class:`NoneType`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            img_info = api.image.get_info_by_id(770918)
            save_path = os.path.join("/home/admin/work/projects/lemons_annotated/ds1/test_imgs/", img_info.name)

            api.image.download_path(770918, save_path)
        """
        response = self._download(id, is_stream=True)
        ensure_base_path(path)
        with open(path, "wb") as fd:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                fd.write(chunk)

    def download(self, id: int, path: str) -> None:
        """
        Downloads Image from Dataset to local path by ID.

        :param id: Image ID in Supervisely.
        :type id: int
        :param path: Local save path for Image.
        :type path: str
        :return: None
        :rtype: :class:`NoneType`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            img_info = api.image.get_info_by_id(770918)
            save_path = os.path.join("/home/admin/work/projects/lemons_annotated/ds1/test_imgs/", img_info.name)

            api.image.download_path(770918, save_path)
        """
        self.download_path(id=id, path=path)

    def _download_batch(
        self,
        dataset_id: int,
        ids: List[int],
        progress_cb: Optional[Union[tqdm, Callable]] = None,
    ):
        """
        Get image id and it content from given dataset and list of images ids.
        """
        if DOWNLOAD_BATCH_SIZE is not None and isinstance(DOWNLOAD_BATCH_SIZE, int):
            batches = batched(ids, DOWNLOAD_BATCH_SIZE)
            logger.debug(
                f"Batch size for func 'ImageApi._download_batch' changed to: {DOWNLOAD_BATCH_SIZE}"
            )
        else:
            batches = batched(ids)
        for batch_ids in batches:
            response = self._api.post(
                "images.bulk.download",
                {ApiField.DATASET_ID: dataset_id, ApiField.IMAGE_IDS: batch_ids},
            )
            decoder = MultipartDecoder.from_response(response)
            for part in decoder.parts:
                content_utf8 = part.headers[b"Content-Disposition"].decode("utf-8")
                # Find name="1245" preceded by a whitespace, semicolon or beginning of line.
                # The regex has 2 capture group: one for the prefix and one for the actual name value.
                img_id = int(re.findall(r'(^|[\s;])name="(\d*)"', content_utf8)[0][1])

                if progress_cb is not None:
                    progress_cb(1)
                yield img_id, part

    def download_paths(
        self,
        dataset_id: int,
        ids: List[int],
        paths: List[str],
        progress_cb: Optional[Union[tqdm, Callable]] = None,
    ) -> None:
        """
        Download Images with given ids and saves them for the given paths.

        :param dataset_id: Dataset ID in Supervisely, where Images are located.
        :type dataset_id: :class:`int`
        :param ids: List of Image IDs in Supervisely.
        :type ids: :class:`List[int]`
        :param paths: Local save paths for Images.
        :type paths: :class:`List[str]`
        :param progress_cb: Function for tracking download progress.
        :type progress_cb: tqdm or callable, optional
        :raises: :class:`ValueError` if len(ids) != len(paths)
        :return: None
        :rtype: :class:`NoneType`

        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            local_save_dir = "/home/admin/work/projects/lemons_annotated/ds1/test_imgs"
            save_paths = []
            image_ids = [771755, 771756, 771757, 771758, 771759, 771760]
            img_infos = api.image.get_info_by_id_batch(image_ids)

            p = tqdm(desc="Images downloaded: ", total=len(img_infos))
            for img_info in img_infos:
                save_paths.append(os.path.join(local_save_dir, img_info.name))

            api.image.download_paths(2573, image_ids, save_paths, progress_cb=p)
            # Progress:
            # {"message": "progress", "event_type": "EventType.PROGRESS", "subtask": "Images downloaded: ", "current": 0, "total": 6, "timestamp": "2021-03-15T19:47:15.406Z", "level": "info"}
            # {"message": "progress", "event_type": "EventType.PROGRESS", "subtask": "Images downloaded: ", "current": 1, "total": 6, "timestamp": "2021-03-15T19:47:16.366Z", "level": "info"}
            # {"message": "progress", "event_type": "EventType.PROGRESS", "subtask": "Images downloaded: ", "current": 2, "total": 6, "timestamp": "2021-03-15T19:47:16.367Z", "level": "info"}
            # {"message": "progress", "event_type": "EventType.PROGRESS", "subtask": "Images downloaded: ", "current": 3, "total": 6, "timestamp": "2021-03-15T19:47:16.367Z", "level": "info"}
            # {"message": "progress", "event_type": "EventType.PROGRESS", "subtask": "Images downloaded: ", "current": 4, "total": 6, "timestamp": "2021-03-15T19:47:16.367Z", "level": "info"}
            # {"message": "progress", "event_type": "EventType.PROGRESS", "subtask": "Images downloaded: ", "current": 5, "total": 6, "timestamp": "2021-03-15T19:47:16.368Z", "level": "info"}
            # {"message": "progress", "event_type": "EventType.PROGRESS", "subtask": "Images downloaded: ", "current": 6, "total": 6, "timestamp": "2021-03-15T19:47:16.368Z", "level": "info"}
        """
        if len(ids) == 0:
            return
        if len(ids) != len(paths):
            raise ValueError('Can not match "ids" and "paths" lists, len(ids) != len(paths)')

        id_to_path = {id: path for id, path in zip(ids, paths)}
        for img_id, resp_part in self._download_batch(dataset_id, ids, progress_cb):
            with open(id_to_path[img_id], "wb") as w:
                w.write(resp_part.content)

    def download_bytes(
        self,
        dataset_id: int,
        ids: List[int],
        progress_cb: Optional[Union[tqdm, Callable]] = None,
    ) -> List[bytes]:
        """
        Download Images with given IDs from Dataset in Binary format.

        :param dataset_id: Dataset ID in Supervisely, where Images are located.
        :type dataset_id: int
        :param ids: List of Image IDs in Supervisely.
        :type ids: List[int]
        :param progress_cb: Function for tracking download progress.
        :type progress_cb: tqdm or callable, optional
        :return: List of Images in binary format
        :rtype: :class:`List[bytes]`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            img_bytes = api.image.download_bytes(dataset_id, [770918])
            print(img_bytes)
            # Output: [b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\ ...']
        """
        if len(ids) == 0:
            return []

        id_to_img = {}
        for img_id, resp_part in self._download_batch(dataset_id, ids, progress_cb):
            id_to_img[img_id] = resp_part.content

        return [id_to_img[id] for id in ids]

    def download_nps(
        self,
        dataset_id: int,
        ids: List[int],
        progress_cb: Optional[Union[tqdm, Callable]] = None,
        keep_alpha: Optional[bool] = False,
    ) -> List[np.ndarray]:
        """
        Download Images with given IDs in numpy format.

        :param dataset_id: Dataset ID in Supervisely, where Images are located.
        :type dataset_id: int
        :param ids: List of Images IDs in Supervisely.
        :type ids: List[int]
        :param progress_cb: Function for tracking download progress.
        :type progress_cb: tqdm or callable, optional
        :param keep_alpha: If True keeps alpha mask for Image, otherwise don't.
        :type keep_alpha: bool, optional
        :return: List of Images in RGB numpy matrix format
        :rtype: :class:`List[np.ndarray]`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            image_ids = [770918, 770919, 770920]
            image_nps = api.image.download_nps(dataset_id, image_ids)
        """
        return [
            sly_image.read_bytes(img_bytes, keep_alpha)
            for img_bytes in self.download_bytes(
                dataset_id=dataset_id, ids=ids, progress_cb=progress_cb
            )
        ]

    def download_nps_generator(
        self,
        dataset_id: int,
        ids: List[int],
        progress_cb: Optional[Union[tqdm, Callable]] = None,
        keep_alpha: Optional[bool] = False,
    ) -> Generator[Tuple[int, np.ndarray], None, None]:
        for img_id, img_part in self._download_batch(dataset_id, ids, progress_cb):
            img_bytes = img_part.content
            yield img_id, sly_image.read_bytes(img_bytes, keep_alpha)

    def check_existing_hashes(
        self, hashes: List[str], progress_cb: Optional[Union[tqdm, Callable]] = None
    ) -> List[str]:
        """
        Checks existing hashes for Images.

        :param hashes: List of hashes.
        :type hashes: List[str]
        :param progress_cb: Function for tracking progress of checking.
        :type progress_cb: tqdm or callable, optional
        :return: List of existing hashes
        :rtype: :class:`List[str]`
        :Usage example: Checkout detailed example `here <https://app.supervisely.com/explore/notebooks/guide-10-check-existing-images-and-upload-only-the-new-ones-1545/overview>`_ (you must be logged into your Supervisely account)

         .. code-block:: python

            # Helpful method when your uploading was interrupted
            # You can check what images has been successfully uploaded by their hashes and what not
            # And continue uploading the rest of the images from that point

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            # Find project
            project = api.project.get_info_by_id(WORKSPACE_ID, PROJECT_ID)

            # Get paths of all images in a directory
            images_paths = sly.fs.list_files('images_to_upload')

            # Calculate hashes for all images paths
            hash_to_image = {}
            images_hashes = []

            for idx, item in enumerate(images_paths):
                item_hash = sly.fs.get_file_hash(item)
                images_hashes.append(item_hash)
                hash_to_image[item_hash] = item

            # Get hashes that are already on server
            remote_hashes = api.image.check_existing_hashes(images_hashes)
            already_uploaded_images = {hh: hash_to_image[hh] for hh in remote_hashes}
        """
        results = []
        if len(hashes) == 0:
            return results
        for hashes_batch in batched(hashes, batch_size=900):
            response = self._api.post("images.internal.hashes.list", hashes_batch)
            results.extend(response.json())

            if progress_cb is not None:
                progress_cb(len(hashes_batch))
        return results

    def check_existing_links(
        self,
        links: List[str],
        progress_cb: Optional[Union[tqdm, Callable]] = None,
        team_id: Optional[int] = None,
    ) -> List[str]:
        """
        Checks existing links for Images.

        :param links: List of links.
        :type links: List[str]
        :param progress_cb: Function for tracking progress of checking.
        :type progress_cb: tqdm or callable, optional
        :param team_id: Team ID in Supervisely (will be used to get remote storage settings).
        :type team_id: int
        :return: List of existing links
        :rtype: List[str]
        """

        if len(links) == 0:
            return []

        def _is_image_available(url, team_id, progress_cb=None):
            if self._api.remote_storage.is_bucket_url(url):
                response = self._api.remote_storage.is_path_exist(url, team_id)
                result = url if response else None
            else:
                response = requests.head(url)
                result = url if response.status_code == 200 else None
            if progress_cb is not None:
                progress_cb(1)
            return result

        _is_image_available_with_progress = partial(
            _is_image_available, team_id=team_id, progress_cb=progress_cb
        )

        with ThreadPoolExecutor(max_workers=20) as executor:
            results = list(executor.map(_is_image_available_with_progress, links))

        return results

    def check_image_uploaded(self, hash: str) -> bool:
        """
        Checks if Image has been uploaded.

        :param hash: Image hash in Supervisely.
        :type hash: str
        :return: True if Image with given hash exist, otherwise False
        :rtype: :class:`bool`
        :Usage example:

         .. code-block::

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            image_check_uploaded = api.image.check_image_uploaded("YZKQrZH5C0rBvGGA3p7hjWahz3/pV09u5m30Bz8GeYs=")
            print(image_check_uploaded)
            # Output: True
        """
        response = self._api.post("images.internal.hashes.list", [hash])
        results = response.json()
        if len(results) == 0:
            return False
        else:
            return True

    def _upload_uniq_images_single_req(self, func_item_to_byte_stream, hashes_items_to_upload):
        """
        Upload images (binary data) to server with single request.
        Expects unique images that aren't exist at server.
        :param func_item_to_byte_stream: converter for "item" to byte stream
        :param hashes_items_to_upload: list of pairs (hash, item)
        :return: list of hashes for successfully uploaded items
        """
        content_dict = {}
        for idx, (_, item) in enumerate(hashes_items_to_upload):
            content_dict["{}-file".format(idx)] = (
                str(idx),
                func_item_to_byte_stream(item),
                "image/*",
            )
        encoder = MultipartEncoder(fields=content_dict)
        resp = self._api.post("images.bulk.upload", encoder)

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
        self, func_item_to_byte_stream, items_hashes, retry_cnt=3, progress_cb=None
    ):
        """
        Upload images (binary data) to server. Works with already existing or duplicating images.
        :param func_item_to_byte_stream: converter for "item" to byte stream
        :param items_hashes: iterable of pairs (item, hash) where "item" is a some descriptor (e.g. image file path)
         for image data, and "hash" is a hash for the image binary data
        :param retry_cnt: int, number of retries to send the whole set of items
        :param progress_cb: callback or tqdm object to account progress (in number of items)
        """
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
        if progress_cb:
            progress_cb(len(remote_hashes))
        pending_hashes = unique_hashes - remote_hashes

        # @TODO: some correlation with sly.io.network_exceptions. Should we perform retries here?
        for retry_idx in range(retry_cnt):
            # single attempt to upload all data which is not uploaded yet

            for hashes in batched(list(pending_hashes)):
                pending_hashes_items = [(h, hash_to_items[h]) for h in hashes]
                hashes_rcv = self._upload_uniq_images_single_req(
                    func_item_to_byte_stream, pending_hashes_items
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
                if progress_cb is not None:
                    progress_cb(items_count_total - len(unique_hashes))
                return

            warning_items = []
            for h in pending_hashes:
                item_data = hash_to_items[h]
                if isinstance(item_data, (bytes, bytearray)):
                    item_data = "some bytes ..."
                warning_items.append((h, item_data))

            logger.warn(
                "Unable to upload images (data).",
                extra={
                    "retry_idx": retry_idx,
                    "items": warning_items,
                },
            )
            # now retry it for the case if it is a shadow server/connection error

        raise ValueError(
            "Unable to upload images (data). "
            "Please check if images are in supported format and if ones aren't corrupted."
        )

    def upload_path(
        self,
        dataset_id: int,
        name: str,
        path: str,
        meta: Optional[Dict] = None,
        validate_meta: Optional[bool] = False,
        use_strict_validation: Optional[bool] = False,
        use_caching_for_validation: Optional[bool] = False,
    ) -> ImageInfo:
        """
        Uploads Image with given name from given local path to Dataset.

        If you include `meta` during the upload, you can add a custom sort parameter for image.
        To achieve this, use the context manager :func:`api.image.add_custom_sort` with the desired key name from the meta dictionary to be used for sorting.
        Refer to the example section for more details.

        :param dataset_id: Dataset ID in Supervisely.
        :type dataset_id: int
        :param name: Image name with extension.
        :type name: str
        :param path: Local Image path.
        :type path: str
        :param meta: Custom additional image info that contain image technical and/or user-generated data.
        :type meta: dict, optional
        :param validate_meta: If True, validates provided meta with saved JSON schema.
        :type validate_meta: bool, optional
        :param use_strict_validation: If True, uses strict validation.
        :type use_strict_validation: bool, optional
        :param use_caching_for_validation: If True, uses caching for validation.
        :type use_caching_for_validation: bool, optional
        :return: Information about Image. See :class:`info_sequence<info_sequence>`
        :rtype: :class:`ImageInfo`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            img_info = api.image.upload_path(dataset_id, name="7777.jpeg", path="/home/admin/Downloads/7777.jpeg")

            # Add custom sort parameter for image
            img_meta = {'my-key':'a'}
            with api.image.add_custom_sort(key="my-key"):
                img_info = api.image.upload_path(dataset_id, name="7777.jpeg", path="/home/admin/Downloads/7777.jpeg", meta=img_meta)

        """
        metas = None if meta is None else [meta]
        return self.upload_paths(
            dataset_id,
            [name],
            [path],
            metas=metas,
            validate_meta=validate_meta,
            use_strict_validation=use_strict_validation,
            use_caching_for_validation=use_caching_for_validation,
        )[0]

    def upload_paths(
        self,
        dataset_id: int,
        names: List[str],
        paths: List[str],
        progress_cb: Optional[Union[tqdm, Callable]] = None,
        metas: Optional[List[Dict]] = None,
        conflict_resolution: Optional[Literal["rename", "skip", "replace"]] = None,
        validate_meta: Optional[bool] = False,
        use_strict_validation: Optional[bool] = False,
        use_caching_for_validation: Optional[bool] = False,
    ) -> List[ImageInfo]:
        """
        Uploads Images with given names from given local path to Dataset.

        If you include `metas` during the upload, you can add a custom sort parameter for images.
        To achieve this, use the context manager :func:`api.image.add_custom_sort` with the desired key name from the meta dictionary to be used for sorting.
        Refer to the example section for more details.

        :param dataset_id: Dataset ID in Supervisely.
        :type dataset_id: int
        :param names: List of Images names with extension.
        :type names: List[str]
        :param paths: List of local Images pathes.
        :type paths: List[str]
        :param progress_cb: Function for tracking the progress of uploading.
        :type progress_cb: tqdm or callable, optional
        :param metas: Custom additional image infos that contain images technical and/or user-generated data as list of separate dicts.
        :type metas: List[dict], optional
        :param conflict_resolution: The strategy to resolve upload conflicts. 'Replace' option will replace the existing images in the dataset with the new images. The images that are being deleted are logged. 'Skip' option will ignore the upload of new images that would result in a conflict. An original image's ImageInfo list will be returned instead. 'Rename' option will rename the new images to prevent any conflict.
        :type conflict_resolution: Optional[Literal["rename", "skip", "replace"]]
        :param validate_meta: If True, validates provided meta with saved JSON schema.
        :type validate_meta: bool, optional
        :param use_strict_validation: If True, uses strict validation.
        :type use_strict_validation: bool, optional
        :param use_caching_for_validation: If True, uses caching for validation.
        :type use_caching_for_validation: bool, optional
        :raises: :class:`ValueError` if len(names) != len(paths)
        :return: List with information about Images. See :class:`info_sequence<info_sequence>`
        :rtype: :class:`List[ImageInfo]`
        :Usage example:

         .. code-block:: python

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            img_names = ["7777.jpeg", "8888.jpeg", "9999.jpeg"]
            image_paths = ["/home/admin/Downloads/img/770918.jpeg", "/home/admin/Downloads/img/770919.jpeg", "/home/admin/Downloads/img/770920.jpeg"]

            img_infos = api.image.upload_paths(dataset_id, names=img_names, paths=img_paths)

            # Add custom sort parameter for images
            img_metas = [{'my-key':'a'}, {'my-key':'b'}, {'my-key':'c'}]
            with api.image.add_custom_sort(key="my-key"):
                img_infos = api.image.upload_paths(dataset_id, names=img_names, paths=img_paths, metas=img_metas)
        """

        def path_to_bytes_stream(path):
            return open(path, "rb")

        hashes = [get_file_hash(x) for x in paths]

        self._upload_data_bulk(path_to_bytes_stream, zip(paths, hashes), progress_cb=progress_cb)

        return self.upload_hashes(
            dataset_id,
            names,
            hashes,
            metas=metas,
            conflict_resolution=conflict_resolution,
            validate_meta=validate_meta,
            use_strict_validation=use_strict_validation,
            use_caching_for_validation=use_caching_for_validation,
        )

    def upload_np(
        self, dataset_id: int, name: str, img: np.ndarray, meta: Optional[Dict] = None
    ) -> ImageInfo:
        """
        Upload given Image in numpy format with given name to Dataset.

        If you include `meta` during the upload, you can add a custom sort parameter for image.
        To achieve this, use the context manager :func:`api.image.add_custom_sort` with the desired key name from the meta dictionary to be used for sorting.
        Refer to the example section for more details.

        :param dataset_id: Dataset ID in Supervisely.
        :type dataset_id: int
        :param name: Image name with extension.
        :type name: str
        :param img: image in RGB format(numpy matrix)
        :type img: np.ndarray
        :param meta: Custom additional image info that contain image technical and/or user-generated data.
        :type meta: dict, optional
        :return: Information about Image. See :class:`info_sequence<info_sequence>`
        :rtype: :class:`ImageInfo`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            img_np = sly.image.read("/home/admin/Downloads/7777.jpeg")
            img_info = api.image.upload_np(dataset_id, name="7777.jpeg", img=img_np)

            # Add custom sort parameter for image
            img_meta = {'my-key':'a'}
            with api.image.add_custom_sort(key="my-key"):
                img_info = api.image.upload_np(dataset_id, name="7777.jpeg", img=img_np, meta=img_meta)
        """
        metas = None if meta is None else [meta]
        return self.upload_nps(dataset_id, [name], [img], metas=metas)[0]

    def upload_nps(
        self,
        dataset_id: int,
        names: List[str],
        imgs: List[np.ndarray],
        progress_cb: Optional[Union[tqdm, Callable]] = None,
        metas: Optional[List[Dict]] = None,
        conflict_resolution: Optional[Literal["rename", "skip", "replace"]] = None,
    ) -> List[ImageInfo]:
        """
        Upload given Images in numpy format with given names to Dataset.

        If you include `metas` during the upload, you can add a custom sort parameter for images.
        To achieve this, use the context manager :func:`api.image.add_custom_sort` with the desired key name from the meta dictionary to be used for sorting.
        Refer to the example section for more details.

        :param dataset_id: Dataset ID in Supervisely.
        :type dataset_id: int
        :param names: Images names with extension.
        :type names: List[str]
        :param imgs: Images in RGB numpy matrix format
        :type imgs: List[np.ndarray]
        :param progress_cb: Function for tracking the progress of uploading.
        :type progress_cb: tqdm or callable, optional
        :param metas: Custom additional image infos that contain images technical and/or user-generated data as list of separate dicts.
        :type metas: List[dict], optional
        :param conflict_resolution: The strategy to resolve upload conflicts. 'Replace' option will replace the existing images in the dataset with the new images. The images that are being deleted are logged. 'Skip' option will ignore the upload of new images that would result in a conflict. An original image's ImageInfo list will be returned instead. 'Rename' option will rename the new images to prevent any conflict.
        :type conflict_resolution: Optional[Literal["rename", "skip", "replace"]]
        :return: List with information about Images. See :class:`info_sequence<info_sequence>`
        :rtype: :class:`List[ImageInfo]`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            img_np_1 = sly.image.read("/home/admin/Downloads/7777.jpeg")
            img_np_2 = sly.image.read("/home/admin/Downloads/8888.jpeg")
            img_np_3 = sly.image.read("/home/admin/Downloads/9999.jpeg")

            img_names = ["7777.jpeg", "8888.jpeg", "9999.jpeg"]
            img_nps = [img_np_1, img_np_2, img_np_3]

            img_infos = api.image.upload_nps(dataset_id, names=img_names, imgs=img_nps)

            # Add custom sort parameter for images
            img_metas = [{'my-key':'a'}, {'my-key':'b'}, {'my-key':'c'}]
            with api.image.add_custom_sort(key="my-key"):
                img_infos = api.image.upload_nps(dataset_id, names=img_names, imgs=img_nps, metas=img_metas)
        """

        def img_to_bytes_stream(item):
            img, name = item[0], item[1]
            img_bytes = sly_image.write_bytes(img, get_file_ext(name))
            return io.BytesIO(img_bytes)

        def img_to_hash(item):
            img, name = item[0], item[1]
            return sly_image.get_hash(img, get_file_ext(name))

        img_name_list = list(zip(imgs, names))
        hashes = [img_to_hash(x) for x in img_name_list]

        self._upload_data_bulk(
            img_to_bytes_stream, zip(img_name_list, hashes), progress_cb=progress_cb
        )
        return self.upload_hashes(
            dataset_id, names, hashes, metas=metas, conflict_resolution=conflict_resolution
        )

    def upload_link(
        self,
        dataset_id: int,
        name: str,
        link: str,
        meta: Optional[Dict] = None,
        force_metadata_for_links=True,
    ) -> ImageInfo:
        """
        Uploads Image from given link to Dataset.

        If you include `meta` during the upload, you can add a custom sort parameter for image.
        To achieve this, use the context manager :func:`api.image.add_custom_sort` with the desired key name from the meta dictionary to be used for sorting.
        Refer to the example section for more details.

        :param dataset_id: Dataset ID in Supervisely.
        :type dataset_id: int
        :param name: Image name with extension.
        :type name: str
        :param link: Link to Image.
        :type link: str
        :param meta: Custom additional image info that contain image technical and/or user-generated data.
        :type meta: dict, optional
        :param force_metadata_for_links: Calculate metadata for link. If False, metadata will be empty.
        :type force_metadata_for_links: bool, optional
        :return: Information about Image. See :class:`info_sequence<info_sequence>`
        :rtype: :class:`ImageInfo`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            img_name = 'Avatar.jpg'
            img_link = 'https://m.media-amazon.com/images/M/MV5BMTYwOTEwNjAzMl5BMl5BanBnXkFtZTcwODc5MTUwMw@@._V1_.jpg'

            img_info = api.image.upload_link(dataset_id, img_name, img_link)

            # Add custom sort parameter for image
            img_meta = {"my-key": "a"}
            with api.image.add_custom_sort(key="my-key"):
                img_info = api.image.upload_link(dataset_id, img_name, img_link, meta=img_meta)
        """
        metas = None if meta is None else [meta]
        return self.upload_links(
            dataset_id,
            [name],
            [link],
            metas=metas,
            force_metadata_for_links=force_metadata_for_links,
        )[0]

    def upload_links(
        self,
        dataset_id: int,
        names: List[str],
        links: List[str],
        progress_cb: Optional[Union[tqdm, Callable]] = None,
        metas: Optional[List[Dict]] = None,
        batch_size: Optional[int] = 50,
        force_metadata_for_links: Optional[bool] = True,
        skip_validation: Optional[bool] = False,
        conflict_resolution: Optional[Literal["rename", "skip", "replace"]] = None,
    ) -> List[ImageInfo]:
        """
        Uploads Images from given links to Dataset.

        If you include `metas` during the upload, you can add a custom sort parameter for images.
        To achieve this, use the context manager :func:`api.image.add_custom_sort` with the desired key name from the meta dictionary to be used for sorting.
        Refer to the example section for more details.

        :param dataset_id: Dataset ID in Supervisely.
        :type dataset_id: int
        :param names: Images names with extension.
        :type names: List[str]
        :param links: Links to Images.
        :type links: List[str]
        :param progress_cb: Function for tracking the progress of uploading.
        :type progress_cb: tqdm or callable, optional
        :param metas: Custom additional image infos that contain images technical and/or user-generated data as list of separate dicts.
        :type metas: List[dict], optional
        :param force_metadata_for_links: Calculate metadata for links. If False, metadata will be empty.
        :type force_metadata_for_links: bool, optional
        :param skip_validation: Skips validation for images, can result in invalid images being uploaded.
        :type skip_validation: bool, optional
        :param conflict_resolution: The strategy to resolve upload conflicts. 'Replace' option will replace the existing images in the dataset with the new images. The images that are being deleted are logged. 'Skip' option will ignore the upload of new images that would result in a conflict. An original image's ImageInfo list will be returned instead. 'Rename' option will rename the new images to prevent any conflict.
        :type conflict_resolution: Optional[Literal["rename", "skip", "replace"]]
        :return: List with information about Images. See :class:`info_sequence<info_sequence>`
        :rtype: :class:`List[ImageInfo]`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            img_names = ['Avatar.jpg', 'Harry Potter.jpg', 'Avengers.jpg']
            img_links = ['https://m.media-amazon.com/images/M/MV5BMTYwOTEwNjAzMl5BMl5BanBnXkFtZTcwODc5MTUwMw@@._V1_.jpg',
                         'https://m.media-amazon.com/images/M/MV5BNDYxNjQyMjAtNTdiOS00NGYwLWFmNTAtNThmYjU5ZGI2YTI1XkEyXkFqcGdeQXVyMTMxODk2OTU@._V1_.jpg',
                         'https://m.media-amazon.com/images/M/MV5BNjQ3NWNlNmQtMTE5ZS00MDdmLTlkZjUtZTBlM2UxMGFiMTU3XkEyXkFqcGdeQXVyNjUwNzk3NDc@._V1_.jpg']

            img_infos = api.image.upload_links(dataset_id, img_names, img_links)

            # Add custom sort parameter for images
            img_metas = [{'my-key':'a'}, {'my-key':'b'}, {'my-key':'c'}]
            with api.image.add_custom_sort(key="my-key"):
                img_infos = api.image.upload_links(dataset_id, names=img_names, links=img_links, metas=img_metas)
        """
        return self._upload_bulk_add(
            lambda item: (ApiField.LINK, item),
            dataset_id,
            names,
            links,
            progress_cb,
            metas=metas,
            batch_size=batch_size,
            force_metadata_for_links=force_metadata_for_links,
            skip_validation=skip_validation,
            conflict_resolution=conflict_resolution,
        )

    def upload_hash(
        self, dataset_id: int, name: str, hash: str, meta: Optional[Dict] = None
    ) -> ImageInfo:
        """
        Upload Image from given hash to Dataset.

        If you include `meta` during the upload, you can add a custom sort parameter for image.
        To achieve this, use the context manager :func:`api.image.add_custom_sort` with the desired key name from the meta dictionary to be used for sorting.
        Refer to the example section for more details.

        :param dataset_id: Dataset ID in Supervisely.
        :type dataset_id: int
        :param name: Image name with extension.
        :type name: str
        :param hash: Image hash.
        :type hash: str
        :param meta: Custom additional image info that contain image technical and/or user-generated data.
        :type meta: dict, optional
        :return: Information about Image. See :class:`info_sequence<info_sequence>`
        :rtype: :class:`ImageInfo`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            dst_dataset_id = 452984
            im_info = api.image.get_info_by_id(193940090)
            hash = im_info.hash
            # It is necessary to upload image with the same name(extention) as in src dataset
            name = im_info.name
            meta = {1: 'meta_example'}
            new_in_info = api.image.upload_hash(dst_dataset_id, name, hash, meta)
            print(json.dumps(new_in_info, indent=4))
            # Output: [
            #     196793586,
            #     "IMG_0748.jpeg",
            #     null,
            #     "NEjmnmdd7DOzaFAKK/nCIl5CtcwZeMkhW3CHe875p9g=",
            #     "image/jpeg",
            #     "jpeg",
            #     66885,
            #     600,
            #     500,
            #     0,
            #     452984,
            #     "2021-03-16T09:09:45.587Z",
            #     "2021-03-16T09:09:45.587Z",
            #     {
            #         "1": "meta_example"
            #     },
            #     "/h5un6l2bnaz1vj8a9qgms4-public/images/original/P/a/kn/W2mzMQg435d6wG0.jpg",
            #     "https://app.supervisely.com/h5un6l2bnaz1vj8a9qgms4-public/images/original/P/a/kn/W2mzMQg435hiHJAPgMU.jpg"
            # ]

            # Add custom sort parameter for image
            new_dataset_id = 452985
            im_info = api.image.get_info_by_id(193940090)
            print(im_info.meta)
            # Output: {'my-key':'a'}
            with api.image.add_custom_sort(key="my-key"):
                img_info = api.image.upload_hash(new_dataset_id, name=im_info.name, hash=im_info.hash, meta=im_info.meta)
        """
        metas = None if meta is None else [meta]
        return self.upload_hashes(dataset_id, [name], [hash], metas=metas)[0]

    def upload_hashes(
        self,
        dataset_id: int,
        names: List[str],
        hashes: List[str],
        progress_cb: Optional[Union[tqdm, Callable]] = None,
        metas: Optional[List[Dict]] = None,
        batch_size: Optional[int] = 50,
        skip_validation: Optional[bool] = False,
        conflict_resolution: Optional[Literal["rename", "skip", "replace"]] = None,
        validate_meta: Optional[bool] = False,
        use_strict_validation: Optional[bool] = False,
        use_caching_for_validation: Optional[bool] = False,
    ) -> List[ImageInfo]:
        """
        Upload images from given hashes to Dataset.

        If you include `metas` during the upload, you can add a custom sort parameter for images.
        To achieve this, use the context manager :func:`api.image.add_custom_sort` with the desired key name from the meta dictionary to be used for sorting.
        Refer to the example section for more details.

        :param dataset_id: Dataset ID in Supervisely.
        :type dataset_id: int
        :param names: Images names with extension.
        :type names: List[str]
        :param hashes: Images hashes.
        :type hashes: List[str]
        :param progress_cb: Function for tracking the progress of uploading.
        :type progress_cb: tqdm or callable, optional
        :param metas: Custom additional image infos that contain images technical and/or user-generated data as list of separate dicts.
        :type metas: List[dict], optional
        :param batch_size: Number of images to upload in one batch.
        :type batch_size: int, optional
        :param skip_validation: Skips validation for images, can result in invalid images being uploaded.
        :type skip_validation: bool, optional
        :param conflict_resolution: The strategy to resolve upload conflicts. 'Replace' option will replace the existing images in the dataset with the new images. The images that are being deleted are logged. 'Skip' option will ignore the upload of new images that would result in a conflict. An original image's ImageInfo list will be returned instead. 'Rename' option will rename the new images to prevent any conflict.
        :type conflict_resolution: Optional[Literal["rename", "skip", "replace"]]
        :param validate_meta: If True, validates provided meta with saved JSON schema.
        :type validate_meta: bool, optional
        :param use_strict_validation: If True, uses strict validation.
        :type use_strict_validation: bool, optional
        :param use_caching_for_validation: If True, uses caching for validation.
        :type use_caching_for_validation: bool, optional
        :return: List with information about Images. See :class:`info_sequence<info_sequence>`
        :rtype: :class:`List[ImageInfo]`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            src_dataset_id = 447130
            hashes = []
            names = []
            metas = []
            imgs_info = api.image.get_list(src_dataset_id)
            # Create lists of hashes, images names and meta information for each image
            for im_info in imgs_info:
                hashes.append(im_info.hash)
                # It is necessary to upload images with the same names(extentions) as in src dataset
                names.append(im_info.name)
                metas.append({im_info.name: im_info.size})

            dst_dataset_id = 452984
            progress = sly.Progress("Images upload: ", len(hashes))
            new_imgs_info = api.image.upload_hashes(dst_dataset_id, names, hashes, progress.iters_done_report, metas)
            # Output:
            # {"message": "progress", "event_type": "EventType.PROGRESS", "subtask": "Images downloaded: ", "current": 0, "total": 10, "timestamp": "2021-03-16T11:59:07.444Z", "level": "info"}
            # {"message": "progress", "event_type": "EventType.PROGRESS", "subtask": "Images downloaded: ", "current": 10, "total": 10, "timestamp": "2021-03-16T11:59:07.644Z", "level": "info"}

            # Add custom sort parameter for images
            new_dataset_id = 452985
            new_metas = [{'my-key':'a'}, {'my-key':'b'}, {'my-key':'c'}]
            with api.image.add_custom_sort(key="my-key"):
                img_infos = api.image.upload_hashes(new_dataset_id, names=names, hashes=hashes, metas=new_metas)
        """
        return self._upload_bulk_add(
            lambda item: (ApiField.HASH, item),
            dataset_id,
            names,
            hashes,
            progress_cb,
            metas=metas,
            batch_size=batch_size,
            skip_validation=skip_validation,
            conflict_resolution=conflict_resolution,
            validate_meta=validate_meta,
            use_strict_validation=use_strict_validation,
            use_caching_for_validation=use_caching_for_validation,
        )

    def upload_id(
        self, dataset_id: int, name: str, id: int, meta: Optional[Dict] = None
    ) -> ImageInfo:
        """
        Upload Image by ID to Dataset.

        If you include `meta` during the upload, you can add a custom sort parameter for image.
        To achieve this, use the context manager :func:`api.image.add_custom_sort` with the desired key name from the meta dictionary to be used for sorting.
        Refer to the example section for more details.

        :param dataset_id: Destination Dataset ID in Supervisely.
        :type dataset_id: int
        :param name: Image name with extension.
        :type name: str
        :param id: Source image ID in Supervisely.
        :type id: int
        :param meta: Custom additional image info that contain image technical and/or user-generated data.
        :type meta: dict, optional
        :return: Information about Image. See :class:`info_sequence<info_sequence>`
        :rtype: :class:`ImageInfo`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            dst_dataset_id = 452984
            im_info = api.image.get_info_by_id(193940090)
            id = im_info.id
            # It is necessary to upload image with the same name(extention) as in src dataset
            name = im_info.name
            meta = {1: 'meta_example'}
            new_in_info = api.image.upload_id(dst_dataset_id, name, id, meta)
            print(json.dumps(new_in_info, indent=4))
            # Output: [
            #     196793605,
            #     "IMG_0748.jpeg",
            #     null,
            #     "NEjmnmdd7DOzaFAKK/nCIl5CtcwZeMkhW3CHe875p9g=",
            #     "image/jpeg",
            #     "jpeg",
            #     66885,
            #     600,
            #     500,
            #     0,
            #     452984,
            #     "2021-03-16T09:27:12.620Z",
            #     "2021-03-16T09:27:12.620Z",
            #     {
            #         "1": "meta_example"
            #     },
            #     "/h5un6l2bnaz1vj8a9qgms4-public/images/original/P/a/kn/W2mzMQg435d6wG0AJGJTOsL1FqMUNOPqu4VdzFAN36LqtGwBIE4AmLOQ1BAxuIyB0bHJAPgMU.jpg",
            #     "https://app.supervisely.com/h5un6l2bnaz1vj8a9qgms4-public/images/original/P/a/kn/iEaDEkejnfnb1Tz56ka0hiHJAPgMU.jpg"
            # ]

            # Add custom sort parameter for image
            new_dataset_id = 452985
            im_info = api.image.get_info_by_id(193940090)
            print(im_info.meta)
            # Output: {"my-key": "a"}
            with api.image.add_custom_sort(key="my-key"):
                img_info = api.image.upload_id(new_dataset_id, name=im_info.name, id=im_info.id, meta=im_info.meta)
        """
        metas = None if meta is None else [meta]
        return self.upload_ids(dataset_id, [name], [id], metas=metas)[0]

    def upload_ids(
        self,
        dataset_id: int,
        names: List[str],
        ids: List[int],
        progress_cb: Optional[Union[tqdm, Callable]] = None,
        metas: Optional[List[Dict]] = None,
        batch_size: Optional[int] = 50,
        force_metadata_for_links: bool = True,
        infos: List[ImageInfo] = None,  # deprecated
        skip_validation: Optional[bool] = False,
        conflict_resolution: Optional[Literal["rename", "skip", "replace"]] = None,
    ) -> List[ImageInfo]:
        """
        Upload Images by IDs to Dataset.

        If you include `metas` during the upload, you can add a custom sort parameter for images.
        To achieve this, use the context manager :func:`api.image.add_custom_sort` with the desired key name from the meta dictionary to be used for sorting.
        Refer to the example section for more details.

        :param dataset_id: Destination Dataset ID in Supervisely.
        :type dataset_id: int
        :param names: Source images names with extension.
        :type names: List[str]
        :param ids: Images IDs.
        :type ids: List[int]
        :param progress_cb: Function for tracking the progress of uploading.
        :type progress_cb: tqdm or callable, optional
        :param metas: Custom additional image infos that contain images technical and/or user-generated data as list of separate dicts.
        :type metas: List[dict], optional
        :param batch_size: Number of images to upload in one batch.
        :type batch_size: int, optional
        :param force_metadata_for_links: Calculate metadata for links. If False, metadata will be empty.
        :type force_metadata_for_links: bool, optional
        :param infos: DEPRECATED: This parameter is not used.
        :type infos: List[ImageInfo], optional
        :param skip_validation: Skips validation for images, can result in invalid images being uploaded.
        :type skip_validation: bool, optional
        :param conflict_resolution: The strategy to resolve upload conflicts. 'Replace' option will replace the existing images in the dataset with the new images. The images that are being deleted are logged. 'Skip' option will ignore the upload of new images that would result in a conflict. An original image's ImageInfo list will be returned instead. 'Rename' option will rename the new images to prevent any conflict.
        :type conflict_resolution: Optional[Literal["rename", "skip", "replace"]]
        :return: List with information about Images. See :class:`info_sequence<info_sequence>`
        :rtype: :class:`List[ImageInfo]`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            src_dataset_id = 447130

            ids = []
            names = []
            metas = []
            imgs_info = api.image.get_list(src_dataset_id)
            # Create lists of ids, images names and meta information for each image
            for im_info in imgs_info:
                ids.append(im_info.id)
                # It is necessary to upload images with the same names(extentions) as in src dataset
                names.append(im_info.name)
                metas.append({im_info.name: im_info.size})

            dst_dataset_id = 452984
            progress = sly.Progress("Images upload: ", len(ids))
            new_imgs_info = api.image.upload_ids(dst_dataset_id, names, ids, progress.iters_done_report, metas)
            # Output:
            # {"message": "progress", "event_type": "EventType.PROGRESS", "subtask": "Images downloaded: ", "current": 0, "total": 10, "timestamp": "2021-03-16T12:31:36.550Z", "level": "info"}
            # {"message": "progress", "event_type": "EventType.PROGRESS", "subtask": "Images downloaded: ", "current": 10, "total": 10, "timestamp": "2021-03-16T12:31:37.119Z", "level": "info"}

            # Add custom sort parameter for images
            new_dataset_id = 452985
            new_metas = [{'my-key':'a'}, {'my-key':'b'}, {'my-key':'c'}]
            with api.image.add_custom_sort(key="my-key"):
                img_infos = api.image.upload_ids(new_dataset_id, names=names, ids=ids, metas=new_metas)
        """
        if metas is None:
            metas = [{}] * len(names)

        return self._upload_bulk_add(
            lambda item: (ApiField.IMAGE_ID, item),
            dataset_id,
            names,
            ids,
            progress_cb,
            metas=metas,
            batch_size=batch_size,
            force_metadata_for_links=force_metadata_for_links,
            skip_validation=skip_validation,
            conflict_resolution=conflict_resolution,
        )

    def upload_by_offsets(
        self,
        dataset: Union[DatasetInfo, int],
        team_file_id: int,
        names: List[str] = None,
        offsets: List[dict] = None,
        progress_cb: Optional[Union[tqdm, Callable]] = None,
        metas: Optional[List[Dict]] = None,
        batch_size: Optional[int] = 50,
        skip_validation: Optional[bool] = False,
        conflict_resolution: Optional[Literal["rename", "skip", "replace"]] = None,
        validate_meta: Optional[bool] = False,
        use_strict_validation: Optional[bool] = False,
        use_caching_for_validation: Optional[bool] = False,
    ) -> List[ImageInfo]:
        """
        Upload images from blob file in Team Files by offsets to Dataset with prepared names.
        To upload large number of images, use :func:`api.image.upload_by_offsets_generator` instead.

        If you include `metas` during the upload, you can add a custom sort parameter for images.
        To achieve this, use the context manager :func:`api.image.add_custom_sort` with the desired key name from the meta dictionary to be used for sorting.

        :param dataset: Dataset ID or DatasetInfo object in Supervisely.
        :type dataset: Union[DatasetInfo,int]
        :param team_file_id: ID of the binary file in the team storage.
        :type team_file_id: int
        :param names: Images names with extension.

                      REQUIRED if there is no file containing offsets in the team storage at the same level as the TAR file.
                      Offset file must be named as the TAR file with the `_offsets.pkl` suffix and must be represented in pickle format.
                      Example: `tar_name_offsets.pkl`
        :type names: List[str], optional
        :param offsets: List of dictionaries with file offsets that define the range of bytes representing the image in the binary.
                        Example: `[{"offsetStart": 0, "offsetEnd": 100}, {"offsetStart": 101, "offsetEnd": 200}]`.

                        REQUIRED if there is no file containing offsets in the team storage at the same level as the TAR file.
                        Offset file must be named as the TAR file with the `_offsets.pkl` suffix and must be represented in pickle format.
                        Example: `tar_name_offsets.pkl`
        :type offsets: List[dict], optional
        :param progress_cb: Function for tracking the progress of uploading.
        :type progress_cb: tqdm or callable, optional
        :param metas: Custom additional image infos that contain images technical and/or user-generated data as list of separate dicts.
        :type metas: List[dict], optional
        :param batch_size: Number of images to upload in one batch.
        :type batch_size: int, optional
        :param skip_validation: Skips validation for images, can result in invalid images being uploaded.
        :type skip_validation: bool, optional
        :param conflict_resolution: The strategy to resolve upload conflicts. 'Replace' option will replace the existing images in the dataset with the new images. The images that are being deleted are logged. 'Skip' option will ignore the upload of new images that would result in a conflict. An original image's ImageInfo list will be returned instead. 'Rename' option will rename the new images to prevent any conflict.
        :type conflict_resolution: Optional[Literal["rename", "skip", "replace"]]
        :param validate_meta: If True, validates provided meta with saved JSON schema.
        :type validate_meta: bool, optional
        :param use_strict_validation: If True, uses strict validation.
        :type use_strict_validation: bool, optional
        :param use_caching_for_validation: If True, uses caching for validation.
        :type use_caching_for_validation: bool, optional
        :return: List with information about Images. See :class:`info_sequence<info_sequence>`
        :rtype: :class:`List[ImageInfo]`
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            from supervisely.api.module_api import ApiField


            server_address = 'https://app.supervisely.com'
            api_token = 'Your Supervisely API Token'
            api = sly.Api(server_address, api_token)

            dataset_id = 452984
            names = ['lemon_1.jpg', 'lemon_1.jpg']
            offsets = [
                {ApiField.OFFSET_START: 0, ApiField.OFFSET_END: 100},
                {ApiField.OFFSET_START: 101, ApiField.OFFSET_END: 200}
            ]
            team_file_id = 123456
            new_imgs_info = api.image.upload_by_offsets(dataset_id, team_file_id, names, offsets,  metas)

            # Output example:
            #   ImageInfo(id=136281,
            #             name='lemon_1.jpg',
            #             link=None,
            #             hash=None,
            #             mime=None,
            #             ext=None,
            #             size=100,
            #             width=None,
            #             height=None,
            #             labels_count=0,
            #             dataset_id=452984,
            #             created_at='2025-03-21T18:30:08.551Z',
            #             updated_at='2025-03-21T18:30:08.551Z',
            #             meta={},
            #             path_original='/h5un6l2.../eyJ0eXBlIjoic291cmNlX2Jsb2I...',
            #             full_storage_url='http://storage:port/h5un6l2...,
            #             tags=[],
            #             created_by_id=user),
            #   ImageInfo(...)
        """

        if isinstance(dataset, int):
            dataset = self._api.dataset.get_info_by_id(dataset)

        items = []
        if len(names) != len(offsets):
            raise ValueError(
                f"The number of images in the offset file does not match the number of offsets: {len(names)} != {len(offsets)}"
            )
        for offset in offsets:
            if not isinstance(offset, dict):
                raise ValueError("Offset should be a dictionary")
            if ApiField.OFFSET_START not in offset or ApiField.OFFSET_END not in offset:
                raise ValueError(
                    f"Offset should contain '{ApiField.OFFSET_START}' and '{ApiField.OFFSET_END}' keys"
                )

            items.append({ApiField.TEAM_FILE_ID: team_file_id, ApiField.SOURCE_BLOB: offset})

        custom_data = self._api.project.get_custom_data(dataset.project_id)
        custom_data[_BLOB_TAG_NAME] = True
        self._api.project.update_custom_data(dataset.project_id, custom_data)

        return self._upload_bulk_add(
            func_item_to_kv=lambda image_data, item: {**image_data, **item},
            dataset_id=dataset.id,
            names=names,
            items=items,
            progress_cb=progress_cb,
            metas=metas,
            batch_size=batch_size,
            skip_validation=skip_validation,
            conflict_resolution=conflict_resolution,
            validate_meta=validate_meta,
            use_strict_validation=use_strict_validation,
            use_caching_for_validation=use_caching_for_validation,
        )

    def upload_by_offsets_generator(
        self,
        dataset: Union[DatasetInfo, int],
        team_file_id: int,
        offsets_file_path: Optional[str] = None,
        progress_cb: Optional[Union[tqdm, Callable]] = None,
        metas: Optional[Dict] = None,
        batch_size: Optional[int] = 10000,
        skip_validation: Optional[bool] = False,
        conflict_resolution: Optional[Literal["rename", "skip", "replace"]] = None,
        validate_meta: Optional[bool] = False,
        use_strict_validation: Optional[bool] = False,
        use_caching_for_validation: Optional[bool] = False,
    ) -> Generator[ImageInfo, None, None]:
        """
        Upload images from blob file in Team Files by offsets to Dataset.
        Generates information about uploaded images in batches of max size 10000.
        File names will be taken from the offset file.

        This method is better suited for large datasets, as it does not require resulting all the images into memory at once.

        If you include `metas` during the upload, you can add a custom sort parameter for images.
        To achieve this, use the context manager :func:`api.image.add_custom_sort` with the desired key name from the meta dictionary to be used for sorting.

        :param dataset: Dataset ID or DatasetInfo object in Supervisely.
        :type dataset: Union[DatasetInfo,int]
        :param team_file_id: ID of the binary file in the team storage.
        :type team_file_id: int
        :param offsets_file_path: Local path to the file with blob images offsets.
        :type offsets_file_path: str, optional
        :param progress_cb: Function for tracking the progress of uploading.
        :type progress_cb: tqdm or callable, optional
        :param metas: Custom additional image infos as dict where:
                     `keys` - image names,
                     `values` - image technical and/or user-generated data dicts
        :type metas: Dict, optional
        :param batch_size: Number of images to upload in one batch.
        :type batch_size: int, optional
        :param skip_validation: Skips validation for images, can result in invalid images being uploaded.
        :type skip_validation: bool, optional
        :param conflict_resolution: The strategy to resolve upload conflicts. 'Replace' option will replace the existing images in the dataset with the new images. The images that are being deleted are logged. 'Skip' option will ignore the upload of new images that would result in a conflict. An original image's ImageInfo list will be returned instead. 'Rename' option will rename the new images to prevent any conflict.
        :type conflict_resolution: Optional[Literal["rename", "skip", "replace"]]
        :param validate_meta: If True, validates provided meta with saved JSON schema.
        :type validate_meta: bool, optional
        :param use_strict_validation: If True, uses strict validation.
        :type use_strict_validation: bool, optional
        :param use_caching_for_validation: If True, uses caching for validation.
        :type use_caching_for_validation: bool, optional
        :return: Generator with information about Images. See :class:`ImageInfo`
        :rtype: :class:`Generator[ImageInfo, None, None]`
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            from supervisely.api.module_api import ApiField


            server_address = 'https://app.supervisely.com'
            api_token = 'Your Supervisely API Token'
            api = sly.Api(server_address, api_token)

            dataset_id = 452984
            team_file_id = 123456
            img_infos = []
            new_imgs_info_generator = api.image.upload_by_offsets_generator(dataset_id, team_file_id)
            for img_infos_batch in new_imgs_info_generator:
                img_infos.extend(img_infos_batch)
        """

        if isinstance(dataset, int):
            dataset = self._api.dataset.get_info_by_id(dataset)

        if offsets_file_path is None:
            offsets_file_path = self.get_blob_offsets_file(team_file_id)
        blob_image_infos_generator = BlobImageInfo.load_from_pickle_generator(
            offsets_file_path, OFFSETS_PKL_BATCH_SIZE
        )

        for batch in blob_image_infos_generator:
            names = [item.name for item in batch]
            metas_batch = (
                [metas[name] for name in names] if metas is not None else [{}] * len(names)
            )
            items = [
                {ApiField.TEAM_FILE_ID: team_file_id, ApiField.SOURCE_BLOB: item.offsets_dict}
                for item in batch
            ]
            yield self._upload_bulk_add(
                func_item_to_kv=lambda image_data, item: {**image_data, **item},
                dataset_id=dataset.id,
                names=names,
                items=items,
                progress_cb=progress_cb,
                metas=metas_batch,
                batch_size=batch_size,
                skip_validation=skip_validation,
                conflict_resolution=conflict_resolution,
                validate_meta=validate_meta,
                use_strict_validation=use_strict_validation,
                use_caching_for_validation=use_caching_for_validation,
            )
        custom_data = self._api.project.get_custom_data(dataset.project_id)
        custom_data[_BLOB_TAG_NAME] = True
        self._api.project.update_custom_data(dataset.project_id, custom_data)

    def get_blob_offsets_file(
        self,
        team_file_id: int,
        progress_cb: Optional[Union[tqdm, Callable]] = None,
    ) -> str:
        """
        Get file with blob images offsets from the team storage.

        :param team_file_id: ID of the binary file in the team storage.
        :type team_file_id: int
        :param progress_cb: Function for tracking the progress of downloading.
        :type progress_cb: tqdm or callable, optional
        :return: Path to the file with blob images offsets in temporary directory.
        :rtype: str

        """
        file_info = self._api.file.get_info_by_id(team_file_id)
        if file_info is None:
            raise ValueError(f"Blob file ID: {team_file_id} with images not found")
        offset_file_name = Path(file_info.path).stem + OFFSETS_PKL_SUFFIX
        offset_file_path = os.path.join(Path(file_info.path).parent, offset_file_name)
        temp_dir = tempfile.mkdtemp()
        local_offset_file_path = os.path.join(temp_dir, offset_file_name)
        self._api.file.download(
            team_id=file_info.team_id,
            remote_path=offset_file_path,
            local_save_path=local_offset_file_path,
            progress_cb=progress_cb,
        )
        return local_offset_file_path

    def _upload_bulk_add(
        self,
        func_item_to_kv,
        dataset_id,
        names,
        items,
        progress_cb=None,
        metas=None,
        batch_size=50,
        force_metadata_for_links=True,
        skip_validation=False,
        conflict_resolution: Optional[Literal["rename", "skip", "replace"]] = None,
        validate_meta: Optional[bool] = False,
        use_strict_validation: Optional[bool] = False,
        use_caching_for_validation: Optional[bool] = False,
    ):
        """ """
        if use_strict_validation and not validate_meta:
            raise ValueError(
                "use_strict_validation is set to True, while validate_meta is set to False. "
                "Please set validate_meta to True to use strict validation "
                "or disable strict validation by setting use_strict_validation to False."
            )
        if validate_meta:
            dataset_info = self._api.dataset.get_info_by_id(dataset_id)

            validation_schema = self._api.project.get_validation_schema(
                dataset_info.project_id, use_caching=use_caching_for_validation
            )

            if validation_schema is None:
                raise ValueError(
                    "Validation schema is not set for the project, while "
                    "validate_meta is set to True. Either disable the validation "
                    "or set the validation schema for the project using the "
                    "api.project.set_validation_schema method."
                )

            for idx, meta in enumerate(metas):
                missing_fields, extra_fields = compare_dicts(
                    validation_schema, meta, strict=use_strict_validation
                )

                if missing_fields or extra_fields:
                    raise ValueError(
                        f"Validation failed for the metadata of the image with index {idx} and name {names[idx]}. "
                        "Please check the metadata and try again. "
                        f"Missing fields: {missing_fields}, Extra fields: {extra_fields}"
                    )

        if (
            conflict_resolution is not None
            and conflict_resolution not in SUPPORTED_CONFLICT_RESOLUTIONS
        ):
            raise ValueError(
                f"Conflict resolution should be one of the following: {SUPPORTED_CONFLICT_RESOLUTIONS}"
            )
        if len(set(names)) != len(names):
            raise ValueError("Some image names are duplicated, only unique images can be uploaded.")

        results = []

        def _add_timestamp(name: str) -> str:
            now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            return f"{get_file_name(name)}_{now}{get_file_ext(name)}"

        def _pack_for_request(names: List[str], items: List[Any], metas: List[Dict]) -> List[Any]:
            images = []
            for name, item, meta in zip(names, items, metas):
                image_data = {ApiField.TITLE: name}
                # Check if the item is a data format for upload by offset
                if (
                    isinstance(item, dict)
                    and ApiField.TEAM_FILE_ID in item
                    and ApiField.SOURCE_BLOB in item
                ):
                    image_data = func_item_to_kv(image_data, item)
                else:
                    item_tuple = func_item_to_kv(item)
                    image_data[item_tuple[0]] = item_tuple[1]
                if hasattr(self, "sort_by") and self.sort_by is not None:
                    meta = self._add_custom_sort(meta, name)
                if len(meta) != 0 and type(meta) == dict:
                    image_data[ApiField.META] = meta
                images.append(image_data)
            return images

        if len(names) == 0:
            return results
        if len(names) != len(items):
            raise ValueError('Can not match "names" and "items" lists, len(names) != len(items)')

        if metas is None:
            metas = [{}] * len(names)
        else:
            if len(names) != len(metas):
                raise ValueError('Can not match "names" and "metas" len(names) != len(metas)')

        idx_to_id = {}
        for batch_count, (batch_names, batch_items, batch_metas) in enumerate(
            zip(
                batched(names, batch_size=batch_size),
                batched(items, batch_size=batch_size),
                batched(metas, batch_size=batch_size),
            )
        ):
            for retry in range(2):
                images = _pack_for_request(batch_names, batch_items, batch_metas)
                try:
                    response = self._api.post(
                        "images.bulk.add",
                        {
                            ApiField.DATASET_ID: dataset_id,
                            ApiField.IMAGES: images,
                            ApiField.FORCE_METADATA_FOR_LINKS: force_metadata_for_links,
                            ApiField.SKIP_VALIDATION: skip_validation,
                        },
                    )
                    if progress_cb is not None:
                        progress_cb(len(images))

                    for info_json in response.json():
                        info_json_copy = info_json.copy()
                        if info_json.get(ApiField.MIME, None) is not None:
                            info_json_copy[ApiField.EXT] = info_json[ApiField.MIME].split("/")[1]
                        # results.append(self.InfoType(*[info_json_copy[field_name] for field_name in self.info_sequence()]))
                        results.append(self._convert_json_info(info_json_copy))

                    try:
                        if "import" in app_categories():
                            ids = [info.id for info in results[-len(batch_names) :]]
                            if len(ids) > 0:
                                increment_upload_count(dataset_id, len(ids))
                                add_uploaded_ids_to_env(dataset_id, ids)
                    except:
                        pass

                    break
                except HTTPError as e:
                    error_details = e.response.json().get("details", {})
                    if isinstance(error_details, list):
                        error_details = error_details[0]
                    if (
                        conflict_resolution is not None
                        and e.response.status_code == 400
                        and error_details.get("type") == "NONUNIQUE"
                    ):
                        logger.info(
                            f"Handling the exception above with '{conflict_resolution}' conflict resolution method"
                        )

                        errors: List[Dict] = error_details.get("errors", [])

                        if conflict_resolution == "replace":
                            ids_to_remove = [error["id"] for error in errors]
                            logger.info(f"Image ids to be removed: {ids_to_remove}")
                            self._api.image.remove_batch(ids_to_remove)
                            continue

                        name_to_index = {name: idx for idx, name in enumerate(batch_names)}
                        errors = sorted(
                            errors, key=lambda x: name_to_index[x["name"]], reverse=True
                        )
                        if conflict_resolution == "rename":
                            for error in errors:
                                error_img_name = error["name"]
                                idx = name_to_index[error_img_name]
                                batch_names[idx] = _add_timestamp(error_img_name)
                        elif conflict_resolution == "skip":
                            for error in errors:
                                error_img_name = error["name"]
                                error_index = name_to_index[error_img_name]

                                idx_to_id[error_index + batch_count * batch_size] = error["id"]
                                for l in [batch_items, batch_metas, batch_names]:
                                    l.pop(error_index)

                        if len(batch_names) == 0:
                            break
                    else:
                        raise

        # name_to_res = {img_info.name: img_info for img_info in results}
        # ordered_results = [name_to_res[name] for name in names]

        if len(idx_to_id) > 0:
            logger.info(
                "Adding ImageInfo of images with the same name that already exist in the dataset to the response."
            )

            idx_to_id = dict(reversed(list(idx_to_id.items())))
            image_infos = self._api.image.get_info_by_id_batch(list(idx_to_id.values()))
            for idx, info in zip(list(idx_to_id.values()), image_infos):
                results.insert(idx, info)

        return results  # ordered_results

    # @TODO: reimplement
    def _convert_json_info(self, info: dict, skip_missing=True):
        """ """
        if info is None:
            return None
        temp_ext = None
        field_values = []
        for field_name in self.info_sequence():
            if field_name == ApiField.EXT:
                continue
            if skip_missing is True:
                val = info.get(field_name, None)
            else:
                val = info[field_name]
            field_values.append(val)
            if field_name == ApiField.MIME:
                if val is not None:
                    temp_ext = val.split("/")[1]
                else:
                    temp_ext = None
                field_values.append(temp_ext)
        for idx, field_name in enumerate(self.info_sequence()):
            if field_name == ApiField.NAME:
                cur_ext = get_file_ext(field_values[idx]).replace(".", "").lower()
                if not cur_ext:
                    field_values[idx] = "{}.{}".format(field_values[idx], temp_ext)
                    break
                if temp_ext is None:
                    break
                if temp_ext == "jpeg" and cur_ext in ["jpg", "jpeg", "mpo"]:
                    break
                if temp_ext != cur_ext:
                    field_values[idx] = "{}.{}".format(field_values[idx], temp_ext)
                break

        res = self.InfoType(*field_values)
        d = res._asdict()

        return ImageInfo(**d)

    def _remove_batch_api_method_name(self):
        """ """
        return "images.bulk.remove"

    def _remove_batch_field_name(self):
        """ """
        return ApiField.IMAGE_IDS

    def copy_batch(
        self,
        dst_dataset_id: int,
        ids: List[int],
        change_name_if_conflict: Optional[bool] = False,
        with_annotations: Optional[bool] = False,
        progress_cb: Optional[Union[tqdm, Callable]] = None,
    ) -> List[ImageInfo]:
        """
        Copies Images with given IDs to Dataset.

        :param dst_dataset_id: Destination Dataset ID in Supervisely.
        :type dst_dataset_id: int
        :param ids: Images IDs in Supervisely.
        :type ids: List[int]
        :param change_name_if_conflict: If True adds suffix to the end of Image name when Dataset already contains an Image with identical name, If False and images with the identical names already exist in Dataset raises error.
        :type change_name_if_conflict: bool, optional
        :param with_annotations: If True Image will be copied to Dataset with annotations, otherwise only Images without annotations.
        :type with_annotations: bool, optional
        :param progress_cb: Function for tracking the progress of copying.
        :type progress_cb: tqdm or callable, optional
        :raises: :class:`TypeError` if type of ids is not list
        :raises: :class:`ValueError` if images ids are from the destination Dataset
        :return: List with information about Images. See :class:`info_sequence<info_sequence>`
        :rtype: :class:`List[ImageInfo]`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            ds_lemon_id = 1780

            ds_lemon_img_infos = api.image.get_list(ds_lemon_id)

            lemons_img_ids = [lemon_img_info.id for lemon_img_info in ds_lemon_img_infos]

            ds_fruit_id = 2574
            ds_fruit_img_infos = api.image.copy_batch(ds_fruit_id, lemons_img_ids, with_annotations=True)
        """
        if type(ids) is not list:
            raise TypeError(
                "ids parameter has type {!r}. but has to be of type {!r}".format(type(ids), list)
            )

        if len(ids) == 0:
            return

        existing_images = self.get_list(dst_dataset_id, force_metadata_for_links=False)
        existing_names = {image.name for image in existing_images}

        ids_info = self.get_info_by_id_batch(ids, force_metadata_for_links=False)
        temp_ds_ids = []
        for info in ids_info:
            if info.dataset_id not in temp_ds_ids:
                temp_ds_ids.append(info.dataset_id)
        if len(temp_ds_ids) > 1:
            raise ValueError("Images ids have to be from the same dataset")

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
                    "Images with the same names already exist in destination dataset. "
                    'Please, use argument "change_name_if_conflict=True" to automatically resolve '
                    "names intersection"
                )

        img_metas = [info.meta or {} for info in ids_info]
        new_images = self.upload_ids(dst_dataset_id, new_names, ids, progress_cb, metas=img_metas)
        new_ids = [new_image.id for new_image in new_images]

        if with_annotations:
            src_project_id = self._api.dataset.get_info_by_id(ids_info[0].dataset_id).project_id
            dst_project_id = self._api.dataset.get_info_by_id(dst_dataset_id).project_id
            self._api.project.merge_metas(src_project_id, dst_project_id)
            self._api.annotation.copy_batch(ids, new_ids)

        return new_images

    def copy_batch_optimized(
        self,
        src_dataset_id: int,
        src_image_infos: List[ImageInfo],
        dst_dataset_id: int,
        with_annotations: Optional[bool] = True,
        progress_cb: Optional[Union[tqdm, Callable]] = None,
        dst_names: List[ImageInfo] = None,
        batch_size: Optional[int] = 500,
        skip_validation: Optional[bool] = False,
        save_source_date: Optional[bool] = True,
    ) -> List[ImageInfo]:
        """
        Copies Images with given IDs to Dataset.

        :param src_dataset_id: Source Dataset ID in Supervisely.
        :type src_dataset_id: int
        :param src_image_infos: ImageInfo objects of images to copy.
        :type src_image_infos: List [ :class:`ImageInfo` ]
        :param dst_dataset_id: Destination Dataset ID in Supervisely.
        :type dst_dataset_id: int
        :param with_annotations: If True Image will be copied to Dataset with annotations, otherwise only Images without annotations.
        :type with_annotations: bool, optional
        :param progress_cb: Function for tracking the progress of copying.
        :type progress_cb: tqdm or callable, optional
        :param dst_names: ImageInfo list with existing items in destination dataset.
        :type dst_names: List [ :class:`ImageInfo` ], optional
        :param batch_size: Number of elements to copy for each request.
        :type batch_size: int, optional
        :param skip_validation: Flag for skipping additinal validations.
        :type skip_validation: bool, optional
        :param save_source_date: Save source annotation dates (creation and modification) or create a new date.
        :type save_source_date: bool, optional
        :raises: :class:`TypeError` if type of src_image_infos is not list
        :return: List with information about Images. See :class:`info_sequence<info_sequence>`
        :rtype: :class:`List[ImageInfo]`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            src_ds_id = 2231
            img_infos = api.image.get_list(src_ds_id)

            dest_ds_id = 2574
            dest_img_infos = api.image.copy_batch_optimized(src_ds_id, img_infos, dest_ds_id)
        """
        if type(src_image_infos) is not list:
            raise TypeError(
                "src_image_infos parameter has type {!r}. but has to be of type {!r}".format(
                    type(src_image_infos), list
                )
            )

        if len(src_image_infos) == 0:
            return

        if dst_names is None:
            existing_images = self.get_list(dst_dataset_id, force_metadata_for_links=False)
            existing_names = {image.name for image in existing_images}
            new_names = [
                generate_free_name(existing_names, info.name, with_ext=True, extend_used_names=True)
                for info in src_image_infos
            ]
        else:
            if len(dst_names) != len(src_image_infos):
                raise RuntimeError("len(dst_names) != len(src_image_infos)")
            new_names = dst_names

        img_metas = [info.meta or {} for info in src_image_infos]
        src_ids = [info.id for info in src_image_infos]
        new_images = self.upload_ids(
            dst_dataset_id,
            new_names,
            src_ids,
            progress_cb,
            metas=img_metas,
            batch_size=batch_size,
            force_metadata_for_links=False,
            infos=src_image_infos,
            skip_validation=skip_validation,
        )
        new_ids = [new_image.id for new_image in new_images]

        if with_annotations:
            src_project_id = self._api.dataset.get_info_by_id(src_dataset_id).project_id
            dst_project_id = self._api.dataset.get_info_by_id(dst_dataset_id).project_id
            self._api.project.merge_metas(src_project_id, dst_project_id)
            self._api.annotation.copy_batch_by_ids(
                src_ids,
                new_ids,
                batch_size=batch_size,
                save_source_date=save_source_date,
            )

        return new_images

    def move_batch(
        self,
        dst_dataset_id: int,
        ids: List[int],
        change_name_if_conflict: Optional[bool] = False,
        with_annotations: Optional[bool] = False,
        progress_cb: Optional[Union[tqdm, Callable]] = None,
    ) -> List[ImageInfo]:
        """
        Moves Images with given IDs to Dataset.

        :param dst_dataset_id: Destination Dataset ID in Supervisely.
        :type dst_dataset_id: int
        :param ids: Images IDs in Supervisely.
        :type ids: List[int]
        :param change_name_if_conflict: If True adds suffix to the end of Image name when Dataset already contains an Image with identical name, If False and images with the identical names already exist in Dataset raises error.
        :type change_name_if_conflict: bool, optional
        :param with_annotations: If True Image will be copied to Dataset with annotations, otherwise only Images without annotations.
        :type with_annotations: bool, optional
        :param progress_cb: Function for tracking the progress of moving.
        :type progress_cb: tqdm or callable, optional
        :raises: :class:`TypeError` if type of ids is not list
        :raises: :class:`ValueError` if images ids are from the destination Dataset
        :return: List with information about Images. See :class:`info_sequence<info_sequence>`
        :rtype: :class:`List[ImageInfo]`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            ds_lemon_id = 1780
            ds_kiwi_id = 1233

            ds_lemon_img_infos = api.image.get_list(ds_lemon_id)
            ds_kiwi_img_infos = api.image.get_list(ds_kiwi_id)

            fruit_img_ids = []
            for lemon_img_info, kiwi_img_info in zip(ds_lemon_img_infos, ds_kiwi_img_infos):
                fruit_img_ids.append(lemon_img_info.id)
                fruit_img_ids.append(kiwi_img_info.id)

            ds_fruit_id = 2574
            ds_fruit_img_infos = api.image.move_batch(ds_fruit_id, fruit_img_ids, with_annotations=True)
        """
        new_images = self.copy_batch(
            dst_dataset_id, ids, change_name_if_conflict, with_annotations, progress_cb
        )
        self.remove_batch(ids)
        return new_images

    def move_batch_optimized(
        self,
        src_dataset_id: int,
        src_image_infos: List[ImageInfo],
        dst_dataset_id: int,
        with_annotations: Optional[bool] = True,
        progress_cb: Optional[Union[tqdm, Callable]] = None,
        dst_names: List[ImageInfo] = None,
        batch_size: Optional[int] = 500,
        skip_validation: Optional[bool] = False,
        save_source_date: Optional[bool] = True,
    ) -> List[ImageInfo]:
        """
        Moves Images with given IDs to Dataset.

        :param src_dataset_id: Source Dataset ID in Supervisely.
        :type src_dataset_id: int
        :param src_image_infos: ImageInfo objects of images to move.
        :type src_image_infos: List [ :class:`ImageInfo` ]
        :param dst_dataset_id: Destination Dataset ID in Supervisely.
        :type dst_dataset_id: int
        :param with_annotations: If True Image will be copied to Dataset with annotations, otherwise only Images without annotations.
        :type with_annotations: bool, optional
        :param progress_cb: Function for tracking the progress of moving.
        :type progress_cb: tqdm or callable, optional
        :param dst_names: ImageInfo list with existing items in destination dataset.
        :type dst_names: List [ :class:`ImageInfo` ], optional
        :param batch_size: Number of elements to copy for each request.
        :type batch_size: int, optional
        :param skip_validation: Flag for skipping additinal validations.
        :type skip_validation: bool, optional
        :param save_source_date: Save source annotation dates (creation and modification) or create a new date.
        :type save_source_date: bool, optional
        :raises: :class:`TypeError` if type of src_image_infos is not list
        :return: List with information about Images. See :class:`info_sequence<info_sequence>`
        :rtype: :class:`List[ImageInfo]`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            src_ds_id = 2231
            img_infos = api.image.get_list(src_ds_id)

            dest_ds_id = 2574
            dest_img_infos = api.image.move_batch_optimized(src_ds_id, img_infos, dest_ds_id)
        """
        new_images = self.copy_batch_optimized(
            src_dataset_id,
            src_image_infos,
            dst_dataset_id,
            with_annotations=with_annotations,
            progress_cb=progress_cb,
            dst_names=dst_names,
            batch_size=batch_size,
            skip_validation=skip_validation,
            save_source_date=save_source_date,
        )
        src_ids = [info.id for info in src_image_infos]
        self.remove_batch(src_ids, batch_size=batch_size)
        return new_images

    def copy(
        self,
        dst_dataset_id: int,
        id: int,
        change_name_if_conflict: Optional[bool] = False,
        with_annotations: Optional[bool] = False,
    ) -> ImageInfo:
        """
        Copies Image with given ID to destination Dataset.

        :param dst_dataset_id: Destination Dataset ID in Supervisely.
        :type dst_dataset_id: int
        :param id: Image ID in Supervisely.
        :type id: int
        :param change_name_if_conflict: If True adds suffix to the end of Image name when Dataset already contains an Image with identical name, If False and images with the identical names already exist in Dataset raises error.
        :type change_name_if_conflict: bool, optional
        :param with_annotations: If True Image will be copied to Dataset with annotations, otherwise only Images without annotations.
        :type with_annotations: bool, optional
        :return: Information about Image. See :class:`info_sequence<info_sequence>`
        :rtype: :class:`ImageInfo`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            dst_ds_id = 365184
            img_id = 121236920

            img_info = api.image.copy(dst_ds_id, img_id, with_annotations=True)
        """
        return self.copy_batch(dst_dataset_id, [id], change_name_if_conflict, with_annotations)[0]

    def move(
        self,
        dst_dataset_id: int,
        id: int,
        change_name_if_conflict: Optional[bool] = False,
        with_annotations: Optional[bool] = False,
    ) -> ImageInfo:
        """
        Moves Image with given ID to destination Dataset.

        :param dst_dataset_id: Destination Dataset ID in Supervisely.
        :type dst_dataset_id: int
        :param id: Image ID in Supervisely.
        :type id: int
        :param change_name_if_conflict: If True adds suffix to the end of Image name when Dataset already contains an Image with identical name, If False and images with the identical names already exist in Dataset raises error.
        :type change_name_if_conflict: bool, optional
        :param with_annotations: If True Image will be copied to Dataset with annotations, otherwise only Images without annotations.
        :type with_annotations: bool, optional
        :return: Information about Image. See :class:`info_sequence<info_sequence>`
        :rtype: :class:`ImageInfo`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            dst_ds_id = 365484
            img_id = 533336920

            img_info = api.image.copy(dst_ds_id, img_id, with_annotations=True)
        """
        return self.move_batch(dst_dataset_id, [id], change_name_if_conflict, with_annotations)[0]

    def url(
        self,
        team_id: int,
        workspace_id: int,
        project_id: int,
        dataset_id: int,
        image_id: int,
    ) -> str:
        """
        Gets Image URL by ID.

        :param team_id: Team ID in Supervisely.
        :type team_id: int
        :param workspace_id: Workspace ID in Supervisely.
        :type workspace_id: int
        :param project_id: Project ID in Supervisely.
        :type project_id: int
        :param dataset_id: Dataset ID in Supervisely.
        :type dataset_id: int
        :param image_id: Image ID in Supervisely.
        :type image_id: int
        :return: Image URL
        :rtype: :class:`str`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            team_id = 16087
            workspace_id = 23821
            project_id = 53939
            dataset_id = 254737
            image_id = 121236920

            img_url = api.image.url(team_id, workspace_id, project_id, dataset_id, image_id)
            print(url)
            # Output: https://app.supervisely.com/app/images/16087/23821/53939/254737#image-121236920
        """
        result = urllib.parse.urljoin(
            self._api.server_address,
            "app/images/{}/{}/{}/{}#image-{}".format(
                team_id, workspace_id, project_id, dataset_id, image_id
            ),
        )

        return result

    def _download_batch_by_hashes(
        self, hashes, retry_attemps=4
    ) -> Generator[Tuple[str, Any, bool], None, None]:
        """
        Download Images with given hashes by batches from Supervisely server.

        :param hashes: List of images hashes in Supervisely.
        :type hashes: List[str]
        :param retry_attemps: Number of attempts to download images.
        :type retry_attemps: int, optional
        :return: Generator with images hashes, images data and verification status.
        :rtype: :class:`Generator[Tuple[str, Any, bool], None, None]`
        """

        for batch_hashes in batched(hashes):
            for attempt in range(retry_attemps + 1):  # the first attempt is not counted as a retry
                if len(batch_hashes) == 0:
                    break
                successful_hashes = []
                response = self._api.post(
                    "images.bulk.download-by-hash", {ApiField.HASHES: batch_hashes}
                )
                decoder = MultipartDecoder.from_response(response)
                for part in decoder.parts:
                    content_utf8 = part.headers[b"Content-Disposition"].decode("utf-8")
                    # Find name="1245" preceded by a whitespace, semicolon or beginning of line.
                    # The regex has 2 capture group: one for the prefix and one for the actual name value.
                    h = content_utf8.replace('form-data; name="', "")[:-1]
                    image_data = io.BytesIO(part.content).getvalue()
                    image_hash = get_bytes_hash(image_data)
                    verified = False
                    if image_hash == h:
                        successful_hashes.append(h)
                        verified = True
                    yield h, part, verified

                batch_hashes = [h for h in batch_hashes if h not in successful_hashes]
                if len(batch_hashes) > 0:
                    if attempt == retry_attemps:
                        logger.warning(
                            "Failed to download images with hashes: %s. Skipping.",
                            batch_hashes,
                        )
                        break
                    else:
                        next_attempt = attempt + 1
                        logger.warning(
                            "Failed to download images with hashes: %s. Retrying (%d/%d).",
                            batch_hashes,
                            next_attempt,
                            retry_attemps,
                        )
                        sleep(2 ** (next_attempt))

    def download_paths_by_hashes(
        self,
        hashes: List[str],
        paths: List[str],
        progress_cb: Optional[Union[tqdm, Callable]] = None,
    ) -> None:
        """
        Download Images with given hashes in Supervisely server and saves them for the given paths.

        :param hashes: List of images hashes in Supervisely.
        :type hashes: List[str]
        :param paths: List of paths to save images.
        :type paths: List[str]
        :param progress_cb: Function for tracking download progress.
        :type progress_cb: tqdm or callable, optional
        :raises: :class:`ValueError` if len(hashes) != len(paths)
        :return: None
        :rtype: :class:`NoneType`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            dataset_id = 447130
            dir_for_save = '/home/admin/Downloads/img'
            hashes = []
            paths = []
            imgs_info = api.image.get_list(dataset_id)
            for im_info in imgs_info:
                hashes.append(im_info.hash)
                # It is necessary to save images with the same names(extentions) as on the server
                paths.append(os.path.join(dir_for_save, im_info.name))
            api.image.download_paths_by_hashes(hashes, paths)
        """
        if len(hashes) == 0:
            return
        if len(hashes) != len(paths):
            raise ValueError('Can not match "hashes" and "paths" lists, len(hashes) != len(paths)')

        h_to_path = {h: path for h, path in zip(hashes, paths)}
        for h, resp_part, verified in self._download_batch_by_hashes(list(set(hashes))):
            ensure_base_path(h_to_path[h])
            with open(h_to_path[h], "wb") as w:
                w.write(resp_part.content)
            if progress_cb is not None:
                if verified:
                    progress_cb(1)

    def download_nps_by_hashes_generator(
        self,
        hashes: List[str],
        keep_alpha: bool = False,
        progress_cb: Optional[Union[tqdm, Callable]] = None,
    ) -> Generator[Tuple[str, np.ndarray], None, None]:
        if len(hashes) == 0:
            return []

        if len(hashes) != len(set(hashes)):
            logger.warn("Found nonunique hashes in download task")

        for im_hash, resp_part, verified in self._download_batch_by_hashes(hashes):
            yield im_hash, sly_image.read_bytes(resp_part.content, keep_alpha)
            if progress_cb is not None:
                if verified:
                    progress_cb(1)

    def download_nps_by_hashes(
        self,
        hashes: List[str],
        keep_alpha: bool = False,
        progress_cb: Optional[Union[tqdm, Callable]] = None,
    ) -> List[np.ndarray]:
        """
        Download Images with given hashes in Supervisely server in numpy format.

        :param hashes: List of images hashes in Supervisely.
        :type hashes: List[str]
        :param progress_cb: Function for tracking download progress.
        :type progress_cb: tqdm or callable, optional
        :return: List of images
        :rtype: :class: List[np.ndarray]
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            image_ids = [770918, 770919, 770920]
            image_hashes = []

            for img_id in image_ids:
                img_info = api.image.get_info_by_id(image_id)
                image_hashes.append(img_info.hash)

            image_nps = api.image.download_nps_by_hashes(image_hashes)
        """
        return [
            img
            for _, img in self.download_nps_by_hashes_generator(
                hashes,
                keep_alpha,
                progress_cb,
            )
        ]

    def get_project_id(self, image_id: int) -> int:
        """
        Gets Project ID by Image ID.

        :param image_id: Image ID in Supervisely.
        :type image_id: int
        :return: Project ID where Image is located.
        :rtype: :class:`int`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            img_id = 121236920
            img_project_id = api.image.get_project_id(img_id)
            print(img_project_id)
            # Output: 53939
        """
        dataset_id = self.get_info_by_id(image_id, force_metadata_for_links=False).dataset_id
        project_id = self._api.dataset.get_info_by_id(dataset_id).project_id
        return project_id

    @staticmethod
    def _get_free_name(exist_check_fn, name):
        """ """
        res_title = name
        suffix = 1

        name_without_ext = get_file_name(name)
        ext = get_file_ext(name)

        while exist_check_fn(res_title):
            res_title = "{}_{:03d}{}".format(name_without_ext, suffix, ext)
            suffix += 1
        return res_title

    def storage_url(self, path_original: str) -> str:
        """
        Get full Image URL link in Supervisely server.

        :param path_original: Original Image path in Supervisely server.
        :type path_original: str
        :return: Full Image URL link in Supervisely server
        :rtype: :class:`str`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            image_id = 376729
            img_info = api.image.get_info_by_id(image_id)
            img_storage_url = api.image.storage_url(img_info.path_original)
        """

        return path_original

    def preview_url(
        self,
        url: str,
        width: Optional[int] = None,
        height: Optional[int] = None,
        quality: Optional[int] = 70,
        ext: Literal["jpeg", "png"] = "jpeg",
        method: Literal["fit", "fill", "fill-down", "force", "auto"] = "auto",
    ) -> str:
        """
        Previews Image with the given resolution parameters.
        Learn more about resize parameters `here <https://docs.imgproxy.net/usage/processing#resize>`_.

        :param url: Full Image storage URL.
        :type url: str
        :param width: Preview Image width.
        :type width: int
        :param height: Preview Image height.
        :type height: int
        :param quality: Preview Image quality.
        :type quality: int
        :param ext: Preview Image extension, available values: "jpeg", "png".
        :type ext: str, optional
        :param method: Preview Image resize method, available values: "fit", "fill", "fill-down", "force", "auto".
        :type method: str, optional
        :return: New URL with resized Image
        :rtype: :class:`str`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            image_id = 376729
            img_info = api.image.get_info_by_id(image_id)
            img_preview_url = api.image.preview_url(img_info.full_storage_url, width=512, height=256)
        """
        return resize_image_url(url, ext, method, width, height, quality)

    def update_meta(self, id: int, meta: Dict) -> Dict[str, Any]:
        """
        It is possible to add custom JSON data to every image for storing some additional information.
        Updates Image metadata by ID. Metadata is visible in Labeling Tool.
        Supervisely also have 2 apps: import metadata and export metadata

        :param id: Image ID in Supervisely.
        :type id: int
        :param meta: Custom additional image info that contain image technical and/or user-generated data.
        :type meta: dict
        :raises: :class:`TypeError` if meta type is not dict
        :return: Image information in dict format with new meta
        :rtype: :class:`dict`
        :Usage example:

         .. code-block:: python

            import os
            import json
            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            image_info = api.image.get_info_by_id(id=3212008)
            print(image_info.meta)
            # Output: {}

            new_meta = {'Camera Make': 'Canon', 'Color Space': 'sRGB', 'Focal Length': '16 mm'}
            new_image_info = api.image.update_meta(id=3212008, meta=new_meta)

            image_info = api.image.get_info_by_id(id=3212008)
            print(json.dumps(obj=image_info.meta, indent=4))
            # Output: {
            #     "Camera Make": "Canon",
            #     "Color Space": "sRGB",
            #     "Focal Length": "16 mm"
            # }
        """
        return self.edit(id=id, meta=meta, return_json=True)

    def edit(
        self,
        id: int,
        name: Optional[str] = None,
        description: Optional[str] = None,
        meta: Optional[Dict] = None,
        return_json: bool = False,
    ) -> Union[ImageInfo, Dict[str, Any]]:
        """Updates the information about the image by given ID with provided parameters.
        At least one parameter must be set, otherwise ValueError will be raised.

        :param id: Image ID in Supervisely.
        :type id: int
        :param name: New Image name.
        :type name: str, optional
        :param description: New Image description.
        :type description: str, optional
        :param meta: New Image metadata. Custom additional image info that contain image technical and/or user-generated data.
        :type meta: dict, optional
        :return_json: If True, return response in JSON format, otherwise convert it ImageInfo object.
            This parameter is only added for backward compatibility for update_meta method.
            It's not recommended to use it in new code.
        :type return_json: bool, optional
        :raises: :class:`ValueError` if at least one parameter is not set
        :raises: :class:`ValueError if meta parameter was set and it is not a dictionary
        :return: Information about updated image as ImageInfo object or as dict if return_json is True
        :rtype: :class:`ImageInfo` or :class:`dict`

        :Usage example:

        .. code-block:: python

            import supervisely as sly

            api = sly.Api.from_env()

            image_id = 123456
            new_image_name = "IMG_3333_new.jpg"

            api.image.edit(id=image_id, name=new_image_name)
        """
        if name is None and description is None and meta is None:
            raise ValueError("At least one parameter must be set")

        if meta is not None and not isinstance(meta, dict):
            raise ValueError("meta parameter must be a dictionary")

        data = {
            ApiField.ID: id,
            ApiField.NAME: name,
            ApiField.DESCRIPTION: description,
            ApiField.META: meta,
        }
        data = {k: v for k, v in data.items() if v is not None}

        response = self._api.post("images.editInfo", data)
        if return_json:
            return response.json()
        return self._convert_json_info(response.json(), skip_missing=True)

    def rename(self, id: int, name: str) -> ImageInfo:
        """
        Renames Image with given ID.

        :param id: Image ID in Supervisely.
        :type id: int
        :param name: New Image name.
        :type name: str
        :return: Information about updated Image.
        :rtype: :class:`ImageInfo`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            image_id = 376729
            new_image_name = 'new_image_name.jpg'
            img_info = api.image.rename(image_id, new_image_name)
        """
        return self.edit(id=id, name=name)

    def add_tag(self, image_id: int, tag_id: int, value: Optional[Union[str, int]] = None) -> None:
        """
        Add tag with given ID to Image by ID.

        :param image_id: Image ID in Supervisely.
        :type image_id: int
        :param tag_id: Tag ID in Supervisely.
        :type tag_id: int
        :param value: Tag value.
        :type value: int or str or None, optional
        :return: :class:`None<None>`
        :rtype: :class:`NoneType<NoneType>`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            image_id = 2389126
            tag_id = 277083
            api.image.add_tag(image_id, tag_id)
        """

        self.add_tag_batch([image_id], tag_id, value)

    def add_tag_batch(
        self,
        image_ids: List[int],
        tag_id: int,
        value: Optional[Union[str, int]] = None,
        progress_cb: Optional[Union[tqdm, Callable]] = None,
        batch_size: Optional[int] = 100,
        tag_meta: Optional[TagMeta] = None,
    ) -> None:
        """
        Add tag with given ID to Images by IDs.

        :param image_ids: List of Images IDs in Supervisely.
        :type image_ids: List[int]
        :param tag_id: Tag ID in Supervisely.
        :type tag_id: int
        :param value: Tag value.
        :type value: int or str or None, optional
        :param progress_cb: Function for tracking progress of adding tag.
        :type progress_cb: tqdm or callable, optional
        :param batch_size: Batch size
        :type batch_size: int, optional
        :param tag_meta: Tag Meta. Needed for value validation, omit to skip validation
        :type tag_meta: TagMeta, optional
        :return: :class:`None<None>`
        :rtype: :class:`NoneType<NoneType>`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            image_ids = [2389126, 2389127]
            tag_id = 277083
            api.image.add_tag_batch(image_ids, tag_id)
        """
        if tag_meta:
            if not (tag_meta.sly_id == tag_id):
                raise ValueError("tag_meta.sly_id and tag_id should be same")
            if not tag_meta.is_valid_value(value):
                raise ValueError("Tag {} can not have value {}".format(tag_meta.name, value))
        else:
            # No value validation
            pass

        for batch_ids in batched(image_ids, batch_size):
            data = {ApiField.TAG_ID: tag_id, ApiField.IDS: batch_ids}
            if value is not None:
                data[ApiField.VALUE] = value
            self._api.post("image-tags.bulk.add-to-image", data)
            if progress_cb is not None:
                progress_cb(len(batch_ids))

    def add_tags_batch(
        self,
        image_ids: List[int],
        tag_ids: Union[int, List[int]],
        values: Optional[Union[str, int, List[Union[str, int, None]]]] = None,
        log_progress: bool = False,
        batch_size: Optional[int] = 100,
        tag_metas: Optional[Union[TagMeta, List[TagMeta]]] = None,
    ) -> List[int]:
        """
        Add tag with given ID to Images by IDs with different values.

        :param image_ids: List of Images IDs in Supervisely.
        :type image_ids: List[int]
        :param tag_ids: Tag IDs in Supervisely.
        :type tag_ids: int or List[int]
        :param values: List of tag values for each image or single value for all images.
        :type values: List[str] or List[int] or str or int, optional
        :param log_progress: If True, will log progress.
        :type log_progress: bool, optional
        :param batch_size: Batch size
        :type batch_size: int, optional
        :param tag_metas: Tag Metas. Needed for values validation, omit to skip validation
        :type tag_metas: TagMeta or List[TagMeta], optional
        :return: List of tags IDs.
        :rtype: List[int]
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()
            image_ids = [2389126, 2389127]
            tag_ids = 277083
            values = ['value1', 'value2']
            api.image.add_tags_batch(image_ids, tag_ids, values)
        """
        if len(image_ids) == 0:
            return []

        if isinstance(tag_ids, int):
            tag_ids = [tag_ids] * len(image_ids)

        if isinstance(tag_metas, TagMeta):
            tag_metas = [tag_metas] * len(image_ids)

        if values is None:
            values = [None] * len(image_ids)
        elif isinstance(values, (str, int)):
            values = [values] * len(image_ids)

        if len(values) != len(image_ids):
            raise ValueError("Length of image_ids and values should be the same")

        if len(tag_ids) != len(image_ids):
            raise ValueError("Length of image_ids and tag_ids should be the same")

        if tag_metas and len(tag_metas) != len(image_ids):
            raise ValueError("Length of image_ids and tag_metas should be the same")

        if tag_metas:
            for tag_meta, tag_id, value in zip(tag_metas, tag_ids, values):
                if not (tag_meta.sly_id == tag_id):
                    raise ValueError(f"{tag_meta.name = } and {tag_id = } should be same")
                if not tag_meta.is_valid_value(value):
                    raise ValueError(f"{tag_meta.name = } can not have value {value = }")

        project_id = self.get_project_id(image_ids[0])
        data = [
            {ApiField.ENTITY_ID: image_id, ApiField.TAG_ID: tag_id, ApiField.VALUE: value}
            for image_id, tag_id, value in zip(image_ids, tag_ids, values)
        ]

        return self.tag.add_to_entities_json(project_id, data, batch_size, log_progress)

    def update_tag_value(self, tag_id: int, value: Union[str, float]) -> Dict:
        """
        Update tag value with given ID.

        :param tag_id: Tag ID in Supervisely.
        :type value: int
        :param value: Tag value.
        :type value: str or float
        :param project_meta: Project Meta.
        :type project_meta: ProjectMeta
        :return: Information about updated tag.
        :rtype: :class:`dict`
        :Usage example:

            .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            tag_id = 277083
            new_value = 'new_value'
            api.image.update_tag_value(tag_id, new_value)

        """
        data = {ApiField.ID: tag_id, ApiField.VALUE: value}
        response = self._api.post("image-tags.update-tag-value", data)
        return response.json()

    def remove_batch(
        self,
        ids: List[int],
        progress_cb: Optional[Union[tqdm, Callable]] = None,
        batch_size: Optional[int] = 50,
    ):
        """
        Remove images from supervisely by IDs.
        IDs must belong to the same project.

        :param ids: List of Images IDs in Supervisely.
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

            image_ids = [2389126, 2389127]
            api.image.remove_batch(image_ids)
        """
        super(ImageApi, self).remove_batch(ids, progress_cb=progress_cb, batch_size=batch_size)

    def remove(self, image_id: int):
        """
        Remove image from supervisely by id.
        All image IDs must belong to the same dataset.
        Therefore, it is necessary to sort IDs before calling this method.

        :param image_id: Images ID in Supervisely.
        :type image_id: int
        :return: :class:`None<None>`
        :rtype: :class:`NoneType<NoneType>`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            image_id = 2389126
            api.image.remove(image_id)
        """
        super(ImageApi, self).remove(image_id)

    def exists(self, parent_id: int, name: str) -> bool:
        """Check if image with given name exists in dataset with given id.

        :param parent_id: Dataset ID in Supervisely.
        :type parent_id: int
        :param name: Image name in Supervisely.
        :type name: str
        :return: True if image exists, False otherwise.
        :rtype: bool"""
        return self.get_info_by_name(parent_id, name, force_metadata_for_links=False) is not None

    def upload_multispectral(
        self,
        dataset_id: int,
        image_name: str,
        channels: Optional[List[np.ndarray]] = None,
        rgb_images: Optional[List[str]] = None,
        progress_cb: Optional[Union[tqdm, Callable]] = None,
    ) -> List[ImageInfo]:
        """Uploads multispectral image to Supervisely, if channels are provided, they will
        be uploaded as separate images. If rgb_images are provided, they will be uploaded without
        splitting into channels as RGB images.

        :param dataset_id: dataset ID to upload images to
        :type dataset_id: int
        :param image_name: name of the image with extension.
        :type image_name: str
        :param channels: list of numpy arrays with image channels
        :type channels: List[np.ndarray], optional
        :param rgb_images: list of paths to RGB images which will be uploaded as is
        :type rgb_images: List[str], optional
        :param progress_cb: function for tracking upload progress
        :type progress_cb: tqdm or callable, optional
        :return: list of uploaded images infos
        :rtype: List[ImageInfo]
        :Usage example:

         .. code-block:: python

            import os
            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'

            # Load secrets and create API object from .env file (recommended)
            # Learn more here: https://developer.supervisely.com/getting-started/basics-of-authentication
            load_dotenv(os.path.expanduser("~/supervisely.env"))

            api = sly.Api.from_env()

            image_name = "demo1.png"
            image = cv2.imread(f"demo_data/{image_name}")

            # Extract channels as 2d numpy arrays: channels = [a, b, c]
            channels = [image[:, :, i] for i in range(image.shape[2])]

            image_infos = api.image.upload_multispectral(api, dataset.id, image_name, channels)
        """
        group_tag_meta = TagMeta(_MULTISPECTRAL_TAG_NAME, TagValueType.ANY_STRING)
        group_tag = Tag(meta=group_tag_meta, value=image_name)
        image_basename = get_file_name(image_name)

        nps_for_upload = []
        if rgb_images is not None:
            for rgb_image in rgb_images:
                nps_for_upload.append(sly_image.read(rgb_image))

        if channels is not None:
            for channel in channels:
                nps_for_upload.append(channel)

        anns = []
        names = []

        for i, np_for_upload in enumerate(nps_for_upload):
            anns.append(Annotation(np_for_upload.shape).add_tag(group_tag))
            names.append(f"{image_basename}_{i}.png")

        image_infos = self.upload_nps(dataset_id, names, nps_for_upload, progress_cb=progress_cb)
        image_ids = [image_info.id for image_info in image_infos]

        self._api.annotation.upload_anns(image_ids, anns)

        return image_infos

    def upload_multiview_images(
        self,
        dataset_id: int,
        group_name: str,
        paths: Optional[List[str]] = None,
        metas: Optional[List[Dict]] = None,
        progress_cb: Optional[Union[tqdm, Callable]] = None,
        links: Optional[List[str]] = None,
        conflict_resolution: Optional[Literal["rename", "skip", "replace"]] = "rename",
        force_metadata_for_links: Optional[bool] = False,
    ) -> List[ImageInfo]:
        """
        Uploads images to Supervisely and adds a tag to them.
        At least one of `paths` or `links` must be provided.

        If you include `metas` during the upload, you can add a custom sort parameter for images.
        To achieve this, use the context manager :func:`api.image.add_custom_sort` with the desired key name from the meta dictionary to be used for sorting.
        Refer to the example section for more details.

        :param dataset_id: Dataset ID in Supervisely.
        :type dataset_id: int
        :param tag_name: Tag name in Supervisely.
                         If tag does not exist in project, create it first.
                         Tag must be of type ANY_STRING.
        :type tag_name: str
        :param group_name: Group name. All images will be assigned by tag with this group name.
        :type group_name: str
        :param paths: List of paths to images.
        :type paths: List[str]
        :param metas: Custom additional image infos that contain images technical and/or user-generated data as list of separate dicts.
        :type metas: Optional[List[Dict]]
        :param progress_cb: Function for tracking upload progress.
        :type progress_cb: Optional[Union[tqdm, Callable]]
        :param links: List of links to images.
        :type links: Optional[List[str]]
        :param conflict_resolution: The strategy to resolve upload conflicts.
            Options:
                - 'replace': Replaces the existing images in the dataset with the new ones if there is a conflict and logs the deletion of existing images.
                - 'skip': Ignores uploading the new images if there is a conflict; the original image's ImageInfo list will be returned instead.
                - 'rename': (default) Renames the new images to prevent name conflicts.
        :type conflict_resolution: Optional[Literal["rename", "skip", "replace"]]
        :param force_metadata_for_links: Specifies whether to force retrieving metadata for images from links.
                                         If False, metadata fields in the response can be empty (if metadata has not been retrieved yet).
        :type force_metadata_for_links: Optional[bool]
        :return: List of uploaded images infos
        :rtype: List[ImageInfo]
        :raises Exception: if tag does not exist in project or tag is not of type ANY_STRING

        :Usage example:

         .. code-block:: python

            import os
            from dotenv import load_dotenv

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'

            # Load secrets and create API object from .env file (recommended)
            # Learn more here: https://developer.supervisely.com/getting-started/basics-of-authentication
            load_dotenv(os.path.expanduser("~/supervisely.env"))

            api = sly.Api.from_env()

            dataset_id = 123456
            paths = ['path/to/audi_01.png', 'path/to/audi_02.png']
            group_name = 'audi'

            image_infos = api.image.upload_multiview_images(dataset_id, group_name, paths)

            # Add custom sort parameter for images
            metas = [{'my-key':'a'}, {'my-key':'b'}]
            with api.image.add_custom_sort(key="my-key"):
                image_infos = api.image.upload_multiview_images(dataset_id, group_name, paths, metas)
        """

        if paths is None and links is None:
            raise ValueError("At least one of 'paths' or 'links' must be provided.")

        group_tag_meta = TagMeta(_MULTIVIEW_TAG_NAME, TagValueType.ANY_STRING)
        group_tag = Tag(meta=group_tag_meta, value=group_name)

        image_infos = []
        if paths is not None:
            for path in paths:
                if get_file_ext(path).lower() not in sly_image.SUPPORTED_IMG_EXTS:
                    raise RuntimeError(
                        f"Image {path!r} has unsupported extension. Supported extensions: {sly_image.SUPPORTED_IMG_EXTS}"
                    )
            names = [get_file_name(path) for path in paths]
            image_infos_by_paths = self.upload_paths(
                dataset_id=dataset_id,
                names=names,
                paths=paths,
                progress_cb=progress_cb,
                metas=metas,
                conflict_resolution=conflict_resolution,
            )
            image_infos.extend(image_infos_by_paths)

        if links is not None:
            names = [get_file_name_with_ext(link) for link in links]
            image_infos_by_links = self.upload_links(
                dataset_id=dataset_id,
                names=names,
                links=links,
                progress_cb=progress_cb,
                conflict_resolution=conflict_resolution,
                force_metadata_for_links=force_metadata_for_links,
            )
            image_infos.extend(image_infos_by_links)

        anns = [Annotation((info.height, info.width)).add_tag(group_tag) for info in image_infos]
        image_ids = [image_info.id for image_info in image_infos]
        self._api.annotation.upload_anns(image_ids, anns)

        uploaded_image_infos = self.get_list(
            dataset_id,
            filters=[
                {
                    ApiField.FIELD: ApiField.ID,
                    ApiField.OPERATOR: "in",
                    ApiField.VALUE: image_ids,
                }
            ],
            force_metadata_for_links=force_metadata_for_links,
        )
        return uploaded_image_infos

    def group_images_for_multiview(
        self,
        image_ids: List[int],
        group_name: str,
        multiview_tag_name: Optional[str] = None,
    ) -> None:
        """
        Group images for multi-view by tag with given name. If tag does not exist in project, will create it first.

        Note:
            * All images must belong to the same project.
            * Tag must be of type ANY_STRING and applicable to images.
            * Recommended number of images in group is 6-12.

        :param image_ids: List of Images IDs in Supervisely.
        :type image_ids: List[int]
        :param group_name: Group name. Images will be assigned by group tag with this value.
        :type group_name: str
        :param multiview_tag_name: Multiview tag name in Supervisely.
                                If None, will use default 'multiview' tag name.
                                If tag does not exist in project, will create it first.
        :type multiview_tag_name: str, optional
        :return: :class:`None<None>`

        :rtype: :class:`NoneType<NoneType>`
        :raises ValueError: if tag is not of type ANY_STRING or not applicable to images

        :Usage example:

         .. code-block:: python

            # ? option 1
            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            BATCH_SIZE = 6
            image_ids = [2389126, 2389127, 2389128, 2389129, 2389130, 2389131, ...]

            # group images for multiview
            for group_name, ids in enumerate(sly.batched(image_ids, batch_size=BATCH_SIZE)):
                api.image.group_images_for_multiview(ids, group_name)


            # ? option 2 (with sly.ApiContext)
            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            BATCH_SIZE = 6
            image_ids = [2389126, 2389127, 2389128, 2389129, 2389130, 2389131, ...]
            project_id = 111111 # change to your project id


            # * make sure that `with_settings=True` is set to get project settings from server
            project_meta_json = api.project.get_meta(project_id, with_settings=True)
            project_meta = sly.ProjectMeta.from_json(project_meta_json)

            # create custom tag meta (optional)
            multiview_tag_name = 'cars'
            tag_meta = sly.TagMeta(multiview_tag_name, sly.TagValueType.ANY_STRING)
            project_meta = project_meta.add_tag_meta(tag_meta)
            project_meta = api.project.update_meta(project_id, project_meta) # update meta on server

            # group images for multiview
            with sly.ApiContext(api, project_id=project_id, project_meta=project_meta):
                for group_name, ids in enumerate(sly.batched(image_ids, batch_size=BATCH_SIZE)):
                    api.image.group_images_for_multiview(ids, group_name, multiview_tag_name)

        """

        # ============= Api Context ===================================
        # using context for optimization (avoiding extra API requests)
        context = self._api.optimization_context
        project_meta = context.get("project_meta")
        project_id = context.get("project_id")
        if project_id is None:
            project_id = self.get_project_id(image_ids[0])
            context["project_id"] = project_id
        if project_meta is None:
            project_meta = ProjectMeta.from_json(
                self._api.project.get_meta(project_id, with_settings=True)
            )
            context["project_meta"] = project_meta
        # =============================================================

        need_update_project_meta = False
        multiview_tag_name = multiview_tag_name or _MULTIVIEW_TAG_NAME
        multiview_tag_meta = project_meta.get_tag_meta(multiview_tag_name)

        if multiview_tag_meta is None:
            multiview_tag_meta = TagMeta(
                multiview_tag_name,
                TagValueType.ANY_STRING,
                applicable_to=TagApplicableTo.IMAGES_ONLY,
            )
            project_meta = project_meta.add_tag_meta(multiview_tag_meta)
            need_update_project_meta = True
        elif multiview_tag_meta.sly_id is None:
            logger.warning(f"`sly_id` is None for group tag, trying to get it from server")
            need_update_project_meta = True

        if multiview_tag_meta.value_type != TagValueType.ANY_STRING:
            raise ValueError(f"Tag '{multiview_tag_name}' is not of type ANY_STRING.")
        elif multiview_tag_meta.applicable_to == TagApplicableTo.OBJECTS_ONLY:
            raise ValueError(f"Tag '{multiview_tag_name}' is not applicable to images.")

        if need_update_project_meta:
            project_meta = self._api.project.update_meta(id=project_id, meta=project_meta)
            context["project_meta"] = project_meta
            multiview_tag_meta = project_meta.get_tag_meta(multiview_tag_name)

        if not project_meta.project_settings.multiview_enabled:
            if multiview_tag_name == _MULTIVIEW_TAG_NAME:
                self._api.project.set_multiview_settings(project_id)
            else:
                self._api.project._set_custom_grouping_settings(
                    id=project_id,
                    group_images=True,
                    tag_name=multiview_tag_name,
                    sync=False,
                )
            project_meta = ProjectMeta.from_json(
                self._api.project.get_meta(project_id, with_settings=True)
            )
            context["project_meta"] = project_meta

        self.add_tag_batch(image_ids, multiview_tag_meta.sly_id, group_name)

    def upload_medical_images(
        self,
        dataset_id: int,
        paths: List[str],
        group_tag_name: Optional[str] = None,
        metas: Optional[List[Dict]] = None,
        progress_cb: Optional[Union[tqdm, Callable]] = None,
    ) -> List[ImageInfo]:
        """
        Upload medical 2D images (DICOM) to Supervisely and group them by specified or default tag.

        If you include `metas` during the upload, you can add a custom sort parameter for images.
        To achieve this, use the context manager :func:`api.image.add_custom_sort` with the desired key name from the meta dictionary to be used for sorting.
        Refer to the example section for more details.

        :param dataset_id: Dataset ID in Supervisely.
        :type dataset_id: int
        :param paths: List of paths to images.
        :type paths: List[str]
        :param group_tag_name: Group name. All images will be assigned by tag with this group name. If `group_tag_name` is None, the images will be grouped by one of the default tags.
        :type group_tag_name: str, optional
        :param metas: Custom additional image infos that contain images technical and/or user-generated data as list of separate dicts.
        :type metas: List[Dict], optional
        :param progress_cb: Function for tracking upload progress.
        :type progress_cb: tqdm or callable, optional

        :return: List of uploaded images infos.
        :rtype: List[ImageInfo]

        :raises Exception: If tag does not exist in project or tag is not of type ANY_STRING
        :raises Exception: If length of `metas` is not equal to the length of `paths`.

        :Usage example:

         .. code-block:: python

            import os
            from dotenv import load_dotenv
            from tqdm import tqdm

            import supervisely as sly

            # Load secrets and create API object from .env file (recommended)
            # Learn more here: https://developer.supervisely.com/getting-started/basics-of-authentication
            if sly.is_development():
               load_dotenv(os.path.expanduser("~/supervisely.env"))

            api = sly.Api.from_env()

            dataset_id = 123456
            paths = ['path/to/medical_01.dcm', 'path/to/medical_02.dcm']
            metas = [{'meta':'01'}, {'meta':'02'}]
            group_tag_name = 'StudyInstanceUID'

            pbar = tqdm(desc="Uploading images", total=len(paths))
            image_infos = api.image.upload_medical_images(dataset_id, paths, group_tag_name, metas)

            # Add custom sort parameter for images
            metas = [{'my-key':'a'}, {'my-key':'b'}]
            with api.image.add_custom_sort(key="my-key"):
                image_infos = api.image.upload_medical_images(dataset_id, paths, group_tag_name, metas)
        """

        if metas is None:
            _metas = [dict() for _ in paths]
        else:
            if len(metas) != len(paths):
                raise ValueError("Length of 'metas' is not equal to the length of 'paths'.")
            _metas = metas.copy()

        dataset = self._api.dataset.get_info_by_id(dataset_id, raise_error=True)
        meta_json = self._api.project.get_meta(dataset.project_id)
        project_meta = ProjectMeta.from_json(meta_json)

        import nrrd

        from supervisely.convert.image.medical2d.medical2d_helper import (
            convert_dcm_to_nrrd,
        )

        image_paths = []
        image_names = []
        anns = []

        converted_dir_name = uuid4().hex
        converted_dir = Path("/tmp/") / converted_dir_name

        group_tag_counter = defaultdict(int)
        for path, image_meta in zip(paths, _metas):
            try:
                nrrd_paths, nrrd_names, group_tags, dcm_meta = convert_dcm_to_nrrd(
                    image_path=path,
                    converted_dir=converted_dir.as_posix(),
                    group_tag_name=group_tag_name,
                )
                image_meta.update(dcm_meta)  # TODO: check update order
                image_paths.extend(nrrd_paths)
                image_names.extend(nrrd_names)

                for nrrd_path in nrrd_paths:
                    tags = []
                    for tag in group_tags:
                        tag_meta = project_meta.get_tag_meta(tag["name"])
                        if tag_meta is None:
                            tag_meta = TagMeta(
                                tag["name"],
                                TagValueType.ANY_STRING,
                                applicable_to=TagApplicableTo.IMAGES_ONLY,
                            )
                            project_meta = project_meta.add_tag_meta(tag_meta)
                        elif tag_meta.value_type != TagValueType.ANY_STRING:
                            raise ValueError(f"Tag '{tag['name']}' is not of type ANY_STRING.")
                        tag = Tag(meta=tag_meta, value=tag["value"])
                        tags.append(tag)
                    img_size = nrrd.read_header(nrrd_path)["sizes"].tolist()[::-1]
                    ann = Annotation(img_size=img_size, img_tags=tags)
                    anns.append(ann)

                for tag in group_tags:
                    group_tag_counter[tag["name"]] += 1
            except Exception as e:
                logger.warning(f"File '{path}' will be skipped due to: {str(e)}")
                continue

        image_infos = self.upload_paths(
            dataset_id=dataset_id,
            names=image_names,
            paths=image_paths,
            progress_cb=progress_cb,
            metas=_metas,
        )
        image_ids = [image_info.id for image_info in image_infos]

        # Update the project metadata and enable image grouping
        self._api.project.update_meta(id=dataset.project_id, meta=project_meta.to_json())
        if len(group_tag_counter) > 0:
            max_used_tag_name = max(group_tag_counter, key=group_tag_counter.get)
            if group_tag_name not in group_tag_counter:
                group_tag_name = max_used_tag_name
            self._api.project.images_grouping(
                id=dataset.project_id, enable=True, tag_name=group_tag_name
            )
        self._api.annotation.upload_anns(image_ids, anns)
        clean_dir(converted_dir.as_posix())

        return image_infos

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

        images_in_dataset = self.get_list(dataset_id, force_metadata_for_links=False)
        used_names = {image_info.name for image_info in images_in_dataset}
        new_names = [
            generate_free_name(used_names, name, with_ext=True, extend_used_names=True)
            for name in names
        ]
        return new_names

    def raise_name_intersections_if_exist(
        self, dataset_id: int, names: List[str], message: str = None
    ):
        """
        Raises error if images with given names already exist in dataset.
        Default error message:
        "Images with the following names already exist in dataset [ID={dataset_id}]: {name_intersections}.
        Please, rename images and try again or set change_name_if_conflict=True to rename automatically on upload."

        :param dataset_id: Dataset ID in Supervisely.
        :type dataset_id: int
        :param names: List of names to check.
        :type names: List[str]
        :param message: Error message.
        :type message: str, optional
        :return: None
        :rtype: None
        """
        images_in_dataset = self.get_list(dataset_id)
        used_names = {image_info.name for image_info in images_in_dataset}
        name_intersections = used_names.intersection(set(names))
        if message is None:
            message = f"Images with the following names already exist in dataset [ID={dataset_id}]: {name_intersections}. Please, rename images and try again or set change_name_if_conflict=True to rename automatically on upload."
        if len(name_intersections) > 0:
            raise ValueError(f"{message}")

    def upload_dir(
        self,
        dataset_id: int,
        dir_path: str,
        recursive: Optional[bool] = True,
        change_name_if_conflict: Optional[bool] = True,
        progress_cb: Optional[Union[tqdm, Callable]] = None,
    ) -> List[ImageInfo]:
        """
        Uploads all images with supported extensions from given directory to Supervisely.
        Optionally, uploads images from subdirectories of given directory.

        :param dataset_id: Dataset ID in Supervisely.
        :type dataset_id: int
        :param dir_path: Path to directory with images.
        :type dir_path: str
        :param recursive: If True uploads images from subdirectories of given directory recursively, otherwise only images from given directory.
        :type recursive: bool, optional
        :param change_name_if_conflict: If True adds suffix to the end of Image name when Dataset already contains an Image with identical name, If False and images with the identical names already exist in Dataset raises error.
        :type change_name_if_conflict: bool, optional
        :param progress_cb: Function for tracking upload progress.
        :type progress_cb: Optional[Union[tqdm, Callable]]
        :return: List of uploaded images infos
        :rtype: List[ImageInfo]
        """

        if recursive:
            paths = list_files_recursively(dir_path, filter_fn=sly_image.is_valid_format)
        else:
            paths = list_files(dir_path, filter_fn=sly_image.is_valid_format)

        names = [get_file_name_with_ext(path) for path in paths]

        if change_name_if_conflict is True:
            names = self.get_free_names(dataset_id, names)
        else:
            self.raise_name_intersections_if_exist(dataset_id, names)

        image_infos = self.upload_paths(dataset_id, names, paths, progress_cb=progress_cb)
        return image_infos

    def upload_dirs(
        self,
        dataset_id: int,
        dir_paths: List[str],
        recursive: Optional[bool] = True,
        change_name_if_conflict: Optional[bool] = True,
        progress_cb: Optional[Union[tqdm, Callable]] = None,
    ) -> List[ImageInfo]:
        """
        Uploads all images with supported extensions from given directories to Supervisely.
        Optionally, uploads images from subdirectories of given directories.

        :param dataset_id: Dataset ID in Supervisely.
        :type dataset_id: int
        :param dir_paths: List of paths to directories with images.
        :type dir_paths: List[str]
        :param recursive: If True uploads images from subdirectories of given directories recursively, otherwise only images from given directories.
        :type recursive: bool, optional
        :param change_name_if_conflict: If True adds suffix to the end of Image name when Dataset already contains an Image with identical name, If False and images with the identical names already exist in Dataset raises error.
        :type change_name_if_conflict: bool, optional
        :param progress_cb: Function for tracking upload progress.
        :type progress_cb: Optional[Union[tqdm, Callable]]
        :return: List of uploaded images infos
        :rtype: List[ImageInfo]
        """

        image_infos = []
        for dir_path in dir_paths:
            image_infos.extend(
                self.upload_dir(
                    dataset_id,
                    dir_path,
                    recursive,
                    change_name_if_conflict,
                    progress_cb,
                )
            )
        return image_infos

    def _validate_project_and_dataset_id(self, project_id: int, dataset_id: int) -> None:
        """
        Check if only one of 'project_id' and 'dataset_id' is provided.

        :param project_id: Project ID in Supervisely.
        :type project_id: int
        :param dataset_id: Dataset ID in Supervisely.
        :type dataset_id: int
        :raises: :class:`ValueError` if both 'project_id' and 'dataset_id' are provided or none of them are provided.
        :return: None
        :rtype: None

        """
        if project_id is None and dataset_id is None:
            raise ValueError("One of 'project_id' or 'dataset_id' should be provided.")

        if project_id is not None and dataset_id is not None:
            raise ValueError("Only one of 'project_id' and 'dataset_id' should be provided.")

    def set_remote(self, images: List[int], links: List[str]):
        """
        Updates the source of existing images by setting new remote links.
        This method is used when an image was initially uploaded as a file or added via a link,
        but later it was decided to change its location (e.g., moved to another storage or re-uploaded elsewhere).
        By updating the link, the image source can be redirected to the new location.

        :param images: List of image ids.
        :type images: List[int]
        :param links: List of new remote links.
        :type links: List[str]
        :return: json-encoded content of a response.

        :Usage example:

            .. code-block:: python

                import supervisely as sly

                api = sly.Api.from_env()

                images = [123, 124, 125]
                links = [
                    "s3://bucket/lemons/ds1/img/IMG_444.jpeg",
                    "s3://bucket/lemons/ds1/img/IMG_445.jpeg",
                    "s3://bucket/lemons/ds1/img/IMG_446.jpeg",
                ]
                result = api.image.set_remote(images, links)
        """

        if len(images) == 0:
            raise ValueError("List of images can not be empty.")

        if len(images) != len(links):
            raise ValueError("Length of 'images' and 'links' should be equal.")

        images_list = []
        for img, lnk in zip(images, links):
            images_list.append({ApiField.ID: img, ApiField.LINK: lnk})

        data = {ApiField.IMAGES: images_list, ApiField.CLEAR_LOCAL_DATA_SOURCE: True}
        r = self._api.post("images.update.links", data)
        return r.json()

    def set_custom_sort(
        self,
        id: int,
        sort_value: str,
    ) -> Dict[str, Any]:
        """
        Sets custom sort value for image with given ID.

        :param id: Image ID in Supervisely.
        :type id: int
        :param sort_value: Sort value.
        :type sort_value: str
        :return: json-encoded content of a response.
        :rtype: Dict[str, Any]
        """
        return self.set_custom_sort_bulk([id], [sort_value])

    def set_custom_sort_bulk(
        self,
        ids: List[int],
        sort_values: List[str],
    ) -> Dict[str, Any]:
        """
        Sets custom sort values for images with given IDs.

        :param ids: Image IDs in Supervisely.
        :type ids: List[int]
        :param sort_values: List of custom sort values that will be set for images. It is stored as a key `customSort` value in the image `meta`.
        :type sort_values: List[str]
        :return: json-encoded content of a response.
        :rtype: Dict[str, Any]
        """
        if len(ids) != len(sort_values):
            raise ValueError(
                f"Length of 'ids' and 'sort_values' is not equal, {len(ids)} != {len(sort_values)}."
            )
        data = {
            ApiField.IMAGES: [
                {ApiField.ID: id, ApiField.CUSTOM_SORT: sort_value}
                for id, sort_value in zip(ids, sort_values)
            ]
        }
        response = self._api.post("images.bulk.set-custom-sort", data)
        return response.json()

    async def _download_async(
        self,
        id: int,
        is_stream: bool = False,
        range_start: Optional[int] = None,
        range_end: Optional[int] = None,
        headers: Optional[dict] = None,
        chunk_size: Optional[int] = None,
    ) -> AsyncGenerator:
        """
        Download Image with given ID asynchronously.
        If is_stream is True, returns stream of bytes, otherwise returns response object.
        For streaming, returns tuple of chunk and hash. Chunk size is 8 MB by default.

        :param id: Image ID in Supervisely.
        :type id: int
        :param is_stream: If True, returns stream of bytes, otherwise returns response object.
        :type is_stream: bool, optional
        :param range_start: Start byte of range for partial download.
        :type range_start: int, optional
        :param range_end: End byte of range for partial download.
        :type range_end: int, optional
        :param headers: Headers for request.
        :type headers: dict, optional
        :param chunk_size: Size of chunk for streaming. Default is 8 MB.
        :type chunk_size: int, optional
        :return: Stream of bytes or response object.
        :rtype: AsyncGenerator
        """
        api_method_name = "images.download"

        if chunk_size is None:
            chunk_size = 8 * 1024 * 1024

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

    async def download_np_async(
        self,
        id: int,
        semaphore: Optional[asyncio.Semaphore] = None,
        keep_alpha: Optional[bool] = False,
        progress_cb: Optional[Union[tqdm, Callable]] = None,
        progress_cb_type: Literal["number", "size"] = "number",
    ) -> np.ndarray:
        """
        Downloads Image with given ID in NumPy format asynchronously.

        :param id: Image ID in Supervisely.
        :type id: int
        :param semaphore: Semaphore for limiting the number of simultaneous downloads.
        :type semaphore: :class:`asyncio.Semaphore`, optional
        :param keep_alpha: If True keeps alpha mask for image, otherwise don't.
        :type keep_alpha: bool, optional
        :param progress_cb: Function for tracking download progress.
        :type progress_cb: tqdm or callable, optional
        :param progress_cb_type: Type of progress callback. Can be "number" or "size". Default is "number".
        :type progress_cb_type: Literal["number", "size"], optional
        :return: Image in RGB numpy matrix format
        :rtype: :class:`np.ndarray`

        :Usage example:

         .. code-block:: python

            import supervisely as sly
            import asyncio
            from tqdm.asyncio import tqdm

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            DATASET_ID = 98357
            semaphore = asyncio.Semaphore(100)
            images = api.image.get_list(DATASET_ID)
            tasks = []
            pbar = tqdm(total=len(images), desc="Downloading images", unit="image")
            for image in images:
                task = api.image.download_np_async(image.id, semaphore, progress_cb=pbar)
                tasks.append(task)
            results = await asyncio.gather(*tasks)
        """
        if semaphore is None:
            semaphore = self._api.get_default_semaphore()

        async with semaphore:
            async for response in self._download_async(id):
                img = sly_image.read_bytes(response.content, keep_alpha)
                if progress_cb is not None:
                    if progress_cb_type == "number":
                        progress_cb(1)
                    elif progress_cb_type == "size":
                        progress_cb(len(response.content))
            return img

    async def download_nps_async(
        self,
        ids: List[int],
        semaphore: Optional[asyncio.Semaphore] = None,
        keep_alpha: Optional[bool] = False,
        progress_cb: Optional[Union[tqdm, Callable]] = None,
        progress_cb_type: Literal["number", "size"] = "number",
    ) -> List[np.ndarray]:
        """
        Downloads Images with given IDs in NumPy format asynchronously.

        :param ids: List of Image IDs in Supervisely.
        :type ids: :class:`List[int]`
        :param semaphore: Semaphore for limiting the number of simultaneous downloads.
        :type semaphore: :class:`asyncio.Semaphore`, optional
        :param keep_alpha: If True keeps alpha mask for images, otherwise don't.
        :type keep_alpha: bool, optional
        :param progress_cb: Function for tracking download progress.
        :type progress_cb: tqdm or callable, optional
        :param progress_cb_type: Type of progress callback. Can be "number" or "size". Default is "number".
        :type progress_cb_type: Literal["number", "size"], optional
        :return: List of Images in RGB numpy matrix format
        :rtype: :class:`List[np.ndarray]`

        :Usage example:

            .. code-block:: python

                import supervisely as sly
                import asyncio

                os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
                os.environ['API_TOKEN'] = 'Your Supervisely API Token'
                api = sly.Api.from_env()

                DATASET_ID = 98357
                semaphore = asyncio.Semaphore(100)
                images = api.image.get_list(DATASET_ID)
                img_ids = [image.id for image in images]
                loop = sly.utils.get_or_create_event_loop()
                results = loop.run_until_complete(
                                api.image.download_nps_async(img_ids, semaphore)
                            )

        """
        if semaphore is None:
            semaphore = self._api.get_default_semaphore()
        tasks = [
            self.download_np_async(id, semaphore, keep_alpha, progress_cb, progress_cb_type)
            for id in ids
        ]
        return await asyncio.gather(*tasks)

    async def download_path_async(
        self,
        id: int,
        path: str,
        semaphore: Optional[asyncio.Semaphore] = None,
        range_start: Optional[int] = None,
        range_end: Optional[int] = None,
        headers: Optional[dict] = None,
        check_hash: bool = True,
        progress_cb: Optional[Union[tqdm, Callable]] = None,
        progress_cb_type: Literal["number", "size"] = "number",
    ) -> None:
        """
        Downloads Image with given ID to local path.

        :param id: Image ID in Supervisely.
        :type id: int
        :param path: Local save path for Image.
        :type path: str
        :param semaphore: Semaphore for limiting the number of simultaneous downloads.
        :type semaphore: :class:`asyncio.Semaphore`, optional
        :param range_start: Start byte of range for partial download.
        :type range_start: int, optional
        :param range_end: End byte of range for partial download.
        :type range_end: int, optional
        :param headers: Headers for request.
        :type headers: dict, optional
        :param check_hash: If True, checks hash of downloaded file.
                        Check is not supported for partial downloads.
                        When range is set, hash check is disabled.
        :type check_hash: bool, optional
        :param progress_cb: Function for tracking download progress.
        :type progress_cb: tqdm or callable, optional
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

            img_info = api.image.get_info_by_id(770918)
            save_path = os.path.join("/path/to/save/", img_info.name)

            semaphore = asyncio.Semaphore(100)
            loop = sly.utils.get_or_create_event_loop()
            loop.run_until_complete(
                    api.image.download_path_async(img_info.id, save_path, semaphore)
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
                ):
                    await fd.write(chunk)
                    hash_to_check = hhash
                    if progress_cb is not None and progress_cb_type == "size":
                        progress_cb(len(chunk))
            if progress_cb is not None and progress_cb_type == "number":
                progress_cb(1)
            if check_hash:
                if hash_to_check is not None:
                    downloaded_file_hash = await get_file_hash_async(path)
                    if hash_to_check != downloaded_file_hash:
                        raise RuntimeError(
                            f"Downloaded hash of image with ID:{id} does not match the expected hash: {downloaded_file_hash} != {hash_to_check}"
                        )

    async def download_paths_async(
        self,
        ids: List[int],
        paths: List[str],
        semaphore: Optional[asyncio.Semaphore] = None,
        headers: Optional[dict] = None,
        check_hash: bool = True,
        progress_cb: Optional[Union[tqdm, Callable]] = None,
        progress_cb_type: Literal["number", "size"] = "number",
    ) -> None:
        """
        Download Images with given IDs and saves them to given local paths asynchronously.

        :param ids: List of Image IDs in Supervisely.
        :type ids: :class:`List[int]`
        :param paths: Local save paths for Images.
        :type paths: :class:`List[str]`
        :param semaphore: Semaphore for limiting the number of simultaneous downloads.
        :type semaphore: :class:`asyncio.Semaphore`, optional
        :param headers: Headers for request.
        :type headers: dict, optional
        :param check_hash: If True, checks hash of downloaded images.
        :type check_hash: bool, optional
        :param progress_cb: Function for tracking download progress.
        :type progress_cb: tqdm or callable, optional
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

            ids = [770918, 770919]
            paths = ["/path/to/save/image1.png", "/path/to/save/image2.png"]
            loop = sly.utils.get_or_create_event_loop()
            loop.run_until_complete(api.image.download_paths_async(ids, paths))
        """
        if len(ids) == 0:
            return
        if len(ids) != len(paths):
            raise ValueError(
                f'Length of "ids" and "paths" should be equal. {len(ids)} != {len(paths)}'
            )
        if semaphore is None:
            semaphore = self._api.get_default_semaphore()
        tasks = []
        for img_id, img_path in zip(ids, paths):
            task = self.download_path_async(
                img_id,
                img_path,
                semaphore,
                headers=headers,
                check_hash=check_hash,
                progress_cb=progress_cb,
                progress_cb_type=progress_cb_type,
            )
            tasks.append(task)
        await asyncio.gather(*tasks)

    async def download_bytes_single_async(
        self,
        id: int,
        semaphore: Optional[asyncio.Semaphore] = None,
        range_start: Optional[int] = None,
        range_end: Optional[int] = None,
        headers: Optional[dict] = None,
        check_hash: bool = True,
        progress_cb: Optional[Union[tqdm, Callable]] = None,
        progress_cb_type: Literal["number", "size"] = "number",
    ) -> bytes:
        """
        Downloads Image bytes with given ID.

        :param id: Image ID in Supervisely.
        :type id: int
        :param semaphore: Semaphore for limiting the number of simultaneous downloads.
        :type semaphore: :class:`asyncio.Semaphore`, optional
        :param range_start: Start byte of range for partial download.
        :type range_start: int, optional
        :param range_end: End byte of range for partial download.
        :type range_end: int, optional
        :param headers: Headers for request.
        :type headers: dict, optional
        :param check_hash: If True, checks hash of downloaded bytes.
                        Check is not supported for partial downloads.
                        When range is set, hash check is disabled.
        :type check_hash: bool, optional
        :param progress_cb: Function for tracking download progress.
        :type progress_cb: Optional[Union[tqdm, Callable]]
        :param progress_cb_type: Type of progress callback. Can be "number" or "size". Default is "number".
        :type progress_cb_type: Literal["number", "size"], optional
        :return: Bytes of downloaded image.
        :rtype: :class:`bytes`
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            import asyncio

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            img_id = 770918
            loop = sly.utils.get_or_create_event_loop()
            img_bytes = loop.run_until_complete(api.image.download_bytes_async(img_id))

        """
        if range_start is not None or range_end is not None:
            check_hash = False  # hash check is not supported for partial downloads
            headers = headers or {}
            headers["Range"] = f"bytes={range_start or ''}-{range_end or ''}"
            logger.debug(f"Image ID: {id}. Setting Range header: {headers['Range']}")

        hash_to_check = None

        if semaphore is None:
            semaphore = self._api.get_default_semaphore()
        async with semaphore:
            content = b""
            async for chunk, hhash in self._download_async(
                id,
                is_stream=True,
                headers=headers,
                range_start=range_start,
                range_end=range_end,
            ):
                content += chunk
                hash_to_check = hhash
                if progress_cb is not None and progress_cb_type == "size":
                    progress_cb(len(chunk))
            if check_hash:
                if hash_to_check is not None:
                    downloaded_bytes_hash = get_bytes_hash(content)
                    if hash_to_check != downloaded_bytes_hash:
                        raise RuntimeError(
                            f"Downloaded hash of image with ID:{id} does not match the expected hash: {downloaded_bytes_hash} != {hash_to_check}"
                        )
            if progress_cb is not None and progress_cb_type == "number":
                progress_cb(1)
            return content

    async def download_bytes_many_async(
        self,
        ids: List[int],
        semaphore: Optional[asyncio.Semaphore] = None,
        headers: Optional[dict] = None,
        check_hash: bool = True,
        progress_cb: Optional[Union[tqdm, Callable]] = None,
        progress_cb_type: Literal["number", "size"] = "number",
    ) -> List[bytes]:
        """
        Downloads Images bytes with given IDs asynchronously
        and returns reults in the same order as in the input list.

        :param ids: List of Image IDs in Supervisely.
        :type ids: :class:`List[int]`
        :param semaphore: Semaphore for limiting the number of simultaneous downloads.
        :type semaphore: :class:`asyncio.Semaphore`, optional
        :param headers: Headers for every request.
        :type headers: dict, optional
        :param check_hash: If True, checks hash of downloaded images.
        :type check_hash: bool, optional
        :param progress_cb: Function for tracking download progress.
        :type progress_cb: Optional[Union[tqdm, Callable]]
        :param progress_cb_type: Type of progress callback. Can be "number" or "size". Default is "number".
        :type progress_cb_type: Literal["number", "size"], optional
        :return: List of bytes of downloaded images.
        :rtype: :class:`List[bytes]`

        :Usage example:

            .. code-block:: python

                import supervisely as sly
                import asyncio

                os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
                os.environ['API_TOKEN
                api = sly.Api.from_env()

                loop = sly.utils.get_or_create_event_loop()
                semaphore = asyncio.Semaphore(100)
                img_bytes_list = loop.run_until_complete(api.image.download_bytes_imgs_async(ids, semaphore))
        """
        if semaphore is None:
            semaphore = self._api.get_default_semaphore()
        tasks = []
        for id in ids:
            task = self.download_bytes_single_async(
                id,
                semaphore,
                headers=headers,
                check_hash=check_hash,
                progress_cb=progress_cb,
                progress_cb_type=progress_cb_type,
            )
            tasks.append(task)
        results = await asyncio.gather(*tasks)
        return results

    async def download_bytes_generator_async(
        self,
        dataset_id: int,
        img_ids: List[int],
        semaphore: Optional[asyncio.Semaphore] = None,
        headers: Optional[dict] = None,
        check_hash: bool = False,
        progress_cb: Optional[Union[tqdm, Callable]] = None,
        progress_cb_type: Literal["number", "size"] = "number",
    ) -> AsyncGenerator[Tuple[int, bytes]]:
        """
        Downloads Image bytes with given ID in batch asynchronously.
        Yields tuple of Image ID and bytes of downloaded image.
        Uses bulk download API method.

        :param dataset_id: Dataset ID in Supervisely.
        :type dataset_id: int
        :param img_ids: List of Image IDs in Supervisely.
        :type img_ids: :class:`List[int]`
        :param semaphore: Semaphore for limiting the number of simultaneous downloads.
        :type semaphore: :class:`asyncio.Semaphore`, optional
        :param headers: Headers for request.
        :type headers: dict, optional
        :param check_hash: If True, checks hash of downloaded bytes. Default is False.
        :type check_hash: bool, optional
        :param progress_cb: Function for tracking download progress.
        :type progress_cb: Optional[Union[tqdm, Callable]]
        :param progress_cb_type: Type of progress callback. Can be "number" or "size". Default is "number".
        :type progress_cb_type: Literal["number", "size"], optional
        :return: Tuple of Image ID and bytes of downloaded image.
        :rtype: :class:`Tuple[int, bytes]`

        :Usage example:

         .. code-block:: python

            import supervisely as sly
            import asyncio

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            dataset_id = 123456
            img_ids = [770918, 770919, 770920, 770921, ... , 770992]
            tasks = []
            for batch in batched(img_ids, 50):
                task = api.image.download_bytes_batch_async(dataset_id, batch)
                tasks.append(task)
            results = await asyncio.gather(*tasks)
        """
        api_method_name = "images.bulk.download"
        json_body = {
            ApiField.DATASET_ID: dataset_id,
            ApiField.IMAGE_IDS: img_ids,
        }

        if semaphore is None:
            semaphore = self._api.get_default_semaphore()
        async with semaphore:
            response = await self._api.post_async(
                api_method_name,
                json=json_body,
                headers=headers,
            )
            decoder = MultipartDecoder.from_response(response)
            for part in decoder.parts:
                content_utf8 = part.headers[b"Content-Disposition"].decode("utf-8")
                # Find name="1245" preceded by a whitespace, semicolon or beginning of line.
                # The regex has 2 capture group: one for the prefix and one for the actual name value.
                img_id = int(re.findall(r'(^|[\s;])name="(\d*)"', content_utf8)[0][1])
                if check_hash:
                    hhash = part.headers.get("x-content-checksum-sha256", None)
                    if hhash is not None:
                        downloaded_bytes_hash = get_bytes_hash(part)
                        if hhash != downloaded_bytes_hash:
                            raise RuntimeError(
                                f"Downloaded hash of image with ID:{img_id} does not match the expected hash: {downloaded_bytes_hash} != {hhash}"
                            )
                if progress_cb is not None and progress_cb_type == "number":
                    progress_cb(1)
                elif progress_cb is not None and progress_cb_type == "size":
                    progress_cb(len(part.content))

                yield img_id, part.content

    async def get_list_generator_async(
        self,
        dataset_id: int = None,
        filters: Optional[List[Dict[str, str]]] = None,
        sort: Optional[str] = "id",
        sort_order: Optional[str] = "asc",
        force_metadata_for_links: Optional[bool] = True,
        only_labelled: Optional[bool] = False,
        fields: Optional[List[str]] = None,
        per_page: Optional[int] = None,
        semaphore: Optional[List[asyncio.Semaphore]] = None,
        **kwargs,
    ) -> AsyncGenerator[List[ImageInfo]]:
        """
        Yields list of images in dataset asynchronously page by page.

        :param dataset_id: Dataset ID in Supervisely.
        :type dataset_id: int
        :param filters: Filters for images.
        :type filters: List[Dict[str, str]], optional
        :param sort: Field name to sort. One of {'id' (default), 'name', 'description', 'labelsCount', 'createdAt', 'updatedAt', 'customSort'}.
        :type sort: str, optional
        :param sort_order: Sort order for images. One of {'asc' (default), 'desc'}
        :type sort_order: str, optional
        :param force_metadata_for_links: If True, forces metadata for links.
        :type force_metadata_for_links: bool, optional
        :param only_labelled: If True, returns only labelled images.
        :type only_labelled: bool, optional
        :param fields: List of fields to return.
        :type fields: List[str], optional
        :param per_page: Number of images to return per page.
        :type per_page: int, optional
        :param semaphore: Semaphore for limiting the number of simultaneous requests.
        :type semaphore: :class:`asyncio.Semaphore`, optional
        :param kwargs: Additional arguments.
        :return: List of images in dataset.
        :rtype: AsyncGenerator[List[ImageInfo]]

        :Usage example:

            .. code-block:: python

                    import supervisely as sly
                    import asyncio

                    os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
                    os.environ['API_TOKEN'] = 'Your Supervisely API Token'
                    api = sly.Api.from_env()

                    loop = sly.utils.get_or_create_event_loop()
                    images = loop.run_until_complete(api.image.get_list_async(123456, per_page=600))
        """

        method = "images.list"
        dataset_info = kwargs.get("dataset_info", None)

        if dataset_info is None:
            dataset_info = self._api.dataset.get_info_by_id(dataset_id, raise_error=True)

        if semaphore is None:
            semaphore = self._api.get_default_semaphore()

        if per_page is None:
            async with semaphore:
                # optimized request to get perPage value that predefined on Supervisely instance
                response = await self._api.post_async(
                    method,
                    {
                        ApiField.DATASET_ID: dataset_info.id,
                        ApiField.FIELDS: [ApiField.ID, ApiField.PATH_ORIGINAL],
                        ApiField.FILTER: [
                            {ApiField.FIELD: ApiField.ID, ApiField.OPERATOR: "=", ApiField.VALUE: 1}
                        ],
                        ApiField.FORCE_METADATA_FOR_LINKS: False,
                    },
                )
                response_json = response.json()
            per_page = response_json.get("perPage", API_DEFAULT_PER_PAGE)

        total_pages = ceil(dataset_info.items_count / per_page)

        data = {
            ApiField.DATASET_ID: dataset_info.id,
            ApiField.PROJECT_ID: dataset_info.project_id,
            ApiField.SORT: sort,
            ApiField.SORT_ORDER: sort_order,
            ApiField.FORCE_METADATA_FOR_LINKS: force_metadata_for_links,
            ApiField.FILTER: filters or [],
            ApiField.PER_PAGE: per_page,
        }
        if fields is not None:
            data[ApiField.FIELDS] = fields
        if only_labelled:
            data[ApiField.FILTERS] = [
                {
                    "type": "objects_class",
                    "data": {
                        "from": 1,
                        "to": 9999,
                        "include": True,
                        "classId": None,
                    },
                }
            ]

        async for page in self.get_list_page_generator_async(method, data, total_pages, semaphore):
            yield page

    @staticmethod
    def update_custom_sort(meta: Dict[str, Any], custom_sort: str) -> Dict[str, Any]:
        """
        Updates a copy of the meta dictionary with a new custom sort value.

        :param meta: Image meta dictionary.
        :type meta: Dict[str, Any]
        :param custom_sort: Custom sort value.
        :type custom_sort: str
        :return: Updated meta dictionary.
        :rtype: Dict[str, Any]
        """
        meta_copy = copy.deepcopy(meta)
        meta_copy[ApiField.CUSTOM_SORT] = custom_sort
        return meta_copy

    def download_blob_file(
        self,
        project_id: int,
        download_id: str,
        path: Optional[str] = None,
        log_progress: bool = True,
        chunk_size: Optional[int] = None,
    ) -> Optional[bytes]:
        """
        Downloads blob file from Supervisely storage by download ID of any Image that belongs to this file.

        :param project_id: Project ID in Supervisely.
        :type project_id: int
        :param download_id: Download ID of any Image that belongs to the blob file in Supervisely storage.
        :type download_id: str
        :param path: Path to save the blob file. If None, returns blob file content as bytes.
        :type path: str, optional
        :param progress_cb: Function for tracking download progress.
        :type progress_cb: tqdm or callable, optional
        :param chunk_size: Size of chunk for streaming. Default is 8 MB.
        :type chunk_size: int, optional
        :return: Blob file content if path is None, otherwise None.
        :rtype: bytes or None

        :Usage example:

         .. code-block:: python


            api = sly.Api.from_env()


            image_id = 6789
            image_info = api.image.get_info_by_id(image_id)
            project_id = api.dataset.get_info_by_id(image_info.dataset_id).project_id

            # Download and save to file
            api.image.download_blob_file(project_id, image_info.download_id, "/path/to/save/archive.tar")

            # Get archive as bytes
            archive_bytes = api.image.download_blob_file(project_id, image_info.download_id)
        """
        if chunk_size is None:
            chunk_size = 8 * 1024 * 1024

        response = self._api.post(
            "images.data.download",
            {ApiField.PROJECT_ID: project_id, ApiField.DOWNLOAD_ID: download_id},
            stream=True,
        )

        if log_progress:
            total_size = int(response.headers.get("Content-Length", 0))
            progress_cb = tqdm(
                total=total_size,
                unit="B",
                unit_scale=True,
                desc="Downloading images blob file",
                leave=True,
            )
        if path is not None:
            ensure_base_path(path)
            with open(path, "wb") as fd:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    fd.write(chunk)
                    if log_progress:
                        progress_cb.update(len(chunk))
            return None
        else:
            content = response.content
            if log_progress:
                progress_cb.update(len(content))
            return content

    def upload_blob_images(
        self,
        dataset: Union[DatasetInfo, int],
        blob_file: Union[FileInfo, str],
        metas: Optional[List[Dict[str, Any]]] = None,
        change_name_if_conflict: bool = True,
        progress_cb: Optional[Union[tqdm, Callable]] = None,
        return_image_infos_generator: bool = False,
    ) -> Union[Generator[ImageInfo, None], None]:
        """
        Uploads images from blob file in Team Files to dataset.

        IMPORTANT: File with image offsets should be in the same directory as the blob file.
        This file should be named as the blob file but with the suffix `_offsets.pkl`.
        It must be a Pickle file with the BlobImageInfos that define the range of bytes representing the image in the binary.
        To prepare the offsets file, use the `supervisely.fs.save_blob_offsets_pkl` function.

        :param dataset: Dataset in Supervisely. Can be DatasetInfo object or dataset ID.
                        It is recommended to use DatasetInfo object to avoid additional API requests.
        :type dataset: Union[DatasetInfo, int]
        :param blob_file: Blob file in Team Files. Can be FileInfo object or path to blob file.
                        It is recommended to use FileInfo object to avoid additional API requests.
        :type blob_file: Union[FileInfo, str]
        :param metas: List of metas for images.
        :type metas: Optional[List[Dict[str, Any]], optional
        :param change_name_if_conflict: If True adds suffix to the end of Image name when Dataset already contains an Image with identical name, If False and images with the identical names already exist in Dataset skips them.
        :type change_name_if_conflict: bool, optional
        :param progress_cb: Function for tracking upload progress. Tracks the count of processed items.
        :type progress_cb: Optional[Union[tqdm, Callable]]
        :param return_image_infos_generator: If True, returns generator of ImageInfo objects. Otherwise, returns None.
        :type return_image_infos_generator: bool, optional

        :return: Generator of ImageInfo objects if return_image_infos_generator is True, otherwise None.
        :rtype: Union[Generator[ImageInfo, None], None]


        """
        if isinstance(dataset, int):
            dataset_id = dataset
            dataset_info = self._api.dataset.get_info_by_id(dataset_id)
        else:
            dataset_id = dataset.id
            dataset_info = dataset

        if isinstance(blob_file, str):
            team_file_info = self._api.file.get_info_by_path(dataset_info.team_id, blob_file)
        else:
            team_file_info = blob_file

        image_infos_generator = self.upload_by_offsets_generator(
            dataset=dataset_info,
            team_file_id=team_file_info.id,
            progress_cb=progress_cb,
            metas=metas,
            conflict_resolution="rename" if change_name_if_conflict else "skip",
        )
        if return_image_infos_generator:
            return image_infos_generator
        else:
            for _ in image_infos_generator:
                pass

    async def download_blob_file_async(
        self,
        project_id: int,
        download_id: str,
        path: str,
        semaphore: Optional[asyncio.Semaphore] = None,
        log_progress: bool = True,
        progress_cb: Optional[Union[tqdm, Callable]] = None,
    ):
        """
        Downloads blob file from Supervisely storage by download ID asynchronously.

        :param project_id: Project ID in Supervisely.
        :type project_id: int
        :param download_id: Download ID of any Image that belongs to the blob file in Supervisely storage.
        :type download_id: str
        :param path: Path to save the blob file.
        :type path: str
        :param semaphore: Semaphore for limiting the number of simultaneous downloads.
        :type semaphore: asyncio.Semaphore, optional
        :param log_progress: If True, shows progress bar.
        :type log_progress: bool, optional
        :param progress_cb: Function for tracking download progress.
        :type progress_cb: tqdm or callable, optional
        """
        api_method_name = "images.data.download"

        if semaphore is None:
            semaphore = self._api.get_default_semaphore()

        async with semaphore:
            ensure_base_path(path)

            if log_progress:
                response = self._api.get(
                    api_method_name,
                    {ApiField.PROJECT_ID: project_id, ApiField.DOWNLOAD_ID: download_id},
                    stream=True,
                )
                total_size = int(response.headers.get("Content-Length", 0))
                response.close()
                name = os.path.basename(path)
                if progress_cb is None:
                    progress_cb = tqdm(
                        total=total_size,
                        unit="B",
                        unit_scale=True,
                        desc=f"Downloading images blob file {name}",
                        leave=True,
                    )

            async with aiofiles.open(path, "wb") as fd:
                async for chunk, _ in self._api.stream_async(
                    method=api_method_name,
                    method_type="POST",
                    data={ApiField.PROJECT_ID: project_id, ApiField.DOWNLOAD_ID: download_id},
                    chunk_size=8 * 1024 * 1024,
                ):
                    if log_progress:
                        progress_cb.update(len(chunk))
                    await fd.write(chunk)

    async def download_blob_files_async(
        self,
        project_id: int,
        download_ids: List[str],
        paths: List[str],
        semaphore: Optional[asyncio.Semaphore] = None,
        log_progress: bool = True,
        progress_cb: Optional[Union[tqdm, Callable]] = None,
    ):
        """
        Downloads multiple blob files from Supervisely storage by download IDs asynchronously.

        :param project_id: Project ID in Supervisely.
        :type project_id: int
        :param download_ids: List of download IDs of any Image that belongs to the blob files in Supervisely storage.
        :type download_ids: List[str]
        :param paths: List of paths to save the blob files.
        :type paths: List[str]
        :param semaphore: Semaphore for limiting the number of simultaneous downloads.
        :type semaphore: asyncio.Semaphore, optional
        :param log_progress: If True, shows progress bar.
        :type log_progress: bool, optional
        :param progress_cb: Function for tracking download progress.
        :type progress_cb: tqdm or callable, optional
        """

        if semaphore is None:
            semaphore = self._api.get_default_semaphore()

        tasks = []
        for download_id, path in zip(download_ids, paths):
            task = self.download_blob_file_async(
                project_id=project_id,
                download_id=download_id,
                path=path,
                semaphore=semaphore,
                log_progress=log_progress,
                progress_cb=progress_cb,
            )
            tasks.append(task)
        await asyncio.gather(*tasks)

    def set_embeddings_updated_at(self, ids: List[int], timestamps: Optional[List[str]] = None):
        """
        Updates the `updated_at` field of images with the timestamp of the embeddings were created.

        :param ids: List of Image IDs in Supervisely.
        :type ids: List[int]
        :param timestamps: List of timestamps in ISO format. If None, uses current time.
                            You could set timestamps to [None, ..., None] if you need to recreate embeddings for images.
        :type timestamps: List[str], optional
        :return: None
        :rtype: NoneType
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            import datetime

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            image_ids = [123, 456, 789]
            timestamps = [datetime.datetime.now().isoformat() for _ in image_ids]
            api.image.set_embeddings_updated_at(image_ids, timestamps)
        """
        method = "images.embeddings-updated-at.update"

        if timestamps is None:
            timestamps = [datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ") for _ in ids]

        if len(ids) != len(timestamps):
            raise ValueError(
                f"Length of ids and timestamps should be equal. {len(ids)} != {len(timestamps)}"
            )
        images = [
            {ApiField.ID: image_id, ApiField.EMBEDDINGS_UPDATED_AT: timestamp}
            for image_id, timestamp in zip(ids, timestamps)
        ]
        self._api.post(
            method,
            {ApiField.IMAGES: images},
        )

    def get_subsequent_image_ids(
        self,
        image_id: int,
        images_count: Optional[int] = None,
        job_id: Optional[int] = None,
        params: Optional[dict] = None,
        dataset_id: Optional[int] = None,
        project_id: Optional[int] = None,
    ) -> List[int]:
        """
        Get list of subsequent image IDs after the specified image ID.

        :param image_id: Image ID in Supervisely.
        :type image_id: int
        :param images_count: Number of subsequent images to retrieve. If None, retrieves all subsequent images.
        :type images_count: int, optional
        :param job_id: Job ID to filter images. If None, does not filter by job ID.
        :type job_id: int, optional
        :param params: Additional parameters for filtering and sorting images.
        :type params: dict, optional
        :param dataset_id: Dataset ID to filter images.
        :type dataset_id: int, optional
        :param project_id: Project ID to filter images. If None, makes a request to retrieve it from the specified image.
        :type project_id: int, optional
        """
        data = {
            "recursive": True,
            "projectId": project_id,
            "filters": [],
            "sort": "name",
            "sort_order": "asc",
        }

        if params is not None:
            data.update(params)

        if data["projectId"] is None:
            image_info = self.get_info_by_id(image_id)
            if image_info is None:
                raise ValueError(f"Image with ID {image_id} not found.")
            project_id = self._api.dataset.get_info_by_id(image_info.dataset_id).project_id
        if job_id is not None:
            self._api.add_header("x-job-id", str(job_id))
        if dataset_id is not None:
            data["datasetId"] = dataset_id

        image_infos = self.get_list_all_pages(
            "images.list",
            data,
            limit=None,
            return_first_response=False,
        )
        self._api.headers.pop("x-job-id", None)
        image_ids = [img_info.id for img_info in image_infos]
        if len(image_ids) == 0:
            raise ValueError("No images found with the specified criteria.")
        elif image_id not in image_ids:
            raise ValueError(f"Image with ID {image_id} not found in the specified entity.")

        target_idx = image_ids.index(image_id) + 1
        to_idx = target_idx + images_count if images_count is not None else len(image_ids)
        return image_ids[target_idx:to_idx]

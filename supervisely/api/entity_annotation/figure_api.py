# coding: utf-8

# docs
from __future__ import annotations

import asyncio
import json
import re
from collections import defaultdict
from typing import (
    AsyncGenerator,
    Callable,
    Dict,
    Generator,
    List,
    NamedTuple,
    Optional,
    Tuple,
    Union,
)

import numpy as np
from requests_toolbelt import MultipartDecoder, MultipartEncoder
from tqdm import tqdm

from supervisely._utils import batched, logger, run_coroutine
from supervisely.annotation.label import LabelingStatus
from supervisely.api.module_api import ApiField, ModuleApi, RemoveableBulkModuleApi
from supervisely.geometry.rectangle import Rectangle
from supervisely.video_annotation.key_id_map import KeyIdMap


class FigureInfo(NamedTuple):
    """
    Represents detailed information about a figure in a scene within the labeling tool.
    It is designed to handle multimodal data.

    Attributes:
        id (int): Unique identifier for the figure.
        class_id (int): Identifier for the class of the figure.
        updated_at (str): Timestamp of the last update.
        created_at (str): Timestamp of creation.
        entity_id (int): Identifier for the entity. Possible entities: image, video, volume, pointcloud, etc.
        object_id (int): Identifier for the object (applicable to videos, volumes, pointclouds).
        project_id (int): Identifier for the project.
        dataset_id (int): Identifier for the dataset.
        frame_index (int): Index of the frame (applicable to videos, volumes (as a slice_index), pointclouds).
        geometry_type (str): Type of geometry.
        geometry (dict): Geometry data.
        geometry_meta (dict): Metadata for the geometry.
        tags (list): List of tags associated with the figure.
        meta (dict): Additional metadata.
        area (str): Area information.
        priority (int): Position of the figure relative to other overlapping or underlying figures.
    """

    id: int
    class_id: int
    updated_at: str
    created_at: str
    entity_id: int
    object_id: int
    project_id: int
    dataset_id: int
    frame_index: int
    geometry_type: str
    geometry: dict
    geometry_meta: dict
    tags: list
    meta: dict
    area: str
    priority: Optional[int] = None
    custom_data: Optional[dict] = None

    @property
    def bbox(self) -> Optional[Rectangle]:
        """
        Get Figure's bounding box.

        :return: Rectangle in supervisely format.
        :rtype: :class: `sly.Rectangle`
        """
        if self.geometry_meta is not None:
            return Rectangle(*self.geometry_meta["bbox"], sly_id=self.id)

    def to_json(self):
        return FigureApi.convert_info_to_json(self)


class FigureApi(RemoveableBulkModuleApi):
    """
    Figure object for :class:`VideoAnnotation<supervisely.video_annotation.video_annotation.VideoAnnotation>`.
    """

    def _remove_batch_api_method_name(self):
        """_remove_batch_api_method_name"""
        return "figures.bulk.remove"

    def _remove_batch_field_name(self):
        """_remove_batch_field_name"""
        return ApiField.FIGURE_IDS

    @staticmethod
    def info_sequence():
        """
        NamedTuple FigureInfo information about Figure.

        :Example:

         .. code-block:: python

            FigureInfo(id=588801373,
                       updated_at='2020-12-22T06:37:13.183Z',
                       created_at='2020-12-22T06:37:13.183Z',
                       entity_id=186648101,
                       object_id=112482,
                       project_id=110366,
                       dataset_id=419886,
                       frame_index=0,
                       geometry_type='bitmap',
                       geometry={'bitmap': {'data': 'eJwdlns8...Cgj4=', 'origin': [335, 205]}})
        """
        return [
            ApiField.ID,
            ApiField.CLASS_ID,
            ApiField.UPDATED_AT,
            ApiField.CREATED_AT,
            ApiField.ENTITY_ID,
            ApiField.OBJECT_ID,
            ApiField.PROJECT_ID,
            ApiField.DATASET_ID,
            ApiField.FRAME_INDEX,
            ApiField.GEOMETRY_TYPE,
            ApiField.GEOMETRY,
            ApiField.GEOMETRY_META,
            ApiField.TAGS,
            ApiField.META,
            ApiField.AREA,
            ApiField.PRIORITY,
            ApiField.CUSTOM_DATA,
        ]

    @staticmethod
    def info_tuple_name():
        """
        Get string name of NamedTuple for class.

        :return: NamedTuple name.
        :rtype: :class:`str`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            tuple_name = api.video.figure.info_tuple_name()
            print(tuple_name) # FigureInfo
        """

        return "FigureInfo"

    def get_info_by_id(self, id: int) -> FigureInfo:
        """
        Get Figure information by ID.

        :param id: Figure ID in Supervisely.
        :type id: int
        :return: Information about Figure. See :class:`info_sequence<info_sequence>`
        :rtype: :class:`NamedTuple`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            figure_id = 588801373

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            figure_info = api.video.figure.get_info_by_id(figure_id)
            print(figure_info)
            # Output: [
            #     588801373,
            #     "2020-12-22T06:37:13.183Z",
            #     "2020-12-22T06:37:13.183Z",
            #     186648101,
            #     112482,
            #     110366,
            #     419886,
            #     0,
            #     "bitmap",
            #     {
            #         "bitmap": {
            #             "data": "eJw...Cgj4=",
            #             "origin": [
            #                 335,
            #                 205
            #             ]
            #         }
            #     }
            # ]
        """
        fields = [
            "id",
            "createdAt",
            "updatedAt",
            "imageId",
            "objectId",
            "classId",
            "projectId",
            "datasetId",
            "geometry",
            "geometryType",
            "geometryMeta",
            "tags",
            "meta",
            "area",
            "priority",
            "nnCreated",
            "nnUpdated",
        ]
        return self._get_info_by_id(id, "figures.info", {ApiField.FIELDS: fields})

    def create(
        self,
        entity_id: int,
        object_id: int,
        meta: Dict,
        geometry_json: Dict,
        geometry_type: str,
        track_id: Optional[int] = None,
        custom_data: Optional[dict] = None,
        status: Optional[LabelingStatus] = None,
    ) -> int:
        """"""
        input_figure = {
            ApiField.META: meta,
            ApiField.OBJECT_ID: object_id,
            ApiField.GEOMETRY_TYPE: geometry_type,
            ApiField.GEOMETRY: geometry_json,
        }

        if status is None:
            status = LabelingStatus.MANUAL

        nn_created, nn_updated = LabelingStatus.to_flags(status)
        input_figure[ApiField.NN_CREATED] = nn_created
        input_figure[ApiField.NN_UPDATED] = nn_updated

        if track_id is not None:
            input_figure[ApiField.TRACK_ID] = track_id

        if custom_data is not None:
            input_figure[ApiField.CUSTOM_DATA] = custom_data

        body = {ApiField.ENTITY_ID: entity_id, ApiField.FIGURES: [input_figure]}

        response = self._api.post("figures.bulk.add", body)
        return response.json()[0][ApiField.ID]

    def get_by_ids(self, dataset_id: int, ids: List[int]) -> List[FigureInfo]:
        """
        Get Figures information by IDs from given dataset ID.

        :param dataset_id: Dataset ID in Supervisely.
        :type dataset_id: int
        :param ids: List of Figures IDs.
        :type ids: List[int]
        :return: List of information about Figures. See :class:`info_sequence<info_sequence>`
        :rtype: :class:`List[NamedTuple]`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            dataset_id = 466642
            figures_ids = [642155547, 642155548, 642155549]
            figures_infos = api.video.figure.get_by_ids(dataset_id, figures_ids)
            print(figures_infos)
            # Output: [
            #     [
            #         642155547,
            #         "2021-03-23T13:25:34.705Z",
            #         "2021-03-23T13:25:34.705Z",
            #         198703211,
            #         152118,
            #         124976,
            #         466642,
            #         0,
            #         "rectangle",
            #         {
            #             "points": {
            #                 "exterior": [
            #                     [
            #                         2240,
            #                         1041
            #                     ],
            #                     [
            #                         2463,
            #                         1187
            #                     ]
            #                 ],
            #                 "interior": []
            #             }
            #         }
            #     ],
            #     [
            #         642155548,
            #         "2021-03-23T13:25:34.705Z",
            #         "2021-03-23T13:25:34.705Z",
            #         198703211,
            #         152118,
            #         124976,
            #         466642,
            #         1,
            #         "rectangle",
            #         {
            #             "points": {
            #                 "exterior": [
            #                     [
            #                         2248,
            #                         1048
            #                     ],
            #                     [
            #                         2455,
            #                         1176
            #                     ]
            #                 ],
            #                 "interior": []
            #             }
            #         }
            #     ],
            #     [
            #         642155549,
            #         "2021-03-23T13:25:34.705Z",
            #         "2021-03-23T13:25:34.705Z",
            #         198703211,
            #         152118,
            #         124976,
            #         466642,
            #         2,
            #         "rectangle",
            #         {
            #             "points": {
            #                 "exterior": [
            #                     [
            #                         2237,
            #                         1046
            #                     ],
            #                     [
            #                         2464,
            #                         1179
            #                     ]
            #                 ],
            #                 "interior": []
            #             }
            #         }
            #     ]
            # ]
        """
        filters = [{"field": "id", "operator": "in", "value": ids}]
        fields = [
            ApiField.ID,
            ApiField.CREATED_AT,
            ApiField.UPDATED_AT,
            ApiField.IMAGE_ID,
            ApiField.OBJECT_ID,
            ApiField.CLASS_ID,
            ApiField.PROJECT_ID,
            ApiField.DATASET_ID,
            ApiField.GEOMETRY,
            ApiField.GEOMETRY_TYPE,
            ApiField.GEOMETRY_META,
            ApiField.TAGS,
            ApiField.META,
            ApiField.AREA,
            ApiField.PRIORITY,
            ApiField.CUSTOM_DATA,
            ApiField.NN_CREATED,
            ApiField.NN_UPDATED,
        ]
        figures_infos = self.get_list_all_pages(
            "figures.list",
            {
                ApiField.DATASET_ID: dataset_id,
                ApiField.FILTER: filters,
                ApiField.FIELDS: fields,
            },
        )

        if len(ids) != len(figures_infos):
            ids_downloaded = [info.id for info in figures_infos]
            raise RuntimeError(
                "Ids don't exist on server: {}".format(set(ids_downloaded) - set(ids))
            )

        id_to_item = {info.id: info for info in figures_infos}

        figures = []
        for input_id in ids:
            figures.append(id_to_item[input_id])

        return figures

    def _append_bulk(
        self,
        entity_id,
        figures_json,
        figures_keys,
        key_id_map: KeyIdMap,
        field_name=ApiField.ENTITY_ID,
    ):
        """"""
        if len(figures_json) == 0:
            return
        for batch_keys, batch_jsons in zip(
            batched(figures_keys, batch_size=100), batched(figures_json, batch_size=100)
        ):
            resp = self._api.post(
                "figures.bulk.add",
                {field_name: entity_id, ApiField.FIGURES: batch_jsons},
            )
            for key, resp_obj in zip(batch_keys, resp.json()):
                figure_id = resp_obj[ApiField.ID]
                key_id_map.add_figure(key, figure_id)

    def create_bulk(
        self,
        figures_json: List[dict],
        entity_id: int = None,
        dataset_id: int = None,
        batch_size=200,
    ) -> List[int]:
        """
        Create figures in Supervisely in bulk.
        To optimize creation of a large number of figures use dataset ID instead of entity ID.
        In this case figure jsons list can contain figures from different entities for the same dataset.
        Every figure json must contain corresponding entity ID.

        *NOTE*: Geometries for AlphaMask must be uploaded separately via `upload_geometries_batch` method.

        :param figures_json: List of figures in Supervisely JSON format.
        :type figures_json: List[dict]
        :param entity_id: Entity ID.
        :type entity_id: int
        :param dataset_id: Dataset ID.
        :type dataset_id: int
        :return: List of figure IDs.
        """
        figure_ids = []
        if len(figures_json) == 0:
            return figure_ids

        if entity_id is None and dataset_id is None:
            raise ValueError("Either entity_id or dataset_id must be provided")
        if dataset_id is not None:
            body = {ApiField.DATASET_ID: dataset_id}
        else:
            body = {ApiField.ENTITY_ID: entity_id}

        for batch_jsons in batched(figures_json, batch_size):
            body[ApiField.FIGURES] = batch_jsons
            response = self._api.post("figures.bulk.add", body)
            for resp_obj in response.json():
                figure_ids.append(resp_obj[ApiField.ID])
        return figure_ids

    def download(
        self, dataset_id: int, image_ids: List[int] = None, skip_geometry: bool = False
    ) -> Dict[int, List[FigureInfo]]:
        """
        Method returns a dictionary with pairs of image ID and list of FigureInfo for the given dataset ID. Can be filtered by image IDs.

        :param dataset_id: Dataset ID in Supervisely.
        :type dataset_id: int
        :param image_ids: Specify the list of image IDs within the given dataset ID. If image_ids is None, the method returns all possible pairs of images with figures. Note: Consider using `sly.batched()` to ensure that no figures are lost in the response.
        :type image_ids: List[int], optional
        :param skip_geometry: Skip the download of figure geometry. May be useful for a significant api request speed increase in the large datasets.
        :type skip_geometry: bool

        :return: A dictionary where keys are image IDs and values are lists of figures.
        :rtype: :class: `Dict[int, List[FigureInfo]]`
        """
        fields = [
            ApiField.ID,
            ApiField.CREATED_AT,
            ApiField.UPDATED_AT,
            ApiField.IMAGE_ID,
            ApiField.OBJECT_ID,
            ApiField.CLASS_ID,
            ApiField.PROJECT_ID,
            ApiField.DATASET_ID,
            ApiField.GEOMETRY,
            ApiField.GEOMETRY_TYPE,
            ApiField.GEOMETRY_META,
            ApiField.TAGS,
            ApiField.META,
            ApiField.AREA,
            ApiField.PRIORITY,
            ApiField.CUSTOM_DATA,
            ApiField.NN_CREATED,
            ApiField.NN_UPDATED,
        ]
        if skip_geometry is True:
            fields = [x for x in fields if x != ApiField.GEOMETRY]

        if image_ids is None:
            filters = []
        else:
            filters = [
                {
                    ApiField.FIELD: ApiField.ENTITY_ID,
                    ApiField.OPERATOR: "in",
                    ApiField.VALUE: image_ids,
                }
            ]
        data = {
            ApiField.DATASET_ID: dataset_id,
            ApiField.FIELDS: fields,
            ApiField.FILTER: filters,
        }
        resp = self._api.post("figures.list", data)
        infos = resp.json()
        images_figures = defaultdict(list)

        total_pages = infos["pagesCount"]
        for page in range(1, total_pages + 1):
            if page > 1:
                data.update({ApiField.PAGE: page})
                resp = self._api.post("figures.list", data)
                infos = resp.json()
            for info in infos["entities"]:
                figure_info = self._convert_json_info(info, True)
                images_figures[figure_info.entity_id].append(figure_info)

        return dict(images_figures)

    def _convert_json_info(self, info: dict, skip_missing=False):
        res = super()._convert_json_info(info, skip_missing=True)
        return FigureInfo(**res._asdict())

    def _download_geometries_generator(
        self, ids: List[int]
    ) -> Generator[Tuple[int, MultipartDecoder.Part], None, None]:
        """
        Private method. Download figures geometries with given IDs from storage.
        """

        for batch_ids in batched(ids):
            response = self._api.post("figures.bulk.download.geometry", {ApiField.IDS: batch_ids})
            decoder = MultipartDecoder.from_response(response)
            for part in decoder.parts:
                content_utf8 = part.headers[b"Content-Disposition"].decode("utf-8")
                # Find name="1245" preceded by a whitespace, semicolon or beginning of line.
                # The regex has 2 capture group: one for the prefix and one for the actual name value.
                figure_id = int(re.findall(r'(^|[\s;])name="(\d*)"', content_utf8)[0][1])
                yield figure_id, part

    def download_geometry(self, figure_id: int) -> dict:
        """
        Download figure geometry with given ID from storage.

        :param figure_id: Figure ID in Supervisely.
        :type figure_id: int
        :return: Figure geometry in Supervisely JSON format.
        :rtype: dict
        """
        return self.download_geometries_batch([figure_id])

    def download_geometries_batch(
        self,
        ids: List[int],
        progress_cb: Optional[Union[tqdm, Callable]] = None,
    ) -> List[dict]:
        """
        Download figure geometries with given IDs from storage.

        :param ids: List of figure IDs in Supervisely.
        :type ids: List[int]
        :param progress_cb: Progress bar to show the download progress. Shows the number of bytes downloaded.
        :type progress_cb: Union[tqdm, Callable], optional
        :return: List of figure geometries in Supervisely JSON format.
        :rtype: List[dict]
        """
        geometries = {}
        for idx, part in self._download_geometries_generator(ids):
            try:
                if progress_cb is not None:
                    progress_cb(len(part.content))
                geometry_json = json.loads(part.content)
                geometries[idx] = geometry_json
            except Exception as e:
                raise RuntimeError(f"Failed to decode geometry for figure ID {idx}") from e

        if len(geometries) != len(ids):
            raise RuntimeError("Not all geometries were downloaded")
        ordered_results = [geometries[i] for i in ids]
        return ordered_results

    def upload_geometry(self, figure_id: int, geometry: dict):
        """
        Upload figure geometry with given figure ID to storage.

        :param figure_id: Figure ID in Supervisely.
        :type figure_id: int
        :param geometry: Figure geometry in Supervisely JSON format.
        :type geometry: dict
        :return: None
        :rtype: None
        """
        self.upload_geometries_batch([figure_id], [geometry])

    def upload_geometries_batch(self, figure_ids: List[int], geometries: List[dict]):
        """
        Upload figure geometries with given figure IDs to storage.

        :param figure_ids: List of figure IDs in Supervisely.
        :type figure_ids: List[int]
        :param geometries: List of figure geometries in Supervisely JSON format.
        :type geometries: List[dict]
        :return: None
        :rtype: None
        """
        geometries = [json.dumps(geometry).encode("utf-8") for geometry in geometries]

        for batch_ids, batch_geometries in zip(
            batched(figure_ids, batch_size=100), batched(geometries, batch_size=100)
        ):
            fields = []
            for figure_id, geometry in zip(batch_ids, batch_geometries):
                fields.append((ApiField.FIGURE_ID, str(figure_id)))
                fields.append(
                    (
                        ApiField.GEOMETRY,
                        (str(figure_id), geometry, "application/octet-stream"),
                    )
                )
            encoder = MultipartEncoder(fields=fields)
            self._api.post("figures.bulk.upload.geometry", encoder)

    async def _download_geometries_generator_async(
        self, ids: List[int], semaphore: Optional[asyncio.Semaphore] = None
    ) -> AsyncGenerator[Tuple[int, bytes], None, None]:
        """
        Private method. Download figures geometries with given IDs from storage asynchronously.

        :param ids: List of figure IDs in Supervisely.
        :type ids: List[int]
        :return: Async generator with pairs of figure ID and figure geometry.
        :rtype: AsyncGenerator[Tuple[int, bytes], None, None]
        """
        if semaphore is None:
            semaphore = self._api.get_default_semaphore()

        for batch_ids in batched(ids):
            async with semaphore:
                response = await self._api.post_async(
                    "figures.bulk.download.geometry", {ApiField.IDS: batch_ids}
                )
                decoder = MultipartDecoder.from_response(response)
                for part in decoder.parts:
                    content_utf8 = part.headers[b"Content-Disposition"].decode("utf-8")
                    # Find name="1245" preceded by a whitespace, semicolon or beginning of line.
                    # The regex has 2 capture group: one for the prefix and one for the actual name value.
                    figure_id = int(re.findall(r'(^|[\s;])name="(\d*)"', content_utf8)[0][1])
                    yield figure_id, part.content

    async def download_geometries_batch_async(
        self,
        ids: List[int],
        progress_cb: Optional[Union[tqdm, Callable]] = None,
        semaphore: Optional[asyncio.Semaphore] = None,
    ) -> List[dict]:
        """
        Download figure geometries with given IDs from storage asynchronously.

        :param ids: List of figure IDs in Supervisely.
        :type ids: List[int]
        :param progress_cb: Progress bar to show the download progress. Shows the number of bytes downloaded.
        :type progress_cb: Union[tqdm, Callable], optional
        :param semaphore: Semaphore to limit the number of concurrent downloads.
        :type semaphore: Optional[asyncio.Semaphore], optional
        :return: List of figure geometries in Supervisely JSON format.
        :rtype: List[dict]

        :Usage example:

            .. code-block:: python

                import asyncio
                import supervisely as sly

                os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
                os.environ['API_TOKEN'] = 'Your Supervisely API Token'
                api = sly.Api.from_env()

                figure_ids = [642155547, 642155548, 642155549]
                loop = sly.utils.get_or_create_event_loop()
                geometries = loop.run_until_complete(
                    api.figure.download_geometries_batch_async(
                        figure_ids,
                        progress_cb=tqdm(total=len(figure_ids), desc="Downloading geometries"),
                        semaphore=asyncio.Semaphore(15),
                    )
                )
        """
        geometries = {}
        async for idx, part in self._download_geometries_generator_async(ids, semaphore):
            if progress_cb is not None:
                progress_cb(len(part))
            geometry_json = json.loads(part)
            geometries[idx] = geometry_json

        if len(geometries) != len(ids):
            raise RuntimeError("Not all geometries were downloaded")
        ordered_results = [geometries[i] for i in ids]
        return ordered_results

    async def upload_geometries_batch_async(
        self,
        figure_ids: List[int],
        geometries: List[dict],
        semaphore: Optional[asyncio.Semaphore] = None,
        progress_cb: Optional[Union[tqdm, Callable]] = None,
    ) -> None:
        """
        Upload figure geometries with given figure IDs to storage asynchronously in batches.

        :param figure_ids: List of figure IDs in Supervisely.
        :type figure_ids: List[int]
        :param geometries: List of figure geometries in Supervisely JSON format.
        :type geometries: List[dict]
        :param semaphore: Semaphore to limit the number of concurrent uploads.
        :type semaphore: Optional[asyncio.Semaphore], optional
        :param progress_cb: Progress bar to show the upload progress. Shows the number of geometries uploaded.
        :type progress_cb: Union[tqdm, Callable], optional
        :return: None
        :rtype: None

        :Usage example:

            .. code-block:: python

                import asyncio
                import supervisely as sly

                os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
                os.environ['API_TOKEN'] = 'Your Supervisely API Token'
                api = sly.Api.from_env()

                figure_ids = [642155547, 642155548, 642155549]
                geometries = [{...}, {...}, {...}]  # Your geometry data

                upload_coroutine = api.figure.upload_geometries_batch_async(
                        figure_ids,
                        geometries,
                        semaphore=asyncio.Semaphore(10),
                    )
                sly.run_coroutine(upload_coroutine)
        """
        if semaphore is None:
            semaphore = self._api.get_default_semaphore()

        encoded_geometries = [json.dumps(geometry).encode("utf-8") for geometry in geometries]

        batch_size = 100
        tasks = []

        for batch_ids, batch_geometries in zip(
            batched(figure_ids, batch_size),
            batched(encoded_geometries, batch_size),
        ):
            fields = []
            for figure_id, geometry in zip(batch_ids, batch_geometries):
                fields.append((ApiField.FIGURE_ID, str(figure_id)))
                fields.append(
                    (
                        ApiField.GEOMETRY,
                        (str(figure_id), geometry, "application/octet-stream"),
                    )
                )

            async def upload_batch(fields, progress_cb, num):
                async with semaphore:
                    encoder = MultipartEncoder(fields=fields)
                    headers = {"Content-Type": encoder.content_type}
                    async for _, _ in self._api.stream_async(
                        "figures.bulk.upload.geometry",
                        "POST",
                        data=encoder,
                        content=encoder.to_string(),
                        headers=headers,
                    ):
                        pass
                if progress_cb is not None:
                    progress_cb.update(num)

            tasks.append(upload_batch(fields, progress_cb, len(batch_ids)))

        if tasks:
            await asyncio.gather(*tasks)

    async def download_async(
        self,
        dataset_id: int,
        image_ids: Optional[List[int]] = None,
        skip_geometry: bool = False,
        semaphore: Optional[asyncio.Semaphore] = None,
        log_progress: bool = True,
        batch_size: int = 300,
    ) -> Dict[int, List[FigureInfo]]:
        """
        Asynchronously download figures for the given dataset ID. Can be filtered by image IDs.
        This method is significantly faster than the synchronous version for large datasets.

        :param dataset_id: Dataset ID in Supervisely.
        :type dataset_id: int
        :param image_ids: Specify the list of image IDs within the given dataset ID. If image_ids is None, the method returns all possible pairs of images with figures.
        :type image_ids: List[int], optional
        :param skip_geometry: Skip the download of figure geometry. May be useful for a significant api request speed increase in the large datasets.
        :type skip_geometry: bool
        :param semaphore: Semaphore to limit the number of concurrent downloads.
        :type semaphore: Optional[asyncio.Semaphore], optional
        :param log_progress: If True, log the progress of the download.
        :type log_progress: bool, optional
        :param batch_size: Size of the batch for downloading figures per 1 request. Default is 300.
                        Used for batching image_ids when filtering by specific images.
                        Adjust this value for optimal performance, value cannot exceed 500.
        :type batch_size: int, optional
        :return: A dictionary where keys are image IDs and values are lists of figures.
        :rtype: Dict[int, List[FigureInfo]]

        :Usage example:

        .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            dataset_id = 12345
            download_coroutine = api.image.figure.download_async(dataset_id)
            figures = sly.run_coroutine(download_coroutine)
        """
        fields = [
            ApiField.ID,
            ApiField.CREATED_AT,
            ApiField.UPDATED_AT,
            ApiField.IMAGE_ID,
            ApiField.OBJECT_ID,
            ApiField.CLASS_ID,
            ApiField.PROJECT_ID,
            ApiField.DATASET_ID,
            ApiField.GEOMETRY,
            ApiField.GEOMETRY_TYPE,
            ApiField.GEOMETRY_META,
            ApiField.TAGS,
            ApiField.META,
            ApiField.AREA,
            ApiField.PRIORITY,
            ApiField.CUSTOM_DATA,
            ApiField.NN_CREATED,
            ApiField.NN_UPDATED,
        ]
        if skip_geometry is True:
            fields = [x for x in fields if x != ApiField.GEOMETRY]

        # Base data setup
        base_data = {
            ApiField.DATASET_ID: dataset_id,
            ApiField.FIELDS: fields,
        }

        if semaphore is None:
            semaphore = self._api.get_default_semaphore()

        async def _get_page_figures(page_data, semaphore, progress_cb: tqdm = None):
            """Helper function to get figures from a single page"""
            async with semaphore:
                response = await self._api.post_async("figures.list", page_data)
                response_json = response.json()

                page_figures = []
                for info in response_json["entities"]:
                    figure_info = self._convert_json_info(info, True)
                    page_figures.append(figure_info)
                if progress_cb is not None:
                    progress_cb.update(len(response_json["entities"]))
                return page_figures

        async def _get_all_pages(ids_filter, progress_cb: tqdm = None):
            """Internal function to process all pages for given filter"""
            data = base_data.copy()
            data[ApiField.FILTER] = ids_filter

            # Get first page to determine pagination
            data[ApiField.PAGE] = 1
            async with semaphore:
                response = await self._api.post_async("figures.list", data)
                response_json = response.json()

            pages_count = response_json["pagesCount"]
            all_figures = []

            # Process first page
            for info in response_json["entities"]:
                figure_info = self._convert_json_info(info, True)
                all_figures.append(figure_info)
            if progress_cb is not None:
                progress_cb.update(len(response_json["entities"]))

            # Process remaining pages in parallel if needed
            if pages_count > 1:
                tasks = []
                for page in range(2, pages_count + 1):
                    page_data = data.copy()
                    page_data[ApiField.PAGE] = page
                    tasks.append(
                        asyncio.create_task(
                            _get_page_figures(page_data, semaphore, progress_cb=progress_cb)
                        )
                    )

                if tasks:
                    page_results = await asyncio.gather(*tasks)
                    for page_figures in page_results:
                        all_figures.extend(page_figures)

            return all_figures

        if log_progress:
            progress_cb = tqdm(desc="Downloading figures", unit="figure", total=0)
        else:
            progress_cb = None

        # Strategy: batch processing based on image_ids
        tasks = []

        if image_ids is None:
            # Single task for all figures in dataset
            filters = []
            tasks.append(_get_all_pages(filters, progress_cb=progress_cb))
        else:
            # Batch image_ids and create tasks for each batch
            for batch_ids in batched(image_ids, batch_size):
                filters = [
                    {
                        ApiField.FIELD: ApiField.ENTITY_ID,
                        ApiField.OPERATOR: "in",
                        ApiField.VALUE: list(batch_ids),
                    }
                ]
                tasks.append(_get_all_pages(filters, progress_cb=progress_cb))
                # Small delay between batches to reduce server load
                await asyncio.sleep(0.02)

        # Execute all tasks in parallel and collect results
        all_results = await asyncio.gather(*tasks)

        # Combine results from all batches
        images_figures = defaultdict(list)

        for batch_figures in all_results:
            for figure in batch_figures:
                images_figures[figure.entity_id].append(figure)

        return dict(images_figures)

    def download_fast(
        self,
        dataset_id: int,
        image_ids: Optional[List[int]] = None,
        skip_geometry: bool = False,
        semaphore: Optional[asyncio.Semaphore] = None,
        log_progress: bool = True,
        batch_size: int = 300,
    ) -> Dict[int, List[FigureInfo]]:
        """
        Download figures for the given dataset ID. Can be filtered by image IDs.
        This method is significantly faster than the synchronous version for large datasets and
        is designed to be used in an asynchronous context.
        Will fallback to the synchronous version if async fails.

        :param dataset_id: Dataset ID in Supervisely.
        :type dataset_id: int
        :param image_ids: Specify the list of image IDs within the given dataset ID. If image_ids is None, the method returns all possible pairs of images with figures.
        :type image_ids: List[int], optional
        :param skip_geometry: Skip the download of figure geometry. May be useful for a significant api request speed increase in the large datasets.
        :type skip_geometry: bool
        :param semaphore: Semaphore to limit the number of concurrent downloads.
        :type semaphore: Optional[asyncio.Semaphore], optional
        :param log_progress: If True, log the progress of the download.
        :type log_progress: bool, optional
        :param batch_size: Size of the batch for downloading figures per 1 request. Default is 300.
                        Used for batching image_ids when filtering by specific images.
                        Adjust this value for optimal performance, value cannot exceed 500.
        :type batch_size: int, optional

        :return: A dictionary where keys are image IDs and values are lists of figures.
        :rtype: Dict[int, List[FigureInfo]]

        :Usage example:

        .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            dataset_id = 12345
            figures = api.image.figure.download_fast(dataset_id)
        """
        try:
            return run_coroutine(
                self.download_async(
                    dataset_id=dataset_id,
                    image_ids=image_ids,
                    skip_geometry=skip_geometry,
                    semaphore=semaphore,
                    log_progress=log_progress,
                    batch_size=batch_size,
                )
            )
        except Exception:
            # Fallback to the synchronous version if async fails
            logger.debug("Async download of figures is failed, falling back to sync download.")
            if log_progress:
                logger.debug("Downloading figures without progress bar.")
            return self.download(
                dataset_id=dataset_id,
                image_ids=image_ids,
                skip_geometry=skip_geometry,
            )

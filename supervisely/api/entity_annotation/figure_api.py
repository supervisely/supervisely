# coding: utf-8

# docs
from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import Dict, Generator, List, NamedTuple, Optional, Tuple

import numpy as np
from requests_toolbelt import MultipartDecoder, MultipartEncoder

from supervisely._utils import batched
from supervisely.api.module_api import ApiField, ModuleApi, RemoveableBulkModuleApi
from supervisely.geometry.rectangle import Rectangle
from supervisely.video_annotation.key_id_map import KeyIdMap


class FigureInfo(NamedTuple):
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

    @property
    def bbox(self) -> Optional[Rectangle]:
        """
        Get Figure's bounding box.

        :return: Rectangle in supervisely format.
        :rtype: :class: `sly.Rectangle`
        """
        if self.geometry_meta is not None:
            return Rectangle(*self.geometry_meta["bbox"])


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
        ]
        return self._get_info_by_id(id, "figures.info", {ApiField.FIELDS: fields})

    def create(
        self,
        entity_id: int,
        object_id: int,
        meta: Dict,
        geometry_json: Dict,
        geometry_type: str,
        track_id: int = None,
    ) -> int:
        """"""
        input_figure = {
            ApiField.META: meta,
            ApiField.OBJECT_ID: object_id,
            ApiField.GEOMETRY_TYPE: geometry_type,
            ApiField.GEOMETRY: geometry_json,
        }

        if track_id is not None:
            input_figure[ApiField.TRACK_ID] = track_id

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

    def download(
        self,
        dataset_id: int,
        image_ids: List[int] = None,
        skip_geometry: bool = False,
    ) -> Dict[int, List[FigureInfo]]:
        """
        Method returns a dictionary with pairs of image ID and list of FigureInfo for the given dataset ID. Can be filtered by image IDs.

        :param dataset_id: Dataset ID in Supervisely.
        :type dataset_id: int
        :param image_ids: Specify the list of image IDs within the given dataset ID. If image_ids is None, the method returns all possible pairs of images with figures. Note: Consider using `sly.batched()` to ensure that no figures are lost in the response.

        :type image_ids: List[int], optional
        :param skip_geometry: Skip the download of figure geometry. May be useful for a significant api requets speed increase in the large datasets.
        :type skip_geometry: bool

        :return: A dictionary where keys are image IDs and values are lists of figures.
        :rtype: :class: `Dict[int, List[FigureInfo]]`
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
        ]
        if skip_geometry is True:
            fields = [x for x in fields if x != "geometry"]

        if image_ids is None:
            filters = []
        else:
            filters = [{ApiField.FIELD: ApiField.ENTITY_ID, "operator": "in", "value": image_ids}]
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

    def download_geometries_batch(self, ids: List[int]) -> List[dict]:
        """
        Download figure geometries with given IDs from storage.

        :param ids: List of figure IDs in Supervisely.
        :type ids: List[int]
        :return: List of figure geometries in Supervisely JSON format.
        :rtype: List[dict]
        """
        geometries = {}
        for idx, part in self._download_geometries_generator(ids):
            geometry_json = json.loads(part.content)
            geometries[idx] = geometry_json

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
                    (ApiField.GEOMETRY, (str(figure_id), geometry, "application/octet-stream"))
                )
            encoder = MultipartEncoder(fields=fields)
            self._api.post("figures.bulk.upload.geometry", encoder)

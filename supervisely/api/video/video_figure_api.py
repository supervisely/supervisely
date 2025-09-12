# coding: utf-8

# docs
from __future__ import annotations

from typing import Dict, List, Optional

from supervisely.api.entity_annotation.figure_api import FigureApi, FigureInfo
from supervisely.api.module_api import ApiField
from supervisely.geometry.geometry import Geometry
from supervisely.video_annotation.key_id_map import KeyIdMap
from supervisely.video_annotation.video_figure import VideoFigure
from supervisely.annotation.label import LabelingStatus

class VideoFigureApi(FigureApi):
    """
    :class:`VideoFigure<supervisely.video_annotation.video_figure.VideoFigure>` for a single video.
    """

    def create(
        self,
        video_id: int,
        object_id: int,
        frame_index: int,
        geometry_json: dict,
        geometry_type: str,
        track_id: Optional[int] = None,
        meta: Optional[dict] = None,
        status: Optional[LabelingStatus] = None,
    ) -> int:
        """
        Create new VideoFigure for given frame in given video ID.

        :param video_id: Video ID in Supervisely.
        :type video_id: int
        :param object_id: ID of the object to which the VideoFigure belongs.
        :type object_id: int
        :param frame_index: Number of the frame to add VideoFigure.
        :type frame_index: int
        :param geometry_json: Parameters of geometry for VideoFigure.
        :type geometry_json: dict
        :param geometry_type: Type of VideoFigure geometry.
        :type geometry_type: str
        :param track_id: int, optional.
        :type track_id: int, optional
        :param meta: Meta data for VideoFigure.
        :type meta: dict, optional
        :param status: Labeling status. Specifies if the VideoFigure was created by NN model, manually or created by NN and then manually corrected.
        :type status: LabelingStatus, optional
        :return: New figure ID
        :rtype: :class:`int`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            video_id = 198703211
            object_id = 152118
            frame_idx = 0
            geometry_json = {'points': {'exterior': [[500, 500], [1555, 1500]], 'interior': []}}
            geometry_type = 'rectangle'

            figure_id = api.video.figure.create(video_id, object_id, frame_idx, geometry_json, geometry_type) # 643182610
        """
        if meta is None:
            meta = {}
        meta = {**(meta or {}), ApiField.FRAME: frame_index}

        return super().create(
            video_id,
            object_id,
            meta,
            geometry_json,
            geometry_type,
            track_id,
            status=status,
        )

    def append_bulk(self, video_id: int, figures: List[VideoFigure], key_id_map: KeyIdMap) -> None:
        """
        Add VideoFigures to given Video by ID.

        :param video_id: Video ID in Supervisely.
        :type video_id: int
        :param figures: List of VideoFigures to append.
        :type figures: List[VideoFigure]
        :param key_id_map: KeyIdMap object.
        :type key_id_map: KeyIdMap
        :return: None
        :rtype: :class:`NoneType`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            project_id = 124976
            meta_json = api.project.get_meta(project_id)
            meta = sly.ProjectMeta.from_json(meta_json)
            key_id_map = KeyIdMap()

            video_id = 198703212
            ann_info = api.video.annotation.download(video_id)
            ann = sly.VideoAnnotation.from_json(ann_info, meta, key_id_map)
            figures = ann.figures[:5]
            api.video.figure.append_bulk(video_id, figures, key_id_map)
        """

        keys = []
        figures_json = []
        for figure in figures:
            keys.append(figure.key())
            figures_json.append(figure.to_json(key_id_map, save_meta=True))

        self._append_bulk(video_id, figures_json, keys, key_id_map)

    def update(self, figure_id: int, geometry: Geometry, status: Optional[LabelingStatus] = None) -> None:
        """Updates figure feometry with given ID in Supervisely with new Geometry object.

        :param figure_id: ID of the figure to update
        :type figure_id: int
        :param geometry: Supervisely Gepmetry object
        :type geometry: Geometry
        :param status: Labeling status. Specifies if the VideoFigure was created by NN model, manually or created by NN and then manually corrected.
        :type status: LabelingStatus, optional
        :Usage example:

         .. code-block:: python

            import os
            from dotenv import load_dotenv

            import supervisely as sly

            # Load secrets and create API object from .env file (recommended)
            # Learn more here: https://developer.supervisely.com/getting-started/basics-of-authentication
            load_dotenv(os.path.expanduser("~/supervisely.env"))
            api = sly.Api.from_env()

            new_geometry: sly.Rectangle(10, 10, 100, 100)
            figure_id = 121236918

            api.video.figure.update(figure_id, new_geometry)
        """
        payload = {
            ApiField.ID: figure_id,
            ApiField.GEOMETRY: geometry.to_json(),
        }

        if status is not None:
            nn_created,nn_updated = LabelingStatus.to_flags(status)
            payload[ApiField.NN_CREATED] = nn_created
            payload[ApiField.NN_UPDATED] = nn_updated

        self._api.post("figures.editInfo", payload)

    def download(
        self, dataset_id: int, video_ids: List[int] = None, skip_geometry: bool = False, **kwargs
    ) -> Dict[int, List[FigureInfo]]:
        """
        Method returns a dictionary with pairs of video ID and list of FigureInfo for the given dataset ID. Can be filtered by video IDs.

        :param dataset_id: Dataset ID in Supervisely.
        :type dataset_id: int
        :param video_ids: Specify the list of video IDs within the given dataset ID. If video_ids is None, the method returns all possible pairs of images with figures. Note: Consider using `sly.batched()` to ensure that no figures are lost in the response.
        :type video_ids: List[int], optional
        :param skip_geometry: Skip the download of figure geometry. May be useful for a significant api request speed increase in the large datasets.
        :type skip_geometry: bool
        :return: A dictionary where keys are video IDs and values are lists of figures.
        :rtype: :class: `Dict[int, List[FigureInfo]]`
        """
        if kwargs.get("image_ids", False) is not False:
            video_ids = kwargs["image_ids"]  # backward compatibility
        return super().download(dataset_id, video_ids, skip_geometry)

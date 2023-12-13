# coding: utf-8
from typing import Optional

from supervisely.api.module_api import ApiField, ModuleApiBase
from supervisely.collection.str_enum import StrEnum


class ImageAnnotationToolAction(StrEnum):
    SET_FIGURE = "figures/setFigure"
    """"""
    NEXT_IMAGE = "images/nextImage"
    """"""
    PREV_IMAGE = "images/prevImage"
    """"""
    SET_IMAGE = "images/setImage"
    """"""
    ZOOM_TO_FIGURE = "scene/zoomToObject"
    """"""


class ImageAnnotationToolApi(ModuleApiBase):
    def set_figure(self, session_id: int, figure_id: int):
        """Sets the figure as the current figure (selected figure) in the annotation tool.

        :param session_id: Annotation tool session id.
        :type session_id: int
        :param figure_id: Figure id.
        :type figure_id: int"""
        return self._act(
            session_id, ImageAnnotationToolAction.SET_FIGURE, {ApiField.FIGURE_ID: figure_id}
        )

    def next_image(self, session_id: int, *args, **kwargs):
        """Changes the current image in the annotation tool to the next image.

        :param session_id: Annotation tool session id.
        :type session_id: int"""
        return self._act(session_id, ImageAnnotationToolAction.NEXT_IMAGE, {})

    def prev_image(self, session_id: int, *args, **kwargs):
        """Changes the current image in the annotation tool to the previous image.

        :param session_id: Annotation tool session id.
        :type session_id: int"""
        return self._act(session_id, ImageAnnotationToolAction.PREV_IMAGE, {})

    def set_image(self, session_id: int, image_id: int):
        """Sets the image as the current image in the annotation tool.
        NOTE: The image must be in the same dataset as the current image.

        :param session_id: Annotation tool session id.
        :type session_id: int
        :param image_id: Image id in the same dataset as the current image.
        :type image_id: int"""
        return self._act(
            session_id, ImageAnnotationToolAction.SET_IMAGE, {ApiField.IMAGE_ID: image_id}
        )

    def zoom_to_figure(self, session_id: int, figure_id: int, zoom_factor: Optional[float] = 1):
        """Zooms the scene to the figure with the given id and zoom factor.

        :param session_id: Annotation tool session id.
        :type session_id: int
        :param figure_id: Figure id.
        :type figure_id: int
        :param zoom_factor: Zoom factor. Default is 1.
        :type zoom_factor: float, optional"""
        return self._act(
            session_id,
            ImageAnnotationToolAction.ZOOM_TO_FIGURE,
            {ApiField.FIGURE_ID: figure_id, ApiField.ZOOM_FACTOR: zoom_factor},
        )

    def _act(self, session_id: int, action: ImageAnnotationToolAction, payload: dict):
        """ """
        data = {
            ApiField.SESSION_ID: session_id,
            ApiField.ACTION: str(action),
            ApiField.PAYLOAD: payload,
        }
        resp = self._api.post("/annotation-tool.run-action", data)
        return resp.json()

    # {
    #     "sessionId": "940c4ec7-3818-420b-9277-ab3c820babe5",
    #     "action": "scene/setViewport",
    #     "payload": {
    #         "viewport": {
    #             "offsetX": -461, # width
    #             "offsetY": -1228, # height
    #             "zoom": 1.7424000000000024
    #         }
    #     }
    # }

    # {
    #     "sessionId": "940c4ec7-3818-420b-9277-ab3c820babe5",
    #     "action": "scene/zoomToObject",
    #     "payload": {
    #         "figureId": 22129,
    #         "zoomFactor": 1.5
    #     }
    # }

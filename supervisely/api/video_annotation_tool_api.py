# coding: utf-8

from typing import Any, Dict, Optional

from supervisely.api.module_api import ApiField, ModuleApiBase
from supervisely.collection.str_enum import StrEnum


class VideoAnnotationToolAction(StrEnum):
    JOBS_DISABLE_CONTROLS = "jobs/disableControls"
    """"""
    JOBS_ENABLE_CONTROLS = "jobs/enableControls"
    """"""
    ENTITIES_SET_INTITY = "entities/setEntity"
    """"""


class VideoAnnotationToolApi(ModuleApiBase):
    def disable_job_controls(self, session_id: str) -> Dict[str, Any]:
        """Disables controls of the labeling jobs. Buttons: Sumbit job, Confirm video.

        :param session_id: ID of the session in the Video Labeling Tool which controls should be disabled.
        :type session_id: str
        :return: Response from API server in JSON format.
        :rtype: Dict[str, Any]
        """
        return self._act(
            session_id,
            VideoAnnotationToolAction.JOBS_DISABLE_CONTROLS,
            {},
        )

    def enable_job_controls(self, session_id: str) -> Dict[str, Any]:
        """Enables controls of the labeling jobs. Buttons: Sumbit job, Confirm video.

        :param session_id: ID of the session in the Video Labeling Tool which controls should be enabled.
        :type session_id: str
        :return: Response from API server in JSON format.
        :rtype: Dict[str, Any]
        """
        return self._act(
            session_id,
            VideoAnnotationToolAction.JOBS_ENABLE_CONTROLS,
            {},
        )

    def set_video(self, session_id: str, video_id: int, frame: Optional[int] = 0) -> Dict[str, Any]:
        """Sets video in the Video Labeling Tool and switches to the specified frame
        if frame number is provided.
        NOTE: Video from the same dataset should be set in the Video Labeling Tool.

        :param session_id: ID of the session in the Video Labeling Tool where video should be set.
        :type session_id: str
        :param video_id: ID of the video in the same dataset which should be set.
        :type video_id: int
        :param frame: Frame number which should be set, defaults to 0.
        :type frame: Optional[int]
        :return: Response from API server in JSON format.
        :rtype: Dict[str, Any]
        """

        return self._act(
            session_id,
            VideoAnnotationToolAction.ENTITIES_SET_INTITY,
            {
                ApiField.ENTITY_ID: video_id,
                ApiField.FRAME: frame,
            },
        )

    def _act(self, session_id: int, action: VideoAnnotationToolAction, payload: dict):
        data = {
            ApiField.SESSION_ID: session_id,
            ApiField.ACTION: str(action),
            ApiField.PAYLOAD: payload,
        }
        resp = self._api.post("/annotation-tool.run-action", data)

        return resp.json()

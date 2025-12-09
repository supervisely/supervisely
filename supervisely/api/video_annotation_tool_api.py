# coding: utf-8

from typing import Any, Dict, Optional, Tuple

from supervisely.api.module_api import ApiField, ModuleApiBase
from supervisely.collection.str_enum import StrEnum


class VideoAnnotationToolAction(StrEnum):
    JOBS_DISABLE_CONTROLS = "jobs/disableControls"
    """"""
    JOBS_ENABLE_CONTROLS = "jobs/enableControls"
    """"""
    JOBS_DISABLE_SUBMIT = "jobs/disableSubmit"
    """"""
    JOBS_ENABLE_SUBMIT = "jobs/enableSubmit"
    """"""
    JOBS_DISABLE_CONFIRM = "jobs/disableConfirm"
    """"""
    JOBS_ENABLE_CONFIRM = "jobs/enableConfirm"
    """"""
    ENTITIES_SET_INTITY = "entities/setEntity"
    """"""
    DIRECT_TRACKING_PROGRESS = "figures/setDirectTrackingProgress"


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

    def disable_submit_button(self, session_id: str) -> Dict[str, Any]:
        """Disables submit button of the labeling jobs.

        :param session_id: ID of the session in the Video Labeling Tool which submit button should be disabled.
        :type session_id: str
        :return: Response from API server in JSON format.
        :rtype: Dict[str, Any]
        """
        return self._act(
            session_id,
            VideoAnnotationToolAction.JOBS_DISABLE_SUBMIT,
            {},
        )

    def enable_submit_button(self, session_id: str) -> Dict[str, Any]:
        """Enables submit button of the labeling jobs.

        :param session_id: ID of the session in the Video Labeling Tool which submit button should be enabled.
        :type session_id: str
        :return: Response from API server in JSON format.
        :rtype: Dict[str, Any]
        """
        return self._act(
            session_id,
            VideoAnnotationToolAction.JOBS_ENABLE_SUBMIT,
            {},
        )

    def disable_confirm_button(self, session_id: str) -> Dict[str, Any]:
        """Disables confirm button of the labeling jobs.

        :param session_id: ID of the session in the Video Labeling Tool which confirm button should be disabled.
        :type session_id: str
        :return: Response from API server in JSON format.
        :rtype: Dict[str, Any]
        """
        return self._act(
            session_id,
            VideoAnnotationToolAction.JOBS_DISABLE_CONFIRM,
            {},
        )

    def enable_confirm_button(self, session_id: str) -> Dict[str, Any]:
        """Enables confirm button of the labeling jobs.

        :param session_id: ID of the session in the Video Labeling Tool which confirm button should be enabled.
        :type session_id: str
        :return: Response from API server in JSON format.
        :rtype: Dict[str, Any]
        """
        return self._act(
            session_id,
            VideoAnnotationToolAction.JOBS_ENABLE_CONFIRM,
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

    def set_direct_tracking_progress(
        self,
        session_id: str,
        video_id: int,
        track_id: str,
        frame_range: Tuple,
        progress_current: int,
        progress_total: int,
    ):
        payload = {
            ApiField.TRACK_ID: track_id,
            ApiField.VIDEO_ID: video_id,
            ApiField.FRAME_RANGE: frame_range,
            ApiField.PROGRESS: {
                ApiField.CURRENT: progress_current,
                ApiField.TOTAL: progress_total,
            },
        }
        return self._act(session_id, VideoAnnotationToolAction.DIRECT_TRACKING_PROGRESS, payload)

    def set_direct_tracking_error(
        self,
        session_id: str,
        video_id: int,
        track_id: str,
        message: str,
    ):
        payload = {
            ApiField.TRACK_ID: track_id,
            ApiField.VIDEO_ID: video_id,
            ApiField.TYPE: "error",
            ApiField.ERROR: {ApiField.MESSAGE: message},
        }
        return self._act(session_id, VideoAnnotationToolAction.DIRECT_TRACKING_PROGRESS, payload)

    def set_direct_tracking_warning(
        self,
        session_id: str,
        video_id: int,
        track_id: str,
        message: str,
    ):
        payload = {
            ApiField.TRACK_ID: track_id,
            ApiField.VIDEO_ID: video_id,
            ApiField.TYPE: "warning",
            ApiField.MESSAGE: message,
        }
        return self._act(session_id, VideoAnnotationToolAction.DIRECT_TRACKING_PROGRESS, payload)

    def _act(self, session_id: int, action: VideoAnnotationToolAction, payload: dict):
        data = {
            ApiField.SESSION_ID: session_id,
            ApiField.ACTION: str(action),
            ApiField.PAYLOAD: payload,
        }
        resp = self._api.post("annotation-tool.run-action", data)

        return resp.json()

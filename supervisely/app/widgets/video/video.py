import copy
from typing import List
import fastapi

import supervisely
from supervisely.app import DataJson
from supervisely.app.widgets import Widget


class Video(Widget):
    class Routes:
        PLAY_CLICKED = "play_clicked_cb"
        STOP_CLICKED = "stop_clicked_cb"
        FRAME_CHANGE_START = "frame_changed_start_cb"
        FRAME_CHANGE_END = "frame_changed_end_cb"

    def __init__(
        self,
        video_id: int = None,
        frame: int = 0,
        intervals: List[List[int]] = [],
        disabled: bool = False,
        widget_id: str = None,
    ):
        self._video_id = video_id
        self._frame = frame
        self._intervals = intervals
        self._disabled = disabled

        self._data = []
        self._annotation: dict = {}

        self._project_meta: supervisely.ProjectMeta = None
        self._loading: bool = False

        #############################
        # video settings
        self._sound_volume: int = 0
        self._playback_rate: int = 0
        self._skip_frames_size: int = 0
        self._intervals_navigation: bool = True
        #############################
        super().__init__(widget_id=widget_id, file_path=__file__)

    # def _generate_project_meta(self):
    #     objects_dict = dict()
    #     annotation: supervisely.Annotation = self._annotation
    #     for label in annotation.labels:
    #         objects_dict[label.obj_class.name] = label.obj_class
    #
    #     objects_list = list(objects_dict.values())
    #     objects_collection = (
    #         supervisely.ObjClassCollection(objects_list)
    #         if len(objects_list) > 0
    #         else None
    #     )
    #
    #     self._project_meta = supervisely.ProjectMeta(obj_classes=objects_collection)
    #     return self._project_meta.to_json()
    #
    # def _update_annotations(self):
    #     annotation = {}
    #     for cell_data in self._data:
    #         annotation[cell_data["cell_uuid"]] = {
    #             "url": cell_data["video_url"],
    #             "figures": [
    #                 label.to_json() for label in cell_data["annotation"].labels
    #             ],
    #             "title": cell_data["title"],
    #         }
    #
    #     self._annotations = copy.deepcopy(annotation)
    #     DataJson()[self.widget_id]["content"]["annotations"] = self._annotations
    #
    # def _update_project_meta(self):
    #     DataJson()[self.widget_id]["content"][
    #         "projectMeta"
    #     ] = self._generate_project_meta()

    # def _update(self):
    #     self._update_annotations()
    #     self._update_project_meta()

    def get_json_data(self):
        return {
            "content": {
                # "projectMeta": self._generate_project_meta(),
                # "annotation": self._annotation,
            },
            "loading": self._loading,
        }

    def get_json_state(self):
        return {
            "intervals": self._intervals,
            "disabled": self._disabled,
            "playPauseFrame": None,
            "inputVideoId": None,
            "videoId": self._video_id,
            "currentFrame": self._frame,
            "options": {
                "soundVolume": self._sound_volume,
                "playbackRate": self._playback_rate,
                "skipFramesSize": self._skip_frames_size,
                "intervalsNavigation": self._intervals_navigation,
            }
        }

    @property
    def video_id(self):
        return self._video_id

    @property
    def loading(self):
        return self._loading

    @loading.setter
    def loading(self, value: bool):
        self._loading = value
        DataJson()[self.widget_id]["loading"] = self._loading
        DataJson().send_changes()

    @property
    def disabled(self):
        return self._disabled

    @disabled.setter
    def disabled(self, value):
        self._disabled = value
        DataJson()[self.widget_id]["disabled"] = self._disabled

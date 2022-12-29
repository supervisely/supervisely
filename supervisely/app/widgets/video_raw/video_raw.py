from supervisely.app import DataJson, StateJson
from supervisely.app.widgets import Widget
from supervisely.api.api import Api
from supervisely.app.widgets_context import JinjaWidgets


class VideoRaw(Widget):
    def __init__(self, video_url: str = None, video_type: str = None, widget_id: str = None):
        self._api = Api()
        self._video_url = video_url
        self._video_type = video_type
        if self._video_type is None:
            self._video_type = "video/mp4"
        self._current_timestamp = 0
        super().__init__(widget_id=widget_id, file_path=__file__)
        JinjaWidgets().context["__widget_scripts__"]["video_raw"] = 'script.js'

    def get_json_data(self):
        return {
            "videoUrl": self._video_url,
            "videoType": self._video_type
        }

    def get_json_state(self):
        return {
            "currentTime": 0,
            "timeToSet": 0,
        }

    @property
    def video_url(self):
        return self._video_url

    @property
    def video_type(self):
        return self._video_type

    def set_video(self, url: str, video_type: str):
        self._video_url = url
        self._video_type = video_type
        DataJson()[self.widget_id]["videoUrl"] = self._video_url
        DataJson()[self.widget_id]["videoType"] = self._video_type
        DataJson().send_changes()
        StateJson()[self.widget_id]["currentTime"] = 0
        StateJson().send_changes()

    def get_current_timestamp(self):
        self._current_timestamp = round(StateJson()[self.widget_id]["currentTime"])
        return self._current_timestamp

    def set_current_timestamp(self, value: int):
        self._current_timestamp = value
        StateJson()[self.widget_id]["timeToSet"] = value
        StateJson().send_changes()
        return self._current_timestamp

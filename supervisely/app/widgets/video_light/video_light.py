from supervisely.app import DataJson, StateJson
from supervisely.app.widgets import Widget
from supervisely.api.api import Api
from supervisely.app.widgets_context import JinjaWidgets


class VideoLight(Widget):
    def __init__(self, url: str = None, mime_type: str = "video/mp4", widget_id: str = None):
        self._api = Api()
        self._url = url
        self._mime_type = mime_type
        self._current_timestamp = 0
        super().__init__(widget_id=widget_id, file_path=__file__)
        JinjaWidgets().context["__widget_scripts__"][
            self.__class__.__name__
        ] = "./sly/css/app/widgets/video_light/script.js"

    def get_json_data(self):
        return {"url": self._url, "mimeType": self._mime_type}

    def get_json_state(self):
        return {
            "currentTime": 0,
            "timeToSet": 0,
        }

    @property
    def url(self):
        return self._url

    @property
    def mime_type(self):
        return self._mime_type

    def set_video(self, url: str, mime_type: str):
        self._url = url
        self._mime_type = mime_type
        DataJson()[self.widget_id]["url"] = self._url
        DataJson()[self.widget_id]["mimeType"] = self._mime_type
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

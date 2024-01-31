from typing import Union

from supervisely.app import DataJson
from supervisely.app.widgets import Widget
from supervisely.app.widgets.video_player.video_player import VideoPlayer
from supervisely.app.widgets_context import JinjaWidgets


class VideoTrackingGates(Widget):
    def __init__(
        self,
        video_url: str = None,
        mime_type: str = "video/mp4",
        disabled: bool = False,
        width: Union[str, int] = 640,
        height: Union[str, int] = 480,
        brush_size: int = 12,
        # lines: List[dict] = None,
        widget_id: str = None,
    ):
        self._video_url = video_url
        self._mime_type = mime_type
        self._video_player = VideoPlayer()
        self._disabled = disabled

        self._regions = []
        self._width = width
        self._height = height
        self._brush_size = brush_size
        # self._lines = lines if lines is not None else []
        # self._widget_width = widget_width
        # self._widget_height = widget_height

        super().__init__(widget_id=widget_id, file_path=__file__)

        script_path = "./sly/css/app/widgets/video_tracking_gates/script.js"
        JinjaWidgets().context["__widget_scripts__"][self.__class__.__name__] = script_path

    def get_json_data(self):
        return {
            "url": self._video_url,
            "mimeType": self._mime_type,
        }

    def get_json_state(self):
        return {
            "brushSize": self._brush_size,
            "height": self._height,
            "width": self._width,
            "outputName": "canvas",
        }

    @property
    def url(self):
        return self._video_player.url

    @property
    def mime_type(self):
        return self._video_player.mime_type

    def set_video(self, url: str, mime_type: str = "video/mp4"):
        self._video_url = url
        self._mime_type = mime_type
        self._video_player.set_video(url, mime_type)
        DataJson()[self.widget_id]["url"] = self._url
        DataJson()[self.widget_id]["mimeType"] = self._mime_type
        DataJson().send_changes()

    def play(self):
        self._video_player.play()

    def pause(self):
        self._video_player.pause()

    def get_current_timestamp(self):
        return self._video_player.get_current_timestamp()

    def set_current_timestamp(self, value: int):
        self._video_player.set_current_timestamp(value)

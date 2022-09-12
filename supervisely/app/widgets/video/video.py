from typing import List

from supervisely.app import DataJson, StateJson
from supervisely.app.widgets import Widget


class Video(Widget):
    class Routes:
        PLAY_CLICKED = "play_clicked_cb"
        PAUSE_CLICKED = "pause_clicked_cb"
        FRAME_CHANGE_START = "frame_changed_start_cb"
        FRAME_CHANGE_END = "frame_changed_end_cb"

    def __init__(
        self,
        video_id: int = None,
        frame: int = 0,
        # intervals: List[List[int]] = [],
        widget_id: str = None,
    ):
        self._video_id = video_id
        self._frame = frame
        self._intervals = []
        self._loading: bool = False
        self._playing: bool = False
        self._changes_handled = False

        #############################
        # video settings
        self._sound_volume: int = 1
        self._playback_rate: int = 1
        self._skip_frames_size: int = 10
        self._intervals_navigation: bool = False
        #############################
        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self):
        return {
            "videoId": self._video_id,
            "intervals": self._intervals,
            "loading": self._loading
        }

    def get_json_state(self):
        return {
            "currentFrame": self._frame,
            "options": {
                "soundVolume": self._sound_volume,
                "playbackRate": self._playback_rate,
                "skipFramesSize": self._skip_frames_size,
                "intervalsNavigation": self._intervals_navigation
            }
        }

    @property
    def video_id(self):
        return self._video_id

    @property
    def frame(self):
        return self._frame

    @frame.setter
    def frame(self, value):
        self._frame = value
        StateJson()[self.widget_id]["currentFrame"] = self._frame
        StateJson().send_changes()

    @property
    def loading(self):
        return self._loading

    @loading.setter
    def loading(self, value: bool):
        self._loading = value
        DataJson()[self.widget_id]["loading"] = self._loading
        DataJson().send_changes()

    @property
    def playing(self):
        return self._playing

    @playing.setter
    def playing(self, value: bool):
        self._playing = value
        DataJson()[self.widget_id]["playing"] = self._playing
        DataJson().send_changes()

    def get_current_frame(self):
        return StateJson()[self.widget_id]["currentFrame"]

    def play_clicked(self, func):
        route_path = self.get_route_path(Video.Routes.PLAY_CLICKED)
        server = self._sly_app.get_server()
        self._playing: bool = True
        self._play_clicked_handled = True
        @server.post(route_path)
        def _click():
            res = self.get_current_frame()
            func(res)
        return _click

    def pause_clicked(self, func):
        route_path = self.get_route_path(Video.Routes.PAUSE_CLICKED)
        server = self._sly_app.get_server()
        self._playing: bool = False
        self._pause_clicked_handled = True
        @server.post(route_path)
        def _click():
            res = self.get_current_frame()
            func(res)
        return _click

    def frame_change_start(self, func):
        route_path = self.get_route_path(Video.Routes.FRAME_CHANGE_START)
        server = self._sly_app.get_server()
        self._frame_changed_start_handled = True
        @server.post(route_path)
        def _click():
            res = self.frame
            if res < 0:
                res = 0
                self.frame = 0
            func(res)
        return _click

    def frame_change_end(self, func):
        route_path = self.get_route_path(Video.Routes.FRAME_CHANGE_END)
        server = self._sly_app.get_server()
        self._frame_changed_end_handled = True
        @server.post(route_path)
        def _click():
            self.frame = self.get_current_frame()
            if self.frame < 0: self.frame = 0
            func(self.frame)
        return _click

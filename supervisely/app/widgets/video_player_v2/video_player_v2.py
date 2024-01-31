from supervisely.annotation.annotation import Annotation
from supervisely.app import DataJson, StateJson
from supervisely.app.widgets import Widget
from supervisely.project.project_meta import ProjectMeta


class Video2(Widget):
    class Routes:
        PLAY_CLICKED = "play_clicked_cb"
        PAUSE_CLICKED = "pause_clicked_cb"
        FRAME_CHANGE_START = "frame_change_started_cb"
        FRAME_CHANGE_END = "frame_change_finished_cb"

    def __init__(
        self,
        url: int = None,
        annotation: Annotation = None,
        project_meta: ProjectMeta = None,
        # intervals: List[List[int]] = [],
        widget_id: str = None,
    ):
        self._video_url = url
        self._annotation = annotation
        self._project_meta = project_meta
        # self._api = Api()
        # self._video_info = None
        # if self._video_id is not None:
        #     self._video_info = self._api.video.get_info_by_id(self._video_id, raise_error=True)

        # self._intervals = []
        # self._loading: bool = False

        self._play_clicked_handled = False
        self._pause_clicked_handled = False
        self._frame_change_started_handled = False
        self._frame_change_finished_handled = False

        #############################
        # video settings
        self._sound_volume: int = 1
        self._playback_rate: int = 1
        self._skip_frames_size: int = 10
        self._intervals_navigation: bool = False
        self._responsive_height: bool = True
        self._enable_zoom: bool = False
        #############################
        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self):
        return {
            "videoUrl": self._video_url,
            "annotation": self._annotation,
            "projectMeta": self._project_meta,
            # "intervals": self._intervals,
            # "loading": self._loading,
        }

    def get_json_state(self):
        return {
            "currentFrame": 0,
            "disabled": False,
            "options": {
                "soundVolume": self._sound_volume,
                "playbackRate": self._playback_rate,
                "skipFramesSize": self._skip_frames_size,
                "intervalsNavigation": self._intervals_navigation,
                "responsiveHeight": self._responsive_height,
                "enableZoom": self._enable_zoom,
            },
        }

    # @property
    # def video_id(self):
    #     return self._video_id

    def set_video(self, url: int, annotation: Annotation = None, project_meta: ProjectMeta = None):
        self._video_url = url
        self._annotation = annotation
        self._project_meta = project_meta
        # self._video_info = self._api.video.get_info_by_id(self._video_id, raise_error=True)
        # DataJson()[self.widget_id]["frames_count"] = self._video_info.frames_count
        DataJson()[self.widget_id]["videoUrl"] = self._video_url
        DataJson()[self.widget_id]["annotation"] = self._annotation
        DataJson()[self.widget_id]["projectMeta"] = self._project_meta
        DataJson().send_changes()
        StateJson()[self.widget_id]["currentFrame"] = 0
        StateJson().send_changes()

    # @property
    # def loading(self):
    #     return self._loading

    # @loading.setter
    # def loading(self, value: bool):
    #     self._loading = value
    #     DataJson()[self.widget_id]["loading"] = self._loading
    #     DataJson().send_changes()

    def set_current_frame(self, value):
        # if self._video_info is None:
        #     raise ValueError("VideoID is not defined yet, use 'set_video' method")
        # if value >= self.get_frames_count():
        #     value = self.get_frames_count() - 1
        StateJson()[self.widget_id]["currentFrame"] = value
        StateJson().send_changes()

    def get_current_frame(self):
        current_frame = StateJson()[self.widget_id]["currentFrame"]
        if not isinstance(current_frame, int):
            current_frame = 0
        return current_frame

    def play_clicked(self, func):
        route_path = self.get_route_path(Video2.Routes.PLAY_CLICKED)
        server = self._sly_app.get_server()
        self._play_clicked_handled = True

        @server.post(route_path)
        def _click():
            res = self.get_current_frame()
            func(res)

        return _click

    def pause_clicked(self, func):
        route_path = self.get_route_path(Video2.Routes.PAUSE_CLICKED)
        server = self._sly_app.get_server()
        self._pause_clicked_handled = True

        @server.post(route_path)
        def _click():
            res = self.get_current_frame()
            func(res)

        return _click

    def frame_change_started(self, func):
        route_path = self.get_route_path(Video2.Routes.FRAME_CHANGE_START)
        server = self._sly_app.get_server()
        self._frame_change_started_handled = True

        @server.post(route_path)
        def _click():
            res = max(0, StateJson()[self.widget_id]["startFrame"])
            func(res)

        return _click

    def frame_change_finished(self, func):
        route_path = self.get_route_path(Video2.Routes.FRAME_CHANGE_END)
        server = self._sly_app.get_server()
        self._frame_change_finished_handled = True

        @server.post(route_path)
        def _click():
            res = self.get_current_frame()
            func(res)

        return _click

    # def get_frames_count(self):
    #     if self._video_info is None:
    #         raise ValueError("VideoID is not defined yet, use 'set_video' method")
    #     return self._video_info.frames_count

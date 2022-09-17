from supervisely.app.widgets import Widget
from supervisely.api.video.video_api import VideoInfo
from supervisely.project.project import Project
from supervisely.video.video import get_labeling_tool_url, get_labeling_tool_link


class VideoThumbnail(Widget):
    def __init__(self, info: VideoInfo, widget_id: str = None):
        self._info = info
        self._description = (
            f"Video length: {info.duration_hms} / {info.frames_count_compact} frames"
        )
        self._url = get_labeling_tool_url(info.dataset_id, info.id)
        self._open_link = get_labeling_tool_link(self._url, "open video")
        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self):
        return {
            "id": self._info.id,
            "name": self._info.name,
            "description": self._description,
            "url": self._url,
            "link": self._open_link,
            "image_preview_url": "",
        }

    def get_json_state(self):
        return None

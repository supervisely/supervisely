from typing import Literal, Optional

from supervisely._utils import abs_url, is_debug_with_sly_net, is_development
from supervisely.api.file_api import FileInfo
from supervisely.app import DataJson
from supervisely.app.widgets import Widget
from supervisely.imaging.color import _validate_hex_color


class ReportThumbnail(Widget):

    def __init__(
        self,
        info: Optional[FileInfo] = None,
        widget_id: Optional[str] = None,
        title: Optional[str] = None,
        color: Optional[str] = "#dcb0ff",
        bg_color: Optional[str] = "#faebff",
        report_type: Literal["model_benchmark", "experiment"] = "model_benchmark",
    ):
        self._id: int = None
        self._info: FileInfo = None
        self._description: str = None
        self._url: str = None
        self._report_type = report_type
        self._set_info(info)
        self._title = title
        self._color = color if _validate_hex_color(color) else "#dcb0ff"
        self._bg_color = bg_color if _validate_hex_color(bg_color) else "#faebff"

        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self):
        return {
            "id": self._id,
            "name": self._title or "Evaluation Report",
            "description": self._description,
            "url": self._url,
            "description": self._description,
            "icon": {
                "className": "zmdi zmdi-assignment",
                "color": self._color,
                "bgColor": self._bg_color,
            },
        }

    def get_json_state(self):
        return None

    def _set_info(self, info: FileInfo = None):
        if info is None:
            self._id: int = None
            self._info: FileInfo = None
            self._description: str = None
            self._url: str = None
            return
        self._id = info.id
        self._info = info
        if self._report_type == "model_benchmark":
            self._description = "Open the Model Benchmark evaluation report."
            lnk = f"/model-benchmark?id={info.id}"
        elif self._report_type == "experiment":
            self._description = "Open the Experiment report."
            lnk = f"/nn/experiments/{info.id}"
        else:
            raise ValueError(f"Invalid report type: {self._report_type}")

        lnk = abs_url(lnk) if is_development() or is_debug_with_sly_net() else lnk
        # self._description = info.path
        self._url = lnk

    def set(
        self,
        info: FileInfo = None,
        report_type: Optional[Literal["model_benchmark", "experiment"]] = None,
    ):
        if report_type is not None:
            self._report_type = report_type
        self._set_info(info)
        self.update_data()
        DataJson().send_changes()

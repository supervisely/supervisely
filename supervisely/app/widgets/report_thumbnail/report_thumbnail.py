import os

from supervisely._utils import abs_url, is_debug_with_sly_net, is_development
from supervisely.api.file_api import FileInfo
from supervisely.app import DataJson
from supervisely.app.widgets import Widget


class ReportThumbnail(Widget):
    def __init__(self, info: FileInfo = None, widget_id: str = None):
        self._id: int = None
        self._info: FileInfo = None
        self._description: str = None
        self._url: str = None
        self._set_info(info)

        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self):
        return {
            "id": self._id,
            "name": "Report",
            "description": self._description,
            "url": self._url,
            "description": self._description,
            "icon": {
                "className": "zmdi zmdi-assignment",
                "color": "#dcb0ff",
                "bgColor": "#faebff",
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
        self._description = "Open the Model Benchmark report for the best model"
        lnk = f"/model-benchmark?id={info.id}"
        lnk = abs_url(lnk) if is_development() or is_debug_with_sly_net() else lnk
        # self._description = info.path
        self._url = lnk

    def set(self, info: FileInfo = None):
        self._set_info(info)
        self.update_data()
        DataJson().send_changes()

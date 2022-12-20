from supervisely.app import DataJson
from supervisely.app.widgets import Widget


class Image(Widget):
    def __init__(
        self,
        url: str = "",
        widget_id: str = None,
    ):
        self._url = url
        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self):
        return {"url": self._url}

    def get_json_state(self):
        return None

    def set(self, url: str):
        self._url = url
        self._update()

    def clean_up(self):
        self._url = ""
        self._update()

    def _update(self):
        DataJson()[self.widget_id]["url"] = self._url
        DataJson().send_changes()

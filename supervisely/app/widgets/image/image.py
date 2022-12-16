from supervisely.app import DataJson
from supervisely.app.widgets import Widget


class Image(Widget):

    def __init__(
        self,
        widget_id: str = None,
    ):

        self._image_url = ""
        self._title = ""
        self._loading = False

        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self):
        return {
            "content": {
                "image_url": self._image_url,
                "title": self._title
            },
            "loading": self._loading,
        }

    def get_json_state(self):
        return {}


    def set_image(
        self,
        image_url: str,
        title: str = "",
    ):

        self._image_url = image_url
        self._title = title

        self._update()

    def clean_up(self):
        self._data = ""
        self._title = ""
        self._update()


    def _update(self):
        DataJson()[self.widget_id]["content"]["image_url"] = self._image_url
        DataJson()[self.widget_id]["content"]["title"] = self._title
        DataJson().send_changes()

    @property
    def loading(self):
        return self._loading

    @loading.setter
    def loading(self, value: bool):
        self._loading = value
        DataJson()[self.widget_id]["loading"] = self._loading
        DataJson().send_changes()

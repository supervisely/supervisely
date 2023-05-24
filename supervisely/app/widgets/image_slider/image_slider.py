from supervisely.app import DataJson, StateJson
from supervisely.app.widgets import Widget
from typing import List


class ImageSlider(Widget):
    def __init__(
        self,
        data: List,
        height: int = 200,
        selectable: bool = False,
        preview_idx: int = 0,
        preview_url: str = None,
        widget_id: str = None,
    ):
        self._data = data
        self._height = height
        self._selectable = selectable
        self._preview_url = preview_url
        self._idx = preview_idx
        self._data_images = []
        self._image_url_to_idx = {}

        if len(self._data) == 0:
            raise ValueError("Input URLs list must not be empty.")

        if self._idx is not None and self._idx >= len(self._data):
            raise ValueError(
                f'"Index {self._idx} can`t be be greater than the length of input URLs list".'
            )

        for image_index, image_url in enumerate(self._data):
            self._data_images.append({"moreExamples": [image_url], "preview": image_url})
            self._image_url_to_idx[image_url] = image_index
            if image_url == self._preview_url:
                self._idx = image_index

        self._preview_url = self._data_images[self._idx]["preview"]
        self._selected = self._data_images[self._idx]["moreExamples"]

        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self):
        return {
            "data": self._data_images,
            "options": {
                "selectable": self._selectable,
                "height": f"{self._height}px",
            },
        }

    def get_json_state(self):
        return {
            "selected": {
                "moreExamples": self._selected,
                "preview": self._preview_url,
                "idx": self._idx,
            }
        }

    def get_preview_url(self):
        self._preview_url = StateJson()[self.widget_id]["selected"]["preview"]
        return self._preview_url

    def set_preview_url(self, value: str):
        self._preview_url = value
        if self._preview_url not in self._image_url_to_idx.keys():
            raise ValueError(f'"There is no {self._preview_url} url in input urls list".')

        set_index = self._image_url_to_idx[self._preview_url]
        StateJson()[self.widget_id]["selected"]["idx"] = set_index
        StateJson()[self.widget_id]["selected"]["preview"] = self._preview_url
        StateJson()[self.widget_id]["selected"]["moreExamples"] = self._preview_url
        StateJson().send_changes()

    def get_preview_idx(self):
        self._idx = StateJson()[self.widget_id]["selected"]["idx"]
        return self._idx

    def set_preview_idx(self, value: int):
        self._idx = value
        if self._idx >= len(self._data):
            raise ValueError(
                f'"Index {self._idx} can`t be be greater than the length of input URLs list".'
            )

        StateJson()[self.widget_id]["selected"]["idx"] = self._idx
        StateJson()[self.widget_id]["selected"]["preview"] = self._data[self._idx]
        StateJson()[self.widget_id]["selected"]["moreExamples"] = self._data[self._idx]
        StateJson().send_changes()

    def get_height(self):
        raw_height = DataJson()[self.widget_id]["options"]["height"]
        self._height = int(raw_height[:-2])
        return self._height

    def set_height(self, value: int):
        if not isinstance(value, int):
            raise TypeError('Argument value type has to be "int" type')
        self._height = value
        DataJson()[self.widget_id]["options"]["height"] = f"{self._height}px"
        DataJson().send_changes()

    @property
    def is_selectable(self):
        self._selectable = DataJson()[self.widget_id]["options"]["selectable"]
        return self._selectable

    def enable_selection(self):
        self._selectable = True
        DataJson()[self.widget_id]["options"]["selectable"] = self._selectable
        DataJson().send_changes()

    def disable_selection(self):
        self._selectable = False
        DataJson()[self.widget_id]["options"]["selectable"] = self._selectable
        DataJson().send_changes()

    def get_data_length(self):
        return len(self._data)

from supervisely.app import DataJson, StateJson
from supervisely.app.widgets import Widget, LabeledImage, Image, Text
from typing import Optional


class CompareImages(Widget):
    def __init__(
        self,
        left: Optional[Image or LabeledImage] = None,
        right: Optional[Image or LabeledImage] = None,
        widget_id: str = None,
    ):
        self._left = left
        self._right = right

        self._check_input_items()

        super().__init__(widget_id=widget_id, file_path=__file__)

    def _check_input_items(self):
        if type(self._left) not in [Image, LabeledImage] and self._left is not None:
            raise TypeError(
                f"Left widget has {type(self._left)} type, Image, LabeledImage only are possible"
            )

        if type(self._right) not in [Image, LabeledImage] and self._right is not None:
            raise TypeError(
                f"Right widget has {type(self._right)} type, Image, LabeledImage only are possible"
            )

    def get_json_data(self):
        # if self._left is None:
        #     self._left_data = None
        # else:
        #     self._left_data = self._left.get_json_data()

        # if self._right is None:
        #     self._right_data = None
        # else:
        #     self._right_data = self._right.get_json_data()

        # return {"left": self._left_data, "right": self._right_data}
        return {}

    def get_json_state(self):
        {}

    def get_left(self):
        return self._left

    def get_right(self):
        return self._right

    def set_left(
        self,
        item: Optional[Image or LabeledImage],
    ):
        self._left = item
        self._check_input_items()
        # DataJson()[self.widget_id]["left"] = self._left.get_json_data()
        # DataJson().send_changes()

    def set_right(
        self,
        item: Optional[Image or LabeledImage],
    ):
        self._right = item
        self._check_input_items()
        # DataJson()[self.widget_id]["right"] = self._right.get_json_data()
        # DataJson().send_changes()

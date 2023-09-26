from supervisely.app import DataJson, StateJson
from supervisely.app.widgets import Widget, LabeledImage, Image
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

        if self._left is None and self._right is None:
            raise TypeError("Both left and right widgets are not set")

        self._correct_input_items()

    def _correct_input_items(self):
        if self._left is None:
            if type(self._right) == LabeledImage:
                self._left = LabeledImage()
            else:
                self._left = Image()

        if self._right is None:
            if type(self._left) == LabeledImage:
                self._right = LabeledImage()
            else:
                self._right = Image()

        if type(self._left) != type(self._right):
            raise TypeError(
                f"You try to compare different content types: {type(self._left)} with {type(self._right)}, check your input data"
            )

    def get_json_data(self):
        return {}

    def get_json_state(self):
        {}

    def get_left(self):
        return self._left

    def get_right(self):
        return self._right

    def set_left(self, *args, **kwargs):
        self._left.set(*args, **kwargs)

    def set_right(self, *args, **kwargs):
        self._right.set(*args, **kwargs)

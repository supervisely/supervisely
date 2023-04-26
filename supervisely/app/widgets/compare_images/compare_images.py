from supervisely.app import DataJson, StateJson
from supervisely.app.widgets import Widget, LabeledImage, Image, Text, Card
from typing import Optional


class CompareImages(Widget):
    def __init__(
        self,
        left: Card = None,
        right: Card = None,
        widget_id: str = None,
    ):
        self._left = left
        self._right = right

        # self._check_input_items()

        super().__init__(widget_id=widget_id, file_path=__file__)

    def _check_input_items(self):
        if type(self._left) is not Card:
            if self._left is not None:
                raise TypeError(f"Left widget has {type(self._left)} type, Card only is possible")
        else:
            self.left_content = self._left._content
            if (
                type(self.left_content) not in [Image, LabeledImage]
                and self.left_content is not None
            ):
                raise TypeError(
                    f"Left widget card content has {type(self.left_content)} type, Image, LabeledImage only are possible"
                )

        if type(self._right) is not Card:
            if self._right is not None:
                raise TypeError(f"Right widget has {type(self._right)} type, Card only is possible")

        else:
            self.right_content = self._right._content
            if (
                type(self.right_content) not in [Image, LabeledImage]
                and self.right_content is not None
            ):
                raise TypeError(
                    f"Right widget card content has {type(self.right_content)} type, Image, LabeledImage only are possible"
                )

        if self._left is None and self._right is None:
            raise TypeError("Both left and right widgets are not set")

        self._correct_input_items()

    def _correct_input_items(self):
        if self._left is None:
            if type(self.right_content) == LabeledImage:
                self.left_content = LabeledImage()
            else:
                self.left_content = Image()
            self._left = Card(content=self.left_content)

        if self._right is None:
            if type(self.left_content) == LabeledImage:
                self.right_content = LabeledImage()
            else:
                self.right_content = Image()
            self._right = Card(content=self.right_content)

        if type(self.left_content) != type(self.right_content):
            raise TypeError(
                f"You try to compare different content types: {type(self.left_content)} with {type(self.right_content)}, check your input data"
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
        self.left_content.set(*args, **kwargs)

    def set_right(self, *args, **kwargs):
        self.right_content.set(*args, **kwargs)

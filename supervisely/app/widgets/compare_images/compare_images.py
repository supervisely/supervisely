from supervisely.app import DataJson, StateJson
from supervisely.app.widgets import Widget, LabeledImage, Image, ImageAnnotationPreview
from typing import Optional


class CompareImages(Widget):
    def __init__(
        self,
        left: Optional[Image or LabeledImage or ImageAnnotationPreview] = None,
        right: Optional[Image or LabeledImage or ImageAnnotationPreview] = None,
        widget_id: str = None,
    ):
        self._set_items(left=left, right=right)
        super().__init__(widget_id=widget_id, file_path=__file__)

    def _set_items(self, left, right):
        self._check_input_items(left=left, right=right)
        if left is None:
            left = LabeledImage() if type(right) == LabeledImage else Image()

        if right is None:
            right = LabeledImage() if type(left) == LabeledImage else Image()

        self._left, self._right = left, right

    def _check_input_items(self, left, right):
        if type(left) not in [Image, LabeledImage, ImageAnnotationPreview, None]:
            raise TypeError(f"Left widget type has to be Image or LabeledImage, got {type(left)}")

        if type(right) not in [Image, LabeledImage, ImageAnnotationPreview, None]:
            raise TypeError(f"Right widget type has to be Image or LabeledImage, got {type(left)}")

        if left is None and right is None:
            raise TypeError("Both left and right widgets are not set")

        if left is not None and right is not None:
            if type(left) != type(right):
                raise TypeError(f"Please provide the same type of widgets for left and right.")

    def get_json_data(self):
        return {}

    def get_json_state(self):
        return {}

    def update_left(self, *args, **kwargs):
        self._left.set(*args, **kwargs)

    def update_right(self, *args, **kwargs):
        self._right.set(*args, **kwargs)

    def clean_up_left(self):
        self._left.clean_up()

    def clean_up_right(self):
        self._right.clean_up()

    def clean_up(self):
        self.clean_up_left()
        self.clean_up_right()

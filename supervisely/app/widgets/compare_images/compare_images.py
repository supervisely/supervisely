from supervisely.app import DataJson
from supervisely.app.widgets import Widget, LabeledImage, Image, Text
from typing import Optional


class CompareImages(Widget):
    def __init__(
        self,
        item_left: Optional[Image or LabeledImage or Text] = Text(
            text="Left image is not selected", status="warning"
        ),
        item_right: Optional[Image or LabeledImage or Text] = Text(
            text="Right image is not selected", status="warning"
        ),
        widget_id: str = None,
    ):
        self._left = item_left
        self._right = item_right

        self._check_input_items()

        super().__init__(widget_id=widget_id, file_path=__file__)

    def _check_input_items(self):
        if type(self._left) not in [Image, LabeledImage, Text]:
            raise TypeError(
                f"Left widget has {type(self._left)} type, Image, LabeledImage, Text only are possible"
            )

        if type(self._right) not in [Image, LabeledImage, Text]:
            raise TypeError(
                f"Right widget has {type(self._right)} type, Image, LabeledImage, Text only are possible"
            )

    def get_json_data(self):
        return {"left": self._left.get_json_data(), "right": self._right.get_json_data()}

    def get_json_state(self):
        return {}

    def set_left(
        self,
        item: Optional[Image or LabeledImage or Text] = Text(
            text="Left image is not set, check your input data", status="warning"
        ),
    ):
        self._left = item
        self._check_input_items()
        DataJson()[self.widget_id] = self._left.get_json_data()
        DataJson().send_changes()

    def set_right(
        self,
        item: Optional[Image or LabeledImage or Text] = Text(
            text="Right image is not set, check your input data", status="warning"
        ),
    ):
        self._right = item
        self._check_input_items()
        DataJson()[self.widget_id] = self._right.get_json_data()
        DataJson().send_changes()

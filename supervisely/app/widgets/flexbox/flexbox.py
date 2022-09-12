from supervisely.app.widgets import Widget
from typing import Dict, List


class Flexbox(Widget):
    # https://www.w3schools.com/css/css3_flexbox.asp
    def __init__(
        self,
        widgets: List[Widget],  # or RadioGroup in future
        widget_id: str = None,
    ):
        self._widgets = widgets
        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self) -> Dict:
        return {}

    def get_json_state(self) -> Dict:
        return {}

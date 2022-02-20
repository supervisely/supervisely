from typing import List
from supervisely.app.widgets import Widget


class RestartDialog(Widget):
    def __init__(
        self,
        steps: List[tuple(str, str)],
        widget_id: str = None,
    ):
        self.steps = steps
        super().__init__(widget_id=widget_id, file_path=__file__)

    def init_data(self):
        return {"steps": {name: endpoint for (name, endpoint) in self.steps}}

    def init_state(self):
        return {"restartName": None}

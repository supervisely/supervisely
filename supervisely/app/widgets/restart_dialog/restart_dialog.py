from typing import List
from fastapi import APIRouter, Request, Depends
from supervisely.app.widgets import Widget
from supervisely.app.content import StateJson


class RestartDialog(Widget):
    def __init__(
        self,
        steps: List,  # (str, callable)
        method_name: str = "/restart",
        widget_id: str = None,
    ):
        self.steps = steps
        self.router = APIRouter()
        self.method_name = method_name
        super().__init__(widget_id=widget_id, file_path=__file__)

        @self.router.post(self.method_name)
        def retart(
            request: Request, state: StateJson = Depends(StateJson.from_request)
        ):
            print(123)
            pass

    def init_data(self):
        return {}  # {"steps": {name: endpoint for (name, endpoint) in self.steps}}

    def init_state(self):
        return {"restartName": None}

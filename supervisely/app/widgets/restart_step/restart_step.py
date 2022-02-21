from typing import List
from fastapi import APIRouter, Request, Depends
from supervisely.app.widgets import Widget
from supervisely.app.content import StateJson, DataJson


class RestartStep(Widget):
    def __init__(
        self,
        steps: List,  # of callable
        method_name: str = "/restart",
        widget_id: str = None,
    ):
        self.steps = steps
        self.router = APIRouter()
        self.method_name = method_name
        super().__init__(widget_id=widget_id, file_path=__file__)

        @self.router.post(self.method_name)
        async def restart(
            request: Request, state: StateJson = Depends(StateJson.from_request)
        ):
            data = DataJson()
            for restart_fn in self.steps:
                restart_fn(data, state)
            state["restart_step"] = None
            await state.synchronize_changes()
            await data.synchronize_changes()

    def init_data(self):
        return {}  # {"steps": {name: endpoint for (name, endpoint) in self.steps}}

    def init_state(self):
        return {"restart_step": None}

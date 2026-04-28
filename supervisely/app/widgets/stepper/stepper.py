from typing import List

from supervisely.app import DataJson, StateJson
from supervisely.app.widgets import Widget


class Stepper(Widget):
    """Step-by-step wizard: displays titles and content for each step; user advances via set_active_step()."""

    def __init__(
        self,
        titles: List = [],
        widgets: List = [],
        active_step: int = 1,
        widget_id: str = None,
    ):
        """:param titles: List of step titles.
        :type titles: List
        :param widgets: List of widgets, one per step.
        :type widgets: List
        :param active_step: Initially active step (1-based).
        :type active_step: int
        :param widget_id: Unique widget identifier.
        :type widget_id: str, optional
        """
        self.titles = titles
        if len(titles) == 0:
            titles = ['' for x in range(len(widgets))]
        self.content = list(zip(titles, widgets))
        self.active_step = active_step
        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self):
        return None

    def get_json_state(self):
        return {'active_step': self.active_step}
    
    def set_active_step(self, value: int):
        self.active_step = value
        StateJson()[self.widget_id]["active_step"] = self.active_step
        StateJson().send_changes()

    def get_active_step(self) -> str:
        return StateJson()[self.widget_id]["active_step"]


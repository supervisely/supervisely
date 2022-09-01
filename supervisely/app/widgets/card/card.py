from typing import List
from supervisely.app import DataJson, StateJson
from supervisely.app.widgets import Widget
from supervisely.sly_logger import logger


class Card(Widget):
    def __init__(
        self,
        title: str = None,
        description: str = None,
        collapsable: bool = False,
        container: Widget = None,
        widget_id: str = None,
    ):
        self._title = title
        self._description = description
        self._collapsable = collapsable
        self._collapsed = False
        self._container = container
        self._options = {"collapsable": self._collapsable, "marginBottom": "0px"}
        self._disabled = {"disabled": False, "message": ""}
        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self):
        return {
            "title": self._title,
            "description": self._description,
            "collapsable": self._collapsable,
            "options": self._options,
        }

    def get_json_state(self):
        return {"disabled": self._disabled, "collapsed": self._collapsed}

    def collapse(self):
        if self._collapsable is False:
            logger.warn(f"Card {self.widget_id} can not be collapsed")
            return
        self._collapsed = True
        StateJson()[self.widget_id]["collapsed"] = self._collapsed
        StateJson().send_changes()

    def uncollapse(self):
        if self._collapsable is False:
            logger.warn(f"Card {self.widget_id} can not be uncollapsed")
            return
        self._collapsed = False
        StateJson()[self.widget_id]["collapsed"] = self._collapsed
        StateJson().send_changes()

    def lock(self, message="Card content is locked"):
        self._disabled = {"disabled": True, "message": message}
        StateJson()[self.widget_id]["disabled"] = self._disabled
        StateJson().send_changes()

    def unlock(self):
        self._disabled["disabled"] = False
        StateJson()[self.widget_id]["disabled"] = self._disabled
        StateJson().send_changes()

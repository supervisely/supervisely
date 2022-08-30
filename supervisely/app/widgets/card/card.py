from typing import List
from supervisely.app import DataJson
from supervisely.app.widgets import Widget


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
        self._options = {"collapsable": self._collapsable}
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

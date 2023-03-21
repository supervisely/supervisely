from supervisely.app import DataJson, StateJson
from supervisely.app.widgets import Widget
from typing import List, Optional, Dict


class Tree(Widget):
    class Routes:
        NODE_CLICK = "node_click"
        CHECK_CHANGE = "check_change"

    def __init__(
        self,
        data: List[Dict] = [],
        label: str = "label",
        children: str = "children",
        disabled: str = "disabled",
        node_key: str = "id",
        show_checkbox: bool = False,
        current_node_key: Optional[str or int] = None,
        empty_text: str = "Input data is void",
        accordion: bool = False,
        widget_id: str = None,
    ):
        self._data = data
        self._label = label
        self._children = children
        self._disabled = disabled
        self._node_key = node_key
        self._show_checkbox = show_checkbox
        self._current_node_key = current_node_key
        self._empty_text = empty_text
        self._accordion = accordion

        self._node_click = False
        self._check_change = False
        self._checked_key = None

        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self):
        return {
            "data": self._data,
            "node_key": self._node_key,
            "show_checkbox": self._show_checkbox,
            "empty_text": self._empty_text,
            "accordion": self._accordion,
            "defaultProps": {
                "label": self._label,
                "children": self._children,
                "disabled": self._disabled,
            },
        }

    def get_json_state(self):
        return {
            "current_node_key": self._current_node_key,
            "checked_key": self._checked_key,
        }

    def get_current_node(self):
        return StateJson()[self.widget_id]["current_node_key"]

    def get_current_check(self):
        return StateJson()[self.widget_id]["checked_key"]

    def get_data(self):
        return DataJson()[self.widget_id]["data"]

    def set_data(self, value: List[Dict]):
        self._data = value
        DataJson()[self.widget_id]["data"] = self._data
        DataJson().send_changes()

    def add_data(self, value: List[Dict]):
        self._data.extend(value)
        DataJson()[self.widget_id]["data"] = self._data
        DataJson().send_changes()

    def set_label(self, value: str):
        self._label = value
        DataJson()[self.widget_id]["label"] = self._label
        DataJson().send_changes()

    def set_children(self, value: str):
        self._children = value
        DataJson()[self.widget_id]["children"] = self._children
        DataJson().send_changes()

    def unable_checkbox(self):
        self._show_checkbox = True
        DataJson()[self.widget_id]["show_checkbox"] = self._show_checkbox
        DataJson().send_changes()

    def disable_checkbox(self):
        self._show_checkbox = False
        DataJson()[self.widget_id]["show_checkbox"] = self._show_checkbox
        DataJson().send_changes()

    def node_click(self, func):
        route_path = self.get_route_path(Tree.Routes.NODE_CLICK)
        server = self._sly_app.get_server()
        self._node_click = True

        @server.post(route_path)
        def _click_node():
            res = self.get_current_node()
            self._current_node_key = res
            func(res)

        return _click_node

    def check_change(self, func1):
        route_path = self.get_route_path(Tree.Routes.CHECK_CHANGE)
        server = self._sly_app.get_server()
        self._check_change = True

        @server.post(route_path)
        def _click_check():
            current_click_id = self.get_current_check()
            func1(current_click_id)

        return _click_check

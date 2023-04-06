from __future__ import annotations
from supervisely.app import DataJson, StateJson
from supervisely.app.widgets import Widget
from typing import List, Optional, Dict

import itertools


class Tree(Widget):
    id: int = itertools.count()

    class Routes:
        NODE_CLICK = "node_click"
        CHECK_CHANGE = "check_change"

    class Children:
        def __init__(self, parent: int, label: str = "", disabled: bool = False) -> Tree.Children:
            self._parent = parent
            self._id = Tree.id.__next__()
            self._label = label
            self._disabled = disabled

        def to_json(self):
            return {
                "parent_id": self._parent_id,
                "id": self._id,
                "label": self._label,
                "disabled": self._disabled,
            }

        def from_json(self, data: dict):
            return Tree.Children(data["label"], data["disabled"])

    class Node:
        def __init__(
            self, label: str, children: List[Tree.Children] = [], disabled: bool = False
        ) -> dict:
            self._id = Tree.id.__next__()
            self._label = label
            self._children = children
            self._disabled = disabled

        def to_json(self):
            res = {
                "id": self._id,
                "label": self._label,
                "children": None,
                "disabled": self._disabled,
            }

            if self._children is not None:
                unpacked_childrens = [child.to_json() for child in self._children]
                res["children"] = unpacked_childrens
            return res

        def from_json(self, json: dict):
            self._label = json["label"]
            self._children = json["children"]
            self._disabled = json["disabled"]
            return Tree.Node(self._label, self._children, self._disabled)

        def get_id(self):
            return self._id

        def add_children(self, children: Tree.Children):
            self._children.append(children)

    def __init__(
        self,
        data: List[Tree.Node] = [],
        node_key: str = "id",
        show_checkbox: bool = False,
        current_node_key: Optional[str or int] = None,
        empty_text: str = "Input data is void",
        accordion: bool = False,
        widget_id: str = None,
    ):
        self._data = data
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
        res = {
            "data": None,
            "node_key": self._node_key,
            "show_checkbox": self._show_checkbox,
            "empty_text": self._empty_text,
            "accordion": self._accordion,
        }

        if self._data is not None:
            res["data"] = [node.to_json() for node in self._data]
        return res

    def get_json_state(self):
        return {
            "current_node_key": self._current_node_key,
            "checked_key": self._checked_key,
        }

    def get_current_node(self):
        node_json = StateJson()[self.widget_id]["current_node_key"]
        return self.get_node(id=node_json["id"])

    def get_current_check(self):
        return StateJson()[self.widget_id]["checked_key"]

    def get_data(self):
        return DataJson()[self.widget_id]["data"]

    def set_data(self, value: List[Dict]):
        self._data = value
        DataJson()[self.widget_id]["data"] = self._data_json
        DataJson().send_changes()

    def add_node(self, value):
        if type(value) is not dict:
            value = value.to_json()
        self._data.append(value)
        DataJson()[self.widget_id]["data"] = self._data_json
        DataJson().send_changes()

    def add_nodes(self, value: List[Dict]):
        self._data.extend(value)
        DataJson()[self.widget_id]["data"] = self._data_json
        DataJson().send_changes()

    def get_node(self, id: int):
        for node in self._data:
            if node.get_id() == id:
                return node

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

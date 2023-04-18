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

    class Node:
        def __init__(
            self, label: str, children: List[Tree.Node] = [], disabled: bool = False
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

        @classmethod
        def from_json(cls, data: dict) -> Tree.Node:
            label = data["label"]
            children = data.get("children", [])
            node_children = []
            if len(children) != 0:
                for child in children:
                    if type(child) is Tree.Node:
                        continue
                    node_children.append(Tree.Node.from_json(child))
            else:
                node_children = children
            disabled = data.get("disabled", False)
            return Tree.Node(label, node_children, disabled)

        def get_id(self):
            return self._id

        def get_children(self):
            return self._children

        def add_children(self, children: Tree.Node):
            self._children = self._children + children
            # self._children.extend(children) - infinite recursion !!!

    def __init__(
        self,
        data: Optional[List[Tree.Node] or List[Dict]] = [],
        node_key: str = "id",
        show_checkbox: bool = False,
        current_node_key: Optional[str or int] = None,
        empty_text: str = "Input data is void",
        accordion: bool = False,
        widget_id: str = None,
    ):
        self._data = []
        if len(data) > 0 and type(data[0]) is dict:
            for curr_data in data:
                self._data.append(Tree.Node.from_json(curr_data))
        else:
            self._data = data
        self._node_key = node_key
        self._show_checkbox = show_checkbox
        self._current_node_key = current_node_key
        self._empty_text = empty_text
        self._accordion = accordion

        self._node_click = False
        self._check_change = False
        self._checked_key = None
        self._data_json = []

        self._all_nodes = []

        self._get_all_nodes()

        super().__init__(widget_id=widget_id, file_path=__file__)

    def _get_all_nodes(self):
        for node in self._data:
            self._all_nodes.append(node)
            self._get_childrens(node)

    def _data_to_json(self):
        return [node.to_json() for node in self._data]

    def _get_childrens(self, curr_node):
        while True:
            curr_children = curr_node.get_children()
            if len(curr_children) == 0:
                break
            else:
                if curr_children[0] in self._all_nodes:
                    break
                for curr_child in curr_children:
                    self._get_childrens(curr_child)
            self._all_nodes.extend(curr_children)

    def get_json_data(self):
        res = {
            "data": None,
            "node_key": self._node_key,
            "show_checkbox": self._show_checkbox,
            "empty_text": self._empty_text,
            "accordion": self._accordion,
        }

        if self._data is not None:
            self._data_json = self._data_to_json()
            res["data"] = self._data_json
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

    def set_data(self, data: Optional[List[Tree.Node] or List[Dict]]):
        if len(data) > 0 and type(data[0]) is dict:
            for curr_data in data:
                self._data.append(Tree.Node.from_json(curr_data))
        else:
            self._data = data

        self._data_json = self._data_to_json()
        self._get_all_nodes()
        DataJson()[self.widget_id]["data"] = self._data_json
        DataJson().send_changes()

    def add_node(self, value: Optional[Tree.Node or Dict]):
        if type(value) is not dict:
            new_node = value
        else:
            new_node = Tree.Node.from_json(value)

        self._data.append(new_node)
        self._data_json = self._data_to_json()
        self._all_nodes.append(new_node)
        self._get_childrens(new_node)
        DataJson()[self.widget_id]["data"] = self._data_json
        DataJson().send_changes()

    def add_nodes(self, values: Optional[List[Tree.Node] or List[Dict]]):
        for value in values:
            self.add_node(value)

    def get_node(self, id: int):
        for node in self._all_nodes:
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

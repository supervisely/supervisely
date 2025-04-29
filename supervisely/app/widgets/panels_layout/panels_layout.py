import random
from typing import Dict, List, Union
from uuid import uuid4

from supervisely._utils import rand_str
from supervisely.app import DataJson, StateJson
from supervisely.app.widgets import Widget


class PanelsLayout(Widget):

    class Panel:
        def __init__(self, name: str, content: Widget, width: int = None, height: int = None):
            self._name = name
            self._content = content
            self._width = width or 100
            self._height = height or 100
            # self._ref = uuid4().hex
            self._ref = rand_str(6)
            # self._ref = random.choice(["panel_1", "panel_2"])

        @property
        def name(self) -> str:
            return self._name

        @property
        def ref(self) -> str:
            return self._ref

        @property
        def content(self) -> Widget:
            return self._content

        def to_json(self) -> dict:
            return {
                "type": "component",
                "width": self._width,
                "height": self._height,
                "componentState": {
                    "tabName": self._name,
                    "ref": self._ref,
                },
                "componentType": "ElementLink",
            }

    class Row:
        def __init__(self, content: list, height: int = None):
            self._content = content
            self._height = height or 100

        def to_json(self) -> dict:
            return {
                "type": "row",
                "height": self._height,
                "content": [item.to_json() for item in self._content],
            }

    class Column:
        def __init__(self, content: list, width: int = None):
            self._content = content
            self._width = width or 100

        def to_json(self) -> dict:
            return {
                "type": "column",
                "width": self._width,
                "content": [item.to_json() for item in self._content],
            }

    class Stack:
        def __init__(self, content: list, width: int = None, height: int = None):
            self._content = content
            self._width = width
            self._height = height

        def to_json(self) -> dict:
            data = {
                "type": "stack",
                "content": [item.to_json() for item in self._content],
            }
            if self._width is not None:
                data["width"] = self._width
            if self._height is not None:
                data["height"] = self._height
            return data

    def __init__(
        self,
        content: list,
        height: Union[str, int] = "100%",
        widget_id: str = None,
    ):
        self._content = content
        if isinstance(height, int):
            height = f"{height}px"
        self._height = height
        self._panels = []  # flat list of panels

        def _get_panels(content):
            for item in content:
                if isinstance(item, self.Panel):
                    self._panels.append(item)
                elif isinstance(item, (self.Row, self.Column, self.Stack)):
                    _get_panels(item._content)

        _get_panels(content)
        """
        Widget for creating a layout with panels (Golden Layout library).

        :param content: Configuration of the layout (recursive structure of rows, columns, and stacks with panels).
        :type content: list
        :param widget_id: Unique identifier for the widget.
        :type widget_id: str
        """
        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self):
        data = {
            "height": self._height,
            "config": {
                "settings": {
                    # "hasHeaders": False,
                    # "reorderEnabled": True,
                    # "selectionEnabled": True,
                    # "popoutWholeStack": False,
                    "showPopoutIcon": False,
                    # "showMaximiseIcon": False,
                    # "showCloseIcon": False,
                },
            },
        }
        data["config"]["content"] = []
        for item in self._content:
            if isinstance(item, (self.Row, self.Column, self.Stack, self.Panel)):
                data["config"]["content"].append(item.to_json())
            else:
                raise ValueError("Content must be a list of Row, Column, Stack, or Panel objects.")
        return data

    def get_json_state(self):
        return {}

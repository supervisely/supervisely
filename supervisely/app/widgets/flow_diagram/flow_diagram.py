from typing import Any, Dict, List, Optional, Union

from supervisely.app.widgets import Widget
from supervisely.app.widgets.flow_node.flow_node import FlowNode
from supervisely.app.widgets_context import JinjaWidgets


class FlowDiagram(Widget):
    """
    A widget for displaying a flow diagram with nodes and connections.
    """

    def __init__(
        self,
        nodes: List[FlowNode] = None,
        connections: List[Dict[str, List[str]]] = None,
        height: Union[str, int] = "500px",
        width: Union[str, int] = "100%",
        widget_id: str = None,
    ):
        self._nodes = nodes or []
        self._connections = connections or {}
        self._height = height
        self._width = width

        super().__init__(widget_id=widget_id, file_path=__file__)

        # Register scripts and styles
        arrow_line = "https://cdn.jsdelivr.net/npm/arrow-line/dist/arrow-line.min.js"
        JinjaWidgets().context["__widget_scripts__"][self.__class__.__name__ + "1"] = arrow_line
        script_path = "./sly/css/app/widgets/flow_diagram/script.js"
        JinjaWidgets().context["__widget_scripts__"][self.__class__.__name__] = script_path

    def get_json_data(self):
        return {
            "height": self._height,
            "width": self._width,
            "nodes": [node.to_json() for node in self._nodes],
            "connections": self._connections,
        }

    def get_json_state(self):
        return {}

from typing import Dict, List, Union

from supervisely.app.widgets import Widget
from supervisely.app.widgets_context import JinjaWidgets


class FlowsView(Widget):
    """
    A widget for displaying a flow diagram with nodes and connections.

    :param nodes: List of nodes to be displayed in the flow diagram.
    :param connections: List of connections between nodes.
    :param height: Height of the widget. Default is "500px".
    :param width: Width of the widget. Default is "100%".

      :Usage:
    .. code-block:: python
    import supervisely_lib as sly
    import supervisely.app.widgets as w

    node_1 = w.FlowsView.Node(
        x=50, y=50, content=w.Card(content=w.Text("This flow processes images using AI."))
    )

    node_2 = w.FlowsView.Node(
        x=200, y=200, content=w.Card("Node 2", content=w.Text("This flow processes images using AI."))
    )

    btn = w.Button("Start", button_size="mini")
    node_3 = w.FlowsView.Node(x=20, y=500, content=btn)


    nodes = [node_1, node_2, node_3]
    connections = {
        node_1.widget_id: [node_2.widget_id, node_3.widget_id],
        node_2.widget_id: [node_3.widget_id],
    }
    flow_diagram = w.FlowsView(nodes=nodes, connections=connections)

    layout = w.Container([flow_diagram])
    app = sly.Application(layout=layout)
    """

    class Node(Widget):
        def __init__(
            self,
            x: int = 0,
            y: int = 0,
            content: Widget = None,
            widget_id: str = None,
        ):
            self.position = {"x": x, "y": y}
            self.x = x
            self.y = y
            self.content = content
            super().__init__(widget_id=widget_id, file_path=__file__)

        def to_json(self):
            return {
                "id": self.widget_id,
                "position": self.position,
                "content_html": self.content.to_html() if self.content else "",
            }

        @property
        def key(self) -> str:
            return f"node-{self.widget_id}"

        def get_json_state(self):
            return self.content.get_json_state() if self.content else {}

        def get_json_data(self):
            return self.content.get_json_data() if self.content else {}

    def __init__(
        self,
        nodes: List[Node] = None,
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

from typing import Any, Dict, List, Literal, Union

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
        connections: Dict[str, List[str]] = None,
        height: Union[str, int] = "500px",
        width: Union[str, int] = "100%",
        arrow_type: Literal["arrow-line", "leader-line"] = "leader-line",
        widget_id: str = None,
    ):
        self._nodes = nodes or []
        self._connections = {}
        self._arrow_type = arrow_type
        if isinstance(connections, dict):
            for k, vs in connections.items():
                if k not in self._connections:
                    self._connections[k] = []
                for v in vs:
                    if type(v) == str:
                        self._connections[k].append([v, self.defualt_connection_options])
                    elif isinstance(v, list):
                        options = self.defualt_connection_options
                        options.update(v[1])
                        self._connections[k].append([v[0], options])
        self._height = height
        self._width = width

        super().__init__(widget_id=widget_id, file_path=__file__)

        # Register scripts and styles
        arrow_line = "https://cdn.jsdelivr.net/npm/arrow-line/dist/arrow-line.min.js"
        JinjaWidgets().context["__widget_scripts__"][self.__class__.__name__ + "1"] = arrow_line
        leader_line = "https://cdn.jsdelivr.net/npm/leader-line@1.0.7/leader-line.min.js"
        JinjaWidgets().context["__widget_scripts__"][self.__class__.__name__ + "2"] = leader_line
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

    @property
    def defualt_connection_options(self) -> Dict[str, Any]:
        if self._arrow_type == "arrow-line":
            options = {
                "thickness": 2,
                "color": "#6199d2",
                "endPlugSize": 1,
                "style": "dot",  # solid, dot, dash, dot-dash
                # endPoint: {type: "circles    ", size: 5, position: "both},
                "endPoint": {
                    "type": "circle",  # circles, squares, arrowHeadFilled, arrowHead
                    "size": 5,
                    "position": "both",  # start, end, both
                },
            }
        else:
            options = {
                "color": "#6199d2",
                "size": 2,
                "path": "straight",  #  straight, grid, arc, magnet, fluid
                "startPlug": "disc",  #  disc, square, arrow1-3, behind, hand
                "startPlugSize": 1,
                "endPlug": "arrow3",  #  disc, square, arrow1-5, behind
                "endPlugSize": 1,
                "middleLabel": "",  # or startLabel, endLabel
                "fontFamily": "JetBrains Mono",
                "fontSize": 12,
                "labelType": "default",
                "startSocket": "bottom",  #  top, right, bottom, left, auto
                "endSocket": "top",  #  top, right, bottom, left, auto
                "dash": True,  #  or {"animation": False, "len": 4, "gap": 8}
            }
        return options.copy()

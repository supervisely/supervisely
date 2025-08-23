import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field

from supervisely.app.content import DataJson, StateJson
from supervisely.app.fastapi import _MainServer
from supervisely.app.fastapi.subapp import Application
from supervisely.app.fastapi.utils import run_sync
from supervisely.app.fastapi.websocket import WebsocketManager
from supervisely.app.widgets import Button, DynamicWidget, Widget
from supervisely.app.widgets_context import JinjaWidgets
from supervisely.io.fs import clean_dir, copy_dir_recursively


class NodeBadge(BaseModel):
    label: str = ""
    value: str = None
    style: Literal["plain", "info", "warning", "error", "success"] = "plain"

    def __init__(self, **data):
        super().__init__(**data)

    @property
    def computed_style(self) -> Dict[str, str]:
        """Returns the computed style based on the badge style type."""
        style_map = {
            "info": {"background": "#1976D2", "color": "#FFFFFF"},
            "warning": {"background": "#FFA000", "color": "#FFFFFF"},
            "error": {"background": "#D32F2F", "color": "#FFFFFF"},
            "success": {"background": "#388E3C", "color": "#FFFFFF"},
        }
        return style_map.get(self.style, {})


class TooltipButton(BaseModel):
    label: str
    link: Optional[Dict[str, str]] = None
    icon: Optional[str] = None

    def __init__(self, **data):
        super().__init__(**data)
        """Post-initialization to convert Button instances to dicts."""
        self.icon = re.sub(r'<i class="(.*?)".*?</i>', r"\1", self.icon) if self.icon else None


class TooltipProperty(BaseModel):
    label: str
    value: str
    link: Optional[Dict[str, str]] = None
    highlight: bool = False


class NodeTooltip(BaseModel):

    description: Optional[str] = ""
    properties: List[TooltipProperty] = Field(default_factory=list)
    buttons: List[TooltipButton] = Field(default_factory=list)


class NodeQueueInfo(BaseModel):
    """Information about the queue."""

    pending: int = 0
    annotating: int = 0
    reviewing: int = 0
    finished: int = 0


class Handle(BaseModel):
    """Represents a handle in the VueFlow node."""

    id: str
    type: Literal["source", "target"] = "source"
    position: Literal["left", "right", "top", "bottom"] = "left"
    label: Optional[str] = None
    connectable: bool = True


class NodeIcon(BaseModel):
    """Represents an icon in the VueFlow node."""

    name: str
    color: Optional[str] = None
    bg_color: Optional[str] = Field(default=None, alias="backgroundColor")


class NodeSettings(BaseModel):
    """Settings for the VueFlow node."""

    type: Literal["project", "action", "queue"] = "action"
    icon: Optional[NodeIcon] = None
    previews: List[Dict[str, str]] = Field(default_factory=list)
    badges: List[NodeBadge] = Field(default_factory=list)
    tooltip: Optional[NodeTooltip] = None
    queue_info: Optional[NodeQueueInfo] = Field(default_factory=NodeQueueInfo, alias="queueInfo")
    handles: List[Handle] = Field(default_factory=list)


class VueFlow(Widget):
    """
    VueFlow widget to wrap VueFlow component.
    This widget is used to create interactive flow diagrams in Supervisely for visualizing workflows.
    """

    class Routes:
        NODE_UPDATED = "node_updated_cb"

    class Edge(BaseModel):
        """Represents an edge in the VueFlow diagram."""

        id: str
        source: str
        target: str
        marker_end: str = Field(default="arrowclosed", alias="markerEnd")
        marker_start: Optional[str] = Field(default=None, alias="markerStart")
        target_handle: Optional[str] = Field(default=None, alias="targetHandle")
        source_handle: Optional[str] = Field(default=None, alias="sourceHandle")
        type: str = "smoothstep"
        label: Optional[str] = None
        animated: bool = False
        curvature: float = 0.5
        style: Optional[Dict[str, Any]] = Field(
            default=None,
            # default_factory=lambda: {
            #     # "stroke": "#B1B1B6",
            #     # "strokeWidth": 1,
            #     "strokeDasharray": "8,8",
            # }
        )

        def to_json(self):
            return self.model_dump(by_alias=True, exclude_none=True)

    class Node:
        """Represents a node in the VueFlow diagram."""

        class Routes:
            NODE_CLICKED = "node_clicked_cb"

        def __init__(
            self,
            id: str,  # * widget ID
            label: str,
            x: Optional[int] = None,
            y: Optional[int] = None,
            position: Optional[Dict[str, int]] = None,
            parent_id: Optional[str] = None,
            data: Optional[NodeSettings] = None,
        ):
            self.id = id
            self.label = label
            if position is not None:
                if "x" not in position or "y" not in position:
                    raise ValueError("Position must contain both 'x' and 'y' keys.")
                self.position = position
            elif x is not None or y is not None:
                self.position = {"x": x, "y": y}
            else:
                raise ValueError("Either position or both x and y must be provided.")
            if data is not None:
                if isinstance(data, dict):
                    self.settings = NodeSettings(**data)
                elif isinstance(data, NodeSettings):
                    self.settings = data
                else:
                    self.settings = NodeSettings()
            self.parent_id = parent_id
            self._server = _MainServer()
            self._click_handled = False

            # ----------------------------------------------------------------
            # --- Connected Nodes --------------------------------------------
            # --- Every node have info about source (parent) nodes -----------
            self._sources = []
            # ----------------------------------------------------------------

        def click(self, func: Callable[[], None]) -> Callable[[], None]:
            """
            Decorator to handle node click events.

            :param func: Function to be called on node click.
            :return: Wrapped function.
            """
            self._click_handled = True
            route_path = f"/{self.parent_id}/{VueFlow.Node.Routes.NODE_CLICKED}"

            @self._server.post(route_path)
            def _click():
                try:
                    func()
                except Exception as e:
                    raise e

            return _click

        def to_json(self):
            """
            Converts the node to a JSON serializable format.
            """
            return {
                "id": self.id,
                "type": "sly-flow",
                "label": self.label,
                "position": self.position,
                "data": self.settings.model_dump(by_alias=True),
            }

        # ------------------------------------------------------------------
        # Tooltip Methods --------------------------------------------------
        # ------------------------------------------------------------------
        def update_property(self, key: str, value: str, link: str = None, highlight: bool = None):
            """Updates the property of the card."""
            for prop in self.settings.tooltip.properties:
                if prop.get("label") == key:
                    prop["value"] = value
                    prop["link"] = {"url": link} if link else None
                    prop["highlight"] = highlight if highlight is not None else False
                    VueFlow.update_node(self)
                    return
            # If property not found, add it
            new_prop = {
                "label": key,
                "value": value,
                "link": {"url": link} if link else None,
                "highlight": highlight if highlight is not None else False,
            }
            self.settings.tooltip.properties.append(new_prop)
            VueFlow.update_node(self)

        def remove_property_by_key(self, key: str, silent: bool = True):
            """Removes the property by key of the card."""
            for idx, prop in enumerate(self.settings.tooltip.properties):
                if prop.get("label") == key:
                    self.settings.tooltip.properties.pop(idx, None)
                    VueFlow.update_node(self)
                    return
            if not silent:
                raise KeyError(f"Property with key '{key}' not found in tooltip properties.")

        # ------------------------------------------------------------------
        # Badge Methods ----------------------------------------------------
        # ------------------------------------------------------------------
        def add_badge(self, badge: Union[dict, NodeBadge]):
            """Adds a badge to the card."""
            if not isinstance(badge, (dict, NodeBadge)):
                raise TypeError("Badge must be an instance of NodeBadge or a dict")
            self.settings.badges.append(badge)
            VueFlow.update_node(self)

        def remove_badge(self, idx: int, silent: bool = True):
            """Removes the badge by index of the card."""
            if not self.settings.badges or idx >= len(self.settings.badges):
                if not silent:
                    raise IndexError("Badge index out of range")
            self.settings.badges.pop(idx, None)
            VueFlow.update_node(self)

        def update_badge(
            self,
            idx: int,
            label: str,
            on_hover: str = None,
            badge_type: Literal["info", "success", "warning", "error"] = None,
            plain: Optional[bool] = None,
        ):
            """Updates the badge by index of the card."""
            if not self.settings.badges or idx >= len(self.settings.badges):
                raise IndexError("Badge index out of range")
            badge = self.settings.badges[idx]
            badge._label = label
            if on_hover is not None:
                badge._on_hover = on_hover
            if plain is not None and plain:
                badge.style = "plain"
            elif badge_type is not None:
                badge.style = badge_type
            VueFlow.update_node(self)

        def update_badge_by_key(
            self,
            key: str,
            label: str,
            badge_type: Literal["info", "success", "warning", "error"] = None,
            new_key: str = None,
            plain: Optional[bool] = None,  # TODO: remove
        ):
            """Updates the badge by key of the card."""
            for badge in self.settings.badges:
                if badge.label == key:
                    badge.value = label
                    if new_key is not None:
                        badge.label = new_key
                    if plain is not None and plain:
                        badge.style = "plain"
                    elif badge_type is not None:
                        badge.style = badge_type
                    VueFlow.update_node(self)
                    return
            # If badge not found, add it
            new_badge = NodeBadge(label=key, value=label, style="info")
            self.add_badge(new_badge)

        def remove_badge_by_key(self, key: str):
            """Removes the badge by key from the card."""
            for idx, badge in enumerate(self.settings.badges):
                if badge.label == key:
                    self.settings.badges.pop(idx, None)
                    VueFlow.update_node(self)
                    return
            raise KeyError(f"Badge with key '{key}' not found in card badges.")

    def __init__(
        self,
        nodes: Optional[List[Node]] = None,
        edges: Optional[List[Edge]] = None,
        widget_id: str = None,
    ):
        self.nodes = nodes if nodes is not None else []
        self.edges = edges if edges is not None else []
        self._url = None
        super().__init__(widget_id=widget_id, file_path=__file__)
        script_path = "./sly/css/app/widgets/vue_flow/script.js"
        JinjaWidgets().context["__widget_scripts__"][self.__class__.__name__] = script_path

        # ! TODO: move vue_flow_ui folder to static folder
        self._prepare_ui_static()

        server = self._sly_app.get_server()
        node_updated_route_path = self.get_route_path(VueFlow.Routes.NODE_UPDATED)

        @server.post(node_updated_route_path)
        def _click(data: Dict):  # {"x": 100, "y": 200}
            try:
                payload = data.get("payload", {})
                node_id = payload.pop("nodeId", None)
                for idx, node in enumerate(StateJson()[self.widget_id]["nodes"]):
                    if node["id"] == node_id:
                        node.update(**payload)  # ? to update data field too
                        new_node = VueFlow.Node(**node, parent_id=self.widget_id)
                        StateJson()[self.widget_id]["nodes"][idx] = new_node.to_json()
                        StateJson().send_changes()
                        self.nodes[idx] = new_node
                        VueFlow.update_node(new_node)
                        break

            except Exception as e:
                raise e

    def get_json_data(self):
        return {
            "nodes": [
                {
                    "type": "action",
                    "className": "sly.solution.CloudImport",
                    "label": "Cloud Import",
                }
            ],
        }

    def get_json_state(self):
        return {
            "nodes": [node.to_json() for node in self.nodes],
            # "nodes": self.nodes,
            "edges": [],  # Edges can be added later if needed
            # "url": f"{self._url}/?showSidebar=true" if self._url is not None else "",
            "url": self._url if self._url is not None else "",
        }

    def add_nodes(self, nodes: List[Node]) -> None:
        """
        Adds nodes to the VueFlow widget.

        :param nodes: List of Node objects to be added.
        """
        serialized_nodes = []
        for node in nodes:
            if isinstance(node, VueFlow.Node):
                node = node.to_json()
            elif not isinstance(node, dict):
                raise TypeError("Each node must be an instance of VueFlow.Node or a dict")
            serialized_nodes.append(node)
        self.nodes.extend(serialized_nodes)
        if "nodes" not in StateJson()[self.widget_id]:
            StateJson()[self.widget_id]["nodes"] = []
        StateJson()[self.widget_id]["nodes"].extend(serialized_nodes)
        StateJson().send_changes()

    def add_node(self, node: Node) -> None:
        """
        Adds a single node to the VueFlow widget.

        :param node: Node object to be added.
        """
        self.add_nodes([node])

    def remove_node(self, node: Union[Node, str]) -> None:
        """
        Removes a node from the VueFlow widget by its ID.

        :param node_id: Widget or node ID to be removed.
        """
        node_id = node.id if isinstance(node, self.Node) else node
        if not isinstance(node_id, str):
            raise ValueError("Node ID must be a string.")
        for idx, node in enumerate(self.nodes):
            if isinstance(node, VueFlow.Node):
                if node.id == node_id:
                    self.nodes.pop(idx)
                    break
            elif isinstance(node, dict):
                if node.get("id") == node_id:
                    self.nodes.pop(idx)
                    break
        else:
            raise ValueError(f"Node with ID '{node_id}' not found.")
        serialized_nodes = []
        for node in self.nodes:
            if isinstance(node, VueFlow.Node):
                node = node.to_json()
            elif not isinstance(node, dict):
                raise TypeError("Each node must be an instance of VueFlow.Node or a dict")
            serialized_nodes.append(node)
        StateJson()[self.widget_id]["nodes"] = serialized_nodes
        StateJson().send_changes()

    def add_edges(self, edges: List[Edge]) -> None:
        """
        Adds edges to the VueFlow widget.

        :param edges: List of Edge objects to be added.
        """
        if "edges" not in StateJson()[self.widget_id]:
            StateJson()[self.widget_id]["edges"] = []
        serialized_edges = []
        for edge in edges:
            if isinstance(edge, VueFlow.Edge):
                edge = edge.to_json()
            elif not isinstance(edge, dict):
                raise TypeError("Each edge must be an instance of VueFlow.Edge or a dict")
            serialized_edges.append(edge)
        self.edges.extend(serialized_edges)
        StateJson()[self.widget_id]["edges"].extend(serialized_edges)
        StateJson().send_changes()

    def add_edge(self, edge: Edge) -> None:
        """
        Adds a single edge to the VueFlow widget.

        :param edge: Edge object to be added.
        """
        self.add_edges([edge])

    def remove_edge(self, edge: Union[Edge, str]) -> None:
        """
        Removes an edge from the VueFlow widget.

        :param edge: Edge object or ID to be removed.
        """
        edge_id = edge.id if isinstance(edge, self.Edge) else edge
        if not isinstance(edge_id, str):
            raise ValueError("Edge ID must be a string.")
        for idx, edge in enumerate(self.edges):
            if isinstance(edge, VueFlow.Edge):
                if edge.id == edge_id:
                    self.edges.pop(idx)
                    break
            elif isinstance(edge, dict):
                if edge.get("id") == edge_id:
                    self.edges.pop(idx)
                    break
        else:
            raise ValueError(f"Edge with ID '{edge_id}' not found.")
        serialized_edges = []
        for edge in self.edges:
            if isinstance(edge, VueFlow.Edge):
                edge = edge.to_json()
            elif not isinstance(edge, dict):
                raise TypeError("Each edge must be an instance of VueFlow.Edge or a dict")
            serialized_edges.append(edge)
        StateJson()[self.widget_id]["edges"] = serialized_edges
        StateJson().send_changes()

    def _prepare_ui_static(self) -> None:
        """
        Prepares the static files for the VueFlow widget.
        This method is called to ensure that the necessary static files are available for the widget.
        """
        # app = Application()
        vue_flow_ui_dir = Path(__file__).parent / "vue_flow_ui"
        # static_dir = Path(app.get_static_dir()) # will not work when initializing app (None)
        static_dir = Path("static")
        new_vue_flow_ui_dir = static_dir / "vue_flow_ui"
        new_vue_flow_ui_dir.mkdir(parents=True, exist_ok=True)
        clean_dir(str(new_vue_flow_ui_dir))
        copy_dir_recursively(str(vue_flow_ui_dir), str(new_vue_flow_ui_dir))
        self._url = f"http://0.0.0.0:8000/{str(new_vue_flow_ui_dir)}/index.html"  # ! TODO: ???
        # self._url = f"{self._url}?showSidebar=true"
        StateJson()[self.widget_id]["url"] = self._url
        StateJson().send_changes()

    def reload(self) -> None:
        """
        Reloads the VueFlow widget to reflect any changes made to the nodes.
        """
        run_sync(
            WebsocketManager().broadcast(
                {
                    "runAction": {
                        "action": f"sly-flow-{self.widget_id}",
                        "payload": {"action": "flow-refresh", "data": {}},
                    }
                }
            )
        )

    @staticmethod
    def update_node(node: Node) -> None:
        """
        Updates a node in the VueFlow widget.

        :param node: Node object or ID to be updated.
        """
        if not isinstance(node, VueFlow.Node):
            raise ValueError("Node must be an instance of VueFlow.Node.")

        for idx, node_json in enumerate(StateJson()[node.parent_id]["nodes"]):
            if node_json["id"] == node.id:
                StateJson()[node.parent_id]["nodes"][idx] = node.to_json()
                StateJson().send_changes()
                break
        run_sync(
            WebsocketManager().broadcast(
                {
                    "runAction": {
                        "action": f"sly-flow-{node.parent_id}",
                        "payload": {"action": "node-update", "data": {"nodeId": node.id}},
                    }
                }
            )
        )

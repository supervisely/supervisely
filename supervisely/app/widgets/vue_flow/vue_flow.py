from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Union

from supervisely._utils import is_development
from supervisely.app.content import DataJson, StateJson
from supervisely.app.fastapi import _MainServer
from supervisely.app.fastapi.subapp import Application
from supervisely.app.fastapi.utils import run_sync
from supervisely.app.fastapi.websocket import WebsocketManager
from supervisely.app.widgets import Widget
from supervisely.app.widgets.vue_flow.edge import Edge as BaseEdge
from supervisely.app.widgets.vue_flow.modal import VueFlowModal
from supervisely.app.widgets.vue_flow.models import NodeLink, NodeSettings
from supervisely.app.widgets.vue_flow.node import Node as BaseNode
from supervisely.app.widgets_context import JinjaWidgets
from supervisely.io.fs import clean_dir, copy_dir_recursively
from supervisely.sly_logger import logger


def _get_payload(data: Dict) -> Dict:
    if isinstance(data, dict):
        return data.get("payload", data)
    return {}


class VueFlow(Widget):
    """
    VueFlow widget to wrap VueFlow component.
    This widget is used to create interactive flow diagrams in Supervisely for visualizing workflows.
    """

    class Routes:
        NODE_ADDED = "node_added_cb"
        NODE_REMOVED = "node_removed_cb"
        NODE_UPDATED = "node_updated_cb"
        EDGE_ADDED = "edge_added_cb"
        EDGE_REMOVED = "edge_removed_cb"
        EDGE_UPDATED = "edge_updated_cb"

    class Edge(BaseEdge):
        """Represents an edge in the VueFlow diagram."""

    class Node(BaseNode):
        """Represents a node in the VueFlow diagram."""

        class Routes:
            NODE_CLICKED = "node_clicked_cb"

        def __init__(self, *args, **kwargs):
            link = kwargs.pop("link", None)

            if hasattr(self, "modal_content") and isinstance(self.modal_content, Widget):
                route_path = f"/{self.id}/{self.Routes.NODE_CLICKED}"
                if isinstance(kwargs.get("data"), NodeSettings):
                    kwargs["data"].link = NodeLink(action=route_path)

            elif link is not None:
                if isinstance(kwargs.get("data"), NodeSettings):
                    kwargs["data"].link = NodeLink(url=link)

            super().__init__(*args, **kwargs)

        def click(self, func: Callable[[], None]) -> Callable[[], None]:
            """
            Decorator to handle node click events.

            :param func: Function to be called on node click.
            :return: Wrapped function.
            """
            self._click_handled = True
            route_path = f"/{self.id}/{self.Routes.NODE_CLICKED}"

            @self._server.post(route_path)
            def _click():
                try:
                    func()
                except Exception as e:
                    raise e

            return _click

        @staticmethod
        def update_node(node):
            return VueFlow.update_node(node)

    def __init__(
        self,
        nodes: Optional[List[Node]] = None,
        edges: Optional[List[Edge]] = None,
        sidebar_nodes: Optional[List[Dict[str, Any]]] = None,
        widget_id: str = None,
        show_sidebar: bool = True,
    ):
        self.nodes = nodes if nodes is not None else []
        self.edges = edges if edges is not None else []
        self._sidebar_nodes = sidebar_nodes if sidebar_nodes is not None else []
        self._url = None
        self._show_sidebar = show_sidebar

        super().__init__(widget_id=widget_id, file_path=__file__)
        self._modal = VueFlowModal(widget_id_prefix=self.widget_id + "_modal")
        script_path = "./sly/css/app/widgets/vue_flow/script.js"
        JinjaWidgets().context["__widget_scripts__"][self.__class__.__name__] = script_path

        # ! TODO: move vue_flow_ui folder to static folder
        # self._prepare_ui_static()

        server = self._sly_app.get_server()
        node_updated_route = self.get_route_path(VueFlow.Routes.NODE_UPDATED)

        @server.post(node_updated_route)
        def _click(data: Dict):
            try:
                payload = data.get("payload", {})
                node_id = payload.pop("nodeId", None)
                for node in self.nodes:
                    if node.id == node_id:
                        node.position = payload.pop("position", node.position)
                        VueFlow.update_node(node)
                        break

            except Exception as e:
                raise e

    def node_added(self, func: Callable[[Dict], None]) -> Callable[[Dict], None]:
        """Decorator to handle node added events."""
        server = self._sly_app.get_server()
        node_added_route = self.get_route_path(self.Routes.NODE_ADDED)

        @server.post(node_added_route)
        def _node_added(node_data: Dict):
            try:
                payload = _get_payload(node_data)
                node_json = payload.get("node", {})
                if not node_json:
                    raise ValueError("Node data is missing in the payload.")
                # Normalize incoming data to expected Node fields
                node_kwargs: Dict[str, Any] = {
                    "id": node_json.get("id"),
                    "label": node_json.get("label", node_json.get("data", {}).get("label", "Node")),
                    "position": node_json.get("position"),
                    "data": node_json.get("data", {}),
                    "parent_id": self.widget_id,
                }
                # Remove unsupported fields inside data to avoid validation errors
                if isinstance(node_kwargs["data"], dict):
                    node_kwargs["data"].pop("initializing", None)
                    node_kwargs["data"].pop("className", None)
                    node_kwargs["data"].pop("label", None)
                    node_kwargs["data"].pop("type", None)

                node_obj = self.Node(**node_kwargs)
                self.add_node(node_obj)
                func(node_obj)
                data = {"nodeId": payload.get("node", {}).get("id")}
                VueFlow.notify_ui(widget_id=self.widget_id, action="node-update", data=data)
            except Exception as e:
                raise e

        return _node_added

    def node_removed(self, func: Callable[[Dict], None]) -> Callable[[Dict], None]:
        """Decorator to handle node removed events."""
        server = self._sly_app.get_server()
        node_removed_route = self.get_route_path(self.Routes.NODE_REMOVED)

        @server.post(node_removed_route)
        def _node_removed(node_data: Dict):
            try:
                payload = _get_payload(node_data)
                node_id = payload.get("nodeId", None)
                self.remove_edges_by_node_id(node_id)
                node = self.pop_node(node_id)
                func(node)
            except Exception as e:
                logger.error(f"Error in node_removed handler: {repr(e)}", exc_info=True)

        return _node_removed

    def edge_added(self, func: Callable[[Dict], None]) -> Callable[[Dict], None]:
        """Decorator to handle edge added events."""
        server = self._sly_app.get_server()
        edge_added_route = self.get_route_path(self.Routes.EDGE_ADDED)

        @server.post(edge_added_route)
        def _edge_added(edge_data: Dict):
            try:
                payload = _get_payload(edge_data)
                edge_json = payload.get("edge", {})
                if not edge_json:
                    raise ValueError("Edge data is missing in the payload.")
                edge_json["id"] = f"edge-{edge_json['source']}-to-{edge_json['target']}"
                edge = VueFlow.Edge(**edge_json)
                self.add_edge(edge)
                func(edge)
                data = {"edgeId": edge.id}
                VueFlow.notify_ui(widget_id=self.widget_id, action="edge-update", data=data)
            except Exception as e:
                raise e

        return _edge_added

    def edge_removed(self, func: Callable[[Dict], None]) -> Callable[[Dict], None]:
        """Decorator to handle edge removed events."""
        server = self._sly_app.get_server()
        edge_removed_route = self.get_route_path(self.Routes.EDGE_REMOVED)

        @server.post(edge_removed_route)
        def _edge_removed(edge_data: Dict):
            try:
                payload = _get_payload(edge_data)
                edge_id = payload.get("edgeId", None)
                edge = self.pop_edge(edge_id)
                func(edge)
            except Exception as e:
                logger.error(f"Error in edge_removed handler: {repr(e)}", exc_info=True)

        return _edge_removed

    @property
    def modal(self) -> VueFlowModal:
        """Returns the modal associated with the VueFlow widget."""
        return self._modal

    def get_json_data(self):
        return {}

    def get_json_state(self):
        url = f"{self._url}?showSidebar={str(self._show_sidebar).lower()}" if self._url else ""
        return {
            "nodes": [node.to_json() for node in self.nodes],
            "edges": [
                edge.to_json() if isinstance(edge, self.Edge) else edge for edge in self.edges
            ],
            "url": url,
            "sidebarNodes": self._sidebar_nodes,
        }

    def add_nodes(self, nodes: List[Node]) -> None:
        """Adds nodes to the VueFlow widget."""
        serialized_nodes = []
        for node in nodes:
            self.nodes.append(node)
            if isinstance(node, VueFlow.Node):
                node = node.to_json()
            else:
                raise TypeError("Node must be an instance of VueFlow.Node")
            serialized_nodes.append(node)
        if "nodes" not in StateJson()[self.widget_id]:
            StateJson()[self.widget_id]["nodes"] = []
        StateJson()[self.widget_id]["nodes"].extend(serialized_nodes)
        StateJson().send_changes()

    def add_node(self, node: Node) -> None:
        """Adds a single node to the VueFlow widget."""
        self.add_nodes([node])

    def pop_node(self, node_id: str) -> None:
        """Removes a node from the VueFlow widget by its ID."""
        if not isinstance(node_id, str):
            raise ValueError("Node ID must be a string.")
        for idx, node in enumerate(self.nodes):
            if node.id == node_id:
                node = self.nodes.pop(idx)
                StateJson()[self.widget_id]["nodes"].pop(idx)
                StateJson().send_changes()
                VueFlow.notify_ui(
                    widget_id=self.widget_id, action="node-remove", data={"nodeIds": [node_id]}
                )
                return node
        else:
            logger.warning(f"Node with ID '{node_id}' not found.")

    def add_edges(self, edges: List[Edge]) -> None:
        """Adds edges to the VueFlow widget."""
        if "edges" not in StateJson()[self.widget_id]:
            StateJson()[self.widget_id]["edges"] = []
        serialized_edges = []
        existing_edge_ids = {edge["id"] for edge in StateJson()[self.widget_id]["edges"]}
        for edge in edges:
            if isinstance(edge, self.Edge):
                edge = edge.to_json()
            else:
                raise TypeError("Edge must be an instance of VueFlow.Edge")
            if edge["id"] in existing_edge_ids:
                raise ValueError(f"Edge with ID '{edge['id']}' already exists.")
            serialized_edges.append(edge)
        self.edges.extend(edges)
        StateJson()[self.widget_id]["edges"].extend(serialized_edges)
        StateJson().send_changes()

    def add_edge(self, edge: Edge) -> None:
        """Adds a single edge to the VueFlow widget."""
        self.add_edges([edge])

    def pop_edge(self, edge_id: str) -> Optional[Edge]:
        """Removes an edge from the VueFlow widget."""
        removed_edges = self.pop_edges([edge_id])
        return removed_edges[0] if removed_edges else None

    def pop_edges(self, edge_ids: List[str]) -> List[Edge]:
        """Removes edges from the VueFlow widget."""
        removed_edges = []
        for edge_id in edge_ids:
            if not isinstance(edge_id, str):
                raise ValueError("Edge ID must be a string.")
            for idx, edge in enumerate(self.edges):
                if edge.id == edge_id:
                    edge = self.edges.pop(idx)
                    StateJson()[self.widget_id]["edges"].pop(idx)
                    StateJson().send_changes()
                    removed_edges.append(edge)
                    data = {"edgeIds": [edge.id for edge in removed_edges]}
                    VueFlow.notify_ui(widget_id=self.widget_id, action="edge-remove", data=data)
                    break
            else:
                logger.warning(f"Edge with ID '{edge_id}' not found.")

        return removed_edges

    def remove_edges_by_node_id(self, node_id: str) -> None:
        """Removes all edges connected to a given node ID."""
        edges_to_remove = [e.id for e in self.edges if e.source == node_id or e.target == node_id]
        self.pop_edges(edges_to_remove)

    def _prepare_ui_static(self, static_dir: str = "static") -> None:
        """
        Prepares the static files for the VueFlow widget.
        This method is called to ensure that the necessary static files are available for the widget.
        """
        self._sly_app
        vue_flow_ui_dir = Path(__file__).parent / "vue_flow_ui"
        logger.info(
            f"VueFlow UI source directory: {vue_flow_ui_dir}.",
            extra={"files": [list(vue_flow_ui_dir.iterdir())]},
        )
        static_dir = Path(static_dir)
        logger.info(f"Preparing VueFlow UI static files in: {static_dir}")

        dst_ui_dir = static_dir / "vue_flow_ui"
        dst_ui_dir.mkdir(parents=True, exist_ok=True)
        clean_dir(str(dst_ui_dir))
        copy_dir_recursively(str(vue_flow_ui_dir), str(dst_ui_dir))
        logger.info(
            f"VueFlow UI static files prepared in: {dst_ui_dir}.",
            extra={"files": [list(dst_ui_dir.iterdir())]},
        )

        self._url = f"{str(dst_ui_dir)}/index.html?showSidebar={str(self._show_sidebar).lower()}"
        if is_development():
            # self._url = f"http://0.0.0.0:8000{self._url}"
            self._url = f"http://localhost:8000/{self._url}"
        StateJson()[self.widget_id]["url"] = self._url
        StateJson().send_changes()

    def reload(self) -> None:
        """Reloads the VueFlow widget to reflect any changes made to the nodes."""
        VueFlow.notify_ui(widget_id=self.widget_id, action="flow-refresh")

    @staticmethod
    def update_node(node: Node) -> None:
        """Updates a node in the VueFlow widget."""
        if not isinstance(node, VueFlow.Node):
            raise ValueError("Node must be an instance of VueFlow.Node.")

        for idx, node_json in enumerate(StateJson()[node.parent_id]["nodes"]):
            if node_json["id"] == node.id:
                StateJson()[node.parent_id]["nodes"][idx] = node.to_json()
                StateJson().send_changes()
                VueFlow.notify_ui(
                    widget_id=node.parent_id, action="node-update", data={"nodeId": node.id}
                )
                break

    @staticmethod
    def notify_ui(widget_id: str, action: str, data: Dict = None) -> None:
        """Sends a notification to the VueFlow widget UI."""
        run_sync(
            WebsocketManager().broadcast(
                {
                    "runAction": {
                        "action": f"sly-flow-{widget_id}",
                        "payload": {"action": action, "data": data or {}},
                    }
                }
            )
        )

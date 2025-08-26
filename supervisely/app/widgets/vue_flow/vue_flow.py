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
            self._click_handled = False
            link = kwargs.pop("link", None)

            if hasattr(self, "modal_content") and isinstance(self.modal_content, Widget):
                if hasattr(self, "_modal") and self._modal is not None:
                    self._click_handled = True
                    route_path = f"/{self.id}/{self.Routes.NODE_CLICKED}"
                    server = _MainServer().get_server()
                    if isinstance(kwargs.get("data"), dict):
                        kwargs["data"]["link"] = {"action": route_path}
                    elif isinstance(kwargs.get("data"), NodeSettings):
                        kwargs["data"].link = NodeLink(action=route_path)

                    @server.post(route_path)
                    def _click():
                        try:
                            self._modal.show()
                            self._modal.loading = True
                            self._modal.set_content(self.modal_content)
                            self._modal.title = self.label
                            self._modal.loading = False
                        except Exception as e:
                            raise e

            elif link is not None:
                if isinstance(kwargs.get("data"), dict):
                    kwargs["data"]["link"] = {"url": link}
                elif isinstance(kwargs.get("data"), NodeSettings):
                    kwargs["data"].link = NodeLink(url=link)

            super().__init__(*args, **kwargs)

        def click(self, func: Callable[[], None]) -> Callable[[], None]:
            """
            Decorator to handle node click events.

            :param func: Function to be called on node click.
            :return: Wrapped function.
            """
            self._click_handled = True
            route_path = f"/{self.parent_id}/{self.Routes.NODE_CLICKED}"

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

        @property
        def modal(self) -> VueFlowModal:
            """Returns the modal associated with this node."""
            return self._modal

        @modal.setter
        def modal(self, modal: VueFlowModal) -> None:
            if not isinstance(modal, VueFlowModal):
                raise ValueError("Modal must be an instance of VueFlowModal.")
            self._modal = modal

    def __init__(
        self,
        nodes: Optional[List[Node]] = None,
        edges: Optional[List[Edge]] = None,
        widget_id: str = None,
        show_sidebar: bool = True,
    ):
        self.nodes = nodes if nodes is not None else []
        self.edges = edges if edges is not None else []
        self._url = None
        self._show_sidebar = show_sidebar

        super().__init__(widget_id=widget_id, file_path=__file__)
        self._modal = VueFlowModal(widget_id_prefix=self.widget_id + "_modal")
        script_path = "./sly/css/app/widgets/vue_flow/script.js"
        JinjaWidgets().context["__widget_scripts__"][self.__class__.__name__] = script_path

        # ! TODO: move vue_flow_ui folder to static folder
        # self._prepare_ui_static()

        server = self._sly_app.get_server()

        def _get_payload(data: Dict) -> Dict:
            if isinstance(data, dict):
                return data.get("payload", data)
            return {}

        node_added_route = self.get_route_path(VueFlow.Routes.NODE_ADDED)
        node_removed_route = self.get_route_path(VueFlow.Routes.NODE_REMOVED)
        node_updated_route = self.get_route_path(VueFlow.Routes.NODE_UPDATED)
        edge_added_route = self.get_route_path(VueFlow.Routes.EDGE_ADDED)
        edge_removed_route = self.get_route_path(VueFlow.Routes.EDGE_REMOVED)
        edge_updated_route = self.get_route_path(VueFlow.Routes.EDGE_UPDATED)

        # -----------------------------------
        # Callbacks -------------------------
        # -----------------------------------
        # Nodes -----------------------------
        # -----------------------------------
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
                node_obj.modal = self.modal
                self.add_node(node_obj)
            except Exception as e:
                raise e

        @server.post(node_removed_route)
        def _node_removed(node_data: Dict):
            try:
                payload = _get_payload(node_data)
                node_id = payload.get("nodeId") or payload.get("id")
                if not node_id:
                    raise ValueError("Node ID is missing in the payload.")
                self.remove_node(node_id)
            except Exception as e:
                raise e

        @server.post(node_updated_route)
        def _click(data: Dict):
            try:
                payload = _get_payload(data)
                node_id = payload.pop("nodeId", None)
                for node in self.nodes:
                    if node.id == node_id:
                        node.position = payload.pop("position", node.position)
                        VueFlow.update_node(node)
                        break
            except Exception as e:
                raise e

        # -----------------------------------
        # Edges -----------------------------
        # -----------------------------------
        @server.post(edge_added_route)
        def _edge_added(edge_data: Dict):  # Edge.to_json()
            try:
                payload = _get_payload(edge_data)
                edge_json = payload.get("edge", {})
                if not edge_json:
                    raise ValueError("Edge data is missing in the payload.")
                # TODO REMOVE: Auto-generate id only if missing
                if "id" not in edge_json or not edge_json["id"]:
                    edge_json["id"] = f"edge-{edge_json['source']}-to-{edge_json['target']}"
                edge = VueFlow.Edge(**edge_json)
                self.add_edge(edge)
            except Exception as e:
                raise e

        @server.post(edge_removed_route)
        def _edge_removed(edge_data: Dict):
            try:
                payload = _get_payload(edge_data)
                edge_id = payload.get("edgeId") or payload.get("id")
                if not edge_id:
                    raise ValueError("Edge ID is missing in the payload.")
                self.remove_edge(edge_id)
            except Exception as e:
                raise e

        @server.post(edge_updated_route)
        def _edge_updated(edge_data: Dict):
            try:
                payload = _get_payload(edge_data)
                edge_json = payload.get("edge", {})
                edge_id = payload.get("edgeId")
                if not edge_json and not edge_id:
                    raise ValueError("Edge data or edgeId is required in the payload.")

                # Build a dict of updates
                updates: Dict[str, Any] = {}
                if edge_json:
                    updates = edge_json
                    edge_id = edge_json.get("id", edge_id)
                # Update in memory edges
                for idx, e in enumerate(StateJson()[self.widget_id].get("edges", [])):
                    if e.get("id") == edge_id:
                        e.update({k: v for k, v in updates.items() if k != "id"})
                        StateJson()[self.widget_id]["edges"][idx] = e
                        break
                # Mirror to self.edges structure
                for idx, e in enumerate(self.edges):
                    if isinstance(e, dict) and e.get("id") == edge_id:
                        self.edges[idx].update({k: v for k, v in updates.items() if k != "id"})
                        break
                StateJson().send_changes()
                # Notify UI
                run_sync(
                    WebsocketManager().broadcast(
                        {
                            "runAction": {
                                "action": f"sly-flow-{self.widget_id}",
                                "payload": {
                                    "action": "edge-update",
                                    "data": {"edge": updates or {"id": edge_id}},
                                },
                            }
                        }
                    )
                )
            except Exception as e:
                raise e

    @property
    def modal(self) -> VueFlowModal:
        """Returns the modal associated with the VueFlow widget."""
        return self._modal

    def get_json_data(self):
        return {}

    def get_json_state(self):
        url = (
            f"{self._url}?showSidebar={str(self._show_sidebar).lower()}"
            if self._url is not None
            else ""
        )
        return {
            "nodes": [node.to_json() for node in self.nodes],
            "edges": [
                edge.to_json() if isinstance(edge, self.Edge) else edge for edge in self.edges
            ],
            "url": url,
            "sidebarNodes": [
                {
                    "type": "action",
                    "className": "sly.solution.AutoImport",
                    "label": "Auto Import",
                },
                {
                    "type": "project",
                    "className": "sly.solution.Project",
                    "label": "Project",
                },
            ],
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
        # @TODO: CHECK AND REMOVE
        # Notify UI about added nodes individually
        for node in serialized_nodes:
            run_sync(
                WebsocketManager().broadcast(
                    {
                        "runAction": {
                            "action": f"sly-flow-{self.widget_id}",
                            "payload": {"action": "node-add", "data": {"node": node}},
                        }
                    }
                )
            )

    def add_node(self, node: Node) -> None:
        """Adds a single node to the VueFlow widget."""
        self.add_nodes([node])

    def remove_node(self, node: Union[Node, str]) -> None:
        """Removes a node from the VueFlow widget by its ID."""
        node_id = node.id if isinstance(node, self.Node) else node
        if not isinstance(node_id, str):
            raise ValueError("Node ID must be a string.")
        for idx, node in enumerate(self.nodes):
            if node.id == node_id:
                self.nodes.pop(idx)
                break
        else:
            raise ValueError(f"Node with ID '{node_id}' not found.")
        StateJson()[self.widget_id]["nodes"].pop(node_id)
        StateJson().send_changes()

        # @TODO:
        # Delete edges
        # Delete subscriptions
        # Delete subscriptions from other nodes to this node

        # Notify UI
        run_sync(
            WebsocketManager().broadcast(
                {
                    "runAction": {
                        "action": f"sly-flow-{self.widget_id}",
                        "payload": {"action": "node-remove", "data": {"nodeId": node_id}},
                    }
                }
            )
        )

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
        # @TODO: CHECK AND REMOVE
        for edge in serialized_edges:
            run_sync(
                WebsocketManager().broadcast(
                    {
                        "runAction": {
                            "action": f"sly-flow-{self.widget_id}",
                            "payload": {"action": "edge-add", "data": {"edge": edge}},
                        }
                    }
                )
            )

    def add_edge(self, edge: Edge) -> None:
        """Adds a single edge to the VueFlow widget."""
        self.add_edges([edge])

    def remove_edge(self, edge: Union[Edge, str]) -> None:
        """Removes an edge from the VueFlow widget."""
        edge_id = edge.id if isinstance(edge, self.Edge) else edge
        if not isinstance(edge_id, str):
            raise ValueError("Edge ID must be a string.")
        for idx, edge in enumerate(self.edges):
            if isinstance(edge, self.Edge):
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
            if isinstance(edge, self.Edge):
                edge = edge.to_json()
            elif not isinstance(edge, dict):
                raise TypeError("Each edge must be an instance of VueFlow.Edge or a dict")
            serialized_edges.append(edge)
        StateJson()[self.widget_id]["edges"] = serialized_edges
        StateJson().send_changes()
        # @TODO: CHECK AND REMOVE
        run_sync(
            WebsocketManager().broadcast(
                {
                    "runAction": {
                        "action": f"sly-flow-{self.widget_id}",
                        "payload": {"action": "edge-remove", "data": {"edgeId": edge_id}},
                    }
                }
            )
        )

    # @TODO: CHECK AND REMOVE
    def refresh_edges(self) -> None:
        """Broadcast the current list of edges to the UI."""
        edges = []
        for edge in StateJson()[self.widget_id].get("edges", []):
            edges.append(edge)
        run_sync(
            WebsocketManager().broadcast(
                {
                    "runAction": {
                        "action": f"sly-flow-{self.widget_id}",
                        "payload": {"action": "edges-refresh", "data": {"edges": edges}},
                    }
                }
            )
        )

    def _prepare_ui_static(self, static_dir: str = "static") -> None:
        """
        Prepares the static files for the VueFlow widget.
        This method is called to ensure that the necessary static files are available for the widget.
        """
        vue_flow_ui_dir = Path(__file__).parent / "vue_flow_ui"
        static_dir = Path(static_dir)

        dst_ui_dir = static_dir / "vue_flow_ui"
        dst_ui_dir.mkdir(parents=True, exist_ok=True)
        clean_dir(str(dst_ui_dir))
        copy_dir_recursively(str(vue_flow_ui_dir), str(dst_ui_dir))

        self._url = f"/{str(dst_ui_dir)}/index.html?showSidebar={str(self._show_sidebar).lower()}"
        if is_development():
            self._url = f"http://localhost:8000{self._url}"
        StateJson()[self.widget_id]["url"] = self._url
        StateJson().send_changes()

    def reload(self) -> None:
        """Reloads the VueFlow widget to reflect any changes made to the nodes."""
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
        """Updates a node in the VueFlow widget."""
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

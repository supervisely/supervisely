import importlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

from supervisely.app.widgets import Dialog, VueFlow
from supervisely.solution.base_node import BaseCardNode, BaseNode
from supervisely.solution.engine.config_parser import YAMLParser


class GraphBuilder(VueFlow):

    class Node(BaseNode):
        pass

    def __init__(
        self,
        nodes: Optional[List[VueFlow.Node]] = None,
        edges: Optional[List[Dict[str, Any]]] = None,
        modals: Optional[List[Dialog]] = None,
        widget_id: Optional[str] = None,
    ):
        self.nodes: Dict[str, VueFlow.Node] = nodes or []
        self.edges: List[Dict[str, Any]] = edges or []
        self._modals: List[Dialog] = modals or []

        sidebar_nodes = []
        from supervisely.solution.nodes import __all__ as all_nodes

        for class_name in all_nodes:
            try:
                module_path = "supervisely.solution.nodes"
                module = importlib.import_module(module_path)
                node_class: BaseNode = getattr(module, class_name)
                if node_class.TITLE is None:
                    continue
                node_info = {
                    "type": node_class.NODE_TYPE,
                    "className": f"{module_path}.{class_name}",
                    "label": node_class.TITLE,
                }
                if node_class.ICON:
                    node_info["icon"] = {
                        "name": node_class.ICON,
                        "color": node_class.ICON_COLOR,
                        "backgroundColor": node_class.ICON_BG_COLOR,
                    }

                sidebar_nodes.append(node_info)
            except Exception as e:
                raise ValueError(f"Error processing node '{class_name}': {repr(e)}")

        super().__init__(
            nodes=self.nodes, edges=self.edges, sidebar_nodes=sidebar_nodes, widget_id=widget_id
        )

        @self.node_added
        def _node_added(node: BaseNode):
            print(f"Node added: {node.id}")

        @self.node_removed
        def _node_removed(node: BaseNode):
            node.disable_publishing(node.id)
            for source in node._sources:
                node.disable_subscriptions(source)
            print(f"Node removed: {node.id}")

        @self.edge_removed
        def _edge_removed(edge):
            # TODO: implement
            print(f"Edge removed: {edge.id}")

        @self.edge_added
        def _edge_added(edge):
            # TODO: implement
            print(f"Edge added: {edge.id}")

    def load_json(self, json_data: Union[str, Path]) -> None:
        if isinstance(json_data, (str, Path)):
            with open(json_data, "r") as f:
                json_data = json.load(f)

        # edges = defaultdict(list)
        nodes = []
        node_id_to_obj = {}
        edges = []

        for node_data in json_data.get("nodes", []):
            node_type = node_data.get("type")
            try:
                module_path, class_name = node_type.rsplit(".", 1)
                module = importlib.import_module(module_path)
                node_class: BaseNode = getattr(module, class_name)
                node = node_class.from_json(node_data, parent_id=self.widget_id)
                if hasattr(node, "modals") and node.modals:
                    self._modals.extend(node.modals)
                nodes.append(node)
                node_id_to_obj[node.id] = node
            except Exception as e:
                raise ValueError(f"Error processing node '{node_type}': {repr(e)}")
        self.add_nodes(nodes)

        to_subscribe = defaultdict(list)
        for edge_data in json_data.get("connections", []):
            edge_id = edge_data.get("id")
            source: str = edge_data.get("source")
            target: str = edge_data.get("target")
            if source not in node_id_to_obj:
                raise ValueError(f"Edge '{edge_id}' has unknown source node id '{source}'")
            if target not in node_id_to_obj:
                raise ValueError(f"Edge '{edge_id}' has unknown target node id '{target}'")
            target_node: BaseNode = node_id_to_obj[target]
            source_node: BaseNode = node_id_to_obj[source]

            target_node._sources.append(source)

            edge = self.Edge(**edge_data)
            for topic, _ in target_node._available_subscribe_methods().items():
                if topic in source_node._available_publish_methods():
                    source_node.enable_publishing(source=source, topic=topic)
                    # target_node.enable_subscriptions(source=source, topic=topic)
                    to_subscribe[source].append((topic, target))
                    edge.source_handle = topic
                    edge.target_handle = topic
                    break
            edges.append(edge)

        for source, items in to_subscribe.items():
            for topic, target in items:
                target_node: BaseNode = node_id_to_obj[target]
                target_node.enable_subscriptions(source=source, topic=topic)

        for node in nodes:
            node.configure_automation()

        self.add_edges(edges)

    def load_yaml(self, yaml_path: Union[str, Path]) -> "GraphBuilder":
        parser = YAMLParser()
        config = parser.load_config(yaml_path)
        if "nodes" not in config:
            raise ValueError("YAML configuration must contain 'nodes' key.")
        return self.load_json(config)

    @property
    def modals(self) -> List[Dialog]:
        """Return a list of modals defined in the graph."""
        return self._modals

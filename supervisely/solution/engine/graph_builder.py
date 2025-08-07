import importlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

from supervisely.app.widgets import Dialog, SolutionGraph
from supervisely.solution.engine.config_parser import YAMLParser


class GraphBuilder(SolutionGraph):

    class Edge:
        def __init__(self, source: str, target: str, settings: Dict[str, Any] = None):
            self.source = source
            self.target = target
            self.settings = {
                "startSocket": "right",
                "endSocket": "left",
                "path": "grid",
                "dash": {"len": 8, "gap": 8},
                "color": "#B1B1B6",
                "size": 1,
                "startPlug": "behind",
                "endPlug": "arrow2",
                "startPlugSize": 3,
                "endPlugSize": 3,
            }
            if settings:
                self.settings.update(settings)

    def __init__(
        self,
        nodes: Optional[List[SolutionGraph.Node]] = None,
        edges: Optional[List[Dict[str, Any]]] = None,
        modals: Optional[List[Dialog]] = None,
    ):
        self.graph = None
        self.nodes: Dict[str, SolutionGraph.Node] = nodes or {}
        self.edges: List[Dict[str, Any]] = edges or defaultdict(list)
        self._modals: List[Dialog] = modals or []

        super().__init__(
            nodes=self.nodes, connections=self.edges, height="100vh", width="100%", show_grid=True
        )

    @classmethod
    def from_json(cls, json_data: Union[str, Path]) -> "GraphBuilder":
        if isinstance(json_data, (str, Path)):
            with open(json_data, "r") as f:
                json_data = json.load(f)

        edges = defaultdict(list)
        nodes = []
        node_id_to_key = {}
        modals = []
        for node_data in json_data:
            node_type = node_data.get("type")
            try:
                module_path, class_name = node_type.rsplit(".", 1)
                module = importlib.import_module(module_path)
                node_class = getattr(module, class_name)
                node = node_class.from_json(node_data)
                if hasattr(node, "modals") and node.modals:
                    modals.extend(node.modals)
                nodes.append(node.node)
                node_id_to_key[node.id] = node.node.key
            except Exception as e:
                raise ValueError(f"Error processing node '{node_type}': {e}")

        for node_data in json_data:
            target_key = node_id_to_key.get(node_data.get("id"))
            for connection in node_data.get("source", []):
                src_key = node_id_to_key.get(connection["id"])
                settings = connection.get("connection_settings", {})
                edge = cls.Edge(source=src_key, target=target_key, settings=settings)
                edges[src_key].append([edge.target, edge.settings])
        return cls(nodes=nodes, edges=edges, modals=modals)

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> "GraphBuilder":
        parser = YAMLParser()
        config = parser.load_config(yaml_path)
        if "nodes" not in config:
            raise ValueError("YAML configuration must contain 'nodes' key.")
        return cls.from_json(config["nodes"])

    @property
    def modals(self) -> List[Dialog]:
        """Return a list of modals defined in the graph."""
        return self._modals

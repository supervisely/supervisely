from collections import defaultdict
from typing import Any, Dict, List, Literal, Union

from supervisely.app.widgets import Container, SolutionGraph
from supervisely.solution.base_node import SolutionElement


class SolutionGraphBuilder:
    """
    Class to create a create graph of nodes and edges for Solution.
    """

    def __init__(
        self,
        height: int = "100vh",
        width: int = "100%",
    ):
        self.height = height
        self.width = width
        self.graph = None

        # Initialize node and edge storage
        self.nodes: Dict[str, SolutionGraph.Node] = {}
        self.edges: List[Dict[str, Any]] = defaultdict(list)
        self.modals: List = []

    def add_node(
        self,
        node: Union[SolutionElement, SolutionGraph.Node],
    ) -> None:
        """
        Add a node to the graph.
        """
        if not isinstance(node, (SolutionElement, SolutionGraph.Node)):
            raise TypeError("node must be an instance of SolutionElement or SolutionGraph.Node")
        if isinstance(node, SolutionElement):
            if hasattr(node, "modals") and node.modals:
                self.modals.extend(node.modals)
            if not hasattr(node, "node"):
                raise ValueError(
                    "SolutionElement must have a 'node' attribute of type SolutionGraph.Node"
                )
            node = node.node

        self.nodes[node.key] = node

    def add_edge(
        self,
        source: Union[SolutionElement, SolutionGraph.Node, str],
        target: Union[SolutionElement, SolutionGraph.Node, str],
        start_socket: Literal["top", "bottom", "left", "right"] = "bottom",
        end_socket: Literal["top", "bottom", "left", "right"] = "top",
        path: Literal["straight", "grid", "arc", "magnet", "fluid"] = "straight",
        dash: Union[Dict, bool] = False,  # {"len": 8, "gap": 8}
        color: str = "#B1B1B6",
        size: int = 1,
        label: Union[str, None] = None,
        font_size: int = 13,
        font_family: str = "JetBrains Mono",
        start_plug: Literal["disc", "square", "arrow1", "arrow2", "arrow3", "behind"] = "behind",
        end_plug: Literal["disc", "square", "arrow1", "arrow2", "arrow3", "behind"] = "arrow2",
        start_plug_size: int = 3,
        end_plug_size: int = 3,
        point_anchor: Union[Dict[str, int], None] = None,
    ) -> None:
        """
        Add an edge to the graph.
        """
        if isinstance(source, (SolutionElement, SolutionGraph.Node)):
            if isinstance(source, SolutionElement):
                if not hasattr(source, "node"):
                    raise ValueError(
                        "SolutionElement must have a 'node' attribute of type SolutionGraph.Node"
                    )
                source = source.node
            source = source.key
        if isinstance(target, (SolutionElement, SolutionGraph.Node)):
            if isinstance(target, SolutionElement):
                if not hasattr(target, "node"):
                    raise ValueError(
                        "SolutionElement must have a 'node' attribute of type SolutionGraph.Node"
                    )
                target = target.node
            target = target.key

        if source not in self.nodes or target not in self.nodes:
            raise ValueError("Both source and target nodes must be added before creating an edge.")

        if dash is True:
            dash = {"len": 8, "gap": 8}

        settings = {
            "startSocket": start_socket,
            "endSocket": end_socket,
            "path": path,
            "dash": dash,
            "color": color,
            "size": size,
            "middleLabel": label,
            "labelType": "default",
            "fontSize": font_size,
            "fontFamily": font_family,
            "startPlug": start_plug,
            "endPlug": end_plug,
            "startPlugSize": start_plug_size,
            "endPlugSize": end_plug_size,
        }
        if point_anchor is not None:
            settings["pointAnchor"] = point_anchor
        edge = [target, settings]
        self.edges[source].append(edge)

    def build(self) -> SolutionGraph:
        """
        Build the SolutionGraph with all nodes and edges added.
        :return: SolutionGraph instance.
        """
        if not self.nodes:
            raise ValueError("No nodes have been added to the graph.")

        # Create the graph with nodes and edges
        self.graph = SolutionGraph(
            nodes=list(self.nodes.values()),
            connections=self.edges,
            height=self.height,
            width=self.width,
            show_grid=True,
        )
        return Container([self.graph, *self.modals])

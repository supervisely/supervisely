from collections import defaultdict
from typing import Any, Dict, List, Literal, Union

from supervisely.app.widgets.solution_graph.solution_graph import SolutionGraph
from supervisely.solution.base_node import SolutionNode, BaseSolutionNode


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

    def add_node(
        self,
        node: Union[BaseSolutionNode, SolutionGraph.Node],
    ) -> None:
        """
        Add a node to the graph.
        """
        if isinstance(node, BaseSolutionNode):
            node = node.node
        if not isinstance(node, SolutionGraph.Node):
            raise TypeError("node must be an instance of SolutionGraph.Node")

        self.nodes[node.key] = node

    def add_edge(
        self,
        source: Union[BaseSolutionNode, SolutionGraph.Node, str],
        target: Union[BaseSolutionNode, SolutionGraph.Node, str],
        start_sockert: Literal["top", "bottom", "left", "right"] = "bottom",
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
    ) -> None:
        """
        Add an edge to the graph.
        """
        if isinstance(source, BaseSolutionNode):
            source = source.node.key
        if isinstance(target, BaseSolutionNode):
            target = target.node.key
        if isinstance(source, SolutionGraph.Node):
            source = source.key
        if isinstance(target, SolutionGraph.Node):
            target = target.key

        if source not in self.nodes or target not in self.nodes:
            raise ValueError("Both source and target nodes must be added before creating an edge.")

        edge = [
            target,
            {
                "startSocket": start_sockert,
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
            },
        ]
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
        return self.graph

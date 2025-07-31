# coding: utf-8

# docs
from __future__ import annotations

from copy import deepcopy
from typing import Dict, List, Optional, Union

import numpy as np

from supervisely.geometry.constants import (
    CLASS_ID,
    CREATED_AT,
    ID,
    LABELER_LOGIN,
    UPDATED_AT,
)
from supervisely.geometry.geometry import Geometry
from supervisely.geometry.graph import EDGES, GraphNodes, Node, _maybe_transform_colors
from supervisely.geometry.point_location import PointLocation
from supervisely.imaging.color import _validate_color, hex2rgb, rgb2hex

VERTICES = "vertices"
CUBOID2D_VERTICES_NAMES = [
    "face1-topleft",
    "face1-topright",
    "face1-bottomright",
    "face1-bottomleft",
    "face2-topleft",
    "face2-topright",
    "face2-bottomright",
    "face2-bottomleft",
]
CUBOID2D_EDGES_MAPPING = [
    CUBOID2D_VERTICES_NAMES[:4],
    CUBOID2D_VERTICES_NAMES[4:],
    ["face1-topleft", "face2-topleft"],
    ["face1-topright", "face2-topright"],
    ["face1-bottomright", "face2-bottomright"],
    ["face1-bottomleft", "face2-bottomleft"],
]


class Cuboid2d(GraphNodes):
    """
    Cuboid2d geometry for a single :class:`Label<supervisely.annotation.label.Label>`. :class:`Cuboid2d<Cuboid2d>` class object is immutable.

    :param nodes: Dict or List containing nodes of graph
    :type nodes: dict
    :param sly_id: Cuboid2d ID in Supervisely server.
    :type sly_id: int, optional
    :param class_id: ID of :class:`ObjClass<supervisely.annotation.obj_class.ObjClass>` to which Cuboid2d belongs.
    :type class_id: int, optional
    :param labeler_login: Login of the user who created Cuboid2d.
    :type labeler_login: str, optional
    :param updated_at: Date and Time when Cuboid2d was modified last. Date Format: Year:Month:Day:Hour:Minute:Seconds. Example: '2021-01-22T19:37:50.158Z'.
    :type updated_at: str, optional
    :param created_at: Date and Time when Cuboid2d was created. Date Format is the same as in "updated_at" parameter.
    :type created_at: str, optional

    :Usage example:

     .. code-block:: python

        import supervisely as sly
        from supervisely.geometry.graph import Node, Cuboid2d

        vertex_1 = Node(sly.PointLocation(5, 5))
        vertex_2 = Node(sly.PointLocation(100, 100))
        vertex_3 = Node(sly.PointLocation(200, 250))
        nodes = {0: vertex_1, 1: vertex_2, 2: vertex_3}
        figure = Cuboid2d(nodes)
    """

    items_json_field = VERTICES

    @staticmethod
    def geometry_name():
        return "cuboid_2d"

    def __init__(
        self,
        nodes: Union[Dict[str, Dict], List],
        sly_id: Optional[int] = None,
        class_id: Optional[int] = None,
        labeler_login: Optional[int] = None,
        updated_at: Optional[str] = None,
        created_at: Optional[str] = None,
        position: Optional[Dict] = None,
        rotation: Optional[Dict] = None,
        dimensions: Optional[Dict] = None,
        face: Optional[List[str]] = None,
    ):
        super().__init__(
            nodes=nodes,
            sly_id=sly_id,
            class_id=class_id,
            labeler_login=labeler_login,
            updated_at=updated_at,
            created_at=created_at,
        )
        self._position = position
        self._rotation = rotation
        self._dimensions = dimensions
        self._face = face

        if len(self._nodes) != 8:
            raise ValueError("Cuboid2d must have exactly 8 vertices")

    @property
    def vertices(self) -> Dict:
        """
        Copy of Cuboid2d vertices.

        :return: Cuboid2d vertices
        :rtype: Optional[Dict]
        """
        return self.nodes

    @property
    def position(self) -> Optional[Dict]:
        """
        Copy of the position of the Cuboid2d.

        :return: Position of the Cuboid2d
        :rtype: Optional[Dict]
        """
        if isinstance(self._position, dict):
            return self._position.copy()

    @property
    def rotation(self) -> Optional[Dict]:
        """
        Copy of the rotation of the Cuboid2d.

        :return: Rotation of the Cuboid2d
        :rtype: Optional[Dict]
        """
        if isinstance(self._rotation, dict):
            return self._rotation.copy()

    @property
    def dimensions(self) -> Optional[Dict]:
        """
        Copy of the dimensions of the Cuboid2d.

        :return: Dimensions of the Cuboid2d
        :rtype: :class:`dict`
        """
        if isinstance(self._dimensions, dict):
            return self._dimensions.copy()

    @property
    def face(self) -> Optional[List[str]]:
        """
        Copy of the face of the Cuboid2d.

        :return: Face of the Cuboid2d
        :rtype: Optional[List[str]]
        """
        if isinstance(self._face, list):
            return self._face.copy()

    @classmethod
    def from_json(cls, data: Dict[str, Dict]) -> Cuboid2d:
        """
        Convert a json dict to Cuboid2d. Read more about `Supervisely format <https://docs.supervisely.com/data-organization/00_ann_format_navi>`_.

        :param data: Cuboid2d in json format as a dict.
        :type data: Dict[str, Dict]
        :return: Cuboid2d object
        :rtype: :class:`Cuboid2d<Cuboid2d>`
        :Usage example:

         .. code-block:: python

            figure_json = {
                "vertices": {
                    "0": {
                        "loc": [5, 5]
                    },
                    "1": {
                        "loc": [100, 100]
                    },
                    "2": {
                        "loc": [250, 200]
                    },
                    "position": {
                        "x": 0.0657651107620552,
                        "y": -0.05634319555373257,
                        "z": 0.7267282757573887
                    },
                    "rotation": { "x": 0, "y": 0, "z": 0 },
                    "dimensions": {
                        "x": 0.1425456564648202,
                        "y": 0.1,
                        "z": 0.36738880874660756
                    },
                    "face": [
                        "face2-topleft",
                        "face2-topright",
                        "face2-bottomright",
                        "face2-bottomleft"
                    ]
                }
            }
            from supervisely.geometry.graph import Cuboid2d
            figure = Cuboid2d.from_json(figure_json)
        """
        nodes = {
            node_id: Node.from_json(node_json)
            for node_id, node_json in data[cls.items_json_field].items()
        }
        labeler_login = data.get(LABELER_LOGIN, None)
        updated_at = data.get(UPDATED_AT, None)
        created_at = data.get(CREATED_AT, None)
        sly_id = data.get(ID, None)
        class_id = data.get(CLASS_ID, None)
        position = data.get("position", None)
        rotation = data.get("rotation", None)
        dimensions = data.get("dimensions", None)
        face = data.get("face", None)
        return cls(
            nodes=nodes,
            sly_id=sly_id,
            class_id=class_id,
            labeler_login=labeler_login,
            updated_at=updated_at,
            created_at=created_at,
            position=position,
            rotation=rotation,
            dimensions=dimensions,
            face=face,
        )

    def to_json(self) -> Dict[str, Dict]:
        """
        Convert the Cuboid2d to list. Read more about `Supervisely format <https://docs.supervisely.com/data-organization/00_ann_format_navi>`_.

        :return: Json format as a dict
        :rtype: Dict[str, Dict]
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            from supervisely.geometry.graph import Node, Cuboid2d

            vertex_1 = Node(sly.PointLocation(5, 5))
            vertex_2 = Node(sly.PointLocation(100, 100))
            vertex_3 = Node(sly.PointLocation(200, 250))
            nodes = {0: vertex_1, 1: vertex_2, 2: vertex_3}
            figure = Cuboid2d(nodes)

            figure_json = figure.to_json()
            print(figure_json)
            # Output: {
            #    "nodes": {
            #        "0": {
            #            "loc": [5, 5]
            #        },
            #        "1": {
            #            "loc": [100, 100]
            #        },
            #        "2": {
            #            "loc": [250, 200]
            #        }
            #    },
            #    "position": {
            #         "x": 0.0657651107620552,
            #         "y": -0.05634319555373257,
            #         "z": 0.7267282757573887
            #     },
            #     "rotation": { "x": 0, "y": 0, "z": 0 },
            #     "dimensions": {
            #         "x": 0.1425456564648202,
            #         "y": 0.1,
            #         "z": 0.36738880874660756
            #     },
            #     "face": [
            #         "face2-topleft",
            #         "face2-topright",
            #         "face2-bottomright",
            #         "face2-bottomleft"
            #     ],
            # }
        """
        res = {
            self.items_json_field: {
                node_id: node.to_json() for node_id, node in self._nodes.items()
            }
        }
        if self._position is not None:
            res["position"] = self._position
        if self._rotation is not None:
            res["rotation"] = self._rotation
        if self._dimensions is not None:
            res["dimensions"] = self._dimensions
        if self._face is not None:
            res["face"] = self._face

        self._add_creation_info(res)
        return res

    def validate(self, name: str, settings: Dict) -> None:
        """
        Checks the graph for correctness and compliance with the template
        """

        missing_nodes = set(settings[self.items_json_field].keys()) - set(self._nodes.keys())
        if len(missing_nodes) > 0:
            raise ValueError(f"Missing vertices in the Cuboid2d: {missing_nodes}.")
        if len(self._nodes) != 8:
            raise ValueError("Cuboid2d must have exactly 8 vertices")

        super().validate(name, settings)

    @staticmethod
    def _transform_config_colors(config, transform_fn):
        """
        Transform colors of edges and nodes in graph template
        :param config: dictionary(graph template)
        :param transform_fn: function to convert
        :return: dictionary(graph template)
        """
        if config is None:
            return None

        result = deepcopy(config)
        _maybe_transform_colors(result.get(EDGES, []), transform_fn)
        _maybe_transform_colors(result[VERTICES].values(), transform_fn)
        return result

    @staticmethod
    def config_from_json(config: Dict) -> Dict:
        """
        Convert graph template from json format
        :param config: dictionary(graph template) in json format
        :return: dictionary(graph template)
        """

        try:
            return Cuboid2d._transform_config_colors(config, hex2rgb)
        except Exception as e:
            raise RuntimeError(
                f"Failed to parse graph template from JSON format. "
                "Check out an example of a graph template in JSON format at: "
                "https://developer.supervisely.com/getting-started/python-sdk-tutorials/images/keypoints#click-to-see-the-example-of-template-in-json-format"
            )

    @staticmethod
    def config_to_json(config: Dict) -> Dict:
        """
        Convert graph template in json format
        :param config: dictionary(graph template)
        :return: dictionary(graph template) in json format
        """
        return Cuboid2d._transform_config_colors(config, rgb2hex)

    @classmethod
    def allowed_transforms(cls):
        """
        allowed_transforms
        """
        from supervisely.geometry.any_geometry import AnyGeometry
        from supervisely.geometry.rectangle import Rectangle

        return [AnyGeometry, Rectangle, GraphNodes]


class Cuboid2dTemplate(Cuboid2d, Geometry):
    """
    Geometry Config Template for a single :class:`Cuboid2d<Cuboid2d>`. :class:`Cuboid2dTemplate<Cuboid2dTemplate>` class object is immutable.
    """

    def __init__(self, color: List[int]):
        _validate_color(color)
        self._point_names = []
        self._config = self._create_template(color)

    def _create_template(self, color: List[int]) -> Cuboid2dTemplate:
        """
        Returns a template for a single :class:`Cuboid2d<Cuboid2d>`.
        """
        config = {VERTICES: {}, EDGES: []}

        x = y = w = h = s = 1  # sample values only for config creation
        base_vertices = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
        shifted_vertices = [(vx + s, vy + s) for vx, vy in base_vertices]
        verices_coords = base_vertices + shifted_vertices

        for label, coords in zip(CUBOID2D_VERTICES_NAMES, verices_coords):
            col, row = coords
            self._point_names.append(label)
            config[VERTICES][label] = {"label": label, "loc": [row, col], "color": color}

        for edges in CUBOID2D_EDGES_MAPPING:
            if len(edges) == 2:
                config[EDGES].append({"src": edges[0], "dst": edges[1], "color": color})
            else:
                for i in range(len(edges)):
                    config[EDGES].append(
                        {"src": edges[i], "dst": edges[(i + 1) % len(edges)], "color": color}
                    )

        return config

    def get_nodes(self):
        self._nodes = {}
        for node in self._config[self.items_json_field]:
            loc = self._config[self.items_json_field][node]["loc"]
            self._nodes[node] = Node(PointLocation(loc[1], loc[0]), label=node)

    def draw(self, image: np.ndarray, thickness=7):
        self.get_nodes()
        self._draw_bool_compatible(
            self._draw_impl,
            bitmap=image,
            color=[0, 255, 0],
            thickness=thickness,
            config=self._config,
        )

    def to_json(self):
        return self.config_to_json(self._config)

    @property
    def config(self):
        return self._config

    @property
    def point_names(self):
        """
        Return point names in order in which they were added
        """
        return self._point_names

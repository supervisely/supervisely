# coding: utf-8

import cv2
from copy import deepcopy

from supervisely_lib.imaging.color import rgb2hex, hex2rgb
from supervisely_lib.io.json import JsonSerializable

from supervisely_lib.geometry.point import Point
from supervisely_lib.geometry.point_location import PointLocation
from supervisely_lib.geometry.rectangle import Rectangle
from supervisely_lib.geometry.geometry import Geometry
from supervisely_lib.geometry.constants import LABELER_LOGIN, CREATED_AT, UPDATED_AT, ID, CLASS_ID


EDGES = 'edges'
NODES = 'nodes'

DISABLED = 'disabled'
LOC = 'loc'

DST = 'dst'
SRC = 'src'
COLOR = 'color'


class Node(JsonSerializable):
    '''
    This is a class for creating and using Nodes
    '''
    def __init__(self, location: PointLocation, disabled=True):
        '''
        :param location: PointLocation class object
        :param disabled: bool
        '''
        self._location = location
        self._disabled = disabled

    @property
    def location(self):
        return self._location

    @property
    def disabled(self):
        return self._disabled

    @classmethod
    def from_json(cls, data):
        '''
        The function from_json convert Node from json format to Node class object.
        :param data: input node in json format
        :return: Node class object
        '''
        # TODO validations
        loc = data[LOC]
        return cls(location=PointLocation(row=loc[1], col=loc[0]), disabled=data.get(DISABLED, False))

    def to_json(self):
        '''
        The function to_json convert node to json format
        :return: node in json format
        '''
        result = {LOC: [self._location.col, self._location.row]}
        if self.disabled:
            result[DISABLED] = True
        return result

    def transform_location(self, transform_fn):
        '''
        :param transform_fn: function to convert location
        :return: Node class object with the changed location attribute using the given function
        '''
        return Node(transform_fn(self._location), disabled=self.disabled)


def _maybe_transform_colors(elements, process_fn):
    '''
    Function _maybe_transform_colors convert some list of parameters using the given function
    :param elements: list of elements
    :param process_fn: function to convert
    '''
    for elem in elements:
        if COLOR in elem:
            elem[COLOR] = process_fn(elem[COLOR])


class GraphNodes(Geometry):
    '''
    This is a class for creating and using GraphNodes
    '''
    @staticmethod
    def geometry_name():
        return 'graph'

    def __init__(self, nodes: dict,
                 sly_id=None, class_id=None, labeler_login=None, updated_at=None, created_at=None):
        '''
        :param nodes: dictionary containing nodes of graph
        '''
        super().__init__(sly_id=sly_id, class_id=class_id, labeler_login=labeler_login, updated_at=updated_at, created_at=created_at)
        self._nodes = nodes

    @property
    def nodes(self):
        return self._nodes.copy()

    @classmethod
    def from_json(cls, data):
        '''
        The function from_json convert GraphNodes from json format to GraphNodes class object.
        :param data: input graph in json format
        :return: GraphNodes class object
        '''
        nodes = {node_id: Node.from_json(node_json) for node_id, node_json in data['nodes'].items()}
        labeler_login = data.get(LABELER_LOGIN, None)
        updated_at = data.get(UPDATED_AT, None)
        created_at = data.get(CREATED_AT, None)
        sly_id = data.get(ID, None)
        class_id = data.get(CLASS_ID, Node)
        return GraphNodes(nodes=nodes, sly_id=sly_id, class_id=class_id,
                          labeler_login=labeler_login, updated_at=updated_at, created_at=created_at)

    def to_json(self):
        '''
        The function to_json convert graph to json format
        :return: graph in json format
        '''
        res = {NODES: {node_id: node.to_json() for node_id, node in self._nodes.items()}}
        self._add_creation_info(res)
        return res

    def crop(self, rect: Rectangle):
        '''
        The function "crop" return list containing graph if all nodes of graph located in given rectangle and an empty list otherwise
        :param rect: Rectangle class object
        :return: list containing GraphNodes class object or empty list
        '''
        is_all_nodes_inside = all(rect.contains_point_location(node.location) for node in self._nodes.values())
        return [self] if is_all_nodes_inside else []

    def relative_crop(self, rect):
        '''
        The function relative_crop calculates new parameters of graph nodes after shifts it with given rectangle(on value of it left top angle)
        :param rect: Rectangle class object
        :return: GraphNodes class object
        '''
        return [geom.translate(drow=-rect.top, dcol=-rect.left) for geom in self.crop(rect)]

    def transform(self, transform_fn):
        '''
        The function "transform" transform graph nodes with given function
        :param transform_fn: function to convert
        :return: GraphNodes class object
        '''
        return GraphNodes(nodes={node_id: transform_fn(node) for node_id, node in self._nodes.items()})

    def transform_locations(self, transform_fn):
        '''
        :param transform_fn: function to convert location
        :return: GraphNodes class object with the changed location attribute of the nodes using the given function
        '''
        return self.transform(lambda kp: kp.transform_location(transform_fn))

    def resize(self, in_size, out_size):
        '''
        The function resize calculates new values of graph nodes after resizing graph
        :param in_size: old image size
        :param out_size: new image size
        :return: GraphNodes class object
        '''
        return self.transform_locations(lambda p: p.resize(in_size, out_size))

    def scale(self, factor):
        '''
        The function scale calculates new values of graph nodes after scaling graph with given factor
        :param factor: float scale parameter
        :return: GraphNodes class object
        '''
        return self.transform_locations(lambda p: p.scale(factor))

    def translate(self, drow, dcol):
        '''
        The function translate calculates new values of graph nodes after shifts it by a certain number of pixels
        :param drow: int
        :param dcol: int
        :return: GraphNodes class object
        '''
        return self.transform_locations(lambda p: p.translate(drow, dcol))

    def rotate(self, rotator):
        '''
        The function rotate calculates new values of graph nodes after rotating graph
        :param rotator: ImageRotator class object
        :return: GraphNodes class object
        '''
        return self.transform_locations(lambda p: p.rotate(rotator))

    def fliplr(self, img_size):
        '''
        The function fliplr calculates new values of graph nodes after fliping graph in horizontal
        :param img_size: tuple or list of integers (image size)
        :return: GraphNodes class object
        '''
        return self.transform_locations(lambda p: p.fliplr(img_size))

    def flipud(self, img_size):
        '''
        The function fliplr calculates new values of graph nodes after fliping graph in vertical
        :param img_size: tuple or list of integers (image size)
        :return: GraphNodes class object
        '''
        return self.transform_locations(lambda p: p.flipud(img_size))

    def _draw_impl(self, bitmap, color, thickness=1, config=None):
        '''
        Draws the graph contour on a given bitmap canvas
        :param bitmap: numpy array
        :param color: tuple or list of integers
        :param thickness: int
        :param config: drawing config specific to a concrete subclass, e.g. per edge colors
        '''
        self.draw_contour(bitmap, color, thickness, config=config)

    @staticmethod
    def _get_nested_or_default(dict, keys_path, default=None):
        result = dict
        for key in keys_path:
            if result is not None:
                result = result.get(key, None)
        return result if result is not None else default

    def _draw_contour_impl(self, bitmap, color=None, thickness=1, config=None):
        if config is not None:
            # If a config with edges and colors is passed, make sure it is
            # consistent with the our set of points.
            self.validate(self.geometry_name(), config)

        # Draw edges first so that nodeas are then drawn on top.
        for edge in self._get_nested_or_default(config, [EDGES], []):
            src = self._nodes.get(edge[SRC], None)
            dst = self._nodes.get(edge[DST], None)
            if (src is not None) and (not src.disabled) and (dst is not None) and (not dst.disabled):
                edge_color = edge.get(COLOR, color)
                cv2.line(bitmap,
                         (src.location.col, src.location.row),
                         (dst.location.col, dst.location.row),
                         tuple(edge_color),
                         thickness)

        nodes_config = self._get_nested_or_default(config, [NODES])
        for node_id, node in self._nodes.items():
            if not node.disabled:
                effective_color = self._get_nested_or_default(nodes_config, [node_id, COLOR], color)
                Point.from_point_location(node.location).draw(
                    bitmap=bitmap, color=effective_color, thickness=thickness, config=None)

    @property
    def area(self):
        return 0.0

    def to_bbox(self):
        '''
        The function to_bbox create Rectangle class object from current GraphNodes class object
        :return: Rectangle class object
        '''
        return Rectangle.from_geometries_list(
            [Point.from_point_location(node.location) for node in self._nodes.values()])

    def clone(self):
        return self

    def validate(self, name, settings):
        '''
        Checks the graph for correctness and compliance with the template
        '''
        super().validate(name, settings)
        # TODO template self-consistency checks.

        nodes_not_in_template = set(self._nodes.keys()) - set(settings[NODES].keys())
        if len(nodes_not_in_template) > 0:
            raise ValueError('Graph contains nodes not declared in the template: {!r}.'.format(nodes_not_in_template))

    @staticmethod
    def _transform_config_colors(config, transform_fn):
        '''
        Transform colors of edges and nodes in graph template
        :param config: dictionary(graph template)
        :param transform_fn: function to convert
        :return: dictionary(graph template)
        '''
        if config is None:
            return None

        result = deepcopy(config)
        _maybe_transform_colors(result.get(EDGES, []), transform_fn)
        _maybe_transform_colors(result[NODES].values(), transform_fn)
        return result

    @staticmethod
    def config_from_json(config):
        '''
        Convert graph template from json format
        :param config: dictionary(graph template) in json format
        :return: dictionary(graph template)
        '''
        return GraphNodes._transform_config_colors(config, hex2rgb)

    @staticmethod
    def config_to_json(config):
        '''
        Convert graph template in json format
        :param config: dictionary(graph template)
        :return: dictionary(graph template) in json format
        '''
        return GraphNodes._transform_config_colors(config, rgb2hex)

    @classmethod
    def allowed_transforms(cls):
        from supervisely_lib.geometry.any_geometry import AnyGeometry
        from supervisely_lib.geometry.rectangle import Rectangle
        return [AnyGeometry, Rectangle]

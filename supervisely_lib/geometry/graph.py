# coding: utf-8

import cv2
from copy import deepcopy

from supervisely_lib.imaging.color import rgb2hex, hex2rgb
from supervisely_lib.io.json import JsonSerializable

from supervisely_lib.geometry.point import Point
from supervisely_lib.geometry.point_location import PointLocation
from supervisely_lib.geometry.rectangle import Rectangle
from supervisely_lib.geometry.geometry import Geometry


EDGES = 'edges'
NODES = 'nodes'

DISABLED = 'disabled'
LOC = 'loc'

DST = 'dst'
SRC = 'src'
COLOR = 'color'


class Node(JsonSerializable):
    def __init__(self, location: PointLocation, disabled=True):
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
        # TODO validations
        loc = data[LOC]
        return cls(location=PointLocation(row=loc[1], col=loc[0]), disabled=data.get(DISABLED, False))

    def to_json(self):
        result = {LOC: [self._location.col, self._location.row]}
        if self.disabled:
            result[DISABLED] = True
        return result

    def transform_location(self, transform_fn):
        return Node(transform_fn(self._location), disabled=self.disabled)


def _maybe_transform_colors(elements, process_fn):
    for elem in elements:
        if COLOR in elem:
            elem[COLOR] = process_fn(elem[COLOR])


class GraphNodes(Geometry):
    @staticmethod
    def geometry_name():
        return 'graph'

    def __init__(self, nodes: dict):
        self._nodes = nodes

    @property
    def nodes(self):
        return self._nodes.copy()

    @classmethod
    def from_json(cls, data):
        nodes = {node_id: Node.from_json(node_json) for node_id, node_json in data['nodes'].items()}
        return GraphNodes(nodes=nodes)

    def to_json(self):
        return {NODES: {node_id: node.to_json() for node_id, node in self._nodes.items()}}

    def crop(self, rect: Rectangle):
        is_all_nodes_inside = all(rect.contains_point_location(node.location) for node in self._nodes.values())
        return [self] if is_all_nodes_inside else []

    def relative_crop(self, rect):
        return [geom.translate(drow=-rect.top, dcol=-rect.left) for geom in self.crop(rect)]

    def transform(self, transform_fn):
        return GraphNodes(nodes={node_id: transform_fn(node) for node_id, node in self._nodes.items()})

    def transform_locations(self, transform_fn):
        return self.transform(lambda kp: kp.transform_location(transform_fn))

    def resize(self, in_size, out_size):
        return self.transform_locations(lambda p: p.resize(in_size, out_size))

    def scale(self, factor):
        return self.transform_locations(lambda p: p.scale(factor))

    def translate(self, drow, dcol):
        return self.transform_locations(lambda p: p.translate(drow, dcol))

    def rotate(self, rotator):
        return self.transform_locations(lambda p: p.rotate(rotator))

    def fliplr(self, img_size):
        return self.transform_locations(lambda p: p.fliplr(img_size))

    def flipud(self, img_size):
        return self.transform_locations(lambda p: p.flipud(img_size))

    def _draw_impl(self, bitmap, color, thickness=1, config=None):
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
        return Rectangle.from_geometries_list(
            [Point.from_point_location(node.location) for node in self._nodes.values()])

    def clone(self):
        return self

    def validate(self, name, settings):
        super().validate(name, settings)
        # TODO template self-consistency checks.

        nodes_not_in_template = set(self._nodes.keys()) - set(settings[NODES].keys())
        if len(nodes_not_in_template) > 0:
            raise ValueError('Graph contains nodes not declared in the template: {!r}.'.format(nodes_not_in_template))

    @staticmethod
    def _transform_config_colors(config, transform_fn):
        if config is None:
            return None

        result = deepcopy(config)
        _maybe_transform_colors(result.get(EDGES, []), transform_fn)
        _maybe_transform_colors(result[NODES].values(), transform_fn)
        return result

    @staticmethod
    def config_from_json(config):
        return GraphNodes._transform_config_colors(config, hex2rgb)

    @staticmethod
    def config_to_json(config):
        return GraphNodes._transform_config_colors(config, rgb2hex)


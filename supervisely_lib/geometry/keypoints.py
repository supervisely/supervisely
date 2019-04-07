# coding: utf-8

from typing import Dict, List
from collections import namedtuple
from copy import deepcopy
from supervisely_lib.io.json import JsonSerializable

import cv2
import numpy as np

from supervisely_lib.geometry.rectangle import Rectangle
from supervisely_lib.geometry.point_location import PointLocation
from supervisely_lib.geometry.geometry import Geometry
from supervisely_lib.geometry.constants import NODES, EDGES, ENABLED, LABEL, COLOR, POINTS, TEMPLATE, LOCATION
from supervisely_lib.imaging.color import hex2rgb, rgb2hex


PointTemplate = namedtuple('PointTemplate', ['label', 'color'])
EdgeTemplate = namedtuple('EdgeTemplate', ['label', 'color', 'start', 'end'])
Keypoint = namedtuple('Keypoint', ['location', 'enabled'])


class KeypointsTemplate(JsonSerializable):
    def __init__(self, points: List[PointTemplate], edges: List[EdgeTemplate]):
        self.points = points
        self.edges = edges

    @classmethod
    def from_json(cls, data):
        points = []
        for point_obj in data[POINTS]:
            points.append(PointTemplate(label=point_obj[LABEL], color=hex2rgb(point_obj[COLOR])))

        edges = []
        for edge_obj in data[EDGES]:
            edges.append(EdgeTemplate(label=edge_obj[LABEL],
                                      color=hex2rgb(edge_obj[COLOR]),
                                      start=edge_obj[POINTS][0],
                                      end=edge_obj[POINTS][1]
                                      ))

        return cls(points=points, edges=edges)

    def to_json(self):
        points_json = []
        for point in self.points:
            points_json.append({
                LABEL: point.label,
                COLOR: rgb2hex(point.color)
            })

        edges_json = []
        for edge in self.edges:
            edges_json.append({
                LABEL: edge.label,
                COLOR: rgb2hex(edge.color),
                POINTS: [edge.start, edge.end]
            })

        packed_obj = {
            POINTS: points_json,
            EDGES: edges_json
        }
        return packed_obj


class Keypoints(Geometry):
    @staticmethod
    def geometry_name():
        return 'keypoints'

    def __init__(self, points: List[Keypoint], template: KeypointsTemplate):
        self._points = points
        self._template = template

        # Extented validation
        points_count = len(self._points)
        template_points_count = len(self._template.points)
        if points_count != template_points_count:
            raise ValueError('Keypoints count ({}) must be equal to template points count ({})!'
                             .format(points_count, template_points_count))
        for edge in self._template.edges:
            if edge.start < 0 or edge.end >= points_count:
                raise ValueError('Keypoints edge refers to wrong keypoint index! {}'.format(edge))


    @property
    def points(self): # TODO: Maybe use tuple instead list?
        return deepcopy(self._points)

    @property
    def template(self):
        return deepcopy(self._template)

    @classmethod
    def from_json(cls, data):
        template = KeypointsTemplate.from_json(data[TEMPLATE])
        points = [Keypoint(location=PointLocation.from_json(point_obj[LOCATION]), enabled=point_obj[ENABLED])
                  for point_obj in data[POINTS]]
        return cls(points=points, template=template)

    def to_json(self):
        packed_obj = {
            POINTS: [{
                LOCATION: point.location.to_json(),
                ENABLED: point.enabled
            } for point in self._points],
            TEMPLATE: self._template.to_json()
        }
        return packed_obj


    def crop(self, rect: Rectangle):
        for i, keypoint in enumerate(self.points):
            if not rect.contains_point_location(keypoint.location):
                # Keypoints is complex object and all nodes are important. Crop element - crop all
                return []
        return [deepcopy(self)]

    def relative_crop(self, rect):
        return [geom.translate(drow=-rect.top, dcol=-rect.left) for geom in self.crop(rect)]

    def _transform(self, transform_fn):
        keypoints = []
        for keypoint in self.points:
            keypoints.append(Keypoint(location=transform_fn(keypoint.location), enabled=keypoint.enabled))
        return self.__class__(points=keypoints, template=self.template)

    def resize(self, in_size, out_size):
        return self._transform(lambda p: p.resize(in_size, out_size))

    def scale(self, factor):
        return self._transform(lambda p: p.scale(factor))

    def translate(self, drow, dcol):
        return self._transform(lambda p: p.translate(drow, dcol))

    def rotate(self, rotator):
        return self._transform(lambda p: p.rotate(rotator))

    def fliplr(self, img_size):
        return self._transform(lambda p: p.fliplr(img_size))

    def flipud(self, img_size):
        return self._transform(lambda p: p.flipud(img_size))

    def draw(self, bitmap, color, thickness=1):
        self.draw_contour(bitmap, color, thickness)

    def draw_contour(self, bitmap, color=None, thickness=1):
        def actual_color(another_color):
            return another_color if color is None else color

        for edge in self._template.edges:
            pt1 = self.point[edge.start].location
            pt2 = self.point[edge.end].location
            cv2.line(bitmap, [pt1.col, pt1.row], [pt2.col, pt2.row], actual_color(edge.color), thickness)

        for idx, point in enumerate(self._points):
            r = round(thickness / 2)
            cv2.circle(bitmap, (point.location.col, point.location.row), radius=r,
                       color=actual_color(self._template.points[idx].color),
                       thickness=cv2.FILLED)

    @property
    def area(self):
        return 0.0

    def to_bbox(self):
        rows = [keypoint.location.row for keypoint in self.points]
        cols = [keypoint.location.col for keypoint in self.points]
        return Rectangle(top=round(min(rows)), left=round(min(cols)), bottom=round(max(rows)), right=round(max(cols)))

    def clone(self):
        return deepcopy(self)

    def validate(self, name, settings):
        super().validate(name, settings)
        if settings[TEMPLATE] != self._template.to_json():
            raise ValueError('Keypoints template not correct!')

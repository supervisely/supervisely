import numpy as np
from typing import List

import supervisely as sly
from supervisely.nn.prediction_dto import PredictionPoint


def numpy_to_dto_point(points: np.ndarray, class_name: str) -> List[PredictionPoint]:
    assert points.shape[-1] == 2
    return [PredictionPoint(class_name=class_name, col=p[1], row=p[0]) for p in points]


def dto_points_to_point_location(points: List[PredictionPoint]) -> List[sly.PointLocation]:
    return [sly.PointLocation(row=p.row, col=p.col) for p in points]


def exteriors_to_sly_polygons(exteriors: List[List[sly.PointLocation]]) -> List[sly.Polygon]:
    return [sly.Polygon(exterior=exterior) for exterior in exteriors]


def dto_points_to_sly_points(points: List[PredictionPoint]) -> List[sly.Point]:
    return [sly.Point(row=p.row, col=p.col) for p in points]

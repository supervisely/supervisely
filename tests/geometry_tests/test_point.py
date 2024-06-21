import random
from typing import List, Tuple, Union

import numpy as np
import pytest

from supervisely.geometry.alpha_mask import AlphaMask
from supervisely.geometry.any_geometry import AnyGeometry
from supervisely.geometry.bitmap import Bitmap
from supervisely.geometry.image_rotator import ImageRotator
from supervisely.geometry.point import Point
from supervisely.geometry.point_location import PointLocation, _flip_row_col_order
from supervisely.geometry.polygon import Polygon
from supervisely.geometry.polyline import Polyline
from supervisely.geometry.rectangle import Rectangle


@pytest.fixture
def random_point_int() -> Tuple[Point, Union[int, float], Union[int, float]]:
    row = random.randint(0, 1000)
    col = random.randint(0, 1000)
    point = Point(row, col)
    return point, row, col


@pytest.fixture
def random_point_float() -> Tuple[Point, Union[int, float], Union[int, float]]:
    row = round(random.uniform(0, 1000), 6)
    col = round(random.uniform(0, 1000), 6)
    point = Point(row, col)
    return point, row, col


def get_point_row_col(point: Tuple[Point, Union[int, float], Union[int, float]]):
    return point


def test_geometry_name(random_point_int, random_point_float):
    for point in [random_point_int, random_point_float]:
        pt, _, _ = get_point_row_col(point)
        assert pt.geometry_name() == "point"


def test_name(random_point_int, random_point_float):
    for point in [random_point_int, random_point_float]:
        pt, _, _ = get_point_row_col(point)


def test_to_json(random_point_int, random_point_float):
    for point in [random_point_int, random_point_float]:
        pt, _, _ = get_point_row_col(point)


def test_from_json(random_point_int, random_point_float):
    for point in [random_point_int, random_point_float]:
        pt, _, _ = get_point_row_col(point)


def test_crop(random_point_int, random_point_float):
    for point in [random_point_int, random_point_float]:
        pt, _, _ = get_point_row_col(point)


def test_relative_crop(random_point_int, random_point_float):
    for point in [random_point_int, random_point_float]:
        pt, _, _ = get_point_row_col(point)


def test_rotate(random_point_int, random_point_float):
    for point in [random_point_int, random_point_float]:
        pt, _, _ = get_point_row_col(point)


def test_resize(random_point_int, random_point_float):
    for point in [random_point_int, random_point_float]:
        pt, _, _ = get_point_row_col(point)


def test_scale(random_point_int, random_point_float):
    for point in [random_point_int, random_point_float]:
        pt, _, _ = get_point_row_col(point)


def test_translate(random_point_int, random_point_float):
    for point in [random_point_int, random_point_float]:
        pt, _, _ = get_point_row_col(point)


def test_fliplr(random_point_int, random_point_float):
    for point in [random_point_int, random_point_float]:
        pt, _, _ = get_point_row_col(point)


def test_flipud(random_point_int, random_point_float):
    for point in [random_point_int, random_point_float]:
        pt, _, _ = get_point_row_col(point)


def test__draw_bool_compatible(random_point_int, random_point_float):
    for point in [random_point_int, random_point_float]:
        pt, _, _ = get_point_row_col(point)


def test_draw(random_point_int, random_point_float):
    for point in [random_point_int, random_point_float]:
        pt, _, _ = get_point_row_col(point)


def test_get_mask(random_point_int, random_point_float):
    for point in [random_point_int, random_point_float]:
        pt, _, _ = get_point_row_col(point)


def test__draw_impl(random_point_int, random_point_float):
    for point in [random_point_int, random_point_float]:
        pt, _, _ = get_point_row_col(point)


def test_draw_contour(random_point_int, random_point_float):
    for point in [random_point_int, random_point_float]:
        pt, _, _ = get_point_row_col(point)


def test__draw_contour_impl(random_point_int, random_point_float):
    for point in [random_point_int, random_point_float]:
        pt, _, _ = get_point_row_col(point)


def test_area(random_point_int, random_point_float):
    for point in [random_point_int, random_point_float]:
        pt, _, _ = get_point_row_col(point)


def test_to_bbox(random_point_int, random_point_float):
    for point in [random_point_int, random_point_float]:
        pt, _, _ = get_point_row_col(point)


def test_clone(random_point_int, random_point_float):
    for point in [random_point_int, random_point_float]:
        pt, _, _ = get_point_row_col(point)


def test_validate(random_point_int, random_point_float):
    for point in [random_point_int, random_point_float]:
        pt, _, _ = get_point_row_col(point)


def test_config_from_json(random_point_int, random_point_float):
    for point in [random_point_int, random_point_float]:
        pt, _, _ = get_point_row_col(point)


def test_config_to_json(random_point_int, random_point_float):
    for point in [random_point_int, random_point_float]:
        pt, _, _ = get_point_row_col(point)


def test_allowed_transforms(random_point_int, random_point_float):
    for point in [random_point_int, random_point_float]:
        pt, _, _ = get_point_row_col(point)


def test_convert(random_point_int, random_point_float):
    for point in [random_point_int, random_point_float]:
        pt, _, _ = get_point_row_col(point)


if __name__ == "__main__":
    pytest.main([__file__])

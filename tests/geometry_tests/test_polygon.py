import random
from typing import List, Tuple, Union

import numpy as np
import pytest

from supervisely import PointLocation, Polygon, Rectangle
from supervisely.geometry.image_rotator import ImageRotator
from supervisely.geometry.point_location import _flip_row_col_order


@pytest.fixture
def random_polygon_int() -> Tuple[
    Polygon,
    List[Tuple[Union[int, float], Union[int, float]]],
    List[Tuple[Union[int, float], Union[int, float]]],
]:
    exterior = [(random.randint(0, 1000), random.randint(0, 1000)) for _ in range(15)]
    interior = []
    poly = Polygon(exterior=exterior, interior=interior)
    return poly, exterior, interior


@pytest.fixture
def random_polygon_float() -> Tuple[
    Polygon,
    List[Tuple[Union[int, float], Union[int, float]]],
    List[Tuple[Union[int, float], Union[int, float]]],
]:
    exterior = [
        (round(random.uniform(0, 1000), 6), round(random.uniform(0, 1000), 6)) for _ in range(15)
    ]
    interior = []
    poly = Polygon(exterior=exterior, interior=interior)
    return poly, exterior, interior


def check_points(polygon_exterior: List[PointLocation], coords: List[Tuple[int, int]]):
    assert len(polygon_exterior) == len(coords)
    for i, point in enumerate(polygon_exterior):
        assert isinstance(point, PointLocation)
        assert point.row == coords[i][0]
        assert point.col == coords[i][1]


def test_points_property(random_polygon_int: Polygon, random_polygon_float: Polygon):
    for polygon in [random_polygon_int, random_polygon_float]:
        poly, exterior, interior = polygon
        check_points(poly.exterior, exterior)
        check_points(poly.interior, interior)


def test_to_json_method(random_polygon_int: Polygon, random_polygon_float: Polygon):
    for polygon in [random_polygon_int, random_polygon_float]:
        poly, exterior, interior = polygon
        expected_json = {
            "points": {"exterior": _flip_row_col_order(exterior), "interior": interior},
            "shape": "polygon",
            "geometryType": "polygon",
        }
        assert poly.to_json() == expected_json


def test_from_json():
    poly_json = {
        "points": {"exterior": [[200, 100], [300, 400], [500, 600]], "interior": []},
        "shape": "polygon",
        "geometryType": "polygon",
    }
    poly = Polygon.from_json(poly_json)
    assert isinstance(poly, Polygon)
    check_points(poly.exterior, [(100, 200), (400, 300), (600, 500)])


def test_scale(random_polygon_int: Polygon, random_polygon_float: Polygon):
    for polygon in [random_polygon_int, random_polygon_float]:
        poly, exterior, interior = polygon
        factor = 0.75
        scale_poly = poly.scale(factor)
        assert isinstance(scale_poly, Polygon)
        check_points(
            scale_poly.exterior,
            [(round(x * factor), round(y * factor)) for x, y in exterior],
        )


def test_translate(random_polygon_int: Polygon, random_polygon_float: Polygon):
    for polygon in [random_polygon_int, random_polygon_float]:
        poly, exterior, interior = polygon
        dx, dy = 150, 350
        translate_poly = poly.translate(dx, dy)
        assert isinstance(translate_poly, Polygon)
        check_points(translate_poly.exterior, [(x + dx, y + dy) for x, y in exterior])


def test_rotate(random_polygon_int: Polygon, random_polygon_float: Polygon):
    for polygon in [random_polygon_int, random_polygon_float]:
        poly, exterior, interior = polygon
        angle = 25
        rotator = ImageRotator((300, 400), angle)
        rotate_poly = poly.rotate(rotator)
        assert isinstance(rotate_poly, Polygon)

        expected_points = []
        for x, y in exterior:
            point_np_uniform = np.array([x, y, 1])
            transformed_np = rotator.affine_matrix.dot(point_np_uniform)
            expected_points.append(
                (round(transformed_np[0].item()), round(transformed_np[1].item()))
            )
        check_points(rotate_poly.exterior, expected_points)


def test_resize(random_polygon_int: Polygon, random_polygon_float: Polygon):
    for polygon in [random_polygon_int, random_polygon_float]:
        poly, exterior, interior = polygon
        in_size = (300, 400)
        out_size = (600, 800)
        resize_poly = poly.resize(in_size, out_size)
        assert isinstance(resize_poly, Polygon)
        check_points(
            resize_poly.exterior,
            [
                (
                    round(x * out_size[0] / in_size[0]),
                    round(y * out_size[1] / in_size[1]),
                )
                for x, y in exterior
            ],
        )


def test_fliplr(random_polygon_int: Polygon, random_polygon_float: Polygon):
    for polygon in [random_polygon_int, random_polygon_float]:
        poly, exterior, interior = polygon
        img_size = (300, 400)
        fliplr_poly = poly.fliplr(img_size)
        assert isinstance(fliplr_poly, Polygon)
        check_points(fliplr_poly.exterior, [(x, img_size[1] - y) for x, y in exterior])


def test_flipud(random_polygon_int: Polygon, random_polygon_float: Polygon):
    for polygon in [random_polygon_int, random_polygon_float]:
        poly, exterior, interior = polygon
        img_size = (300, 400)
        flipud_poly = poly.flipud(img_size)
        assert isinstance(flipud_poly, Polygon)
        check_points(flipud_poly.exterior, [(img_size[0] - x, y) for x, y in exterior])


def test_clone(random_polygon_int: Polygon, random_polygon_float: Polygon):
    for polygon in [random_polygon_int, random_polygon_float]:
        poly, exterior, interior = polygon
        clone_poly = poly.clone()
        assert isinstance(clone_poly, Polygon)
        check_points(clone_poly.exterior, exterior)
        check_points(clone_poly.interior, interior)


def test_crop(random_polygon_int: Polygon, random_polygon_float: Polygon):
    for polygon in [random_polygon_int, random_polygon_float]:
        poly, exterior, interior = polygon
        rect = Rectangle(100, 200, 300, 400)
        cropped_polygons = poly.crop(rect)
        for cropped_poly in cropped_polygons:
            assert isinstance(cropped_poly, Polygon)
            for point in cropped_poly.exterior:
                assert 0 <= point.row <= 500
                assert 0 <= point.col <= 500


def test_area(random_polygon_int: Polygon, random_polygon_float: Polygon):
    for polygon in [random_polygon_int, random_polygon_float]:
        poly, exterior, interior = polygon
        area = poly.area
        assert isinstance(area, float)
        assert area >= 0


def test_approx_dp(random_polygon_int: Polygon, random_polygon_float: Polygon):
    # Create a polygon
    for polygon in [random_polygon_int, random_polygon_float]:
        poly, exterior, interior = polygon

        # Approximate the polygon
        epsilon = 0.75
        approx_poly = poly.approx_dp(epsilon)

        assert isinstance(approx_poly, Polygon)
        assert len(approx_poly.exterior) <= len(poly.exterior)
        for approx_interior, original_interior in zip(approx_poly.interior, poly.interior):
            assert len(approx_interior) <= len(original_interior)

        assert approx_poly.exterior != poly.exterior
        for approx_interior, original_interior in zip(approx_poly.interior, poly.interior):
            assert approx_interior != original_interior


if __name__ == "__main__":
    pytest.main([__file__])

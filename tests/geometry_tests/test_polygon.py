import random
from typing import List, Tuple, Union

import numpy as np
import pytest

from supervisely.geometry.alpha_mask import AlphaMask
from supervisely.geometry.any_geometry import AnyGeometry
from supervisely.geometry.bitmap import Bitmap
from supervisely.geometry.image_rotator import ImageRotator
from supervisely.geometry.point_location import PointLocation, _flip_row_col_order
from supervisely.geometry.polygon import Polygon
from supervisely.geometry.polyline import Polyline
from supervisely.geometry.rectangle import Rectangle


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


def get_polygon_exterior_interior(polygon) -> Tuple[
    Polygon,
    List[Tuple[Union[int, float], Union[int, float]]],
    List[Tuple[Union[int, float], Union[int, float]]],
]:
    return polygon


def check_points_equal(polygon_exterior: List[PointLocation], coords: List[Tuple[int, int]]):
    assert len(polygon_exterior) == len(coords)
    for i, point in enumerate(polygon_exterior):
        assert isinstance(point, PointLocation)
        assert point.row == coords[i][0]
        assert point.col == coords[i][1]


def test_geometry_name(random_polygon_int, random_polygon_float):
    for polygon in [random_polygon_int, random_polygon_float]:
        poly, _, _ = get_polygon_exterior_interior(polygon)
        assert poly.geometry_name() == "polygon"


def test_name(random_polygon_int, random_polygon_float):
    for polygon in [random_polygon_int, random_polygon_float]:
        poly, _, _ = get_polygon_exterior_interior(polygon)
        assert poly.name() == "polygon"


def test_to_json(random_polygon_int, random_polygon_float):
    for polygon in [random_polygon_int, random_polygon_float]:
        poly, exterior, interior = get_polygon_exterior_interior(polygon)
        expected_json = {
            "points": {"exterior": _flip_row_col_order(exterior), "interior": interior},
            "shape": "polygon",
            "geometryType": "polygon",
        }
        assert poly.to_json() == expected_json


def test_from_json(random_polygon_int, random_polygon_float):
    for polygon in [random_polygon_int, random_polygon_float]:
        poly, exterior, interior = get_polygon_exterior_interior(polygon)
        poly_json = {
            "points": {"exterior": _flip_row_col_order(exterior), "interior": interior},
            "shape": "polygon",
            "geometryType": "polygon",
        }
        poly_from_json = Polygon.from_json(poly_json)
        assert isinstance(poly_from_json, Polygon)
        check_points_equal(poly_from_json.exterior, exterior)
        check_points_equal(poly_from_json.interior, interior)


def test_crop(random_polygon_int, random_polygon_float):
    for polygon in [random_polygon_int, random_polygon_float]:
        poly, exterior, interior = get_polygon_exterior_interior(polygon)
        rect = Rectangle(100, 200, 300, 400)
        cropped_polygons = poly.crop(rect)
        for cropped_poly in cropped_polygons:
            assert isinstance(cropped_poly, Polygon)
            for point in cropped_poly.exterior:
                assert 0 <= point.row <= 500
                assert 0 <= point.col <= 500


def test_relative_crop(random_polygon_int, random_polygon_float):
    for polygon in [random_polygon_int, random_polygon_float]:
        poly, exterior, interior = get_polygon_exterior_interior(polygon)
        rect = Rectangle(100, 200, 300, 400)
        cropped_polygons = poly.relative_crop(rect)
        for cropped_poly in cropped_polygons:
            assert isinstance(cropped_poly, Polygon)
            for point in cropped_poly.exterior:
                assert 0 <= point.row <= 1
                assert 0 <= point.col <= 1


def test_rotate(random_polygon_int, random_polygon_float):
    angle = 25
    image_rotate_center = (300, 400)
    for polygon in [random_polygon_int, random_polygon_float]:
        poly, exterior, interior = get_polygon_exterior_interior(polygon)
        rotator = ImageRotator(image_rotate_center, angle)
        rotate_poly = poly.rotate(rotator)
        assert isinstance(rotate_poly, Polygon)
        expected_points = []
        for x, y in exterior:
            point_np_uniform = np.array([x, y, 1])
            transformed_np = rotator.affine_matrix.dot(point_np_uniform)
            expected_points.append(
                (round(transformed_np[0].item()), round(transformed_np[1].item()))
            )
        check_points_equal(rotate_poly.exterior, expected_points)


def test_resize(random_polygon_int, random_polygon_float):
    in_size = (300, 400)
    out_size = (600, 800)
    for polygon in [random_polygon_int, random_polygon_float]:
        poly, exterior, interior = get_polygon_exterior_interior(polygon)
        resize_poly = poly.resize(in_size, out_size)
        assert isinstance(resize_poly, Polygon)
        check_points_equal(
            resize_poly.exterior,
            [
                (
                    round(x * out_size[0] / in_size[0]),
                    round(y * out_size[1] / in_size[1]),
                )
                for x, y in exterior
            ],
        )


def test_scale(random_polygon_int, random_polygon_float):
    factor = 0.75
    for polygon in [random_polygon_int, random_polygon_float]:
        poly, exterior, interior = get_polygon_exterior_interior(polygon)
        scale_poly = poly.scale(factor)
        assert isinstance(scale_poly, Polygon)
        check_points_equal(
            scale_poly.exterior,
            [(round(x * factor), round(y * factor)) for x, y in exterior],
        )


def test_translate(random_polygon_int, random_polygon_float):
    dx, dy = 150, 350
    for polygon in [random_polygon_int, random_polygon_float]:
        poly, exterior, interior = get_polygon_exterior_interior(polygon)
        translate_poly = poly.translate(dx, dy)
        assert isinstance(translate_poly, Polygon)
        check_points_equal(translate_poly.exterior, [(x + dx, y + dy) for x, y in exterior])


def test_fliplr(random_polygon_int, random_polygon_float):
    img_size = (800, 800)
    for polygon in [random_polygon_int, random_polygon_float]:
        poly, exterior, interior = get_polygon_exterior_interior(polygon)
        fliplr_poly = poly.fliplr(img_size)
        assert isinstance(fliplr_poly, Polygon)
        check_points_equal(fliplr_poly.exterior, [(x, img_size[1] - y) for x, y in exterior])


def test_flipud(random_polygon_int, random_polygon_float):
    img_size = (800, 800)
    for polygon in [random_polygon_int, random_polygon_float]:
        poly, exterior, interior = get_polygon_exterior_interior(polygon)
        flipud_poly = poly.flipud(img_size)
        assert isinstance(flipud_poly, Polygon)
        check_points_equal(flipud_poly.exterior, [(img_size[0] - x, y) for x, y in exterior])


def test_draw_bool_compatible(random_polygon_int, random_polygon_float):
    img_size = (800, 800)
    for polygon in [random_polygon_int, random_polygon_float]:
        poly, exterior, interior = get_polygon_exterior_interior(polygon)
        bitmap = np.zeros(img_size, dtype=np.uint8)
        poly._draw_bool_compatible(poly._draw_impl, bitmap, 255, 1)
        assert np.any(bitmap)


def test_draw(random_polygon_int, random_polygon_float):
    img_size = (800, 800)
    for polygon in [random_polygon_int, random_polygon_float]:
        poly, exterior, interior = get_polygon_exterior_interior(polygon)
        bitmap = np.zeros(img_size, dtype=np.uint8)
        color = [255, 255, 255]
        poly.draw(bitmap, color, thickness=1)
        assert np.any(bitmap)


def test_get_mask(random_polygon_int, random_polygon_float):
    img_size = (800, 800)
    for polygon in [random_polygon_int, random_polygon_float]:
        poly, exterior, interior = get_polygon_exterior_interior(polygon)
        mask = poly.get_mask(img_size)
        assert mask.shape == img_size
        assert mask.dtype == np.bool
        assert np.any(mask)


def test__draw_impl(random_polygon_int, random_polygon_float):
    for polygon in [random_polygon_int, random_polygon_float]:
        poly, exterior, interior = get_polygon_exterior_interior(polygon)


def test_draw_contour(random_polygon_int, random_polygon_float):
    for polygon in [random_polygon_int, random_polygon_float]:
        poly, exterior, interior = get_polygon_exterior_interior(polygon)


def test_draw_contour_impl(random_polygon_int, random_polygon_float):
    for polygon in [random_polygon_int, random_polygon_float]:
        poly, exterior, interior = get_polygon_exterior_interior(polygon)


def test_area(random_polygon_int, random_polygon_float):
    for polygon in [random_polygon_int, random_polygon_float]:
        poly, exterior, interior = get_polygon_exterior_interior(polygon)
        area = poly.area
        assert isinstance(area, float)
        assert area >= 0


def test_to_bbox(random_polygon_int, random_polygon_float):
    for polygon in [random_polygon_int, random_polygon_float]:
        poly, exterior, interior = get_polygon_exterior_interior(polygon)


def test_clone(random_polygon_int, random_polygon_float):
    for polygon in [random_polygon_int, random_polygon_float]:
        poly, exterior, interior = get_polygon_exterior_interior(polygon)
        clone_poly = poly.clone()
        assert isinstance(clone_poly, Polygon)
        check_points_equal(clone_poly.exterior, exterior)
        check_points_equal(clone_poly.interior, interior)


def test_validate(random_polygon_int, random_polygon_float):
    for polygon in [random_polygon_int, random_polygon_float]:
        poly, exterior, interior = get_polygon_exterior_interior(polygon)


def test_config_from_json(random_polygon_int, random_polygon_float):
    for polygon in [random_polygon_int, random_polygon_float]:
        poly, exterior, interior = get_polygon_exterior_interior(polygon)


def test_config_to_json(random_polygon_int, random_polygon_float):
    for polygon in [random_polygon_int, random_polygon_float]:
        poly, exterior, interior = get_polygon_exterior_interior(polygon)


def test_allowed_transforms(random_polygon_int, random_polygon_float):
    for polygon in [random_polygon_int, random_polygon_float]:
        poly, exterior, interior = get_polygon_exterior_interior(polygon)
        allowed_transforms = poly.allowed_transforms()
        assert set(allowed_transforms) == set([AnyGeometry, Rectangle, Bitmap, AlphaMask])


def test_convert(random_polygon_int, random_polygon_float):
    for polygon in [random_polygon_int, random_polygon_float]:
        poly, exterior, interior = get_polygon_exterior_interior(polygon)
        assert poly.convert(type(poly)) == [poly]
        assert poly.convert(AnyGeometry) == [poly]
        for new_geometry in poly.allowed_transforms():
            converted = poly.convert(new_geometry)
            assert all(isinstance(g, new_geometry) for g in converted)

        class NotAllowedGeometry:
            pass

        with pytest.raises(
            NotImplementedError,
            match="from {!r} to {!r}".format(poly.geometry_name(), "NotAllowedGeometry"),
        ):
            poly.convert(NotAllowedGeometry)


# Polygon specific methods
# ------------------------


def test_approx_dp(random_polygon_int, random_polygon_float):
    epsilon = 0.75
    for polygon in [random_polygon_int, random_polygon_float]:
        poly, exterior, interior = get_polygon_exterior_interior(polygon)
        approx_poly = poly.approx_dp(epsilon)
        assert isinstance(approx_poly, Polygon)
        assert len(approx_poly.exterior) <= len(poly.exterior)
        for approx_interior, original_interior in zip(approx_poly.interior, poly.interior):
            assert len(approx_interior) <= len(original_interior)
        assert approx_poly.exterior != poly.exterior
        for approx_interior, original_interior in zip(approx_poly.interior, poly.interior):
            assert approx_interior != original_interior


def test_exterior_interior(random_polygon_int, random_polygon_float):
    for polygon in [random_polygon_int, random_polygon_float]:
        poly, exterior, interior = get_polygon_exterior_interior(polygon)
        check_points_equal(poly.exterior, exterior)
        check_points_equal(poly.interior, interior)


def test_exterior_interior_np(random_polygon_int, random_polygon_float):
    for polygon in [random_polygon_int, random_polygon_float]:
        poly, exterior, interior = get_polygon_exterior_interior(polygon)
        exterior_np = poly.exterior_np
        interior_np = poly.interior_np
        assert exterior_np.shape == (len(exterior), 2)
        for i, (x, y) in enumerate(exterior):
            assert exterior_np[i, 0] == x
            assert exterior_np[i, 1] == y
        if len(interior) > 0:
            assert interior_np.shape == (len(interior), 2)
            for i, (x, y) in enumerate(interior):
                assert interior_np[i, 0] == x
                assert interior_np[i, 1] == y


if __name__ == "__main__":
    pytest.main([__file__])

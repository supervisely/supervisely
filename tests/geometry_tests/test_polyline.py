import inspect
import os
import random
from typing import List, Tuple, Union

import numpy as np
import pytest
from test_geometry import draw_test

from supervisely.geometry.alpha_mask import AlphaMask
from supervisely.geometry.any_geometry import AnyGeometry
from supervisely.geometry.bitmap import Bitmap
from supervisely.geometry.image_rotator import ImageRotator
from supervisely.geometry.point import Point
from supervisely.geometry.point_location import PointLocation, _flip_row_col_order
from supervisely.geometry.polygon import Polygon
from supervisely.geometry.polyline import Polyline
from supervisely.geometry.rectangle import Rectangle
from supervisely.io.fs import get_file_name

dir_name = get_file_name(os.path.abspath(__file__))
# Draw Settings
color = [255, 255, 255]
thickness = 1


def get_random_image() -> np.ndarray:
    image_shape = (random.randint(801, 2000), random.randint(801, 2000), 3)
    background_color = [0, 0, 0]
    bitmap = np.full(image_shape, background_color, dtype=np.uint8)
    return bitmap


@pytest.fixture
def random_polyline_int() -> Tuple[
    Polyline,
    List[Tuple[Union[int, float], Union[int, float]]],
]:
    exterior = [(random.randint(0, 500), random.randint(0, 500)) for _ in range(10)]
    poly = Polyline(exterior=exterior)
    return poly, exterior


@pytest.fixture
def random_polyline_float() -> Tuple[
    Polyline,
    List[Tuple[Union[int, float], Union[int, float]]],
]:
    exterior = [
        (round(random.uniform(0, 500), 6), round(random.uniform(0, 500), 6)) for _ in range(10)
    ]
    poly = Polyline(exterior=exterior)
    return poly, exterior


def get_polyline_and_exterior(
    polyline,
) -> Tuple[Polyline, List[Tuple[Union[int, float], List[Union[int, float]]]]]:
    return polyline


def check_points_equal(
    polygon_exterior: List[PointLocation],
    coords: List[Tuple[Union[int, float], Union[int, float]]],
):
    assert len(polygon_exterior) == len(coords)
    for i, point in enumerate(polygon_exterior):
        assert isinstance(point, PointLocation)
        assert point.row == coords[i][0]
        assert point.col == coords[i][1]


def test_geometry_name(random_polyline_int, random_polyline_float):
    for idx, polyline in enumerate([random_polyline_int, random_polyline_float], 1):
        poly, exterior = get_polyline_and_exterior(polyline)
        assert poly.geometry_name() == "line"


def test_name(random_polyline_int, random_polyline_float):
    for idx, polyline in enumerate([random_polyline_int, random_polyline_float], 1):
        poly, exterior = get_polyline_and_exterior(polyline)
        assert poly.name() == "line"


def test_to_json(random_polyline_int, random_polyline_float):
    for idx, polyline in enumerate([random_polyline_int, random_polyline_float], 1):
        poly, exterior = get_polyline_and_exterior(polyline)
        expected_json = {
            "points": {"exterior": _flip_row_col_order(exterior), "interior": []},
            "shape": poly.name(),
            "geometryType": poly.geometry_name(),
        }
        assert poly.to_json() == expected_json


def test_from_json(random_polyline_int, random_polyline_float):
    for idx, polyline in enumerate([random_polyline_int, random_polyline_float], 1):
        poly, exterior = get_polyline_and_exterior(polyline)

        polyline_json = {
            "points": {"exterior": exterior, "interior": []},
            "labelerLogin": "test",
            "updatedAt": "2022-01-01T00:00:00Z",
            "createdAt": "2022-01-01T00:00:00Z",
            "id": 1,
            "classId": 1,
        }

        returned_polyline = Polyline.from_json(polyline_json)
        check_points_equal(returned_polyline.exterior, _flip_row_col_order(exterior))
        assert returned_polyline.labeler_login == polyline_json["labelerLogin"]
        assert returned_polyline.updated_at == polyline_json["updatedAt"]
        assert returned_polyline.created_at == polyline_json["createdAt"]
        assert returned_polyline.sly_id == polyline_json["id"]
        assert returned_polyline.class_id == polyline_json["classId"]


def test_crop(random_polyline_int, random_polyline_float):
    for idx, polyline in enumerate([random_polyline_int, random_polyline_float], 1):
        poly, exterior = get_polyline_and_exterior(polyline)
        rect = Rectangle(100, 200, 300, 400)
        cropped_polylines = poly.crop(rect)
        for cidx, cropped_poly in enumerate(cropped_polylines, 1):
            assert isinstance(cropped_poly, Polyline)
            for point in cropped_poly.exterior:
                assert 0 <= point.row <= 500
                assert 0 <= point.col <= 500

            random_image = get_random_image()
            function_name = inspect.currentframe().f_code.co_name
            draw_test(
                dir_name, f"{function_name}_{cidx}_geometry_{idx}", random_image, cropped_poly
            )


def test_relative_crop(random_polyline_int, random_polyline_float):
    for idx, polyline in enumerate([random_polyline_int, random_polyline_float], 1):
        poly, exterior = get_polyline_and_exterior(polyline)
        rect = Rectangle(100, 200, 300, 400)
        cropped_polylines = poly.relative_crop(rect)
        for cidx, cropped_poly in enumerate(cropped_polylines, 1):
            assert isinstance(cropped_poly, Polyline)
            for point in cropped_poly.exterior:
                assert 0 <= point.row <= 300
                assert 0 <= point.col <= 200

            random_image = get_random_image()
            function_name = inspect.currentframe().f_code.co_name
            draw_test(
                dir_name, f"{function_name}_{cidx}_geometry_{idx}", random_image, cropped_poly
            )


def test_rotate(random_polyline_int, random_polyline_float):
    for idx, polyline in enumerate([random_polyline_int, random_polyline_float], 1):
        poly, exterior = get_polyline_and_exterior(polyline)
        random_image = get_random_image()
        img_size, angle = random_image.shape[:2], random.randint(0, 360)
        rotator = ImageRotator(img_size, angle)
        rotated_poly = poly.rotate(rotator)
        assert isinstance(rotated_poly, Polyline)

        expected_points = []
        for x, y in exterior:
            point_np_uniform = np.array([x, y, 1])
            transformed_np = rotator.affine_matrix.dot(point_np_uniform)
            expected_points.append(
                (round(transformed_np[0].item()), round(transformed_np[1].item()))
            )
        check_points_equal(rotated_poly.exterior, expected_points)

        function_name = inspect.currentframe().f_code.co_name
        draw_test(dir_name, f"{function_name}_geometry_{idx}", random_image, rotated_poly)


def test_resize(random_polyline_int, random_polyline_float):
    for idx, polyline in enumerate([random_polyline_int, random_polyline_float], 1):
        poly, exterior = get_polyline_and_exterior(polyline)
        random_image = get_random_image()
        in_size = random_image.shape[:2]
        out_size = (random.randint(1000, 1200), random.randint(1000, 1200))
        resized_poly = poly.resize(in_size, out_size)
        assert isinstance(resized_poly, Polyline)
        check_points_equal(
            resized_poly.exterior,
            [
                (
                    round(x * out_size[0] / in_size[0]),
                    round(y * out_size[1] / in_size[1]),
                )
                for x, y in exterior
            ],
        )

        function_name = inspect.currentframe().f_code.co_name
        draw_test(dir_name, f"{function_name}_geometry_{idx}", random_image, resized_poly)


def test_scale(random_polyline_int, random_polyline_float):
    for idx, polyline in enumerate([random_polyline_int, random_polyline_float], 1):
        poly, exterior = get_polyline_and_exterior(polyline)
        factor = round(random.uniform(0, 1), 3)
        scaled_poly = poly.scale(factor)
        assert isinstance(scaled_poly, Polyline)
        check_points_equal(
            scaled_poly.exterior,
            [(round(x * factor), round(y * factor)) for x, y in exterior],
        )

        random_image = get_random_image()
        function_name = inspect.currentframe().f_code.co_name
        draw_test(dir_name, f"{function_name}_geometry_{idx}", random_image, scaled_poly)


def test_translate(random_polyline_int, random_polyline_float):
    for idx, polyline in enumerate([random_polyline_int, random_polyline_float], 1):
        poly, exterior = get_polyline_and_exterior(polyline)
        dx, dy = random.randint(10, 150), random.randint(10, 350)
        translated_poly = poly.translate(dx, dy)
        assert isinstance(translated_poly, Polyline)
        check_points_equal(translated_poly.exterior, [(x + dx, y + dy) for x, y in exterior])

        random_image = get_random_image()
        function_name = inspect.currentframe().f_code.co_name
        draw_test(dir_name, f"{function_name}_geometry_{idx}", random_image, translated_poly)


def test_fliplr(random_polyline_int, random_polyline_float):
    for idx, polyline in enumerate([random_polyline_int, random_polyline_float], 1):
        poly, exterior = get_polyline_and_exterior(polyline)
        random_image = get_random_image()
        img_size = random_image.shape[:2]
        fliplr_poly = poly.fliplr(img_size)
        assert isinstance(fliplr_poly, Polyline)
        check_points_equal(fliplr_poly.exterior, [(x, img_size[1] - y) for x, y in exterior])

        function_name = inspect.currentframe().f_code.co_name
        draw_test(dir_name, f"{function_name}_geometry_{idx}", random_image, fliplr_poly)


def test_flipud(random_polyline_int, random_polyline_float):
    for idx, polyline in enumerate([random_polyline_int, random_polyline_float], 1):
        poly, exterior = get_polyline_and_exterior(polyline)
        random_image = get_random_image()
        img_size = random_image.shape[:2]
        flipud_poly = poly.flipud(img_size)
        assert isinstance(flipud_poly, Polyline)
        check_points_equal(flipud_poly.exterior, [(img_size[0] - x, y) for x, y in exterior])

        function_name = inspect.currentframe().f_code.co_name
        draw_test(dir_name, f"{function_name}_geometry_{idx}", random_image, flipud_poly)


def test_draw_bool_compatible(random_polyline_int, random_polyline_float):
    for idx, polyline in enumerate([random_polyline_int, random_polyline_float], 1):
        poly, exterior = get_polyline_and_exterior(polyline)
        random_image = get_random_image()
        poly._draw_bool_compatible(poly._draw_impl, random_image, color, thickness)
        assert np.any(random_image == color)

        function_name = inspect.currentframe().f_code.co_name
        draw_test(dir_name, f"{function_name}_geometry_{idx}", random_image)


def test_draw(random_polyline_int, random_polyline_float):
    for idx, polyline in enumerate([random_polyline_int, random_polyline_float], 1):
        poly, exterior = get_polyline_and_exterior(polyline)
        random_image = get_random_image()
        poly.draw(random_image, color, thickness)
        assert np.any(random_image == color)

        function_name = inspect.currentframe().f_code.co_name
        draw_test(dir_name, f"{function_name}_geometry_{idx}", random_image)


def test_get_mask(random_polyline_int, random_polyline_float):
    for idx, polyline in enumerate([random_polyline_int, random_polyline_float], 1):
        poly, exterior = get_polyline_and_exterior(polyline)
        random_image = get_random_image()
        img_size = random_image.shape[:2]
        mask = poly.get_mask(img_size)

        assert mask.shape == img_size, "Mask shape should match image size"
        assert mask.dtype == bool, "Mask dtype should be boolean"
        assert np.any(mask), "Mask should have at least one True value"

        new_bitmap = Bitmap(mask)
        function_name = inspect.currentframe().f_code.co_name
        draw_test(dir_name, f"{function_name}_geometry_{idx}", random_image, new_bitmap)


def test_draw_impl(random_polyline_int, random_polyline_float):
    for idx, polyline in enumerate([random_polyline_int, random_polyline_float], 1):
        poly, exterior = get_polyline_and_exterior(polyline)
        random_image = get_random_image()
        poly._draw_impl(random_image, color, thickness)
        assert np.any(random_image == color)

        function_name = inspect.currentframe().f_code.co_name
        draw_test(dir_name, f"{function_name}_geometry_{idx}", random_image)


def test_draw_contour(random_polyline_int, random_polyline_float):
    for idx, polyline in enumerate([random_polyline_int, random_polyline_float], 1):
        poly, exterior = get_polyline_and_exterior(polyline)
        random_image = get_random_image()
        poly.draw_contour(random_image, color, thickness)
        assert np.any(random_image == color)

        function_name = inspect.currentframe().f_code.co_name
        draw_test(dir_name, f"{function_name}_geometry_{idx}", random_image)


def test_draw_contour_impl(random_polyline_int, random_polyline_float):
    for idx, polyline in enumerate([random_polyline_int, random_polyline_float], 1):
        poly, exterior = get_polyline_and_exterior(polyline)
        random_image = get_random_image()
        poly._draw_contour_impl(random_image, color, thickness)
        assert np.any(random_image == color)

        function_name = inspect.currentframe().f_code.co_name
        draw_test(dir_name, f"{function_name}_geometry_{idx}", random_image)


def test_area(random_polyline_int, random_polyline_float):
    for idx, polyline in enumerate([random_polyline_int, random_polyline_float], 1):
        poly, exterior = get_polyline_and_exterior(polyline)
        area = poly.area
        assert isinstance(area, float)
        assert area >= 0


def test_to_bbox(random_polyline_int, random_polyline_float):
    for idx, polyline in enumerate([random_polyline_int, random_polyline_float], 1):
        poly, exterior = get_polyline_and_exterior(polyline)
        poly_bbox = poly.to_bbox()
        assert isinstance(poly_bbox, Rectangle)
        assert poly_bbox.top == min(x for x, y in exterior)
        assert poly_bbox.left == min(y for x, y in exterior)
        assert poly_bbox.bottom == max(x for x, y in exterior)
        assert poly_bbox.right == max(y for x, y in exterior)

        random_image = get_random_image()
        function_name = inspect.currentframe().f_code.co_name
        draw_test(dir_name, f"{function_name}_geometry_{idx}", random_image, poly_bbox)


def test_clone(random_polyline_int, random_polyline_float):
    for idx, polyline in enumerate([random_polyline_int, random_polyline_float], 1):
        poly, exterior = get_polyline_and_exterior(polyline)
        clone_poly = poly.clone()
        assert isinstance(clone_poly, Polyline)
        check_points_equal(clone_poly.exterior, exterior)


def test_validate(random_polyline_int, random_polyline_float):
    for idx, polyline in enumerate([random_polyline_int, random_polyline_float], 1):
        poly, exterior = get_polyline_and_exterior(polyline)
        assert poly.validate(poly.geometry_name(), settings=None) is None
        with pytest.raises(
            ValueError, match="Geometry validation error: shape names are mismatched!"
        ):
            poly.validate("different_shape_name", settings=None)


def test_config_from_json(random_polyline_int, random_polyline_float):
    for idx, polyline in enumerate([random_polyline_int, random_polyline_float], 1):
        poly, exterior = get_polyline_and_exterior(polyline)
        config = {"key": "value"}
        returned_config = poly.config_from_json(config)
        assert returned_config == config


def test_config_to_json(random_polyline_int, random_polyline_float):
    for idx, polyline in enumerate([random_polyline_int, random_polyline_float], 1):
        poly, exterior = get_polyline_and_exterior(polyline)
        config = {"key": "value"}
        returned_config = poly.config_to_json(config)
        assert returned_config == config


def test_allowed_transforms(random_polyline_int, random_polyline_float):
    for idx, polyline in enumerate([random_polyline_int, random_polyline_float], 1):
        poly, exterior = get_polyline_and_exterior(polyline)
        allowed_transforms = poly.allowed_transforms()
        assert set(allowed_transforms) == set([AnyGeometry, Rectangle, Bitmap, Polygon, AlphaMask])


def test_convert(random_polyline_int, random_polyline_float):
    for idx, polyline in enumerate([random_polyline_int, random_polyline_float], 1):
        poly, exterior = get_polyline_and_exterior(polyline)
        assert poly.convert(type(poly)) == [poly]
        assert poly.convert(AnyGeometry) == [poly]
        for new_geometry in poly.allowed_transforms():
            converted = poly.convert(new_geometry)
            for g in converted:
                assert isinstance(g, new_geometry) or isinstance(g, Polyline)

                random_image = get_random_image()
                function_name = inspect.currentframe().f_code.co_name
                draw_test(
                    dir_name,
                    f"{function_name}_geometry_{idx}_converted_{g.name()}",
                    random_image,
                    g,
                )

        with pytest.raises(
            NotImplementedError,
            match="from {!r} to {!r}".format(poly.geometry_name(), Point.geometry_name()),
        ):
            poly.convert(Point)


# Polyline specific methods
# -------------------------


def test_approx_dp(random_polyline_int, random_polyline_float):
    for idx, polyline in enumerate([random_polyline_int, random_polyline_float], 1):
        poly, exterior = get_polyline_and_exterior(polyline)
        epsilon = round(random.uniform(0, 1), 3)
        approx_poly = poly.approx_dp(epsilon)
        assert isinstance(approx_poly, Polyline)
        assert len(approx_poly.exterior) <= len(poly.exterior)

        random_image = get_random_image()
        function_name = inspect.currentframe().f_code.co_name
        draw_test(dir_name, f"{function_name}_geometry_{idx}", random_image, approx_poly)


if __name__ == "__main__":
    pytest.main([__file__])

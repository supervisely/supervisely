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
def random_point_int() -> Tuple[Point, Union[int, float], Union[int, float]]:
    row = random.randint(0, 500)
    col = random.randint(0, 500)
    point = Point(row, col)
    return point, row, col


@pytest.fixture
def random_point_float() -> Tuple[Point, Union[int, float], Union[int, float]]:
    row = round(random.uniform(0, 500), 6)
    col = round(random.uniform(0, 500), 6)
    point = Point(row, col)
    return point, row, col


def check_points_equal(
    point_loc_1: PointLocation,
    point_loc_2: PointLocation,
):
    assert isinstance(point_loc_1, PointLocation)
    assert isinstance(point_loc_2, PointLocation)
    assert point_loc_1.row == point_loc_2.row
    assert point_loc_1.col == point_loc_2.col


def get_point_row_col(point: Tuple[Point, Union[int, float], Union[int, float]]):
    return point


def test_geometry_name(random_point_int, random_point_float):
    for idx, point in enumerate([random_point_int, random_point_float], 1):
        pt, row, col = get_point_row_col(point)
        assert pt.geometry_name() == "point"


def test_name(random_point_int, random_point_float):
    for idx, point in enumerate([random_point_int, random_point_float], 1):
        pt, row, col = get_point_row_col(point)
        assert pt.name() == "point"


def test_to_json(random_point_int, random_point_float):
    for idx, point in enumerate([random_point_int, random_point_float], 1):
        pt, row, col = get_point_row_col(point)
        expected_json = {
            "points": {"exterior": _flip_row_col_order([[row, col]]), "interior": []},
        }
        assert pt.to_json() == expected_json


def test_from_json(random_point_int, random_point_float):
    for idx, point in enumerate([random_point_int, random_point_float], 1):
        pt, row, col = get_point_row_col(point)

        pt_json = {
            "points": {"exterior": _flip_row_col_order([[row, col]]), "interior": []},
        }

        pt_from_json = Point.from_json(pt_json)
        assert isinstance(pt_from_json, Point)
        check_points_equal(pt_from_json.point_location, pt.point_location)


def test_crop(random_point_int, random_point_float):
    for idx, point in enumerate([random_point_int, random_point_float], 1):
        pt, row, col = get_point_row_col(point)

        containing_rect = Rectangle(row - 1, col - 1, row + 1, col + 1)
        cropped_points = pt.crop(containing_rect)
        assert len(cropped_points) == 1, "Point should be within the rectangle"
        assert (
            cropped_points[0].row == row and cropped_points[0].col == col
        ), "Cropped point should have the same coordinates"

        non_containing_rect = Rectangle(col + 2, row + 2, col + 3, row + 3)
        cropped_points = pt.crop(non_containing_rect)
        assert len(cropped_points) == 0, "Point should not be within the rectangle"

        for cidx, cropped_pt in enumerate(cropped_points, 1):
            random_image = get_random_image()
            function_name = inspect.currentframe().f_code.co_name
            draw_test(dir_name, f"{function_name}_{cidx}_geometry_{idx}", random_image, cropped_pt)


def test_relative_crop(random_point_int, random_point_float):
    for idx, point in enumerate([random_point_int, random_point_float], 1):
        pt, row, col = get_point_row_col(point)
        containing_rect = Rectangle(row - 1, col - 1, row + 1, col + 1)
        cropped_points = pt.relative_crop(containing_rect)
        assert len(cropped_points) == 1, "Point should be within the rectangle"
        assert (
            cropped_points[0].row == 1 and cropped_points[0].col == 1
        ), "Cropped point should have the same coordinates"
        non_containing_rect = Rectangle(col + 2, row + 2, col + 3, row + 3)
        cropped_points = pt.relative_crop(non_containing_rect)
        assert len(cropped_points) == 0, "Point should not be within the rectangle"

        for cidx, cropped_pt in enumerate(cropped_points, 1):
            random_image = get_random_image()
            function_name = inspect.currentframe().f_code.co_name
            draw_test(dir_name, f"{function_name}_{cidx}_geometry_{idx}", random_image, cropped_pt)


def test_rotate(random_point_int, random_point_float):
    for idx, point in enumerate([random_point_int, random_point_float], 1):
        pt, row, col = get_point_row_col(point)
        random_image = get_random_image()
        img_size, angle = random_image.shape[:2], random.randint(0, 360)
        rotator = ImageRotator(img_size, angle)
        rotated_point = pt.rotate(rotator)
        assert isinstance(rotated_point, Point)
        point_np_uniform = np.array([row, col, 1])
        transformed_np = rotator.affine_matrix.dot(point_np_uniform)
        expected_row = round(transformed_np[0].item())
        expected_col = round(transformed_np[1].item())
        assert rotated_point.row == expected_row
        assert rotated_point.col == expected_col

        function_name = inspect.currentframe().f_code.co_name
        draw_test(dir_name, f"{function_name}_geometry_{idx}", random_image, rotated_point)


def test_resize(random_point_int, random_point_float):
    for idx, point in enumerate([random_point_int, random_point_float], 1):
        pt, row, col = get_point_row_col(point)
        random_image = get_random_image()
        in_size = random_image.shape[:2]
        out_size = (random.randint(1000, 1200), random.randint(1000, 1200))
        resized_pt = pt.resize(in_size, out_size)
        assert isinstance(resized_pt, Point)
        assert resized_pt.row == round(pt.row * out_size[0] / in_size[0])
        assert resized_pt.col == round(pt.col * out_size[1] / in_size[1])

        function_name = inspect.currentframe().f_code.co_name
        draw_test(dir_name, f"{function_name}_geometry_{idx}", random_image, resized_pt)


def test_scale(random_point_int, random_point_float):
    for idx, point in enumerate([random_point_int, random_point_float], 1):
        pt, row, col = get_point_row_col(point)
        factor = round(random.uniform(0, 1), 3)
        scaled_pt = pt.scale(factor)
        assert isinstance(scaled_pt, Point)
        assert scaled_pt.row == round(row * factor)
        assert scaled_pt.col == round(col * factor)

        random_image = get_random_image()
        function_name = inspect.currentframe().f_code.co_name
        draw_test(dir_name, f"{function_name}_geometry_{idx}", random_image, scaled_pt)


def test_translate(random_point_int, random_point_float):
    for idx, point in enumerate([random_point_int, random_point_float], 1):
        pt, row, col = get_point_row_col(point)
        drow, dcol = random.randint(10, 150), random.randint(10, 350)
        translated_pt = pt.translate(drow, dcol)
        assert isinstance(translated_pt, Point)
        assert translated_pt.row == row + drow
        assert translated_pt.col == col + dcol

        random_image = get_random_image()
        function_name = inspect.currentframe().f_code.co_name
        draw_test(dir_name, f"{function_name}_geometry_{idx}", random_image, translated_pt)


def test_fliplr(random_point_int, random_point_float):
    for idx, point in enumerate([random_point_int, random_point_float], 1):
        pt, row, col = get_point_row_col(point)
        random_image = get_random_image()
        img_size = random_image.shape[:2]
        fliplr_pt = pt.fliplr(img_size)
        assert isinstance(fliplr_pt, Point)
        assert fliplr_pt.row == pt.row
        assert fliplr_pt.col == img_size[1] - pt.col

        function_name = inspect.currentframe().f_code.co_name
        draw_test(dir_name, f"{function_name}_geometry_{idx}", random_image, fliplr_pt)


def test_flipud(random_point_int, random_point_float):
    for idx, point in enumerate([random_point_int, random_point_float], 1):
        pt, row, col = get_point_row_col(point)
        random_image = get_random_image()
        img_size = random_image.shape[:2]
        flipud_pt = pt.flipud(img_size)
        assert isinstance(flipud_pt, Point)
        assert flipud_pt.row == img_size[0] - pt.row
        assert flipud_pt.col == pt.col

        function_name = inspect.currentframe().f_code.co_name
        draw_test(dir_name, f"{function_name}_geometry_{idx}", random_image, flipud_pt)


def test__draw_bool_compatible(random_point_int, random_point_float):
    for idx, point in enumerate([random_point_int, random_point_float], 1):
        pt, row, col = get_point_row_col(point)
        random_image = get_random_image()
        pt._draw_bool_compatible(pt._draw_impl, random_image, 255, 1)
        assert np.any(random_image == color)

        function_name = inspect.currentframe().f_code.co_name
        draw_test(dir_name, f"{function_name}_geometry_{idx}", random_image)


def test_draw(random_point_int, random_point_float):
    for idx, point in enumerate([random_point_int, random_point_float], 1):
        pt, row, col = get_point_row_col(point)
        random_image = get_random_image()
        pt.draw(random_image, color, thickness)
        assert np.any(random_image == color)

        function_name = inspect.currentframe().f_code.co_name
        draw_test(dir_name, f"{function_name}_geometry_{idx}", random_image)


def test_get_mask(random_point_int, random_point_float):
    for idx, point in enumerate([random_point_int, random_point_float], 1):
        pt, row, col = get_point_row_col(point)
        random_image = get_random_image()
        img_size = random_image.shape[:2]
        mask = pt.get_mask(img_size)
        assert mask.shape == img_size
        assert mask.dtype == np.bool
        assert np.any(mask)

        new_bitmap = Bitmap(mask)
        function_name = inspect.currentframe().f_code.co_name
        draw_test(dir_name, f"{function_name}_geometry_{idx}", random_image, new_bitmap)


def test__draw_impl(random_point_int, random_point_float):
    for idx, point in enumerate([random_point_int, random_point_float], 1):
        pt, row, col = get_point_row_col(point)
        random_image = get_random_image()
        pt._draw_impl(random_image, color, thickness)
        assert np.any(random_image == color)

        function_name = inspect.currentframe().f_code.co_name
        draw_test(dir_name, f"{function_name}_geometry_{idx}", random_image)


def test_draw_contour(random_point_int, random_point_float):
    for idx, point in enumerate([random_point_int, random_point_float], 1):
        pt, row, col = get_point_row_col(point)
        random_image = get_random_image()
        pt.draw_contour(random_image, color, thickness)
        assert np.any(random_image == color)

        function_name = inspect.currentframe().f_code.co_name
        draw_test(dir_name, f"{function_name}_geometry_{idx}", random_image)


def test__draw_contour_impl(random_point_int, random_point_float):
    for idx, point in enumerate([random_point_int, random_point_float], 1):
        pt, row, col = get_point_row_col(point)
        random_image = get_random_image()
        pt._draw_contour_impl(random_image, color, thickness)
        assert np.any(random_image == color)

        function_name = inspect.currentframe().f_code.co_name
        draw_test(dir_name, f"{function_name}_geometry_{idx}", random_image)


def test_area(random_point_int, random_point_float):
    for idx, point in enumerate([random_point_int, random_point_float], 1):
        pt, row, col = get_point_row_col(point)
        area = pt.area
        assert isinstance(area, float)
        assert area >= 0


def test_to_bbox(random_point_int, random_point_float):
    for idx, point in enumerate([random_point_int, random_point_float], 1):
        pt, row, col = get_point_row_col(point)
        bbox = pt.to_bbox()
        assert bbox.top == row, "Top of the bounding box does not match the point's row"
        assert bbox.left == col, "Left of the bounding box does not match the point's col"
        assert bbox.bottom == row, "Bottom of the bounding box does not match the point's row"
        assert bbox.right == col, "Right of the bounding box does not match the point's col"

        random_image = get_random_image()
        function_name = inspect.currentframe().f_code.co_name
        draw_test(dir_name, f"{function_name}_geometry_{idx}", random_image, bbox)


def test_clone(random_point_int, random_point_float):
    for idx, point in enumerate([random_point_int, random_point_float], 1):
        pt, row, col = get_point_row_col(point)
        clone_pt = pt.clone()
        check_points_equal(pt.point_location, clone_pt.point_location)


def test_validate(random_point_int, random_point_float):
    for idx, point in enumerate([random_point_int, random_point_float], 1):
        pt, row, col = get_point_row_col(point)
        pt.validate("point", {})


def test_config_from_json(random_point_int, random_point_float):
    for idx, point in enumerate([random_point_int, random_point_float], 1):
        pt, row, col = get_point_row_col(point)
        config = {"key": "value"}
        returned_config = pt.config_from_json(config)
        assert returned_config == config


def test_config_to_json(random_point_int, random_point_float):
    for idx, point in enumerate([random_point_int, random_point_float], 1):
        pt, row, col = get_point_row_col(point)
        config = {"key": "value"}
        returned_config = pt.config_to_json(config)
        assert returned_config == config


def test_allowed_transforms(random_point_int, random_point_float):
    for idx, point in enumerate([random_point_int, random_point_float], 1):
        pt, row, col = get_point_row_col(point)
        allowed_transforms = pt.allowed_transforms()
        assert set(allowed_transforms) == set([AnyGeometry])


def test_convert(random_point_int, random_point_float):
    for idx, point in enumerate([random_point_int, random_point_float], 1):
        pt, row, col = get_point_row_col(point)
        assert pt.convert(type(pt)) == [pt]
        assert pt.convert(AnyGeometry) == [pt]
        for new_geometry in pt.allowed_transforms():
            converted = pt.convert(new_geometry)
            for g in converted:
                assert isinstance(g, new_geometry) or isinstance(g, Point)

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
            match="from {!r} to {!r}".format(pt.geometry_name(), Rectangle.geometry_name()),
        ):
            pt.convert(Rectangle)


# Points specific methods


def from_point_location(random_point_int, random_point_float):
    for idx, point in enumerate([random_point_int, random_point_float], 1):
        pt, row, col = get_point_row_col(point)
        assert row == pt.row
        assert col == pt.col


def test_point_location(random_point_int, random_point_float):
    for idx, point in enumerate([random_point_int, random_point_float], 1):
        pt, row, col = get_point_row_col(point)
        point_loc = pt.point_location
        assert point_loc.row == pt.row
        assert point_loc.col == pt.col


if __name__ == "__main__":
    pytest.main([__file__])

import inspect
import os
import random
from typing import List, Tuple, Union

import numpy as np
import pytest  # pylint: disable=import-error
from test_geometry import draw_test

from supervisely.geometry.alpha_mask import AlphaMask
from supervisely.geometry.any_geometry import AnyGeometry
from supervisely.geometry.bitmap import Bitmap, SkeletonizeMethod
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
def random_rect_int() -> Tuple[
    Rectangle,
    List[Tuple[Union[int, float], Union[int, float]]],
]:
    top = random.randint(0, 400)
    left = random.randint(0, 400)
    bottom = random.randint(401, 800)
    right = random.randint(401, 800)

    rect = Rectangle(top=top, left=left, bottom=bottom, right=right)
    coords = [(top, left), (top, right), (bottom, right), (bottom, left)]
    return rect, coords


@pytest.fixture
def random_rect_float() -> Tuple[
    Rectangle,
    List[Tuple[Union[int, float], Union[int, float]]],
]:
    top = round(random.uniform(0, 400), 6)
    left = round(random.uniform(0, 400), 6)
    bottom = round(random.uniform(401, 800), 6)
    right = round(random.uniform(401, 800), 6)

    rect = Rectangle(top=top, left=left, bottom=bottom, right=right)
    coords = [(top, left), (top, right), (bottom, right), (bottom, left)]
    return rect, coords


def get_rect_and_coords(rectangle) -> Tuple[
    Rectangle,
    List[Tuple[Union[int, float], Union[int, float]]],
]:
    return rectangle


def check_corners_equal(rect: Rectangle, expected_rect: Rectangle):
    assert rect.top == expected_rect.top
    assert rect.left == expected_rect.left
    assert rect.bottom == expected_rect.bottom
    assert rect.right == expected_rect.right


def check_corners_transformed(
    actual_corners: List[PointLocation],
    expected_corners: List[Tuple[Union[int, float], Union[int, float]]],
):
    actual_corners = [(corner.row, corner.col) for corner in actual_corners]
    for expected, actual in zip(expected_corners, actual_corners):
        assert (
            round(actual[0]) == expected[0] and round(actual[1]) == expected[1]
        ), "Corners do not match after resize."


def test_geometry_name(random_rect_int, random_rect_float):
    for idx, rectangle in enumerate([random_rect_int, random_rect_float], 1):
        rect, _ = get_rect_and_coords(rectangle)
        assert rect.geometry_name() == "rectangle"


def test_name(random_rect_int, random_rect_float):
    for idx, rectangle in enumerate([random_rect_int, random_rect_float], 1):
        rect, _ = get_rect_and_coords(rectangle)
        assert rect.name() == "rectangle"


def test_to_json(random_rect_int, random_rect_float):
    for idx, rectangle in enumerate([random_rect_int, random_rect_float], 1):
        rect, _ = get_rect_and_coords(rectangle)
        json_data = rect.to_json()
        expected_json = {
            "points": {
                "exterior": [[rect.left, rect.top], [rect.right, rect.bottom]],
                "interior": [],
            }
        }
        assert json_data == expected_json


def test_from_json(random_rect_int, random_rect_float):
    for idx, rectangle in enumerate([random_rect_int, random_rect_float], 1):
        rect, _ = get_rect_and_coords(rectangle)
        json_data = {
            "points": {
                "exterior": [[rect.left, rect.top], [rect.right, rect.bottom]],
                "interior": [],
            }
        }
        rect_from_json = Rectangle.from_json(json_data)
        check_corners_equal(rect, rect_from_json)


def test_crop(random_rect_int, random_rect_float):
    for idx, rectangle in enumerate([random_rect_int, random_rect_float], 1):
        rect, _ = get_rect_and_coords(rectangle)
        cropper = Rectangle(150, 250, 300, 400)
        cropped_rectangles = rect.crop(cropper)
        if (
            rect.right < cropper.left
            or rect.left > cropper.right
            or rect.bottom < cropper.top
            or rect.top > cropper.bottom
        ):
            assert cropped_rectangles == []
        else:
            assert len(cropped_rectangles) == 1
            cropped_rectangle = cropped_rectangles[0]

            if rect._integer_coords:
                expected_top = round(max(rect.top, cropper.top))
                expected_left = round(max(rect.left, cropper.left))
                expected_bottom = round(min(rect.bottom, cropper.bottom))
                expected_right = round(min(rect.right, cropper.right))
            else:
                expected_top = max(rect.top, cropper.top)
                expected_left = max(rect.left, cropper.left)
                expected_bottom = min(rect.bottom, cropper.bottom)
                expected_right = min(rect.right, cropper.right)

            assert cropped_rectangle.top == expected_top
            assert cropped_rectangle.left == expected_left
            assert cropped_rectangle.bottom == expected_bottom
            assert cropped_rectangle.right == expected_right

        for cidx, cropped_rectangle in enumerate(cropped_rectangles, 1):
            random_image = get_random_image()
            function_name = inspect.currentframe().f_code.co_name
            draw_test(
                dir_name, f"{function_name}_{cidx}_geometry_{idx}", random_image, cropped_rectangle
            )


def test_relative_crop(random_rect_int, random_rect_float):
    for idx, rectangle in enumerate([random_rect_int, random_rect_float], 1):
        rect, _ = get_rect_and_coords(rectangle)
        cropper = Rectangle(150, 250, 300, 400)
        cropped_rectangles = rect.relative_crop(cropper)
        if (
            rect.right < cropper.left
            or rect.left > cropper.right
            or rect.bottom < cropper.top
            or rect.top > cropper.bottom
        ):
            assert cropped_rectangles == []
        else:
            assert len(cropped_rectangles) == 1
            cropped_rectangle = cropped_rectangles[0]

            if rect._integer_coords:
                expected_top = round(max(rect.top, cropper.top) - cropper.top)
                expected_left = round(max(rect.left, cropper.left) - cropper.left)
                expected_bottom = round(min(rect.bottom, cropper.bottom) - cropper.top)
                expected_right = round(min(rect.right, cropper.right) - cropper.left)
            else:
                expected_top = max(rect.top, cropper.top) - cropper.top
                expected_left = max(rect.left, cropper.left) - cropper.left
                expected_bottom = min(rect.bottom, cropper.bottom) - cropper.top
                expected_right = min(rect.right, cropper.right) - cropper.left

            assert cropped_rectangle.top == expected_top
            assert cropped_rectangle.left == expected_left
            assert cropped_rectangle.bottom == expected_bottom
            assert cropped_rectangle.right == expected_right

        for cidx, cropped_rectangle in enumerate(cropped_rectangles, 1):
            random_image = get_random_image()
            function_name = inspect.currentframe().f_code.co_name
            draw_test(
                dir_name, f"{function_name}_{cidx}_geometry_{idx}", random_image, cropped_rectangle
            )


def test_rotate(random_rect_int, random_rect_float):
    for idx, rectangle in enumerate([random_rect_int, random_rect_float], 1):
        rect, _ = get_rect_and_coords(rectangle)
        random_image = get_random_image()

        img_size, angle = random_image.shape[:2], random.randint(0, 360)
        rotator = ImageRotator(img_size, angle)
        rotated_rect = rect.rotate(rotator)

        expected_corners = [rotator.transform_point(p) for p in rect.corners]
        if rect._integer_coords:
            rows, cols = zip(*[(round(p.row), round(p.col)) for p in expected_corners])
        else:
            rows, cols = zip(*[(p.row, p.col) for p in expected_corners])
        assert rotated_rect.top == min(rows)
        assert rotated_rect.left == min(cols)
        assert rotated_rect.bottom == max(rows)
        assert rotated_rect.right == max(cols)

        function_name = inspect.currentframe().f_code.co_name
        draw_test(dir_name, f"{function_name}_geometry_{idx}", random_image, rotated_rect)


def test_resize(random_rect_int, random_rect_float):
    for idx, rectangle in enumerate([random_rect_int, random_rect_float], 1):
        rect, _ = get_rect_and_coords(rectangle)
        random_image = get_random_image()
        in_size = random_image.shape[:2]
        out_size = (random.randint(1000, 1200), random.randint(1000, 1200))
        resized_rect = rect.resize(in_size, out_size)

        frow = out_size[0] / in_size[0]
        fcol = out_size[1] / in_size[1]

        expected_transformed_corners = [
            (round(corner.row * frow), round(corner.col * fcol)) for corner in rect.corners
        ]
        check_corners_transformed(resized_rect.corners, expected_transformed_corners)

        function_name = inspect.currentframe().f_code.co_name
        draw_test(dir_name, f"{function_name}_geometry_{idx}", random_image, resized_rect)


def test_scale(random_rect_int, random_rect_float):
    for idx, rectangle in enumerate([random_rect_int, random_rect_float], 1):
        rect, _ = get_rect_and_coords(rectangle)
        factor = round(random.uniform(0, 1), 3)
        scaled_rect = rect.scale(factor)

        expected_transformed_corners = [
            (round(corner.row * factor), round(corner.col * factor)) for corner in rect.corners
        ]
        check_corners_transformed(scaled_rect.corners, expected_transformed_corners)

        random_image = get_random_image()
        function_name = inspect.currentframe().f_code.co_name
        draw_test(dir_name, f"{function_name}_geometry_{idx}", random_image, scaled_rect)


def test_translate(random_rect_int, random_rect_float):
    for idx, rectangle in enumerate([random_rect_int, random_rect_float], 1):
        rect, _ = get_rect_and_coords(rectangle)
        drow, dcol = random.randint(10, 150), random.randint(10, 350)
        translated_rect = rect.translate(drow, dcol)

        if rect._integer_coords:
            expected_top = round(rect.top + drow)
            expected_left = round(rect.left + dcol)
            expected_bottom = round(rect.bottom + drow)
            expected_right = round(rect.right + dcol)
        else:
            expected_top = rect.top + drow
            expected_left = rect.left + dcol
            expected_bottom = rect.bottom + drow
            expected_right = rect.right + dcol

        assert translated_rect.top == expected_top
        assert translated_rect.left == expected_left
        assert translated_rect.bottom == expected_bottom
        assert translated_rect.right == expected_right

        random_image = get_random_image()
        function_name = inspect.currentframe().f_code.co_name
        draw_test(dir_name, f"{function_name}_geometry_{idx}", random_image, translated_rect)


def test_fliplr(random_rect_int, random_rect_float):
    for idx, rectangle in enumerate([random_rect_int, random_rect_float], 1):
        rect, _ = get_rect_and_coords(rectangle)
        random_image = get_random_image()
        img_size = random_image.shape[:2]
        flipped_rect = rect.fliplr(img_size)
        assert flipped_rect.top == rect.top
        assert flipped_rect.left == img_size[1] - rect.right
        assert flipped_rect.bottom == rect.bottom
        assert flipped_rect.right == img_size[1] - rect.left

        function_name = inspect.currentframe().f_code.co_name
        draw_test(dir_name, f"{function_name}_geometry_{idx}", random_image, flipped_rect)


def test_flipud(random_rect_int, random_rect_float):
    for idx, rectangle in enumerate([random_rect_int, random_rect_float], 1):
        rect, _ = get_rect_and_coords(rectangle)
        random_image = get_random_image()
        img_size = random_image.shape[:2]
        flipped_rect = rect.flipud(img_size)
        assert flipped_rect.top == img_size[0] - rect.bottom
        assert flipped_rect.left == rect.left
        assert flipped_rect.bottom == img_size[0] - rect.top
        assert flipped_rect.right == rect.right

        function_name = inspect.currentframe().f_code.co_name
        draw_test(dir_name, f"{function_name}_geometry_{idx}", random_image, flipped_rect)


def test_draw_bool_compatible(random_rect_int, random_rect_float):
    for idx, rectangle in enumerate([random_rect_int, random_rect_float], 1):
        rect, _ = get_rect_and_coords(rectangle)
        random_image = get_random_image()
        rect._draw_bool_compatible(rect._draw_impl, random_image, color, thickness)
        assert np.any(random_image == color)

        function_name = inspect.currentframe().f_code.co_name
        draw_test(dir_name, f"{function_name}_geometry_{idx}", random_image)


def test_draw(random_rect_int, random_rect_float):
    for idx, rectangle in enumerate([random_rect_int, random_rect_float], 1):
        rect, _ = get_rect_and_coords(rectangle)
        random_image = get_random_image()
        rect.draw(random_image, color, thickness)
        assert np.any(random_image == color)

        function_name = inspect.currentframe().f_code.co_name
        draw_test(dir_name, f"{function_name}_geometry_{idx}", random_image)


def test_get_mask(random_rect_int, random_rect_float):
    for idx, rectangle in enumerate([random_rect_int, random_rect_float], 1):
        rect, _ = get_rect_and_coords(rectangle)
        random_image = get_random_image()
        img_size = random_image.shape[:2]
        mask = rect.get_mask(img_size)
        assert mask.shape == img_size, "Mask shape should match image size"
        assert mask.dtype == bool, "Mask dtype should be boolean"
        assert np.any(mask), "Mask should have at least one True value"

        new_bitmap = Bitmap(mask)
        function_name = inspect.currentframe().f_code.co_name
        draw_test(dir_name, f"{function_name}_geometry_{idx}", random_image, new_bitmap)


def test_draw_impl(random_rect_int, random_rect_float):
    for idx, rectangle in enumerate([random_rect_int, random_rect_float], 1):
        rect, _ = get_rect_and_coords(rectangle)
        random_image = get_random_image()
        rect._draw_impl(random_image, color, thickness)
        assert np.any(random_image == color)

        function_name = inspect.currentframe().f_code.co_name
        draw_test(dir_name, f"{function_name}_geometry_{idx}", random_image)


def test_draw_contour(random_rect_int, random_rect_float):
    for idx, rectangle in enumerate([random_rect_int, random_rect_float], 1):
        rect, _ = get_rect_and_coords(rectangle)
        random_image = get_random_image()
        rect.draw_contour(random_image, color, thickness)
        assert np.any(random_image == color)

        function_name = inspect.currentframe().f_code.co_name
        draw_test(dir_name, f"{function_name}_geometry_{idx}", random_image)


def test_draw_contour_impl(random_rect_int, random_rect_float):
    for idx, rectangle in enumerate([random_rect_int, random_rect_float], 1):
        rect, _ = get_rect_and_coords(rectangle)
        random_image = get_random_image()
        rect._draw_contour_impl(random_image, color, thickness)
        assert np.any(random_image == color)
        function_name = inspect.currentframe().f_code.co_name
        draw_test(dir_name, f"{function_name}_geometry_{idx}", random_image)


def test_area(random_rect_int, random_rect_float):
    for idx, rectangle in enumerate([random_rect_int, random_rect_float], 1):
        rect, _ = get_rect_and_coords(rectangle)
        expected_area = rect.width * rect.height
        assert rect.area == expected_area


def test_to_bbox(random_rect_int, random_rect_float):
    for idx, rectangle in enumerate([random_rect_int, random_rect_float], 1):
        rect, _ = get_rect_and_coords(rectangle)
        bbox_rect = rect.to_bbox()
        check_corners_equal(rect, bbox_rect)

        random_image = get_random_image()
        function_name = inspect.currentframe().f_code.co_name
        draw_test(dir_name, f"{function_name}_geometry_{idx}", random_image, bbox_rect)


def test_clone(random_rect_int, random_rect_float):
    for idx, rectangle in enumerate([random_rect_int, random_rect_float], 1):
        rect, _ = get_rect_and_coords(rectangle)
        clone_rect = rect.clone()
        check_corners_equal(rect, clone_rect)


def test_validate(random_rect_int, random_rect_float):
    for idx, rectangle in enumerate([random_rect_int, random_rect_float], 1):
        rect, _ = get_rect_and_coords(rectangle)
        assert rect.validate(rect.geometry_name(), settings=None) is None
        with pytest.raises(
            ValueError, match="Geometry validation error: shape names are mismatched!"
        ):
            rect.validate("different_shape_name", settings=None)


def test_config_from_json(random_rect_int, random_rect_float):
    for idx, rectangle in enumerate([random_rect_int, random_rect_float], 1):
        rect, _ = get_rect_and_coords(rectangle)
        config = {"key": "value"}
        returned_config = rect.config_from_json(config)
        assert returned_config == config


def test_config_to_json(random_rect_int, random_rect_float):
    for idx, rectangle in enumerate([random_rect_int, random_rect_float], 1):
        rect, _ = get_rect_and_coords(rectangle)
        config = {"key": "value"}
        returned_config = rect.config_to_json(config)
        assert returned_config == config


def test_allowed_transforms(random_rect_int, random_rect_float):
    for idx, rectangle in enumerate([random_rect_int, random_rect_float], 1):
        rect, _ = get_rect_and_coords(rectangle)
        allowed_transforms = rect.allowed_transforms()
        assert set(allowed_transforms) == set([AlphaMask, AnyGeometry, Bitmap, Polygon])


def test_convert(random_rect_int, random_rect_float):
    for idx, rectangle in enumerate([random_rect_int, random_rect_float], 1):
        rect, _ = get_rect_and_coords(rectangle)
        assert rect.convert(type(rect)) == [rect]
        assert rect.convert(AnyGeometry) == [rect]
        for new_geometry in rect.allowed_transforms():
            converted = rect.convert(new_geometry)
            for g in converted:
                assert isinstance(g, new_geometry) or isinstance(g, Rectangle)

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
            match="from {!r} to {!r}".format(rect.geometry_name(), Point.geometry_name()),
        ):
            rect.convert(Point)


# Rectangle specific methods
# --------------------------


def test_corners(random_rect_int, random_rect_float):
    for idx, rectangle in enumerate([random_rect_int, random_rect_float], 1):
        rect, coords = get_rect_and_coords(rectangle)
        corners = rect.corners
        for corner, coord in zip(corners, coords):
            assert corner.row == coord[0]
            assert corner.col == coord[1]


def test_center(random_rect_int, random_rect_float):
    for idx, rectangle in enumerate([random_rect_int, random_rect_float], 1):
        rect, coords = get_rect_and_coords(rectangle)
        center = rect.center

        expected_center_row = (rect.top + rect.bottom) // 2
        expected_center_col = (rect.left + rect.right) // 2

        assert center.row == expected_center_row
        assert center.col == expected_center_col


def test_width(random_rect_int, random_rect_float):
    for idx, rectangle in enumerate([random_rect_int, random_rect_float], 1):
        rect, _ = get_rect_and_coords(rectangle)
        assert rect.width == rect.right - rect.left + 1


def test_height(random_rect_int, random_rect_float):
    for idx, rectangle in enumerate([random_rect_int, random_rect_float], 1):
        rect, _ = get_rect_and_coords(rectangle)
        assert rect.height == rect.bottom - rect.top + 1


def test_top(random_rect_int, random_rect_float):
    for idx, rectangle in enumerate([random_rect_int, random_rect_float], 1):
        rect, _ = get_rect_and_coords(rectangle)
        top = rect.top
        assert top == rect.top


def test_left(random_rect_int, random_rect_float):
    for idx, rectangle in enumerate([random_rect_int, random_rect_float], 1):
        rect, _ = get_rect_and_coords(rectangle)
        left = rect.left
        assert left == rect.left


def test_bottom(random_rect_int, random_rect_float):
    for idx, rectangle in enumerate([random_rect_int, random_rect_float], 1):
        rect, _ = get_rect_and_coords(rectangle)
        bottom = rect.bottom
        assert bottom == rect.bottom


def test_right(random_rect_int, random_rect_float):
    for idx, rectangle in enumerate([random_rect_int, random_rect_float], 1):
        rect, _ = get_rect_and_coords(rectangle)
        right = rect.right
        assert right == rect.right


def from_array(random_rect_int, random_rect_float):
    height, width = random.randint(100, 800), random.randint(100, 800)
    np_array = np.zeros((height, width), dtype=np.uint64)
    rect_from_array = Rectangle.from_array(np_array)
    expected_rectangle_from_array = Rectangle(0, 0, height - 1, width - 1)
    check_corners_equal(rect_from_array, expected_rectangle_from_array)

    random_image = get_random_image()
    function_name = inspect.currentframe().f_code.co_name
    draw_test(dir_name, f"{function_name}_geometry", random_image, rect_from_array)


def from_size(random_rect_int, random_rect_float):
    size_int = (800, 800)
    rectangle_from_size_int = Rectangle.from_size(size_int)
    expected_rectangle_from_size_int = Rectangle(0, 0, 799, 799)
    check_corners_equal(rectangle_from_size_int, expected_rectangle_from_size_int)

    random_image = get_random_image()
    function_name = inspect.currentframe().f_code.co_name
    draw_test(
        dir_name, f"{function_name}_geometry_int_corners", random_image, rectangle_from_size_int
    )

    size_float = (800.525468, 800.57894163)
    rectangle_from_size_float = Rectangle.from_size(size_float)
    expected_rectangle_from_size_float = Rectangle(0, 0, 799.525468, 799.57894163)
    check_corners_equal(rectangle_from_size_float, expected_rectangle_from_size_float)

    random_image = get_random_image()
    function_name = inspect.currentframe().f_code.co_name
    draw_test(
        dir_name, f"{function_name}_geometry_float_corners", random_image, rectangle_from_size_float
    )


def from_geometries_list(random_rect_int, random_rect_float):
    geom_objs_int = [
        Point(100, 200),
        Polyline([PointLocation(730, 2104), PointLocation(2479, 402)]),
    ]
    rectangle_from_geom_objs_int = Rectangle.from_geometries_list(geom_objs_int)
    expected_rectangle_from_geom_objs_int = Rectangle(100, 200, 2479, 2104)
    check_corners_equal(rectangle_from_geom_objs_int, expected_rectangle_from_geom_objs_int)

    random_image = get_random_image()
    function_name = inspect.currentframe().f_code.co_name
    draw_test(dir_name, f"{function_name}_geometry_int", random_image, rectangle_from_geom_objs_int)

    geom_objs_float = [
        Point(100.124563, 200.724563),
        Polyline(
            [
                PointLocation(730.324563, 2104.3454643),
                PointLocation(2479.62345, 402.554336),
            ]
        ),
    ]
    rectangle_from_geom_objs_float = Rectangle.from_geometries_list(geom_objs_float)
    expected_rectangle_from_geom_objs_float = Rectangle(100, 201, 2480, 2104)
    check_corners_equal(rectangle_from_geom_objs_float, expected_rectangle_from_geom_objs_float)

    random_image = get_random_image()
    function_name = inspect.currentframe().f_code.co_name
    draw_test(
        dir_name,
        f"{function_name}_geometry_float",
        random_image,
        rectangle_from_geom_objs_float,
    )


def test_contains(random_rect_int, random_rect_float):
    for idx, rectangle in enumerate([random_rect_int, random_rect_float], 1):
        rect, _ = get_rect_and_coords(rectangle)
        assert rect.contains(rect) == True
        rect2 = Rectangle(rect.top + 1, rect.left + 1, rect.bottom - 1, rect.right - 1)
        assert rect.contains(rect2) == True
        rect3 = Rectangle(rect.top - 1, rect.left - 1, rect.bottom + 1, rect.right + 1)
        assert rect.contains(rect3) == False


def test_contains_point_location(random_rect_int, random_rect_float):
    for idx, rectangle in enumerate([random_rect_int, random_rect_float], 1):
        rect, _ = get_rect_and_coords(rectangle)
        pt = PointLocation(rect.top + 1, rect.left + 1)
        assert rect.contains_point_location(pt) == True


def test_to_size(random_rect_int, random_rect_float):
    for idx, rectangle in enumerate([random_rect_int, random_rect_float], 1):
        rect, _ = get_rect_and_coords(rectangle)
        size = rect.to_size()
        assert size == (rect.height, rect.width)


def test_get_cropped_numpy_slice(random_rect_int, random_rect_float):
    for idx, rectangle in enumerate([random_rect_int, random_rect_float], 1):
        rect, _ = get_rect_and_coords(rectangle)
        random_image = get_random_image()
        cropped_data = rect.get_cropped_numpy_slice(random_image)
        expected_height = round(rect.bottom) - round(rect.top) + 1
        expected_width = round(rect.right) - round(rect.left) + 1
        assert cropped_data.shape[:2] == (expected_height, expected_width)

        # cropped_bitmap = Bitmap(data=cropped_data, origin=PointLocation(rect.top, rect.left))
        # function_name = inspect.currentframe().f_code.co_name
        # draw_test(dir_name, f"{function_name}_geometry_{idx}", random_image, cropped_bitmap)


def test_intersects_with(random_rect_int, random_rect_float):
    for idx, rectangle in enumerate([random_rect_int, random_rect_float], 1):
        rect, _ = get_rect_and_coords(rectangle)
        rect2 = Rectangle(rect.top + 10, rect.left + 10, rect.bottom - 10, rect.right - 10)
        assert rect.intersects_with(rect2) == True, "Rectangles should intersect"
        rect3 = Rectangle(rect.bottom + 10, rect.right + 10, rect.bottom + 110, rect.right + 110)
        assert rect.intersects_with(rect3) == False, "Rectangles should not intersect"

        random_image = get_random_image()
        function_name = inspect.currentframe().f_code.co_name
        draw_test(dir_name, f"{function_name}_geometry_{idx}", random_image, rect, [255, 0, 0])
        draw_test(dir_name, f"{function_name}_geometry_{idx}", random_image, rect2, [0, 255, 0])
        draw_test(dir_name, f"{function_name}_geometry_{idx}", random_image, rect3, [0, 0, 255])


if __name__ == "__main__":
    pytest.main([__file__])

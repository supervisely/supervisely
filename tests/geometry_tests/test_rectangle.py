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
from supervisely.geometry.point import Point


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


def test_geometry_name(random_rect_int, random_rect_float):
    for rectangle in [random_rect_int, random_rect_float]:
        rect, _ = get_rect_and_coords(rectangle)
        assert rect.geometry_name() == "rectangle"


def test_name(random_rect_int, random_rect_float):
    for rectangle in [random_rect_int, random_rect_float]:
        rect, _ = get_rect_and_coords(rectangle)
        assert rect.name() == "rectangle"


def test_to_json(random_rect_int, random_rect_float):
    for rectangle in [random_rect_int, random_rect_float]:
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
    for rectangle in [random_rect_int, random_rect_float]:
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
    for rectangle in [random_rect_int, random_rect_float]:
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
            assert cropped_rectangle.top == max(rect.top, cropper.top)
            assert cropped_rectangle.left == max(rect.left, cropper.left)
            assert cropped_rectangle.bottom == min(rect.bottom, cropper.bottom)
            assert cropped_rectangle.right == min(rect.right, cropper.right)


def test_relative_crop(random_rect_int, random_rect_float):
    for rectangle in [random_rect_int, random_rect_float]:
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
            assert cropped_rectangle.top == max(rect.top, cropper.top) - cropper.top
            assert cropped_rectangle.left == max(rect.left, cropper.left) - cropper.left
            assert (
                cropped_rectangle.bottom
                == min(rect.bottom, cropper.bottom) - cropper.top
            )
            assert (
                cropped_rectangle.right == min(rect.right, cropper.right) - cropper.left
            )


def test_rotate(random_rect_int, random_rect_float):
    for rectangle in [random_rect_int, random_rect_float]:
        rect, _ = get_rect_and_coords(rectangle)
        rotator = ImageRotator((800, 800), 45)
        rotated_rect = rect.rotate(rotator)

        rotated_corners = [rotator.transform_point(p) for p in rect.corners]
        rows, cols = zip(*[(p.row, p.col) for p in rotated_corners])
        assert rotated_rect.top == min(rows)
        assert rotated_rect.left == min(cols)
        assert rotated_rect.bottom == max(rows)
        assert rotated_rect.right == max(cols)


def test_resize(random_rect_int, random_rect_float):
    for rectangle in [random_rect_int, random_rect_float]:
        rect, _ = get_rect_and_coords(rectangle)
        in_size = (300, 400)
        out_size = (600, 800)
        resized_rect = rect.resize(in_size, out_size)
        assert resized_rect.width == round((rect.width * out_size[1] / in_size[1]))
        assert resized_rect.height == round((rect.height * out_size[0] / in_size[0]))


def test_scale(random_rect_int, random_rect_float):
    for rectangle in [random_rect_int, random_rect_float]:
        rect, _ = get_rect_and_coords(rectangle)
        factor = 0.75
        scaled_rect = rect.scale(factor)
        assert scaled_rect.width == round((rect.width * factor))
        assert scaled_rect.height == round((rect.height * factor))


def test_translate(random_rect_int, random_rect_float):
    for rectangle in [random_rect_int, random_rect_float]:
        rect, _ = get_rect_and_coords(rectangle)
        drow, dcol = 50, 100
        translated_rect = rect.translate(drow, dcol)
        assert translated_rect.top == rect.top + drow
        assert translated_rect.left == rect.left + dcol
        assert translated_rect.bottom == rect.bottom + drow
        assert translated_rect.right == rect.right + dcol


def test_fliplr(random_rect_int, random_rect_float):
    img_size = (800, 800)
    for rectangle in [random_rect_int, random_rect_float]:
        rect, _ = get_rect_and_coords(rectangle)
        flipped_rect = rect.fliplr(img_size)
        assert flipped_rect.top == rect.top
        assert flipped_rect.left == img_size[1] - rect.right
        assert flipped_rect.bottom == rect.bottom
        assert flipped_rect.right == img_size[1] - rect.left


def test_flipud(random_rect_int, random_rect_float):
    img_size = (800, 800)
    for rectangle in [random_rect_int, random_rect_float]:
        rect, _ = get_rect_and_coords(rectangle)
        flipped_rect = rect.flipud(img_size)
        assert flipped_rect.top == img_size[0] - rect.bottom
        assert flipped_rect.left == rect.left
        assert flipped_rect.bottom == img_size[0] - rect.top
        assert flipped_rect.right == rect.right


def test_draw_bool_compatible(random_rect_int, random_rect_float):
    img_size = (800, 800)
    for rectangle in [random_rect_int, random_rect_float]:
        rect, _ = get_rect_and_coords(rectangle)
        bitmap = np.zeros(img_size, dtype=np.uint8)
        rect._draw_bool_compatible(rect._draw_impl, bitmap, 255, 1)
        assert np.any(bitmap)


def test_draw(random_rect_int, random_rect_float):
    img_size = (800, 800)
    for rectangle in [random_rect_int, random_rect_float]:
        rect, _ = get_rect_and_coords(rectangle)
        bitmap = np.zeros(img_size, dtype=np.uint8)
        color = [255, 255, 255]
        rect.draw(bitmap, color, thickness=1)
        assert np.any(bitmap)


def test_get_mask(random_rect_int, random_rect_float):
    img_size = (800, 800)
    for rectangle in [random_rect_int, random_rect_float]:
        rect, _ = get_rect_and_coords(rectangle)
        mask = rect.get_mask(img_size)
        assert mask.shape == img_size
        assert mask.dtype == np.bool
        assert np.any(mask)


def test_draw_impl(random_rect_int, random_rect_float):
    img_size = (800, 800)
    for rectangle in [random_rect_int, random_rect_float]:
        rect, _ = get_rect_and_coords(rectangle)
        bitmap = np.zeros((img_size[0], img_size[1], 3), dtype=np.uint8)
        color = (255, 255, 255)
        rect._draw_impl(bitmap, color)
        assert np.any(bitmap[rect.top : rect.bottom, rect.left : rect.right] == color)


def test_draw_contour(random_rect_int, random_rect_float):
    img_size = (800, 800)
    for rectangle in [random_rect_int, random_rect_float]:
        rect, _ = get_rect_and_coords(rectangle)
        bitmap = np.zeros(img_size, dtype=np.uint8)
        color = [255, 255, 255]
        rect.draw_contour(bitmap, color, thickness=1)
        assert np.any(bitmap)


def test_draw_contour_impl(random_rect_int, random_rect_float):
    img_size = (800, 800)
    for rectangle in [random_rect_int, random_rect_float]:
        rect, _ = get_rect_and_coords(rectangle)
        bitmap = np.zeros((img_size[0], img_size[1], 3), dtype=np.uint8)
        color = (255, 255, 255)
        rect._draw_contour_impl(bitmap, color)
        assert np.any(bitmap[rect.top : rect.bottom, rect.left] == color)
        assert np.any(bitmap[rect.top : rect.bottom, rect.right] == color)
        assert np.any(bitmap[rect.top, rect.left : rect.right] == color)
        assert np.any(bitmap[rect.bottom, rect.left : rect.right] == color)


def test_area(random_rect_int, random_rect_float):
    for rectangle in [random_rect_int, random_rect_float]:
        rect, _ = get_rect_and_coords(rectangle)
        expected_area = rect.width * rect.height
        assert rect.area == expected_area


def test_to_bbox(random_rect_int, random_rect_float):
    for rectangle in [random_rect_int, random_rect_float]:
        rect, _ = get_rect_and_coords(rectangle)
        bbox_rect = rect.to_bbox()
        check_corners_equal(rect, bbox_rect)


def test_clone(random_rect_int, random_rect_float):
    for rectangle in [random_rect_int, random_rect_float]:
        rect, _ = get_rect_and_coords(rectangle)
        clone_rect = rect.clone()
        check_corners_equal(rect, clone_rect)


def test_validate(random_rect_int, random_rect_float):
    for rectangle in [random_rect_int, random_rect_float]:
        rect, _ = get_rect_and_coords(rectangle)
        assert rect.validate(rect.geometry_name(), settings=None) is None
        with pytest.raises(
            ValueError, match="Geometry validation error: shape names are mismatched!"
        ):
            rect.validate("different_shape_name", settings=None)


def test_config_from_json(random_rect_int, random_rect_float):
    for rectangle in [random_rect_int, random_rect_float]:
        rect, _ = get_rect_and_coords(rectangle)
        config = {"key": "value"}
        returned_config = rect.config_from_json(config)
        assert returned_config == config


def test_config_to_json(random_rect_int, random_rect_float):
    for rectangle in [random_rect_int, random_rect_float]:
        rect, _ = get_rect_and_coords(rectangle)
        config = {"key": "value"}
        returned_config = rect.config_to_json(config)
        assert returned_config == config


def test_allowed_transforms(random_rect_int, random_rect_float):
    for rectangle in [random_rect_int, random_rect_float]:
        rect, _ = get_rect_and_coords(rectangle)
        allowed_transforms = rect.allowed_transforms()
        assert set(allowed_transforms) == set([AlphaMask, AnyGeometry, Bitmap, Polygon])


def test_convert(random_rect_int, random_rect_float):
    for rectangle in [random_rect_int, random_rect_float]:
        rect, _ = get_rect_and_coords(rectangle)
        assert rect.convert(type(rect)) == [rect]
        assert rect.convert(AnyGeometry) == [rect]
        for new_geometry in rect.allowed_transforms():
            converted = rect.convert(new_geometry)
            assert all(isinstance(g, new_geometry) for g in converted)

        class NotAllowedGeometry:
            pass

        with pytest.raises(
            NotImplementedError,
            match="from {!r} to {!r}".format(
                rect.geometry_name(), "NotAllowedGeometry"
            ),
        ):
            rect.convert(NotAllowedGeometry)


# Rectangle specific methods
# --------------------------


def test_corners(random_rect_int, random_rect_float):
    for rectangle in [random_rect_int, random_rect_float]:
        rect, coords = get_rect_and_coords(rectangle)
        corners = rect.corners
        for corner, coord in zip(corners, coords):
            assert corner.row == coord[0]
            assert corner.col == coord[1]


def test_center(random_rect_int, random_rect_float):
    for rectangle in [random_rect_int, random_rect_float]:
        rect, coords = get_rect_and_coords(rectangle)
        center = rect.center
        assert center.row == (rect.top + rect.bottom) / 2
        assert center.col == (rect.left + rect.right) / 2


def test_width(random_rect_int, random_rect_float):
    for rectangle in [random_rect_int, random_rect_float]:
        rect, _ = get_rect_and_coords(rectangle)
        assert rect.width == rect.right - rect.left


def test_height(random_rect_int, random_rect_float):
    for rectangle in [random_rect_int, random_rect_float]:
        rect, _ = get_rect_and_coords(rectangle)
        assert rect.height == rect.bottom - rect.top


def test_top(random_rect_int, random_rect_float):
    for rectangle in [random_rect_int, random_rect_float]:
        rect, _ = get_rect_and_coords(rectangle)
        top = rect.top
        assert top == rect.top


def test_left(random_rect_int, random_rect_float):
    for rectangle in [random_rect_int, random_rect_float]:
        rect, _ = get_rect_and_coords(rectangle)
        left = rect.left
        assert left == rect.left


def test_bottom(random_rect_int, random_rect_float):
    for rectangle in [random_rect_int, random_rect_float]:
        rect, _ = get_rect_and_coords(rectangle)
        bottom = rect.bottom
        assert bottom == rect.bottom


def test_right(random_rect_int, random_rect_float):
    for rectangle in [random_rect_int, random_rect_float]:
        rect, _ = get_rect_and_coords(rectangle)
        right = rect.right
        assert right == rect.right


def from_array(random_rect_int, random_rect_float):
    np_array = np.zeros((800, 800), dtype=np.uint64)
    rect_from_array = Rectangle.from_array(np_array)
    expected_rectangle_from_array = Rectangle(0, 0, 799, 799)
    check_corners_equal(rect_from_array, expected_rectangle_from_array)


def from_size(random_rect_int, random_rect_float):
    size_int = (800, 800)
    rectangle_from_size_int = Rectangle.from_size(size_int)
    expected_rectangle_from_size_int = Rectangle(0, 0, 799, 799)
    check_corners_equal(rectangle_from_size_int, expected_rectangle_from_size_int)

    size_float = (800.525468, 800.57894163)
    rectangle_from_size_float = Rectangle.from_size(size_float)
    expected_rectangle_from_size_float = Rectangle(0, 0, 799.525468, 799.57894163)
    check_corners_equal(rectangle_from_size_float, expected_rectangle_from_size_float)


def from_geometries_list(random_rect_int, random_rect_float):
    geom_objs_int = [
        Point(100, 200),
        Polyline([PointLocation(730, 2104), PointLocation(2479, 402)]),
    ]
    rectangle_from_geom_objs_int = Rectangle.from_geometries_list(geom_objs_int)
    expected_rectangle_from_geom_objs_int = Rectangle(100, 200, 2479, 2104)
    check_corners_equal(
        rectangle_from_geom_objs_int, expected_rectangle_from_geom_objs_int
    )

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
    check_corners_equal(
        rectangle_from_geom_objs_float, expected_rectangle_from_geom_objs_float
    )


def test_contains(random_rect_int, random_rect_float):
    for rectangle in [random_rect_int, random_rect_float]:
        rect, _ = get_rect_and_coords(rectangle)
        assert rect.contains(rect) == True
        rect2 = Rectangle(rect.top + 1, rect.left + 1, rect.bottom - 1, rect.right - 1)
        assert rect.contains(rect2) == True
        rect3 = Rectangle(rect.top - 1, rect.left - 1, rect.bottom + 1, rect.right + 1)
        assert rect.contains(rect3) == False


def test_contains_point_location(random_rect_int, random_rect_float):
    for rectangle in [random_rect_int, random_rect_float]:
        rect, _ = get_rect_and_coords(rectangle)
        pt = PointLocation(rect.top + 1, rect.left + 1)
        assert rect.contains_point_location(pt) == True


def test_to_size(random_rect_int, random_rect_float):
    for rectangle in [random_rect_int, random_rect_float]:
        rect, _ = get_rect_and_coords(rectangle)
        size = rect.to_size()
        assert size == (rect.width, rect.height)


def test_get_cropped_numpy_slice(random_rect_int, random_rect_float):
    img_size = (800, 800)
    for rectangle in [random_rect_int, random_rect_float]:
        rect, _ = get_rect_and_coords(rectangle)
        data = np.zeros(img_size)
        cropped_data = rect.get_cropped_numpy_slice(data)
        assert cropped_data.shape == (rect.height, rect.width)


def test_intersects_with(random_rect_int, random_rect_float):
    for rectangle in [random_rect_int, random_rect_float]:
        rect, _ = get_rect_and_coords(rectangle)
        rect2 = Rectangle(
            rect.top + 100, rect.left + 100, rect.bottom + 100, rect.right + 100
        )
        assert rect.intersects_with(rect2) == True
        rect3 = Rectangle(
            rect.top - 100, rect.left - 100, rect.bottom - 100, rect.right - 100
        )
        assert rect.intersects_with(rect3) == False


if __name__ == "__main__":
    pytest.main([__file__])

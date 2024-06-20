import random

import numpy as np
import pytest

import supervisely as sly
from supervisely.geometry.image_rotator import ImageRotator


def check_corners(rect: sly.Rectangle, expected_rect: sly.Rectangle):
    assert rect.top == expected_rect.top
    assert rect.left == expected_rect.left
    assert rect.bottom == expected_rect.bottom
    assert rect.right == expected_rect.right


def test_constructor():
    rectangle_int = sly.Rectangle(100, 200, 300, 400)
    assert rectangle_int.top == 100
    assert rectangle_int.left == 200
    assert rectangle_int.bottom == 300
    assert rectangle_int.right == 400

    rectangle_float = sly.Rectangle(100.123456, 200.123456, 300.123456, 400.123456)
    assert rectangle_float.top == 100.123456
    assert rectangle_float.left == 200.123456
    assert rectangle_float.bottom == 300.123456
    assert rectangle_float.right == 400.123456


def test_to_json():
    rectangle_int = sly.Rectangle(100, 200, 300, 400)
    json_data_int = rectangle_int.to_json()
    expected_json_int = {"points": {"exterior": [[200, 100], [400, 300]], "interior": []}}
    assert json_data_int == expected_json_int

    rectangle_float = sly.Rectangle(100.123456, 200.123456, 300.123456, 400.123456)
    json_data_float = rectangle_float.to_json()
    expected_json_float = {
        "points": {
            "exterior": [[200.123456, 100.123456], [400.123456, 300.123456]],
            "interior": [],
        }
    }
    assert json_data_float == expected_json_float


def test_crop():
    rectangle_int = sly.Rectangle(100, 200, 300, 400)
    other_rectangle_int = sly.Rectangle(150, 250, 350, 450)
    cropped_rectangles_int = rectangle_int.crop(other_rectangle_int)
    expected_cropped_rectangles_int = [sly.Rectangle(150, 250, 300, 400)]
    assert len(cropped_rectangles_int) == len(expected_cropped_rectangles_int)
    assert cropped_rectangles_int[0].top == expected_cropped_rectangles_int[0].top
    check_corners(cropped_rectangles_int[0], expected_cropped_rectangles_int[0])

    rectangle_float = sly.Rectangle(100.123456, 200.123456, 300.123456, 400.123456)
    other_rectangle_float = sly.Rectangle(150.123456, 250.123456, 350.123456, 450.123456)
    cropped_rectangles_float = rectangle_float.crop(other_rectangle_float)
    expected_cropped_rectangles_float = [
        sly.Rectangle(150.123456, 250.123456, 300.123456, 400.123456)
    ]
    assert len(cropped_rectangles_float) == len(expected_cropped_rectangles_float)
    check_corners(cropped_rectangles_float[0], expected_cropped_rectangles_float[0])


def test_rotate():
    rectangle_int = sly.Rectangle(100, 200, 300, 400)
    rotator_int = ImageRotator((600, 800), 90)
    rotated_rectangle_int = rectangle_int.rotate(rotator_int)
    expected_rotated_rectangle_int = sly.Rectangle(399, 100, 599, 300)
    check_corners(rotated_rectangle_int, expected_rotated_rectangle_int)

    rectangle_float = sly.Rectangle(100.123456, 200.123456, 300.123456, 400.123456)
    rotator_float = ImageRotator((600, 800), 90)
    rotated_rectangle_float = rectangle_float.rotate(rotator_float)
    expected_rotated_rectangle_float = sly.Rectangle(399, 100, 599, 300)
    check_corners(rotated_rectangle_float, expected_rotated_rectangle_float)


def test_area():
    rectangle_int = sly.Rectangle(100, 200, 300, 400)
    assert rectangle_int.area == 40401

    rectangle_float = sly.Rectangle(100.123456, 200.123456, 300.123456, 400.123456)
    assert rectangle_float.area == 40400.999999999985


def test_center():
    rectangle_int = sly.Rectangle(100, 200, 300, 400)
    expected_center_int = sly.PointLocation(200, 300)
    assert rectangle_int.center.col == expected_center_int.col
    assert rectangle_int.center.row == expected_center_int.row

    rectangle_float = sly.Rectangle(100.123456, 200.123456, 300.123456, 400.123456)
    expected_center_float = sly.PointLocation(200.0, 300.0)
    assert rectangle_float.center.col == expected_center_float.col
    assert rectangle_float.center.row == expected_center_float.row


def test_corners():
    rectangle_int = sly.Rectangle(100, 200, 300, 400)
    corners_int = rectangle_int.corners
    expected_corners_int = [
        sly.PointLocation(row=100, col=200),
        sly.PointLocation(row=100, col=400),
        sly.PointLocation(row=300, col=400),
        sly.PointLocation(row=300, col=200),
    ]
    assert len(corners_int) == len(expected_corners_int)
    for corner, expected_corner in zip(corners_int, expected_corners_int):
        assert corner.row == expected_corner.row
        assert corner.col == expected_corner.col

    rectangle_float = sly.Rectangle(100.123456, 200.123456, 300.123456, 400.123456)
    corners_float = rectangle_float.corners
    expected_corners_float = [
        sly.PointLocation(row=100.123456, col=200.123456),
        sly.PointLocation(row=100.123456, col=400.123456),
        sly.PointLocation(row=300.123456, col=400.123456),
        sly.PointLocation(row=300.123456, col=200.123456),
    ]
    assert len(corners_float) == len(expected_corners_float)
    for corner, expected_corner in zip(corners_float, expected_corners_float):
        assert corner.row == expected_corner.row
        assert corner.col == expected_corner.col


def test_from_json():
    figure_json_int = {"points": {"exterior": [[100, 100], [900, 700]], "interior": []}}
    figure_int = sly.Rectangle.from_json(figure_json_int)
    assert figure_int.top == 100
    assert figure_int.left == 100
    assert figure_int.bottom == 700
    assert figure_int.right == 900

    figure_json_float = {
        "points": {
            "exterior": [[100.123456, 100.123456], [900.123456, 700.123456]],
            "interior": [],
        }
    }
    figure_float = sly.Rectangle.from_json(figure_json_float)
    assert figure_float.top == 100.123456
    assert figure_float.left == 100.123456
    assert figure_float.bottom == 700.123456
    assert figure_float.right == 900.123456


def test_resize():
    rectangle_int = sly.Rectangle(100, 200, 300, 400)
    resized_rect_int = rectangle_int.resize((800, 600), (400, 300))
    expected_resized_rect_int = sly.Rectangle(50, 100, 150, 200)
    check_corners(resized_rect_int, expected_resized_rect_int)

    rectangle_float = sly.Rectangle(100.123456, 200.123456, 300.123456, 400.123456)
    resized_rect_float = rectangle_float.resize((800, 600), (400, 300))
    expected_resized_rect_float = sly.Rectangle(50, 100, 150, 200)
    check_corners(resized_rect_float, expected_resized_rect_float)


def test_scale():
    rectangle_int = sly.Rectangle(100, 200, 300, 400)
    scaled_rect_int = rectangle_int.scale(0.5)
    expected_scaled_rect_int = sly.Rectangle(50, 100, 150, 200)
    check_corners(scaled_rect_int, expected_scaled_rect_int)

    rectangle_float = sly.Rectangle(100.123456, 200.123456, 300.123456, 400.123456)
    scaled_rect_float = rectangle_float.scale(0.5)
    expected_scaled_rect_float = sly.Rectangle(50, 100, 150, 200)
    check_corners(scaled_rect_float, expected_scaled_rect_float)


def test_translate():
    rectangle = sly.Rectangle(100, 200, 300, 400)
    translated_rect = rectangle.translate(50, 100)
    expected_translated_rect = sly.Rectangle(150, 300, 350, 500)
    check_corners(translated_rect, expected_translated_rect)

    translated_rect_negative = rectangle.translate(-50, -100)
    expected_translated_rect_negative = sly.Rectangle(50, 100, 250, 300)
    check_corners(translated_rect_negative, expected_translated_rect_negative)


def test_fliplr():
    rectangle_int = sly.Rectangle(100, 200, 300, 400)
    img_size_int = (800, 600)
    fliplr_rect_int = rectangle_int.fliplr(img_size_int)
    expected_fliplr_rect_int = sly.Rectangle(100, 200, 300, 400)
    check_corners(fliplr_rect_int, expected_fliplr_rect_int)

    rectangle_float = sly.Rectangle(100.123456, 200.123456, 300.123456, 400.123456)
    img_size_float = (800, 600)
    fliplr_rect_float = rectangle_float.fliplr(img_size_float)
    expected_fliplr_rect_float = sly.Rectangle(
        100.123456, 199.87654400000002, 300.123456, 399.87654399999997
    )
    check_corners(fliplr_rect_float, expected_fliplr_rect_float)


def test_flipud():
    rectangle_int = sly.Rectangle(100, 200, 300, 400)
    img_size_int = (800, 600)
    flipud_rect_int = rectangle_int.flipud(img_size_int)
    expected_flipud_rect_int = sly.Rectangle(500, 200, 700, 400)
    check_corners(flipud_rect_int, expected_flipud_rect_int)

    rectangle_float = sly.Rectangle(100.123456, 200.123456, 300.123456, 400.123456)
    img_size_float = (500.123456, 600.123456)
    flipud_rect_float = rectangle_float.flipud(img_size_float)
    expected_flipud_rect_float = sly.Rectangle(200.0, 200.123456, 400.0, 400.123456)
    check_corners(flipud_rect_float, expected_flipud_rect_float)


def test_to_bbox():
    rectangle_int = sly.Rectangle(100, 200, 300, 400)
    bbox_rect_int = rectangle_int.to_bbox()
    expected_bbox_rect_int = sly.Rectangle(100, 200, 300, 400)
    check_corners(bbox_rect_int, expected_bbox_rect_int)

    rectangle_float = sly.Rectangle(100.123456, 200.123456, 300.123456, 400.123456)
    bbox_rect_float = rectangle_float.to_bbox()
    expected_bbox_rect_float = sly.Rectangle(100.123456, 200.123456, 300.123456, 400.123456)
    check_corners(bbox_rect_float, expected_bbox_rect_float)


def test_from_array():
    # numpy arrays cannot have float dimensions
    np_array_int = np.zeros((800, 600), dtype=np.uint64)
    rectangle_from_array_int = sly.Rectangle.from_array(np_array_int)
    expected_rectangle_from_array_int = sly.Rectangle(0, 0, 799, 599)
    check_corners(rectangle_from_array_int, expected_rectangle_from_array_int)


def test_from_size():
    size_int = (300, 400)
    rectangle_from_size_int = sly.Rectangle.from_size(size_int)
    expected_rectangle_from_size_int = sly.Rectangle(0, 0, 299, 399)
    check_corners(rectangle_from_size_int, expected_rectangle_from_size_int)

    size_float = (300.525468, 400.57894163)
    rectangle_from_size_float = sly.Rectangle.from_size(size_float)
    expected_rectangle_from_size_float = sly.Rectangle(0, 0, 299.525468, 399.57894163)
    check_corners(rectangle_from_size_float, expected_rectangle_from_size_float)


def test_from_geometries_list():
    geom_objs_int = [
        sly.Point(100, 200),
        sly.Polyline([sly.PointLocation(730, 2104), sly.PointLocation(2479, 402)]),
    ]
    rectangle_from_geom_objs_int = sly.Rectangle.from_geometries_list(geom_objs_int)
    expected_rectangle_from_geom_objs_int = sly.Rectangle(100, 200, 2479, 2104)
    check_corners(rectangle_from_geom_objs_int, expected_rectangle_from_geom_objs_int)

    geom_objs_float = [
        sly.Point(100.124563, 200.724563),
        sly.Polyline(
            [
                sly.PointLocation(730.324563, 2104.3454643),
                sly.PointLocation(2479.62345, 402.554336),
            ]
        ),
    ]
    rectangle_from_geom_objs_float = sly.Rectangle.from_geometries_list(geom_objs_float)
    expected_rectangle_from_geom_objs_float = sly.Rectangle(100, 201, 2480, 2104)
    check_corners(rectangle_from_geom_objs_float, expected_rectangle_from_geom_objs_float)


def test_width():
    rect_int = sly.Rectangle(200, 250, 400, 500)
    assert rect_int.width == 251

    rect_float = sly.Rectangle(200.352234, 250.8795651, 400.1235536, 500.87464)
    assert rect_float.width == 250.9950749


def test_height():
    rect_int = sly.Rectangle(200, 250, 400, 500)
    assert rect_int.height == 201

    rect_float = sly.Rectangle(200.352234, 250.8795651, 400.1235536, 500.87464)
    assert rect_float.height == 200.77131959999997


def test_contains():
    rect_int = sly.Rectangle(200, 250, 400, 500)
    contained_rect_int = sly.Rectangle(250, 300, 350, 400)
    assert rect_int.contains(contained_rect_int) == True

    rect_float = sly.Rectangle(200.5, 250.5, 400.5, 500.5)
    contained_rect_float = sly.Rectangle(250.5, 300.5, 350.5, 400.5)
    assert rect_float.contains(contained_rect_float) == True


def test_contains_point_location():
    rect_int = sly.Rectangle(200, 250, 400, 500)
    pt_int = sly.PointLocation(250, 300)
    assert rect_int.contains_point_location(pt_int) == True

    rect_float = sly.Rectangle(200.5, 250.5, 400.5, 500.5)
    pt_float = sly.PointLocation(250.5, 300.5)
    assert rect_float.contains_point_location(pt_float) == True


def test_to_size():
    rect_int = sly.Rectangle(200, 250, 400, 500)
    assert rect_int.to_size() == (201, 251)

    rect_float = sly.Rectangle(200.352234, 250.8795651, 400.1235536, 500.87464)
    assert rect_float.to_size() == (200.77131959999997, 250.9950749)


def test_get_cropped_numpy_slice():
    rect_int = sly.Rectangle(200, 250, 400, 500)
    data_int = np.zeros((600, 600))
    cropped_data_int = rect_int.get_cropped_numpy_slice(data_int)
    assert cropped_data_int.shape == (201, 251)

    rect_float = sly.Rectangle(200.52345, 250.652342, 400.734555, 500.23445)
    data_float = np.zeros((600, 600))
    cropped_data_float = rect_float.get_cropped_numpy_slice(data_float)
    assert cropped_data_float.shape == (201, 250)


def test_intersects_with():
    rect1_int = sly.Rectangle(200, 250, 400, 500)
    rect2_int = sly.Rectangle(300, 350, 500, 600)
    assert rect1_int.intersects_with(rect2_int) == True

    rect3_int = sly.Rectangle(0, 0, 100, 100)
    assert rect1_int.intersects_with(rect3_int) == False

    rect1_float = sly.Rectangle(200.5, 250.5, 400.5, 500.5)
    rect2_float = sly.Rectangle(300.5, 350.5, 500.5, 600.5)
    assert rect1_float.intersects_with(rect2_float) == True

    rect3_float = sly.Rectangle(0.0, 0.0, 100.0, 100.0)
    assert rect1_float.intersects_with(rect3_float) == False


if __name__ == "__main__":
    pytest.main([__file__])
if __name__ == "__main__":
    pytest.main([__file__])

import inspect
import os
import random
from typing import List, Tuple, Union

import cv2
import numpy as np
import pytest
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
def random_mask_int() -> Tuple[Bitmap, np.ndarray, Tuple[Union[int, float], Union[int, float]]]:
    height = random.randint(200, 400)
    width = random.randint(200, 400)
    data_shape = (height, width)
    data = np.zeros(data_shape, dtype=np.bool_)
    origin_coords = [random.randint(0, 10), random.randint(0, 10)]
    origin = PointLocation(row=origin_coords[0], col=origin_coords[1])

    start_row, start_col = origin.row, origin.col
    end_row, end_col = height - random.randint(0, 50), width - random.randint(0, 50)
    data[start_row:end_row, start_col:end_col] = True

    bitmap = Bitmap(data=data, origin=origin)
    data = bitmap.data
    origin_coords = [bitmap.origin.row, bitmap.origin.col]
    return bitmap, data, origin_coords


@pytest.fixture
def random_mask_float() -> Tuple[Bitmap, np.ndarray, Tuple[Union[int, float], Union[int, float]]]:
    height = random.randint(200, 400)
    width = random.randint(200, 400)
    data_shape = (height, width)
    data = np.zeros(data_shape, dtype=np.bool_)

    origin_coords = [round(random.uniform(0, 10), 6), round(random.uniform(0, 10), 6)]
    origin = PointLocation(row=origin_coords[0], col=origin_coords[1])

    start_row, start_col = round(origin.row), round(origin.col)
    end_row, end_col = height - random.randint(0, 50), width - random.randint(0, 50)
    data[start_row:end_row, start_col:end_col] = True

    bitmap = Bitmap(data=data, origin=origin)
    data = bitmap.data
    origin_coords = [bitmap.origin.row, bitmap.origin.col]
    return bitmap, data, origin_coords


def get_bitmap_data_origin(
    mask: Tuple[Bitmap, np.ndarray, Tuple[Union[int, float], Union[int, float]]]
) -> Tuple[Bitmap, np.ndarray, Tuple[Union[int, float], Union[int, float]]]:
    return mask


def check_origin_equal(bitmap: Bitmap, origin: List[Union[int, float]]):
    assert [bitmap.origin.row, bitmap.origin.col] == origin


def test_geometry_name(random_mask_int, random_mask_float):
    for idx, mask in enumerate([random_mask_int, random_mask_float], 1):
        bitmap, data, origin = get_bitmap_data_origin(mask)
        assert bitmap.geometry_name() == "bitmap"


def test_name(random_mask_int, random_mask_float):
    for idx, mask in enumerate([random_mask_int, random_mask_float], 1):
        bitmap, data, origin = get_bitmap_data_origin(mask)
        assert bitmap.name() == "bitmap"


def test_to_json(random_mask_int, random_mask_float):
    for idx, mask in enumerate([random_mask_int, random_mask_float], 1):
        bitmap, data, origin = get_bitmap_data_origin(mask)
        base_64_data = bitmap.data_2_base64(data)
        expected_json = {
            "bitmap": {"origin": origin[::-1], "data": base_64_data},
            "shape": bitmap.name(),
            "geometryType": bitmap.name(),
        }
        assert bitmap.to_json() == expected_json


def test_from_json(random_mask_int, random_mask_float):
    for idx, mask in enumerate([random_mask_int, random_mask_float], 1):
        bitmap, data, origin = get_bitmap_data_origin(mask)
        base_64_data = bitmap.data_2_base64(data)
        bitmap_json = {
            "bitmap": {"origin": origin[::-1], "data": base_64_data},
            "shape": bitmap.name(),
            "geometryType": bitmap.name(),
        }
        bitmap = Bitmap.from_json(bitmap_json)
        assert isinstance(bitmap, Bitmap)
        check_origin_equal(bitmap, origin)


def test_crop(random_mask_int, random_mask_float):
    for idx, mask in enumerate([random_mask_int, random_mask_float], 1):
        bitmap, data, origin = get_bitmap_data_origin(mask)
        rect = Rectangle(top=10, left=10, bottom=20, right=20)
        cropped_bitmaps = bitmap.crop(rect)
        for cidx, cropped_bitmap in enumerate(cropped_bitmaps, 1):

            bitmap_height, bitmap_width = bitmap.data.shape[0], bitmap.data.shape[1]

            intersect_top = max(bitmap.origin.row, rect.top)
            intersect_left = max(bitmap.origin.col, rect.left)
            intersect_bottom = min(bitmap.origin.row + bitmap_height, rect.bottom)
            intersect_right = min(bitmap.origin.col + bitmap_width, rect.right)

            expected_height = max(0, intersect_bottom - intersect_top)
            expected_width = max(0, intersect_right - intersect_left)

            assert isinstance(cropped_bitmap, Bitmap)
            assert cropped_bitmap.data.size > 0, "Cropped bitmap data is empty."
            assert cropped_bitmap.data.shape[0] == round(expected_height) + 1
            assert cropped_bitmap.data.shape[1] == round(expected_width) + 1

            random_image = get_random_image()
            function_name = inspect.currentframe().f_code.co_name
            draw_test(
                dir_name, f"{function_name}_{cidx}_geometry_{idx}", random_image, cropped_bitmap
            )


def test_relative_crop(random_mask_int, random_mask_float):
    for idx, mask in enumerate([random_mask_int, random_mask_float], 1):
        bitmap, data, origin = get_bitmap_data_origin(mask)
        cropper = Rectangle(top=10, left=10, bottom=20, right=20)
        cropped_bitmaps = bitmap.relative_crop(cropper)
        for cidx, cropped_bitmap in enumerate(cropped_bitmaps):
            assert isinstance(cropped_bitmap, Bitmap)
            assert cropped_bitmap.data.size > 0, "Cropped bitmap data is empty."
            assert bitmap.data.shape != cropped_bitmap.data.shape
            assert bitmap.data.shape[0] != cropped_bitmap.data.shape[0]
            assert bitmap.data.shape[1] != cropped_bitmap.data.shape[1]

            random_image = get_random_image()
            function_name = inspect.currentframe().f_code.co_name
            draw_test(
                dir_name, f"{function_name}_{cidx}_geometry_{idx}", random_image, cropped_bitmap
            )


def test_rotate(random_mask_int, random_mask_float):
    for idx, mask in enumerate([random_mask_int, random_mask_float], 1):
        bitmap, data, origin = get_bitmap_data_origin(mask)
        random_image = get_random_image()
        img_size, angle = random_image.shape[:2], random.randint(0, 360)
        rotator = ImageRotator(img_size, angle)
        rotated_bitmap = bitmap.rotate(rotator)

        assert isinstance(rotated_bitmap, Bitmap)
        assert rotated_bitmap.data.shape != data.shape

        function_name = inspect.currentframe().f_code.co_name
        draw_test(dir_name, f"{function_name}_geometry_{idx}", random_image, rotated_bitmap)


def test_resize(random_mask_int, random_mask_float):
    for idx, mask in enumerate([random_mask_int, random_mask_float], 1):
        bitmap, data, origin = get_bitmap_data_origin(mask)
        in_size = data.shape[:2]
        out_size = (in_size[0] // 2, in_size[1] // 2)
        resized_bitmap = bitmap.resize(in_size, out_size)
        assert resized_bitmap.data.shape == out_size

        random_image = get_random_image()
        function_name = inspect.currentframe().f_code.co_name
        draw_test(dir_name, f"{function_name}_geometry_{idx}", random_image, resized_bitmap)


def test_scale(random_mask_int, random_mask_float):
    for idx, mask in enumerate([random_mask_int, random_mask_float], 1):
        factor = round(random.uniform(0, 1), 3)
        bitmap, data, origin = get_bitmap_data_origin(mask)
        scaled_bitmap = bitmap.scale(factor)
        assert scaled_bitmap.data.shape == (
            round(data.shape[0] * factor),
            round(data.shape[1] * factor),
        )

        random_image = get_random_image()
        function_name = inspect.currentframe().f_code.co_name
        draw_test(dir_name, f"{function_name}_geometry_{idx}", random_image, scaled_bitmap)


def test_translate(random_mask_int, random_mask_float):
    for idx, mask in enumerate([random_mask_int, random_mask_float], 1):
        bitmap, data, origin = get_bitmap_data_origin(mask)
        drow, dcol = random.randint(10, 150), random.randint(10, 350)
        translated_bitmap = bitmap.translate(drow, dcol)
        expected_trans_origin = [origin[0] + drow, origin[1] + dcol]
        check_origin_equal(translated_bitmap, expected_trans_origin)

        random_image = get_random_image()
        function_name = inspect.currentframe().f_code.co_name
        draw_test(dir_name, f"{function_name}_geometry_{idx}", random_image, translated_bitmap)


def test_fliplr(random_mask_int, random_mask_float):
    for idx, mask in enumerate([random_mask_int, random_mask_float], 1):
        bitmap, data, origin = get_bitmap_data_origin(mask)
        random_image = get_random_image()
        img_size = random_image.shape[:2]
        flipped_bitmap = bitmap.fliplr(img_size)
        assert np.array_equal(flipped_bitmap.data, np.fliplr(data))

        function_name = inspect.currentframe().f_code.co_name
        draw_test(dir_name, f"{function_name}_geometry_{idx}", random_image, flipped_bitmap)


def test_flipud(random_mask_int, random_mask_float):
    for idx, mask in enumerate([random_mask_int, random_mask_float], 1):
        bitmap, data, origin = get_bitmap_data_origin(mask)
        random_image = get_random_image()
        img_size = random_image.shape[:2]
        flipped_bitmap = bitmap.flipud(img_size)
        assert np.array_equal(flipped_bitmap.data, np.flipud(data))

        function_name = inspect.currentframe().f_code.co_name
        draw_test(dir_name, f"{function_name}_geometry_{idx}", random_image, flipped_bitmap)


def test_draw_bool_compatible(random_mask_int, random_mask_float):
    for idx, mask in enumerate([random_mask_int, random_mask_float], 1):
        bitmap, data, origin = get_bitmap_data_origin(mask)
        random_image = get_random_image()

        bitmap._draw_bool_compatible(bitmap._draw_impl, random_image, color, thickness)
        assert np.any(random_image == color)

        function_name = inspect.currentframe().f_code.co_name
        draw_test(dir_name, f"{function_name}_geometry_{idx}", random_image)


def test_draw(random_mask_int, random_mask_float):
    for idx, mask in enumerate([random_mask_int, random_mask_float], 1):
        bitmap, data, origin = get_bitmap_data_origin(mask)
        random_image = get_random_image()
        bitmap.draw(random_image, color, thickness)
        assert np.any(random_image == color)

        function_name = inspect.currentframe().f_code.co_name
        draw_test(dir_name, f"{function_name}_geometry_{idx}", random_image)


def test_get_mask(random_mask_int, random_mask_float):
    for idx, mask in enumerate([random_mask_int, random_mask_float], 1):
        bitmap, data, origin = get_bitmap_data_origin(mask)
        random_image = get_random_image()
        bmask = bitmap.get_mask(random_image.shape[:2])
        assert bmask.shape == random_image.shape[:2]
        assert bmask.dtype == np.bool
        assert np.any(bmask == True)

        new_bitmap = Bitmap(bmask)
        function_name = inspect.currentframe().f_code.co_name
        draw_test(dir_name, f"{function_name}_geometry_{idx}", random_image, new_bitmap)


def test_draw_impl(random_mask_int, random_mask_float):
    for idx, mask in enumerate([random_mask_int, random_mask_float], 1):
        bitmap, data, origin = get_bitmap_data_origin(mask)
        random_image = get_random_image()
        bitmap._draw_impl(random_image, color, thickness)
        assert np.any(random_image == color)

        function_name = inspect.currentframe().f_code.co_name
        draw_test(dir_name, f"{function_name}_geometry_{idx}", random_image)


def test_draw_contour(random_mask_int, random_mask_float):
    for idx, mask in enumerate([random_mask_int, random_mask_float], 1):
        bitmap, data, origin = get_bitmap_data_origin(mask)
        random_image = get_random_image()
        bitmap.draw_contour(random_image, color, thickness)
        assert np.any(random_image == color)

        function_name = inspect.currentframe().f_code.co_name
        draw_test(dir_name, f"{function_name}_geometry_{idx}", random_image)


def test__draw_contour_impl(random_mask_int, random_mask_float):
    for idx, mask in enumerate([random_mask_int, random_mask_float], 1):
        bitmap, data, origin = get_bitmap_data_origin(mask)
        random_image = get_random_image()
        bitmap._draw_contour_impl(random_image, color, thickness)
        assert np.any(random_image == color)

        function_name = inspect.currentframe().f_code.co_name
        draw_test(dir_name, f"{function_name}_geometry_{idx}", random_image)


def test_area(random_mask_int, random_mask_float):
    for idx, mask in enumerate([random_mask_int, random_mask_float], 1):
        bitmap, data, origin = get_bitmap_data_origin(mask)
        assert bitmap.area == np.sum(data)


def test_to_bbox(random_mask_int, random_mask_float):
    for idx, mask in enumerate([random_mask_int, random_mask_float], 1):
        bitmap, data, origin = get_bitmap_data_origin(mask)
        bitmap_bbox = bitmap.to_bbox()
        assert isinstance(bitmap_bbox, Rectangle)
        assert round(bitmap_bbox.height) == data.shape[0]
        assert round(bitmap_bbox.width) == data.shape[1]

        random_image = get_random_image()
        function_name = inspect.currentframe().f_code.co_name
        draw_test(dir_name, f"{function_name}_geometry_{idx}", random_image, bitmap_bbox)


def test_clone(random_mask_int, random_mask_float):
    for idx, mask in enumerate([random_mask_int, random_mask_float], 1):
        bitmap, data, origin = get_bitmap_data_origin(mask)
        cloned_bitmap = bitmap.clone()
        assert isinstance(cloned_bitmap, Bitmap)
        check_origin_equal(cloned_bitmap, origin)


def test_validate(random_mask_int, random_mask_float):
    for idx, mask in enumerate([random_mask_int, random_mask_float], 1):
        bitmap, data, origin = get_bitmap_data_origin(mask)
        assert bitmap.validate(bitmap.geometry_name(), settings=None) is None
        with pytest.raises(
            ValueError, match="Geometry validation error: shape names are mismatched!"
        ):
            bitmap.validate("different_shape_name", settings=None)


def test_config_from_json(random_mask_int, random_mask_float):
    for idx, mask in enumerate([random_mask_int, random_mask_float], 1):
        bitmap, data, origin = get_bitmap_data_origin(mask)
        config = {"key": "value"}
        returned_config = bitmap.config_from_json(config)
        assert returned_config == config


def test_config_to_json(random_mask_int, random_mask_float):
    for idx, mask in enumerate([random_mask_int, random_mask_float], 1):
        bitmap, data, origin = get_bitmap_data_origin(mask)
        config = {"key": "value"}
        returned_config = bitmap.config_to_json(config)
        assert returned_config == config


def test_allowed_transforms(random_mask_int, random_mask_float):
    for idx, mask in enumerate([random_mask_int, random_mask_float], 1):
        bitmap, data, origin = get_bitmap_data_origin(mask)
        allowed_transforms = bitmap.allowed_transforms()
        assert set(allowed_transforms) == set([AlphaMask, AnyGeometry, Polygon, Rectangle])


def test_convert(random_mask_int, random_mask_float):
    for idx, mask in enumerate([random_mask_int, random_mask_float], 1):
        bitmap, data, origin = get_bitmap_data_origin(mask)
        assert bitmap.convert(type(bitmap)) == [bitmap]
        assert bitmap.convert(AnyGeometry) == [bitmap]
        for new_geometry in bitmap.allowed_transforms():
            converted = bitmap.convert(new_geometry)
            for g in converted:
                assert isinstance(g, new_geometry) or isinstance(g, Bitmap)

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
            match="from {!r} to {!r}".format(bitmap.geometry_name(), Point.geometry_name()),
        ):
            bitmap.convert(Point)


# Bitmap specific methods
# ------------------------


def test_data(random_mask_int, random_mask_float):
    for idx, mask in enumerate([random_mask_int, random_mask_float], 1):
        bitmap, data, origin = get_bitmap_data_origin(mask)
        assert np.array_equal(bitmap.data, data)


def test_origin(random_mask_int, random_mask_float):
    for idx, mask in enumerate([random_mask_int, random_mask_float], 1):
        bitmap, data, origin = get_bitmap_data_origin(mask)
        check_origin_equal(bitmap, origin)


def test_base64_2_data(random_mask_int, random_mask_float):
    for idx, mask in enumerate([random_mask_int, random_mask_float], 1):
        bitmap, data, origin = get_bitmap_data_origin(mask)
        encoded = bitmap.data_2_base64(data)
        decoded = Bitmap.base64_2_data(encoded)
        assert np.array_equal(decoded, data)


def test_data_2_base64(random_mask_int, random_mask_float):
    for idx, mask in enumerate([random_mask_int, random_mask_float], 1):
        bitmap, data, origin = get_bitmap_data_origin(mask)
        encoded = bitmap.data_2_base64(data)
        decoded = Bitmap.base64_2_data(encoded)
        assert np.array_equal(decoded, data)


def test_skeletonize(random_mask_int, random_mask_float):
    for idx, mask in enumerate([random_mask_int, random_mask_float], 1):
        bitmap, data, origin = get_bitmap_data_origin(mask)
        for method in SkeletonizeMethod:
            skeleton = bitmap.skeletonize(method)
            assert isinstance(skeleton, Bitmap)

            random_image = get_random_image()
            function_name = inspect.currentframe().f_code.co_name
            draw_test(
                dir_name, f"{function_name}_geometry_{idx}_{method.name}", random_image, skeleton
            )


def test_to_contours(random_mask_int, random_mask_float):
    for idx, mask in enumerate([random_mask_int, random_mask_float], 1):
        bitmap, data, origin = get_bitmap_data_origin(mask)
        contours = bitmap.to_contours()
        assert isinstance(contours, list), "Output should be a list"
        assert len(contours) > 0, "List should not be empty"
        for cidx, contour in enumerate(contours, 1):
            assert isinstance(contour, Polygon), "All elements in the list should be Polygons"

            random_image = get_random_image()
            function_name = inspect.currentframe().f_code.co_name
            draw_test(dir_name, f"{function_name}_{cidx}_geometry_{idx}", random_image, contour)


def test_bitwise_mask(random_mask_int, random_mask_float):
    for idx, mask in enumerate([random_mask_int, random_mask_float], 1):
        bitmap, data, origin = get_bitmap_data_origin(mask)
        return
        # @TODO: Fix this test
        full_expected_size = (
            round(max(bitmap.origin.row + data.shape[0], data.shape[0])),
            round(max(bitmap.origin.col + data.shape[1], data.shape[1])),
        )
        mask = np.zeros(full_expected_size, dtype=data.dtype)
        assert (
            mask.shape[0] >= data.shape[0] and mask.shape[1] >= data.shape[1]
        ), "Mask and data shapes must be compatible"
        result = bitmap.bitwise_mask(mask, np.logical_and)
        assert result != [], "Result should not be an empty list."
        assert isinstance(result, Bitmap), "Result is not an instance of Bitmap."
        expected_shape = (origin.row + data.shape[0], origin.col + data.shape[1])
        assert (
            result.data.shape == expected_shape
        ), f"Result shape {result.data.shape} does not match expected shape {expected_shape}."


def test_from_path(random_mask_int, random_mask_float):
    pass


if __name__ == "__main__":
    pytest.main([__file__])

import random
from typing import List, Tuple, Union

import cv2
import numpy as np
import pytest

from supervisely.geometry.alpha_mask import AlphaMask
from supervisely.geometry.any_geometry import AnyGeometry
from supervisely.geometry.bitmap import Bitmap, SkeletonizeMethod
from supervisely.geometry.image_rotator import ImageRotator
from supervisely.geometry.point import Point
from supervisely.geometry.point_location import PointLocation, _flip_row_col_order
from supervisely.geometry.polygon import Polygon
from supervisely.geometry.polyline import Polyline
from supervisely.geometry.rectangle import Rectangle

# Draw Settings
color = [255, 255, 255]
thickness = 1


def draw_circle(image, circle_radius=50, color=(255, 255, 255), thickness=-1):
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    cv2.circle(image, center, circle_radius, color, thickness)


@pytest.fixture
def random_image() -> np.ndarray:
    image_shape = (random.randint(801, 1000), random.randint(801, 1000), 3)
    background_color = [0, 0, 0]
    bitmap = np.full(image_shape, background_color, dtype=np.uint8)
    return bitmap


@pytest.fixture
def random_mask_int() -> Tuple[Bitmap, np.ndarray, Tuple[Union[int, float], Union[int, float]]]:
    height = random.randint(200, 800)
    width = random.randint(200, 800)
    data_shape = (height, width)
    # data = np.zeros(data_shape, dtype=np.uint8)
    # draw_circle(data)
    data = np.ones(data_shape, dtype=np.uint8)
    origin_coords = [random.randint(200, 800), random.randint(200, 800)]
    origin = PointLocation(row=origin_coords[0], col=origin_coords[1])
    bitmap = Bitmap(data=data, origin=origin)
    return bitmap, data, origin_coords


@pytest.fixture
def random_mask_float() -> Tuple[Bitmap, np.ndarray, Tuple[Union[int, float], Union[int, float]]]:
    height = random.randint(200, 800)
    width = random.randint(200, 800)
    data_shape = (height, width)
    data = np.ones(data_shape, dtype=np.uint8)
    origin_coords = [round(random.uniform(200, 800), 6), round(random.uniform(200, 800), 6)]
    origin = PointLocation(row=origin_coords[0], col=origin_coords[1])
    bitmap = Bitmap(data=data, origin=origin)
    return bitmap, data, origin_coords


def get_bitmap_data_origin(
    mask: Tuple[Bitmap, np.ndarray, Tuple[Union[int, float], Union[int, float]]]
) -> Tuple[Bitmap, np.ndarray, Tuple[Union[int, float], Union[int, float]]]:
    return mask


def check_origin_equal(bitmap: Bitmap, origin: List[Union[int, float]]):
    assert [bitmap.origin.row, bitmap.origin.col] == origin


def test_geometry_name(random_mask_int, random_mask_float, random_image):
    for mask in [random_mask_int, random_mask_float]:
        bitmap, data, origin = get_bitmap_data_origin(mask)
        assert bitmap.geometry_name() == "bitmap"


def test_name(random_mask_int, random_mask_float, random_image):
    for mask in [random_mask_int, random_mask_float]:
        bitmap, data, origin = get_bitmap_data_origin(mask)
        assert bitmap.name() == "bitmap"


def test_to_json(random_mask_int, random_mask_float, random_image):
    for mask in [random_mask_int, random_mask_float]:
        bitmap, data, origin = get_bitmap_data_origin(mask)
        base_64_data = bitmap.data_2_base64(data)
        expected_json = {
            "bitmap": {"origin": origin[::-1], "data": base_64_data},
            "shape": bitmap.name(),
            "geometryType": bitmap.name(),
        }
        assert bitmap.to_json() == expected_json


def test_from_json(random_mask_int, random_mask_float, random_image):
    for mask in [random_mask_int, random_mask_float]:
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


def test_crop(random_mask_int, random_mask_float, random_image):
    for mask in [random_mask_int, random_mask_float]:
        bitmap, data, origin = get_bitmap_data_origin(mask)
        rect = Rectangle(top=10, left=10, bottom=20, right=20)
        cropped_bitmaps = bitmap.crop(rect)
        for cropped_bitmap in cropped_bitmaps:
            assert isinstance(cropped_bitmap, Bitmap)
            assert cropped_bitmap.origin == PointLocation(row=rect.top, col=rect.left)
            assert np.array_equal(
                cropped_bitmap.data, data[rect.top : rect.bottom, rect.left : rect.right]
            )


def test_relative_crop(random_mask_int, random_mask_float, random_image):
    for mask in [random_mask_int, random_mask_float]:
        bitmap, data, origin = get_bitmap_data_origin(mask)
        rect = Rectangle(top=10, left=10, bottom=20, right=20)
        cropped_bitmaps = bitmap.relative_crop(rect)
        for cropped_bitmap in cropped_bitmaps:
            assert isinstance(cropped_bitmap, Bitmap)
            assert cropped_bitmap.origin == PointLocation(row=0, col=0)
            assert np.array_equal(
                cropped_bitmap.data, data[rect.top : rect.bottom, rect.left : rect.right]
            )


def test_rotate(random_mask_int, random_mask_float, random_image):
    for mask in [random_mask_int, random_mask_float]:
        bitmap, data, origin = get_bitmap_data_origin(mask)
        img_size, angle = bitmap.data.shape[:2], random.randint(0, 360)

        print(f"Bitmap data shape: {bitmap.data.shape}")
        print(f"Image size: {img_size}")

        rotator = ImageRotator(img_size, angle)
        rotated_bitmap = bitmap.rotate(rotator)

        print(f"Rotated bitmap data shape: {rotated_bitmap.data.shape}")
        print(f"Expected new image size: {rotator.new_imsize}")

        assert isinstance(rotated_bitmap, Bitmap)
        assert rotated_bitmap.data.shape == rotator.new_imsize


def test_resize(random_mask_int, random_mask_float, random_image):
    for mask in [random_mask_int, random_mask_float]:
        bitmap, data, origin = get_bitmap_data_origin(mask)
        in_size = random_image.shape[:2]
        out_size = (in_size[0] // 2, in_size[1] // 2)
        resized_bitmap = bitmap.resize(in_size, out_size)
        assert resized_bitmap.data.shape == out_size


def test_scale(random_mask_int, random_mask_float, random_image):
    for mask in [random_mask_int, random_mask_float]:
        factor = round(random.uniform(0, 1), 3)
        bitmap, data, origin = get_bitmap_data_origin(mask)
        scaled_bitmap = bitmap.scale(factor)
        assert scaled_bitmap.data.shape == (
            round(data.shape[0] * factor),
            round(data.shape[1] * factor),
        )


def test_translate(random_mask_int, random_mask_float, random_image):
    for mask in [random_mask_int, random_mask_float]:
        bitmap, data, origin = get_bitmap_data_origin(mask)
        drow, dcol = random.randint(10, 150), random.randint(10, 350)
        translated_bitmap = bitmap.translate(drow, dcol)
        expected_trans_origin = [origin[0] + drow, origin[1] + dcol]
        check_origin_equal(translated_bitmap, expected_trans_origin)


def test_fliplr(random_mask_int, random_mask_float, random_image):
    for mask in [random_mask_int, random_mask_float]:
        bitmap, data, origin = get_bitmap_data_origin(mask)
        img_size = random_image.shape[:2]
        flipped_bitmap = bitmap.fliplr(img_size)
        assert np.array_equal(flipped_bitmap.data, np.fliplr(data))


def test_flipud(random_mask_int, random_mask_float, random_image):
    for mask in [random_mask_int, random_mask_float]:
        bitmap, data, origin = get_bitmap_data_origin(mask)
        img_size = random_image.shape[:2]
        flipped_bitmap = bitmap.flipud(img_size)
        assert np.array_equal(flipped_bitmap.data, np.flipud(data))


def test_draw_bool_compatible(random_mask_int, random_mask_float, random_image):
    for mask in [random_mask_int, random_mask_float]:
        bitmap, data, origin = get_bitmap_data_origin(mask)
        bitmap._draw_bool_compatible(bitmap._draw_impl, random_image, color, thickness)
        assert np.any(random_image == color)


def test_draw(random_mask_int, random_mask_float, random_image):
    for mask in [random_mask_int, random_mask_float]:
        bitmap, data, origin = get_bitmap_data_origin(mask)
        bitmap.draw(random_image, color, thickness)
        assert np.any(random_image == color)


def test_get_mask(random_mask_int, random_mask_float, random_image):
    for mask in [random_mask_int, random_mask_float]:
        bitmap, data, origin = get_bitmap_data_origin(mask)
        bmask = bitmap.get_mask(random_image.shape)
        assert bmask.shape == random_image.shape
        assert bmask.dtype == np.bool
        assert np.any(bmask == True)


def test_draw_impl(random_mask_int, random_mask_float, random_image):
    for mask in [random_mask_int, random_mask_float]:
        bitmap, data, origin = get_bitmap_data_origin(mask)
        original_bitmap_data = bitmap.data.copy()
        bitmap._draw_impl(random_image, color, thickness)
        assert np.any(random_image == original_bitmap_data)


def test_draw_contour(random_mask_int, random_mask_float, random_image):
    for mask in [random_mask_int, random_mask_float]:
        bitmap, data, origin = get_bitmap_data_origin(mask)
        original_bitmap_data = bitmap.data.copy()
        bitmap.draw_contour(random_image, color, thickness)
        assert np.any(random_image != original_bitmap_data)


def test__draw_contour_impl(random_mask_int, random_mask_float, random_image):
    for mask in [random_mask_int, random_mask_float]:
        bitmap, data, origin = get_bitmap_data_origin(mask)
        original_bitmap_data = bitmap.data.copy()
        bitmap._draw_contour_impl(random_image, color, thickness)
        assert np.any(random_image != original_bitmap_data)


def test_area(random_mask_int, random_mask_float, random_image):
    for mask in [random_mask_int, random_mask_float]:
        bitmap, data, origin = get_bitmap_data_origin(mask)
        assert bitmap.area == np.sum(data)


def test_to_bbox(random_mask_int, random_mask_float, random_image):
    for mask in [random_mask_int, random_mask_float]:
        bitmap, data, origin = get_bitmap_data_origin(mask)
        bitmap_bbox = bitmap.to_bbox()
        assert isinstance(bitmap_bbox, Rectangle)
        assert round(bitmap_bbox.height) == data.shape[0]
        assert round(bitmap_bbox.width) == data.shape[1]


def test_clone(random_mask_int, random_mask_float, random_image):
    for mask in [random_mask_int, random_mask_float]:
        bitmap, data, origin = get_bitmap_data_origin(mask)
        cloned_bitmap = bitmap.clone()
        assert isinstance(cloned_bitmap, Bitmap)
        check_origin_equal(cloned_bitmap, origin)


def test_validate(random_mask_int, random_mask_float, random_image):
    for mask in [random_mask_int, random_mask_float]:
        bitmap, data, origin = get_bitmap_data_origin(mask)
        assert bitmap.validate(bitmap.geometry_name(), settings=None) is None
        with pytest.raises(
            ValueError, match="Geometry validation error: shape names are mismatched!"
        ):
            bitmap.validate("different_shape_name", settings=None)


def test_config_from_json(random_mask_int, random_mask_float, random_image):
    for mask in [random_mask_int, random_mask_float]:
        bitmap, data, origin = get_bitmap_data_origin(mask)
        config = {"key": "value"}
        returned_config = bitmap.config_from_json(config)
        assert returned_config == config


def test_config_to_json(random_mask_int, random_mask_float, random_image):
    for mask in [random_mask_int, random_mask_float]:
        bitmap, data, origin = get_bitmap_data_origin(mask)
        config = {"key": "value"}
        returned_config = bitmap.config_to_json(config)
        assert returned_config == config


def test_allowed_transforms(random_mask_int, random_mask_float, random_image):
    for mask in [random_mask_int, random_mask_float]:
        bitmap, data, origin = get_bitmap_data_origin(mask)
        allowed_transforms = bitmap.allowed_transforms()
        assert set(allowed_transforms) == set([AlphaMask, AnyGeometry, Polygon, Rectangle])


def test_convert(random_mask_int, random_mask_float, random_image):
    for mask in [random_mask_int, random_mask_float]:
        bitmap, data, origin = get_bitmap_data_origin(mask)
        assert bitmap.convert(type(bitmap)) == [bitmap]
        assert bitmap.convert(AnyGeometry) == [bitmap]
        for new_geometry in bitmap.allowed_transforms():
            converted = bitmap.convert(new_geometry)
            for g in converted:
                assert isinstance(g, new_geometry) or isinstance(g, Bitmap)

        with pytest.raises(
            NotImplementedError,
            match="from {!r} to {!r}".format(bitmap.geometry_name(), Point.geometry_name()),
        ):
            bitmap.convert(Point)


# Bitmap specific methods
# ------------------------


def test_data(random_mask_int, random_mask_float, random_image):
    for mask in [random_mask_int, random_mask_float]:
        bitmap, data, origin = get_bitmap_data_origin(mask)
        assert np.array_equal(bitmap.data, data)


def test_origin(random_mask_int, random_mask_float, random_image):
    for mask in [random_mask_int, random_mask_float]:
        bitmap, data, origin = get_bitmap_data_origin(mask)
        check_origin_equal(bitmap, origin)


def test_base64_2_data(random_mask_int, random_mask_float, random_image):
    for mask in [random_mask_int, random_mask_float]:
        bitmap, data, origin = get_bitmap_data_origin(mask)
        encoded = bitmap.data_2_base64(data)
        decoded = Bitmap.base64_2_data(encoded)
        assert np.array_equal(decoded, data)


def test_data_2_base64(random_mask_int, random_mask_float, random_image):
    for mask in [random_mask_int, random_mask_float]:
        bitmap, data, origin = get_bitmap_data_origin(mask)
        encoded = bitmap.data_2_base64(data)
        decoded = Bitmap.base64_2_data(encoded)
        assert np.array_equal(decoded, data)


def test_skeletonize(random_mask_int, random_mask_float, random_image):
    for mask in [random_mask_int, random_mask_float]:
        bitmap, data, origin = get_bitmap_data_origin(mask)
        for method in SkeletonizeMethod:
            skeleton = bitmap.skeletonize(method)
            assert isinstance(skeleton, Bitmap)


def test_to_contours(random_mask_int, random_mask_float, random_image):
    for mask in [random_mask_int, random_mask_float]:
        bitmap, data, origin = get_bitmap_data_origin(mask)
        contours = bitmap.to_contours()
        assert isinstance(contours, list), "Output should be a list"
        assert len(contours) > 0, "List should not be empty"
        assert all(
            isinstance(contour, Polygon) for contour in contours
        ), "All elements in the list should be numpy arrays"


def test_bitwise_mask(random_mask_int, random_mask_float, random_image):
    for mask in [random_mask_int, random_mask_float]:
        bitmap, data, origin = get_bitmap_data_origin(mask)

        # Ensure mask is created with the correct shape and dtype right before use
        mask = np.ones(data.shape, dtype=data.dtype)

        # Double-checking shapes match right before the operation
        assert mask.shape == data.shape, "Mask and data shapes must match"

        # Apply the bitwise operation
        result = bitmap.bitwise_mask(mask, np.logical_and)

        assert (
            isinstance(result, Bitmap) or result == []
        ), "Output should be a Bitmap instance or an empty list"

        if isinstance(result, Bitmap):
            assert (
                result.data.shape == data.shape
            ), "Resulting data should have the same shape as input data"
            assert (
                result.data.dtype == data.dtype
            ), "Resulting data should have the same data type as input data"

            # Correctly applying the logical_and operation
            # Ensure mask is correctly referenced if it's a property of `bitmap` or `result`
            expected_result = np.logical_and(
                data, mask
            )  # Using mask directly since it's a local variable
            assert np.all(
                expected_result == result.data
            ), "Resulting data should match the expected bitwise AND result"


def test_from_path(random_mask_int, random_mask_float, random_image):
    pass


if __name__ == "__main__":
    pytest.main([__file__])

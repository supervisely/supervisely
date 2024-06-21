import random
from typing import List, Tuple, Union

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


@pytest.fixture
def random_mask_int() -> Tuple[Bitmap, np.ndarray, Tuple[Union[int, float], Union[int, float]]]:
    height = random.randint(200, 1000)
    width = random.randint(200, 1000)
    data = np.ones((height, width), dtype=np.uint8)
    origin_coords = [random.randint(200, 1000), random.randint(200, 1000)]
    origin = PointLocation(row=origin_coords[0], col=origin_coords[1])
    alpha_mask = Bitmap(data=data, origin=origin)
    return alpha_mask, data, origin_coords


@pytest.fixture
def random_mask_float() -> Tuple[Bitmap, np.ndarray, Tuple[Union[int, float], Union[int, float]]]:
    height = random.randint(200, 1000)
    width = random.randint(200, 1000)
    data = np.ones((height, width), dtype=np.uint8)
    origin_coords = [round(random.uniform(200, 1000), 6), round(random.uniform(200, 1000), 6)]
    origin = PointLocation(row=origin_coords[0], col=origin_coords[1])
    alpha_mask = Bitmap(data=data, origin=origin)
    return alpha_mask, data, origin_coords


def get_bitmap_data_origin(
    mask: Tuple[Bitmap, np.ndarray, Tuple[Union[int, float], Union[int, float]]]
) -> Tuple[Bitmap, np.ndarray, Tuple[Union[int, float], Union[int, float]]]:
    return mask


def check_origin_equal(bitmap: Bitmap, origin: List[Union[int, float]]):
    assert [bitmap.origin.row, bitmap.origin.col] == origin


def test_geometry_name(random_mask_int, random_mask_float):
    for mask in [random_mask_int, random_mask_float]:
        bitmap, data, origin = get_bitmap_data_origin(mask)
        assert bitmap.geometry_name() == "bitmap"


def test_name(random_mask_int, random_mask_float):
    for mask in [random_mask_int, random_mask_float]:
        bitmap, data, origin = get_bitmap_data_origin(mask)
        assert bitmap.name() == "bitmap"


def test_to_json(random_mask_int, random_mask_float):
    for mask in [random_mask_int, random_mask_float]:
        bitmap, data, origin = get_bitmap_data_origin(mask)
        base_64_data = bitmap.data_2_base64(data)
        expected_json = {
            "bitmap": {"origin": origin[::-1], "data": base_64_data},
            "shape": bitmap.name(),
            "geometryType": bitmap.name(),
        }
        assert bitmap.to_json() == expected_json


def test_from_json(random_mask_int, random_mask_float):
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


def test_crop(random_mask_int, random_mask_float):
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


def test_relative_crop(random_mask_int, random_mask_float):
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


def test_rotate(random_mask_int, random_mask_float):
    for mask in [random_mask_int, random_mask_float]:
        bitmap, data, origin = get_bitmap_data_origin(mask)

        if len(data.shape) != 2:
            data = data.reshape((data.shape[0], -1))

        rotator = ImageRotator(data.shape, 45)
        rotated_bitmap = bitmap.rotate(rotator)
        assert isinstance(rotated_bitmap, Bitmap)
        assert rotated_bitmap.data.shape == rotator.dst_imsize


def test_resize(random_mask_int, random_mask_float):
    for mask in [random_mask_int, random_mask_float]:
        bitmap, data, origin = get_bitmap_data_origin(mask)
        in_size = data.shape
        out_size = (in_size[0] // 2, in_size[1] // 2)
        resized_bitmap = bitmap.resize(in_size, out_size)
        assert resized_bitmap.data.shape == out_size


def test_scale(random_mask_int, random_mask_float):
    factor = 2
    for mask in [random_mask_int, random_mask_float]:
        bitmap, data, origin = get_bitmap_data_origin(mask)
        scaled_bitmap = bitmap.scale(factor)
        assert scaled_bitmap.data.shape == (data.shape[0] * factor, data.shape[1] * factor)


def test_translate(random_mask_int, random_mask_float):
    for mask in [random_mask_int, random_mask_float]:
        bitmap, data, origin = get_bitmap_data_origin(mask)
        translated_bitmap = bitmap.translate(drow=10, dcol=10)
        expected_trans_origin = [origin[0] + 10, origin[1] + 10]
        check_origin_equal(translated_bitmap, expected_trans_origin)


def test_fliplr(random_mask_int, random_mask_float):
    for mask in [random_mask_int, random_mask_float]:
        bitmap, data, origin = get_bitmap_data_origin(mask)
        flipped_bitmap = bitmap.fliplr(data.shape)
        assert np.array_equal(flipped_bitmap.data, np.fliplr(data))


def test_flipud(random_mask_int, random_mask_float):
    for mask in [random_mask_int, random_mask_float]:
        bitmap, data, origin = get_bitmap_data_origin(mask)
        flipped_bitmap = bitmap.flipud(data.shape)
        assert np.array_equal(flipped_bitmap.data, np.flipud(data))


def test_draw_bool_compatible(random_mask_int, random_mask_float):
    for mask in [random_mask_int, random_mask_float]:
        bitmap, data, origin = get_bitmap_data_origin(mask)
        bitmap._draw_bool_compatible(bitmap._draw_impl, data, 255, 1)
        assert np.any(data)


def test_draw(random_mask_int, random_mask_float):
    for mask in [random_mask_int, random_mask_float]:
        bitmap, data, origin = get_bitmap_data_origin(mask)
        empty_bitmap = np.zeros(data.shape, dtype=np.uint8)
        bitmap.draw(empty_bitmap, 255, 1)
        assert np.any(empty_bitmap)


def test_get_mask(random_mask_int, random_mask_float):
    for mask in [random_mask_int, random_mask_float]:
        bitmap, data, origin = get_bitmap_data_origin(mask)
        bmask = bitmap.get_mask(data.shape)
        assert np.array_equal(bmask, data)


def test_draw_impl(random_mask_int, random_mask_float):
    for mask in [random_mask_int, random_mask_float]:
        bitmap, data, origin = get_bitmap_data_origin(mask)
        original_bitmap_data = bitmap.data.copy()
        bitmap._draw_impl(bitmap, color=255)
        assert np.any(bitmap.data == original_bitmap_data)


def test_draw_contour(random_mask_int, random_mask_float):
    for mask in [random_mask_int, random_mask_float]:
        bitmap, data, origin = get_bitmap_data_origin(mask)
        original_bitmap_data = bitmap.data.copy()
        bitmap.draw_contour(bitmap, color=255)
        assert np.any(bitmap.data != original_bitmap_data)


def test__draw_contour_impl(random_mask_int, random_mask_float):
    for mask in [random_mask_int, random_mask_float]:
        bitmap, data, origin = get_bitmap_data_origin(mask)
        original_bitmap_data = bitmap.data.copy()
        bitmap._draw_contour_impl(bitmap, color=255)
        assert np.any(bitmap.data != original_bitmap_data)


def test_area(random_mask_int, random_mask_float):
    for mask in [random_mask_int, random_mask_float]:
        bitmap, data, origin = get_bitmap_data_origin(mask)
        assert bitmap.area == np.sum(data)


def test_to_bbox(random_mask_int, random_mask_float):
    for mask in [random_mask_int, random_mask_float]:
        bitmap, data, origin = get_bitmap_data_origin(mask)
        bitmap_bbox = bitmap.to_bbox()
        assert isinstance(bitmap_bbox, Rectangle)
        assert bitmap_bbox.bottom == data.shape[0]
        assert bitmap_bbox.right == data.shape[1]


def test_clone(random_mask_int, random_mask_float):
    for mask in [random_mask_int, random_mask_float]:
        bitmap, data, origin = get_bitmap_data_origin(mask)
        cloned_bitmap = bitmap.clone()
        assert isinstance(cloned_bitmap, Bitmap)
        check_origin_equal(cloned_bitmap, origin)


def test_validate(random_mask_int, random_mask_float):
    for mask in [random_mask_int, random_mask_float]:
        bitmap, data, origin = get_bitmap_data_origin(mask)
        assert bitmap.validate(bitmap.geometry_name(), settings=None) is None
        with pytest.raises(
            ValueError, match="Geometry validation error: shape names are mismatched!"
        ):
            bitmap.validate("different_shape_name", settings=None)


def test_config_from_json(random_mask_int, random_mask_float):
    for mask in [random_mask_int, random_mask_float]:
        bitmap, data, origin = get_bitmap_data_origin(mask)
        config = {"key": "value"}
        returned_config = bitmap.config_from_json(config)
        assert returned_config == config


def test_config_to_json(random_mask_int, random_mask_float):
    for mask in [random_mask_int, random_mask_float]:
        bitmap, data, origin = get_bitmap_data_origin(mask)
        config = {"key": "value"}
        returned_config = bitmap.config_to_json(config)
        assert returned_config == config


def test_allowed_transforms(random_mask_int, random_mask_float):
    for mask in [random_mask_int, random_mask_float]:
        bitmap, data, origin = get_bitmap_data_origin(mask)
        allowed_transforms = bitmap.allowed_transforms()
        assert set(allowed_transforms) == set([AlphaMask, AnyGeometry, Polygon, Rectangle])


def test_convert(random_mask_int, random_mask_float):
    for mask in [random_mask_int, random_mask_float]:
        bitmap, data, origin = get_bitmap_data_origin(mask)
        assert bitmap.convert(type(bitmap)) == [bitmap]
        assert bitmap.convert(AnyGeometry) == [bitmap]
        for new_geometry in bitmap.allowed_transforms():
            converted = bitmap.convert(new_geometry)
            assert all(isinstance(g, new_geometry) for g in converted)

        class NotAllowedGeometry:
            pass

        with pytest.raises(
            NotImplementedError,
            match="from {!r} to {!r}".format(bitmap.geometry_name(), "NotAllowedGeometry"),
        ):
            bitmap.convert(NotAllowedGeometry)


# Bitmap specific methods
# ------------------------


def test_data(random_alpha_mask_int, random_alpha_mask_float):
    for mask in [random_alpha_mask_int, random_alpha_mask_float]:
        bitmap, data, origin = get_bitmap_data_origin(mask)
        assert np.array_equal(bitmap.data, data)


def test_origin(random_alpha_mask_int, random_alpha_mask_float):
    for mask in [random_alpha_mask_int, random_alpha_mask_float]:
        bitmap, data, origin = get_bitmap_data_origin(mask)
        check_origin_equal(bitmap, origin)


def test_base64_2_data(random_mask_int, random_mask_float):
    for mask in [random_mask_int, random_mask_float]:
        bitmap, data, origin = get_bitmap_data_origin(mask)
        encoded = bitmap.data_2_base64(data)
        decoded = Bitmap.base64_2_data(encoded)
        assert np.array_equal(decoded, data)


def test_data_2_base64(random_mask_int, random_mask_float):
    for mask in [random_mask_int, random_mask_float]:
        bitmap, data, origin = get_bitmap_data_origin(mask)
        encoded = bitmap.data_2_base64(data)
        decoded = Bitmap.base64_2_data(encoded)
        assert np.array_equal(decoded, data)


def test_skeletonize(random_mask_int, random_mask_float):
    for mask in [random_mask_int, random_mask_float]:
        bitmap, data, origin = get_bitmap_data_origin(mask)
        for method in SkeletonizeMethod:
            skeleton = bitmap.skeletonize(method)
            assert isinstance(skeleton, Bitmap)


def test_to_contours(random_mask_int, random_mask_float):
    for mask in [random_mask_int, random_mask_float]:
        bitmap, data, origin = get_bitmap_data_origin(mask)
        contours = bitmap.to_contours()
        assert isinstance(contours, list), "Output should be a list"
        assert len(contours) > 0, "List should not be empty"
        assert all(
            isinstance(contour, np.ndarray) for contour in contours
        ), "All elements in the list should be numpy arrays"
        assert all(
            contour.ndim == 2 for contour in contours
        ), "All contours should be 2-dimensional"
        assert all(
            contour.shape[1] == 2 for contour in contours
        ), "All contours should have 2 coordinates (x, y)"


def test_bitwise_mask(random_mask_int, random_mask_float):
    for mask in [random_mask_int, random_mask_float]:
        bitmap, data, origin = get_bitmap_data_origin(mask)
        result = bitmap.bitwise_mask(data, np.logical_and)
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
            assert np.all(
                np.logical_and(data, data) == result.data
            ), "Resulting data should be the bitwise AND of input data"


def test_from_path(random_mask_int, random_mask_float):
    pass


if __name__ == "__main__":
    pytest.main([__file__])

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

# Draw Settings
color = [255, 255, 255]
thickness = 1


@pytest.fixture
def random_image() -> np.ndarray:
    image_shape = (random.randint(801, 1000), random.randint(801, 1000), 3)
    background_color = [0, 0, 0]
    bitmap = np.full(image_shape, background_color, dtype=np.uint8)
    return bitmap


@pytest.fixture
def random_alpha_mask_int() -> (
    Tuple[AlphaMask, np.ndarray, Tuple[Union[int, float], Union[int, float]]]
):
    height = random.randint(0, 1000)
    width = random.randint(0, 1000)
    data = np.ones((height, width), dtype=np.uint8)
    origin_coords = [random.randint(0, 1000), random.randint(0, 1000)]
    origin = PointLocation(row=origin_coords[0], col=origin_coords[1])
    alpha_mask = AlphaMask(data=data, origin=origin)
    return alpha_mask, data, origin_coords


@pytest.fixture
def random_alpha_mask_float() -> (
    Tuple[AlphaMask, np.ndarray, Tuple[Union[int, float], Union[int, float]]]
):
    height = random.randint(0, 1000)
    width = random.randint(0, 1000)
    data = np.ones((height, width), dtype=np.uint8)
    origin_coords = [round(random.uniform(0, 1000), 6), round(random.uniform(0, 1000), 6)]
    origin = PointLocation(row=origin_coords[0], col=origin_coords[1])
    alpha_mask = AlphaMask(data=data, origin=origin)
    return alpha_mask, data, origin_coords


def get_mask_data_origin(
    alpha_mask: Tuple[AlphaMask, np.ndarray, Tuple[Union[int, float], Union[int, float]]]
) -> Tuple[AlphaMask, np.ndarray, Tuple[Union[int, float], Union[int, float]]]:
    return alpha_mask


def check_origin_equal(alpha_mask: AlphaMask, origin: Tuple[int, int]):
    assert [alpha_mask.origin.row, alpha_mask.origin.col] == origin


def test_geometry_name(random_alpha_mask_int, random_alpha_mask_float, random_image):
    for a_mask in [random_alpha_mask_int, random_alpha_mask_float]:
        alpha_mask, data, origin = get_mask_data_origin(a_mask)
        assert alpha_mask.geometry_name() == "alpha_mask"


def test_name(random_alpha_mask_int, random_alpha_mask_float, random_image):
    for a_mask in [random_alpha_mask_int, random_alpha_mask_float]:
        alpha_mask, data, origin = get_mask_data_origin(a_mask)
        assert alpha_mask.name() == "alpha_mask"


def test_to_json(random_alpha_mask_int, random_alpha_mask_float, random_image):
    for a_mask in [random_alpha_mask_int, random_alpha_mask_float]:
        alpha_mask, data, origin = get_mask_data_origin(a_mask)
        base_64_data = alpha_mask.data_2_base64(data)
        expected_json = {
            "bitmap": {"origin": origin[::-1], "data": base_64_data},
            "shape": alpha_mask.name(),
            "geometryType": alpha_mask.name(),
        }
        assert alpha_mask.to_json() == expected_json


def test_from_json(random_alpha_mask_int, random_alpha_mask_float, random_image):
    for a_mask in [random_alpha_mask_int, random_alpha_mask_float]:
        alpha_mask, data, origin = get_mask_data_origin(a_mask)
        base_64_data = alpha_mask.data_2_base64(data)
        alpha_mask_json = {
            "bitmap": {"origin": origin[::-1], "data": base_64_data},
            "shape": alpha_mask.name(),
            "geometryType": alpha_mask.name(),
        }
        alpha_mask = AlphaMask.from_json(alpha_mask_json)
        assert isinstance(alpha_mask, AlphaMask)
        check_origin_equal(alpha_mask, origin)


def test_crop(random_alpha_mask_int, random_alpha_mask_float, random_image):
    for a_mask in [random_alpha_mask_int, random_alpha_mask_float]:
        alpha_mask, data, origin = get_mask_data_origin(a_mask)
        rect = Rectangle(top=10, left=10, bottom=20, right=20)
        cropped_alpha_masks = alpha_mask.crop(rect)
        for cropped_alpha_mask in cropped_alpha_masks:
            assert isinstance(cropped_alpha_mask, AlphaMask)
            assert cropped_alpha_mask.origin == PointLocation(row=rect.top, col=rect.left)
            assert np.array_equal(
                cropped_alpha_mask.data, data[rect.top : rect.bottom, rect.left : rect.right]
            )


def test_relative_crop(random_alpha_mask_int, random_alpha_mask_float, random_image):
    for a_mask in [random_alpha_mask_int, random_alpha_mask_float]:
        alpha_mask, data, origin = get_mask_data_origin(a_mask)
        rect = Rectangle(top=10, left=10, bottom=20, right=20)
        cropped_alpha_masks = alpha_mask.relative_crop(rect)
        for cropped_alpha_mask in cropped_alpha_masks:
            assert isinstance(cropped_alpha_mask, AlphaMask)
            assert cropped_alpha_mask.origin == PointLocation(row=0, col=0)
            assert np.array_equal(
                cropped_alpha_mask.data, data[rect.top : rect.bottom, rect.left : rect.right]
            )


def test_rotate(random_alpha_mask_int, random_alpha_mask_float, random_image):
    for a_mask in [random_alpha_mask_int, random_alpha_mask_float]:
        alpha_mask, data, origin = get_mask_data_origin(a_mask)

        img_size, angle = random_image.shape[:2], random.randint(0, 360)
        rotator = ImageRotator(img_size, angle)
        rotated_alpha_mask = alpha_mask.rotate(rotator)
        assert isinstance(rotated_alpha_mask, AlphaMask)
        assert rotated_alpha_mask.data.shape == rotator.dst_imsize


def test_resize(random_alpha_mask_int, random_alpha_mask_float, random_image):
    for a_mask in [random_alpha_mask_int, random_alpha_mask_float]:
        alpha_mask, data, origin = get_mask_data_origin(a_mask)
        in_size = random_image.shape[:2]
        out_size = (in_size[0] // 2, in_size[1] // 2)
        resized_alpha_mask = alpha_mask.resize(in_size, out_size)
        assert resized_alpha_mask.data.shape == out_size


def test_scale(random_alpha_mask_int, random_alpha_mask_float, random_image):
    for a_mask in [random_alpha_mask_int, random_alpha_mask_float]:
        alpha_mask, data, origin = get_mask_data_origin(a_mask)
        factor = round(random.uniform(0, 1), 3)
        scaled_alpha_mask = alpha_mask.scale(factor)
        assert scaled_alpha_mask.data.shape == (data.shape[0] * factor, data.shape[1] * factor)


def test_translate(random_alpha_mask_int, random_alpha_mask_float, random_image):
    for a_mask in [random_alpha_mask_int, random_alpha_mask_float]:
        alpha_mask, data, origin = get_mask_data_origin(a_mask)
        drow, dcol = random.randint(10, 150), random.randint(10, 350)
        translated_alpha_mask = alpha_mask.translate(drow, dcol)
        expected_trans_origin = [origin[0] + 10, origin[1] + 10]
        check_origin_equal(translated_alpha_mask, expected_trans_origin)


def test_fliplr(random_alpha_mask_int, random_alpha_mask_float, random_image):
    for a_mask in [random_alpha_mask_int, random_alpha_mask_float]:
        alpha_mask, data, origin = get_mask_data_origin(a_mask)
        img_size = random_image.shape[:2]
        flipped_alpha_mask = alpha_mask.fliplr(img_size)
        assert np.array_equal(flipped_alpha_mask.data, np.fliplr(data))


def test_flipud(random_alpha_mask_int, random_alpha_mask_float, random_image):
    for a_mask in [random_alpha_mask_int, random_alpha_mask_float]:
        alpha_mask, data, origin = get_mask_data_origin(a_mask)
        img_size = random_image.shape[:2]
        flipped_alpha_mask = alpha_mask.flipud(img_size)
        assert np.array_equal(flipped_alpha_mask.data, np.flipud(data))


def test__draw_bool_compatible(random_alpha_mask_int, random_alpha_mask_float, random_image):
    for a_mask in [random_alpha_mask_int, random_alpha_mask_float]:
        alpha_mask, data, origin = get_mask_data_origin(a_mask)
        alpha_mask._draw_bool_compatible(alpha_mask._draw_impl, random_image, color, thickness)
        assert np.any(random_image == color)


def test_draw(random_alpha_mask_int, random_alpha_mask_float, random_image):
    for a_mask in [random_alpha_mask_int, random_alpha_mask_float]:
        alpha_mask, data, origin = get_mask_data_origin(a_mask)
        alpha_mask.draw(random_image, color, thickness)
        assert np.any(random_image == color)


def test_get_mask(random_alpha_mask_int, random_alpha_mask_float, random_image):
    for a_mask in [random_alpha_mask_int, random_alpha_mask_float]:
        alpha_mask, data, origin = get_mask_data_origin(a_mask)
        amask = alpha_mask.get_mask(random_image.shape)
        assert amask.shape == random_image.shape
        assert amask.dtype == np.bool
        assert np.any(amask == True)


def test__draw_impl(random_alpha_mask_int, random_alpha_mask_float, random_image):
    for a_mask in [random_alpha_mask_int, random_alpha_mask_float]:
        alpha_mask, data, origin = get_mask_data_origin(a_mask)
        original_alpha_mask_data = alpha_mask.data.copy()
        alpha_mask._draw_impl(random_image, color, thickness)
        assert np.any(random_image == original_alpha_mask_data)


def test_draw_contour(random_alpha_mask_int, random_alpha_mask_float, random_image):
    for a_mask in [random_alpha_mask_int, random_alpha_mask_float]:
        alpha_mask, data, origin = get_mask_data_origin(a_mask)
        original_alpha_mask_data = alpha_mask.data.copy()
        alpha_mask.draw_contour(random_image, color, thickness)
        assert np.any(random_image != original_alpha_mask_data)


def test__draw_contour_impl(random_alpha_mask_int, random_alpha_mask_float, random_image):
    for a_mask in [random_alpha_mask_int, random_alpha_mask_float]:
        alpha_mask, data, origin = get_mask_data_origin(a_mask)
        original_alpha_mask_data = alpha_mask.data.copy()
        alpha_mask._draw_contour_impl(random_image, color, thickness)
        assert np.any(random_image != original_alpha_mask_data)


def test_area(random_alpha_mask_int, random_alpha_mask_float, random_image):
    for a_mask in [random_alpha_mask_int, random_alpha_mask_float]:
        alpha_mask, data, origin = get_mask_data_origin(a_mask)
        assert alpha_mask.area == np.sum(data)


def test_to_bbox(random_alpha_mask_int, random_alpha_mask_float, random_image):
    for a_mask in [random_alpha_mask_int, random_alpha_mask_float]:
        alpha_mask, data, origin = get_mask_data_origin(a_mask)
        alpha_mask_bbox = alpha_mask.to_bbox()
        assert isinstance(alpha_mask_bbox, Rectangle)
        assert alpha_mask_bbox.bottom == data.shape[0]
        assert alpha_mask_bbox.right == data.shape[1]


def test_clone(random_alpha_mask_int, random_alpha_mask_float, random_image):
    for a_mask in [random_alpha_mask_int, random_alpha_mask_float]:
        alpha_mask, data, origin = get_mask_data_origin(a_mask)
        cloned_alpha_mask = alpha_mask.clone()
        assert isinstance(cloned_alpha_mask, AlphaMask)
        check_origin_equal(cloned_alpha_mask, origin)


def test_validate(random_alpha_mask_int, random_alpha_mask_float, random_image):
    for a_mask in [random_alpha_mask_int, random_alpha_mask_float]:
        alpha_mask, data, origin = get_mask_data_origin(a_mask)
        assert alpha_mask.validate(alpha_mask.geometry_name(), settings=None) is None
        with pytest.raises(
            ValueError, match="Geometry validation error: shape names are mismatched!"
        ):
            alpha_mask.validate("different_shape_name", settings=None)


def test_config_from_json(random_alpha_mask_int, random_alpha_mask_float, random_image):
    for a_mask in [random_alpha_mask_int, random_alpha_mask_float]:
        alpha_mask, data, origin = get_mask_data_origin(a_mask)
        config = {"key": "value"}
        returned_config = alpha_mask.config_from_json(config)
        assert returned_config == config


def test_config_to_json(random_alpha_mask_int, random_alpha_mask_float, random_image):
    for a_mask in [random_alpha_mask_int, random_alpha_mask_float]:
        alpha_mask, data, origin = get_mask_data_origin(a_mask)
        config = {"key": "value"}
        returned_config = alpha_mask.config_to_json(config)
        assert returned_config == config


def test_allowed_transforms(random_alpha_mask_int, random_alpha_mask_float, random_image):
    for a_mask in [random_alpha_mask_int, random_alpha_mask_float]:
        alpha_mask, data, origin = get_mask_data_origin(a_mask)
        allowed_transforms = alpha_mask.allowed_transforms()
        assert set(allowed_transforms) == set([AnyGeometry, Bitmap, Polygon, Rectangle])


def test_convert(random_alpha_mask_int, random_alpha_mask_float, random_image):
    for a_mask in [random_alpha_mask_int, random_alpha_mask_float]:
        alpha_mask, data, origin = get_mask_data_origin(a_mask)
        assert alpha_mask.convert(type(alpha_mask)) == [alpha_mask]
        assert alpha_mask.convert(AnyGeometry) == [alpha_mask]
        for new_geometry in alpha_mask.allowed_transforms():
            converted = alpha_mask.convert(new_geometry)
            assert all(isinstance(g, new_geometry) for g in converted)

        class NotAllowedGeometry:
            pass

        with pytest.raises(
            NotImplementedError,
            match="from {!r} to {!r}".format(alpha_mask.geometry_name(), "NotAllowedGeometry"),
        ):
            alpha_mask.convert(NotAllowedGeometry)


# Alpha Mask specific methods
# ------------------------


def test_data(random_alpha_mask_int, random_alpha_mask_float, random_image):
    for a_mask in [random_alpha_mask_int, random_alpha_mask_float]:
        alpha_mask, data, origin = get_mask_data_origin(a_mask)
        assert np.array_equal(alpha_mask.data, data)


def test_origin(random_alpha_mask_int, random_alpha_mask_float, random_image):
    for a_mask in [random_alpha_mask_int, random_alpha_mask_float]:
        alpha_mask, data, origin = get_mask_data_origin(a_mask)
        check_origin_equal(alpha_mask, origin)


def test_base64_2_data(random_alpha_mask_int, random_alpha_mask_float, random_image):
    for mask in [random_alpha_mask_int, random_alpha_mask_float]:
        alpha_mask, data, origin = get_mask_data_origin(mask)
        encoded = alpha_mask.data_2_base64(data)
        decoded = AlphaMask.base64_2_data(encoded)
        assert np.array_equal(decoded, data)


def test_data_2_base64(random_alpha_mask_int, random_alpha_mask_float, random_image):
    for mask in [random_alpha_mask_int, random_alpha_mask_float]:
        alpha_mask, data, origin = get_mask_data_origin(mask)
        encoded = alpha_mask.data_2_base64(data)
        decoded = AlphaMask.base64_2_data(encoded)
        assert np.array_equal(decoded, data)


def test_skeletonize(random_alpha_mask_int, random_alpha_mask_float, random_image):
    for mask in [random_alpha_mask_int, random_alpha_mask_float]:
        alpha_mask, data, origin = get_mask_data_origin(mask)
        for method in SkeletonizeMethod:
            skeleton = alpha_mask.skeletonize(method)
            assert isinstance(skeleton, AlphaMask)


def test_to_contours(random_alpha_mask_int, random_alpha_mask_float, random_image):
    for mask in [random_alpha_mask_int, random_alpha_mask_float]:
        alpha_mask, data, origin = get_mask_data_origin(mask)
        contours = alpha_mask.to_contours()
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


def test_bitwise_mask(random_alpha_mask_int, random_alpha_mask_float, random_image):
    for mask in [random_alpha_mask_int, random_alpha_mask_float]:
        alpha_mask, data, origin = get_mask_data_origin(mask)
        result = alpha_mask.bitwise_mask(data, np.logical_and)
        assert (
            isinstance(result, AlphaMask) or result == []
        ), "Output should be a Alpha instance or an empty list"
        if isinstance(result, AlphaMask):
            assert (
                result.data.shape == data.shape
            ), "Resulting data should have the same shape as input data"
            assert (
                result.data.dtype == data.dtype
            ), "Resulting data should have the same data type as input data"
            assert np.all(
                np.logical_and(data, data) == result.data
            ), "Resulting data should be the bitwise AND of input data"


def test_from_path(random_alpha_mask_int, random_alpha_mask_float, random_image):
    pass


if __name__ == "__main__":
    pytest.main([__file__])

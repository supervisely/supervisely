import inspect
import os
import random
from typing import List, Tuple, Union

import numpy as np
import pytest  # pylint: disable=import-error
from test_geometry import draw_test, get_random_image

from supervisely.geometry.alpha_mask import AlphaMask
from supervisely.geometry.any_geometry import AnyGeometry
from supervisely.geometry.bitmap import Bitmap, SkeletonizeMethod
from supervisely.geometry.image_rotator import ImageRotator
from supervisely.geometry.point import Point
from supervisely.geometry.point_location import PointLocation
from supervisely.geometry.polygon import Polygon
from supervisely.geometry.rectangle import Rectangle
from supervisely.io.fs import get_file_name

dir_name = get_file_name(os.path.abspath(__file__))
# Draw Settings
color = [255, 255, 255]
thickness = 1


@pytest.fixture
def random_alpha_mask_int_1() -> (
    Tuple[AlphaMask, np.ndarray, Tuple[Union[int, float], Union[int, float]]]
):
    height = random.randint(200, 400)
    width = random.randint(200, 400)
    data_shape = (height, width)
    data = np.ones(data_shape, dtype=np.uint8) * 255
    origin_coords = [random.randint(0, 10), random.randint(0, 10)]
    origin = PointLocation(row=origin_coords[0], col=origin_coords[1])
    alpha_mask = AlphaMask(data=data, origin=origin)
    data = alpha_mask.data
    origin_coords = [alpha_mask.origin.row, alpha_mask.origin.col]
    return alpha_mask, data, origin_coords


@pytest.fixture
def random_alpha_mask_int_2() -> (
    Tuple[AlphaMask, np.ndarray, Tuple[Union[int, float], Union[int, float]]]
):
    height = random.randint(200, 400)
    width = random.randint(200, 400)
    data_shape = (height, width)
    data = np.ones(data_shape, dtype=np.uint8) * 255
    origin_coords = [
        random.randint(height // 2 - 50, height // 2 + 50),
        random.randint(width // 2 - 50, width // 2 + 50),
    ]
    origin = PointLocation(row=origin_coords[0], col=origin_coords[1])
    alpha_mask = AlphaMask(data=data, origin=origin)
    data = alpha_mask.data
    origin_coords = [alpha_mask.origin.row, alpha_mask.origin.col]
    return alpha_mask, data, origin_coords


@pytest.fixture
def random_alpha_mask_float() -> (
    Tuple[AlphaMask, np.ndarray, Tuple[Union[int, float], Union[int, float]]]
):
    height = random.randint(200, 400)
    width = random.randint(200, 400)
    data_shape = (height, width)
    data = np.ones(data_shape, dtype=np.uint8) * 255
    origin_coords = [round(random.uniform(0, 10), 6), round(random.uniform(0, 10), 6)]
    origin = PointLocation(row=origin_coords[0], col=origin_coords[1])
    alpha_mask = AlphaMask(data=data, origin=origin)
    data = alpha_mask.data
    origin_coords = [alpha_mask.origin.row, alpha_mask.origin.col]
    return alpha_mask, data, origin_coords


def get_mask_data_origin(
    alpha_mask: Tuple[AlphaMask, np.ndarray, Tuple[Union[int, float], Union[int, float]]]
) -> Tuple[AlphaMask, np.ndarray, Tuple[Union[int, float], Union[int, float]]]:
    return alpha_mask


def check_origin_equal(alpha_mask: AlphaMask, origin: Tuple[int, int]):
    assert [alpha_mask.origin.row, alpha_mask.origin.col] == origin


def test_geometry_name(random_alpha_mask_int_1, random_alpha_mask_int_2):
    for idx, a_mask in enumerate([random_alpha_mask_int_1, random_alpha_mask_int_2], 1):
        alpha_mask, data, origin = get_mask_data_origin(a_mask)
        assert alpha_mask.geometry_name() == "alpha_mask"


def test_name(random_alpha_mask_int_1, random_alpha_mask_int_2):
    for idx, a_mask in enumerate([random_alpha_mask_int_1, random_alpha_mask_int_2], 1):
        alpha_mask, data, origin = get_mask_data_origin(a_mask)
        assert alpha_mask.name() == "alpha_mask"


def test_to_json(random_alpha_mask_int_1, random_alpha_mask_int_2):
    for idx, a_mask in enumerate([random_alpha_mask_int_1, random_alpha_mask_int_2], 1):
        alpha_mask, data, origin = get_mask_data_origin(a_mask)
        base_64_data = alpha_mask.data_2_base64(data)
        origin = [alpha_mask.origin.row, alpha_mask.origin.col]
        expected_json = {
            "bitmap": {"origin": origin[::-1], "data": base_64_data},
            "shape": alpha_mask.name(),
            "geometryType": alpha_mask.name(),
        }
        assert alpha_mask.to_json() == expected_json


def test_from_json(random_alpha_mask_int_1, random_alpha_mask_int_2):
    for idx, a_mask in enumerate([random_alpha_mask_int_1, random_alpha_mask_int_2], 1):
        alpha_mask, data, origin = get_mask_data_origin(a_mask)
        base_64_data = alpha_mask.data_2_base64(data)
        origin = [alpha_mask.origin.row, alpha_mask.origin.col]
        alpha_mask_json = {
            "bitmap": {"origin": origin[::-1], "data": base_64_data},
            "shape": alpha_mask.name(),
            "geometryType": alpha_mask.name(),
        }
        alpha_mask = AlphaMask.from_json(alpha_mask_json)
        assert isinstance(alpha_mask, AlphaMask)
        origin = [alpha_mask.origin.row, alpha_mask.origin.col]
        check_origin_equal(alpha_mask, origin)


def test_crop(random_alpha_mask_int_1, random_alpha_mask_int_2):
    for idx, a_mask in enumerate([random_alpha_mask_int_1, random_alpha_mask_int_2], 1):
        alpha_mask, data, origin = get_mask_data_origin(a_mask)
        rect = Rectangle(top=10, left=10, bottom=20, right=20)
        cropped_alpha_masks = alpha_mask.crop(rect)
        for cidx, cropped_alpha_mask in enumerate(cropped_alpha_masks, 1):
            assert isinstance(cropped_alpha_mask, AlphaMask)
            assert [cropped_alpha_mask.origin.row, cropped_alpha_mask.origin.col] == [
                rect.top,
                rect.left,
            ]

            random_image = get_random_image()
            function_name = inspect.currentframe().f_code.co_name
            draw_test(
                dir_name, f"{function_name}_{cidx}_geometry_{idx}", random_image, cropped_alpha_mask
            )


def test_relative_crop(random_alpha_mask_int_1, random_alpha_mask_int_2):
    for idx, a_mask in enumerate([random_alpha_mask_int_1, random_alpha_mask_int_2], 1):
        alpha_mask, data, origin = get_mask_data_origin(a_mask)
        cropper = Rectangle(top=10, left=10, bottom=20, right=20)
        cropped_alpha_masks = alpha_mask.relative_crop(cropper)
        for cidx, cropped_alpha_mask in enumerate(cropped_alpha_masks, 1):
            assert isinstance(cropped_alpha_mask, AlphaMask)
            assert (
                0 <= cropped_alpha_mask.origin.row < 10
            ), "Cropped bitmap origin row is out of bounds."
            assert (
                0 <= cropped_alpha_mask.origin.col < 10
            ), "Cropped bitmap origin col is out of bounds."
            assert cropped_alpha_mask.data.size > 0, "Cropped bitmap data is empty."
            height, width = data.shape[:2]
            assert cropped_alpha_mask.data.shape[0] == min(height, cropper.top) + 1
            assert cropped_alpha_mask.data.shape[1] == min(width, cropper.left) + 1

            random_image = get_random_image()
            function_name = inspect.currentframe().f_code.co_name
            draw_test(
                dir_name, f"{function_name}_{cidx}_geometry_{idx}", random_image, cropped_alpha_mask
            )


def test_rotate(random_alpha_mask_int_1, random_alpha_mask_int_2):
    for idx, a_mask in enumerate([random_alpha_mask_int_1, random_alpha_mask_int_2], 1):
        alpha_mask, data, origin = get_mask_data_origin(a_mask)
        random_image = get_random_image([255, 255, 255])

        function_name = inspect.currentframe().f_code.co_name
        draw_test(
            dir_name,
            f"{function_name}_geometry_{idx}_original",
            random_image,
            alpha_mask,
            [255, 0, 0],
        )

        img_size, angle = random_image.shape[:2], random.randint(0, 360)
        rotator = ImageRotator(img_size, angle)
        rotated_alpha_mask = alpha_mask.rotate(rotator)
        rotated_image = rotator.rotate_img(random_image, True)

        assert isinstance(rotated_alpha_mask, AlphaMask)
        assert rotated_alpha_mask.data.shape != data.shape

        function_name = inspect.currentframe().f_code.co_name
        draw_test(
            dir_name,
            f"{function_name}_geometry_{idx}",
            rotated_image,
            rotated_alpha_mask,
            [0, 0, 255],
        )


def test_resize(random_alpha_mask_int_1, random_alpha_mask_int_2):
    for idx, a_mask in enumerate([random_alpha_mask_int_1, random_alpha_mask_int_2], 1):
        alpha_mask, data, origin = get_mask_data_origin(a_mask)
        random_image = get_random_image()
        in_size = data.shape[:2]
        out_size = (in_size[0] // 2, in_size[1] // 2)
        resized_alpha_mask = alpha_mask.resize(in_size, out_size)
        assert resized_alpha_mask.data.shape == out_size

        function_name = inspect.currentframe().f_code.co_name
        draw_test(dir_name, f"{function_name}_geometry_{idx}", random_image, resized_alpha_mask)


def test_scale(random_alpha_mask_int_1, random_alpha_mask_int_2):
    for idx, a_mask in enumerate([random_alpha_mask_int_1, random_alpha_mask_int_2], 1):
        alpha_mask, data, origin = get_mask_data_origin(a_mask)
        random_image = get_random_image()
        factor = round(random.uniform(0, 1), 3)
        scaled_alpha_mask = alpha_mask.scale(factor)
        assert scaled_alpha_mask.data.shape == (
            round(data.shape[0] * factor),
            round(data.shape[1] * factor),
        )

        function_name = inspect.currentframe().f_code.co_name
        draw_test(dir_name, f"{function_name}_geometry_{idx}", random_image, scaled_alpha_mask)


def test_translate(random_alpha_mask_int_1, random_alpha_mask_int_2):
    for idx, a_mask in enumerate([random_alpha_mask_int_1, random_alpha_mask_int_2], 1):
        alpha_mask, data, origin = get_mask_data_origin(a_mask)
        random_image = get_random_image()
        drow, dcol = random.randint(10, 150), random.randint(10, 350)
        translated_alpha_mask = alpha_mask.translate(drow, dcol)
        origin = [alpha_mask.origin.row, alpha_mask.origin.col]
        expected_trans_origin = [origin[0] + drow, origin[1] + dcol]
        check_origin_equal(translated_alpha_mask, expected_trans_origin)

        function_name = inspect.currentframe().f_code.co_name
        draw_test(dir_name, f"{function_name}_geometry_{idx}", random_image, translated_alpha_mask)


def test_fliplr(random_alpha_mask_int_1, random_alpha_mask_int_2):
    for idx, a_mask in enumerate([random_alpha_mask_int_1, random_alpha_mask_int_2], 1):
        alpha_mask, data, origin = get_mask_data_origin(a_mask)
        random_image = get_random_image()
        img_size = random_image.shape[:2]
        flipped_alpha_mask = alpha_mask.fliplr(img_size)
        assert np.array_equal(flipped_alpha_mask.data, np.fliplr(data))

        function_name = inspect.currentframe().f_code.co_name
        draw_test(dir_name, f"{function_name}_geometry_{idx}", random_image, flipped_alpha_mask)


def test_flipud(random_alpha_mask_int_1, random_alpha_mask_int_2):
    for idx, a_mask in enumerate([random_alpha_mask_int_1, random_alpha_mask_int_2], 1):
        alpha_mask, data, origin = get_mask_data_origin(a_mask)
        random_image = get_random_image()
        img_size = random_image.shape[:2]
        flipped_alpha_mask = alpha_mask.flipud(img_size)
        assert np.array_equal(flipped_alpha_mask.data, np.flipud(data))

        function_name = inspect.currentframe().f_code.co_name
        draw_test(dir_name, f"{function_name}_geometry_{idx}", random_image, flipped_alpha_mask)


def test__draw_bool_compatible(random_alpha_mask_int_1, random_alpha_mask_int_2):
    for idx, a_mask in enumerate([random_alpha_mask_int_1, random_alpha_mask_int_2], 1):
        alpha_mask, data, origin = get_mask_data_origin(a_mask)
        random_image = get_random_image()
        alpha_mask._draw_bool_compatible(alpha_mask._draw_impl, random_image, color, thickness)
        assert np.any(random_image == color)

        function_name = inspect.currentframe().f_code.co_name
        draw_test(dir_name, f"{function_name}_geometry_{idx}", random_image)


def test_draw(random_alpha_mask_int_1, random_alpha_mask_int_2):
    for idx, a_mask in enumerate([random_alpha_mask_int_1, random_alpha_mask_int_2], 1):
        alpha_mask, data, origin = get_mask_data_origin(a_mask)
        random_image = get_random_image()
        alpha_mask.draw(random_image, color, thickness)
        assert np.any(random_image == color)

        function_name = inspect.currentframe().f_code.co_name
        draw_test(dir_name, f"{function_name}_geometry_{idx}", random_image)


def test_get_mask(random_alpha_mask_int_1, random_alpha_mask_int_2):
    for idx, a_mask in enumerate([random_alpha_mask_int_1, random_alpha_mask_int_2], 1):
        alpha_mask, data, origin = get_mask_data_origin(a_mask)
        random_image = get_random_image()
        amask = alpha_mask.get_mask(random_image.shape[:2])
        assert amask.shape == random_image.shape[:2]
        assert amask.dtype == np.bool
        assert np.any(amask == True)

        new_amask = AlphaMask(amask)
        function_name = inspect.currentframe().f_code.co_name
        draw_test(dir_name, f"{function_name}_geometry_{idx}", random_image, new_amask)


def test__draw_impl(random_alpha_mask_int_1, random_alpha_mask_int_2):
    for idx, a_mask in enumerate([random_alpha_mask_int_1, random_alpha_mask_int_2], 1):
        alpha_mask, data, origin = get_mask_data_origin(a_mask)
        random_image = get_random_image()
        alpha_mask._draw_impl(random_image, color, thickness)
        assert np.any(random_image == color)
        assert np.unique(random_image).size == 2

        function_name = inspect.currentframe().f_code.co_name
        draw_test(dir_name, f"{function_name}_geometry_{idx}", random_image)


def test_draw_contour(random_alpha_mask_int_1, random_alpha_mask_int_2):
    for idx, a_mask in enumerate([random_alpha_mask_int_1, random_alpha_mask_int_2], 1):
        alpha_mask, data, origin = get_mask_data_origin(a_mask)
        random_image = get_random_image()
        alpha_mask.draw_contour(random_image, color, thickness)
        assert np.any(random_image == color)

        function_name = inspect.currentframe().f_code.co_name
        draw_test(dir_name, f"{function_name}_geometry_{idx}", random_image)


def test__draw_contour_impl(random_alpha_mask_int_1, random_alpha_mask_int_2):
    for idx, a_mask in enumerate([random_alpha_mask_int_1, random_alpha_mask_int_2], 1):
        alpha_mask, data, origin = get_mask_data_origin(a_mask)
        random_image = get_random_image()
        alpha_mask._draw_contour_impl(random_image, color, thickness)
        assert np.any(random_image == color)

        function_name = inspect.currentframe().f_code.co_name
        draw_test(dir_name, f"{function_name}_geometry_{idx}", random_image)


def test_area(random_alpha_mask_int_1, random_alpha_mask_int_2):
    for idx, a_mask in enumerate([random_alpha_mask_int_1, random_alpha_mask_int_2], 1):
        alpha_mask, data, origin = get_mask_data_origin(a_mask)
        assert alpha_mask.area == float(np.count_nonzero(data))


def test_to_bbox(random_alpha_mask_int_1, random_alpha_mask_int_2):
    for idx, a_mask in enumerate([random_alpha_mask_int_1, random_alpha_mask_int_2], 1):
        alpha_mask, data, origin = get_mask_data_origin(a_mask)
        alpha_mask_bbox = alpha_mask.to_bbox()
        assert isinstance(alpha_mask_bbox, Rectangle)
        assert alpha_mask_bbox.height == data.shape[0]
        assert alpha_mask_bbox.width == data.shape[1]

        random_image = get_random_image()
        function_name = inspect.currentframe().f_code.co_name
        draw_test(dir_name, f"{function_name}_geometry_{idx}", random_image, alpha_mask_bbox)


def test_clone(random_alpha_mask_int_1, random_alpha_mask_int_2):
    for idx, a_mask in enumerate([random_alpha_mask_int_1, random_alpha_mask_int_2], 1):
        alpha_mask, data, origin = get_mask_data_origin(a_mask)
        cloned_alpha_mask = alpha_mask.clone()
        assert isinstance(cloned_alpha_mask, AlphaMask)
        origin = [alpha_mask.origin.row, alpha_mask.origin.col]
        check_origin_equal(cloned_alpha_mask, origin)


def test_validate(random_alpha_mask_int_1, random_alpha_mask_int_2):
    for idx, a_mask in enumerate([random_alpha_mask_int_1, random_alpha_mask_int_2], 1):
        alpha_mask, data, origin = get_mask_data_origin(a_mask)
        assert alpha_mask.validate(alpha_mask.geometry_name(), settings=None) is None
        with pytest.raises(
            ValueError, match="Geometry validation error: shape names are mismatched!"
        ):
            alpha_mask.validate("different_shape_name", settings=None)


def test_config_from_json(random_alpha_mask_int_1, random_alpha_mask_int_2):
    for idx, a_mask in enumerate([random_alpha_mask_int_1, random_alpha_mask_int_2], 1):
        alpha_mask, data, origin = get_mask_data_origin(a_mask)
        config = {"key": "value"}
        returned_config = alpha_mask.config_from_json(config)
        assert returned_config == config


def test_config_to_json(random_alpha_mask_int_1, random_alpha_mask_int_2):
    for idx, a_mask in enumerate([random_alpha_mask_int_1, random_alpha_mask_int_2], 1):
        alpha_mask, data, origin = get_mask_data_origin(a_mask)
        config = {"key": "value"}
        returned_config = alpha_mask.config_to_json(config)
        assert returned_config == config


def test_allowed_transforms(random_alpha_mask_int_1, random_alpha_mask_int_2):
    for idx, a_mask in enumerate([random_alpha_mask_int_1, random_alpha_mask_int_2], 1):
        alpha_mask, data, origin = get_mask_data_origin(a_mask)
        allowed_transforms = alpha_mask.allowed_transforms()
        assert set(allowed_transforms) == set([Bitmap, AnyGeometry, Polygon, Rectangle])


def test_convert(random_alpha_mask_int_1, random_alpha_mask_int_2):
    for idx, a_mask in enumerate([random_alpha_mask_int_1, random_alpha_mask_int_2], 1):
        alpha_mask, data, origin = get_mask_data_origin(a_mask)
        assert alpha_mask.convert(type(alpha_mask)) == [alpha_mask]
        assert alpha_mask.convert(AnyGeometry) == [alpha_mask]
        for new_geometry in alpha_mask.allowed_transforms():
            converted = alpha_mask.convert(new_geometry)
            for g in converted:
                assert isinstance(g, new_geometry) or isinstance(g, AlphaMask)

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
            match="from {!r} to {!r}".format(alpha_mask.geometry_name(), Point.geometry_name()),
        ):
            alpha_mask.convert(Point)


# Alpha Mask specific methods
# ------------------------


def test_data(random_alpha_mask_int_1, random_alpha_mask_int_2):
    for idx, a_mask in enumerate([random_alpha_mask_int_1, random_alpha_mask_int_2], 1):
        alpha_mask, data, origin = get_mask_data_origin(a_mask)
        assert np.array_equal(alpha_mask.data, data)


def test_origin(random_alpha_mask_int_1, random_alpha_mask_int_2):
    for idx, a_mask in enumerate([random_alpha_mask_int_1, random_alpha_mask_int_2], 1):
        alpha_mask, data, origin = get_mask_data_origin(a_mask)
        origin = [alpha_mask.origin.row, alpha_mask.origin.col]
        check_origin_equal(alpha_mask, origin)


def test_base64_2_data(random_alpha_mask_int_1, random_alpha_mask_int_2):
    for idx, a_mask in enumerate([random_alpha_mask_int_1, random_alpha_mask_int_2], 1):
        alpha_mask, data, origin = get_mask_data_origin(a_mask)
        encoded = alpha_mask.data_2_base64(data)
        decoded = AlphaMask.base64_2_data(encoded)
        assert np.array_equal(decoded, data)


def test_data_2_base64(random_alpha_mask_int_1, random_alpha_mask_int_2):
    for idx, a_mask in enumerate([random_alpha_mask_int_1, random_alpha_mask_int_2], 1):
        alpha_mask, data, origin = get_mask_data_origin(a_mask)
        encoded = alpha_mask.data_2_base64(data)
        decoded = AlphaMask.base64_2_data(encoded)
        assert np.array_equal(decoded, data)


def test_skeletonize(random_alpha_mask_int_1, random_alpha_mask_int_2):
    for idx, a_mask in enumerate([random_alpha_mask_int_1, random_alpha_mask_int_2], 1):
        alpha_mask, data, origin = get_mask_data_origin(a_mask)
        alpha_mask = AlphaMask(data=data / 255)
        for method in SkeletonizeMethod:
            skeleton = alpha_mask.skeletonize(method)
            assert isinstance(skeleton, AlphaMask)

            random_image = get_random_image()
            function_name = inspect.currentframe().f_code.co_name
            draw_test(
                dir_name, f"{function_name}_geometry_{idx}_{method.name}", random_image, skeleton
            )


def test_to_contours(random_alpha_mask_int_1, random_alpha_mask_int_2):
    for idx, a_mask in enumerate([random_alpha_mask_int_1, random_alpha_mask_int_2], 1):
        alpha_mask, data, origin = get_mask_data_origin(a_mask)
        contours = alpha_mask.to_contours()
        assert isinstance(contours, list), "Output should be a list"
        assert len(contours) > 0, "List should not be empty"
        for cidx, contour in enumerate(contours, 1):
            assert isinstance(contour, Polygon), "All elements in the list should be Polygons"

            random_image = get_random_image()
            function_name = inspect.currentframe().f_code.co_name
            draw_test(dir_name, f"{function_name}_{cidx}_geometry_{idx}", random_image, contour)


def test_bitwise_mask(random_alpha_mask_int_1, random_alpha_mask_int_2):
    for idx, a_mask in enumerate([random_alpha_mask_int_1, random_alpha_mask_int_2], 1):
        alpha_mask, data, origin = get_mask_data_origin(a_mask)
        return
        # @TODO: Fix this test
        mask = np.ones(data.shape, dtype=data.dtype)
        assert mask.shape == data.shape, "Mask and data shapes must match"
        result = alpha_mask.bitwise_mask(mask, np.logical_and)

        assert (
            isinstance(result, AlphaMask) or result == []
        ), "Output should be a Bitmap instance or an empty list"

        if isinstance(result, AlphaMask):
            assert (
                result.data.shape == data.shape
            ), "Resulting data should have the same shape as input data"
            assert (
                result.data.dtype == data.dtype
            ), "Resulting data should have the same data type as input data"
            expected_result = np.logical_and(data, mask)
            assert np.all(
                expected_result == result.data
            ), "Resulting data should match the expected bitwise AND result"


def test_from_path(random_alpha_mask_int_1, random_alpha_mask_int_2):
    pass


if __name__ == "__main__":
    pytest.main([__file__])

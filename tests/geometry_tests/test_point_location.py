import random
from typing import Tuple, Union

import numpy as np
import pytest

from supervisely.geometry.image_rotator import ImageRotator
from supervisely.geometry.point_location import PointLocation

# Draw Settings
color = [255, 255, 255]
thickness = 1


@pytest.fixture
def random_image() -> np.ndarray:
    image_shape = (random.randint(501, 1000), random.randint(501, 1000), 3)
    background_color = [0, 0, 0]
    bitmap = np.full(image_shape, background_color, dtype=np.uint8)
    return bitmap


@pytest.fixture
def random_point_location_int() -> Tuple[PointLocation, Union[int, float], Union[int, float]]:
    row = random.randint(0, 500)
    col = random.randint(0, 500)
    loc = PointLocation(row, col)
    return loc, row, col


@pytest.fixture
def random_point_location_float() -> Tuple[PointLocation, Union[int, float], Union[int, float]]:
    row = round(random.uniform(0, 500), 6)
    col = round(random.uniform(0, 500), 6)
    loc = PointLocation(row, col)
    return loc, row, col


def get_loc_row_col(
    point_location,
) -> Tuple[PointLocation, Union[int, float], Union[int, float]]:
    return point_location


def check_points_equal(
    point_loc_1: PointLocation,
    point_loc_2: PointLocation,
):
    assert isinstance(point_loc_1, PointLocation)
    assert isinstance(point_loc_2, PointLocation)
    assert point_loc_1.row == point_loc_2.row
    assert point_loc_1.col == point_loc_2.col


def test_geometry_name(random_point_location_int, random_point_location_float, random_image):
    # 'PointLocation' object has no attribute 'geometry_name'
    pass


def test_name(random_point_location_int, random_point_location_float, random_image):
    # 'PointLocation' object has no attribute 'name'
    pass


def test_row_property(random_point_location_int, random_point_location_float, random_image):
    for point_location in [random_point_location_int, random_point_location_float]:
        loc, row, _ = get_loc_row_col(point_location)
        assert loc.row == row


def test_col_property(random_point_location_int, random_point_location_float, random_image):
    for point_location in [random_point_location_int, random_point_location_float]:
        loc, _, col = get_loc_row_col(point_location)
        assert loc.col == col


def test_to_json(random_point_location_int, random_point_location_float, random_image):
    for point_location in [random_point_location_int, random_point_location_float]:
        loc, row, col = get_loc_row_col(point_location)
        expected_json = {"points": {"exterior": [[col, row]], "interior": []}}
        assert loc.to_json() == expected_json


def test_from_json(random_point_location_int, random_point_location_float, random_image):
    for point_location in [random_point_location_int, random_point_location_float]:
        loc, row, col = get_loc_row_col(point_location)
        loc_json = {"points": {"exterior": [[col, row]], "interior": []}}
        loc_from_json = PointLocation.from_json(loc_json)
        check_points_equal(loc, loc_from_json)


def test_crop(random_point_location_int, random_point_location_float, random_image):
    # 'PointLocation' object has no attribute 'crop'
    pass


def test_relative_crop(random_point_location_int, random_point_location_float, random_image):
    # 'PointLocation' object has no attribute 'relative_crop'
    pass


def test_rotate(random_point_location_int, random_point_location_float, random_image):
    for point_location in [random_point_location_int, random_point_location_float]:
        loc, row, col = get_loc_row_col(point_location)
        img_size, angle = random_image.shape[:2], random.randint(0, 360)
        rotator = ImageRotator(img_size, angle)
        rotate_loc = loc.rotate(rotator)
        assert isinstance(rotate_loc, PointLocation)

        point_np_uniform = np.array([row, col, 1])
        transformed_np = rotator.affine_matrix.dot(point_np_uniform)
        expected_row = round(transformed_np[0].item())
        expected_col = round(transformed_np[1].item())

        assert rotate_loc.row == expected_row
        assert rotate_loc.col == expected_col


def test_resize(random_point_location_int, random_point_location_float, random_image):
    for point_location in [random_point_location_int, random_point_location_float]:
        loc, row, col = get_loc_row_col(point_location)
        in_size = random_image.shape[:2]
        out_size = (random.randint(1000, 1200), random.randint(1000, 1200))
        resize_loc = loc.resize(in_size, out_size)
        assert isinstance(resize_loc, PointLocation)
        assert resize_loc.row == round(loc.row * out_size[0] / in_size[0])
        assert resize_loc.col == round(loc.col * out_size[1] / in_size[1])


def test_scale(random_point_location_int, random_point_location_float, random_image):
    for point_location in [random_point_location_int, random_point_location_float]:
        loc, row, col = get_loc_row_col(point_location)
        factor = round(random.uniform(0, 1), 3)
        scale_loc = loc.scale(factor)
        assert isinstance(scale_loc, PointLocation)
        assert scale_loc.row == round(row * factor)
        assert scale_loc.col == round(col * factor)


def test_scale_frow_fcol(random_point_location_int, random_point_location_float, random_image):
    for point_location in [random_point_location_int, random_point_location_float]:
        loc, row, col = get_loc_row_col(point_location)
        frow, fcol = round(random.uniform(0, 1), 3), round(random.uniform(2, 3), 3)
        loc_scale_rc = loc.scale_frow_fcol(frow, fcol)
        assert isinstance(loc_scale_rc, PointLocation)
        assert loc_scale_rc.row == round(row * frow)
        assert loc_scale_rc.col == round(col * fcol)


def test_translate(random_point_location_int, random_point_location_float, random_image):
    for point_location in [random_point_location_int, random_point_location_float]:
        loc, row, col = get_loc_row_col(point_location)
        drow, dcol = random.randint(10, 150), random.randint(10, 350)
        translate_loc = loc.translate(drow, dcol)
        assert isinstance(translate_loc, PointLocation)
        assert translate_loc.row == row + drow
        assert translate_loc.col == col + dcol


def test_fliplr(random_point_location_int, random_point_location_float, random_image):
    for point_location in [random_point_location_int, random_point_location_float]:
        loc, row, col = get_loc_row_col(point_location)
        img_size = random_image.shape[:2]
        fliplr_loc = loc.fliplr(img_size)
        assert isinstance(fliplr_loc, PointLocation)
        assert fliplr_loc.row == loc.row
        assert fliplr_loc.col == img_size[1] - loc.col


def test_flipud(random_point_location_int, random_point_location_float, random_image):
    for point_location in [random_point_location_int, random_point_location_float]:
        loc, row, col = get_loc_row_col(point_location)
        img_size = random_image.shape[:2]
        flipud_loc = loc.flipud(img_size)
        assert isinstance(flipud_loc, PointLocation)
        assert flipud_loc.row == img_size[0] - loc.row
        assert flipud_loc.col == loc.col


def test_draw_bool_compatible(random_point_location_int, random_point_location_float, random_image):
    # 'PointLocation' object has no attribute '_draw_bool_compatible'
    pass


def test_draw(random_point_location_int, random_point_location_float, random_image):
    # 'PointLocation' object has no attribute 'draw'
    pass


def test_get_mask(random_point_location_int, random_point_location_float, random_image):
    # 'PointLocation' object has no attribute 'get_mask'
    pass


def test_draw_impl(random_point_location_int, random_point_location_float, random_image):
    # 'PointLocation' object has no attribute '_draw_impl'
    pass


def test_draw_contour(random_point_location_int, random_point_location_float, random_image):
    # 'PointLocation' object has no attribute 'draw_contour'
    pass


def test_draw_contour_impl(random_point_location_int, random_point_location_float, random_image):
    # 'PointLocation' object has no attribute '_draw_contour_impl'
    pass


def test_area(random_point_location_int, random_point_location_float, random_image):
    # 'PointLocation' object has no attribute 'area'
    pass


def test_to_bbox(random_point_location_int, random_point_location_float, random_image):
    # 'PointLocation' object has no attribute 'to_bbox'
    pass


def test_clone(random_point_location_int, random_point_location_float, random_image):
    for point_location in [random_point_location_int, random_point_location_float]:
        loc, _, _ = get_loc_row_col(point_location)
        clone_loc = loc.clone()
        assert isinstance(clone_loc, PointLocation)
        assert clone_loc.row == loc.row
        assert clone_loc.col == loc.col


def test_validate(random_point_location_int, random_point_location_float, random_image):
    # 'PointLocation' object has no attribute 'validate'
    pass


def test_config_from_json(random_point_location_int, random_point_location_float, random_image):
    # 'PointLocation' object has no attribute 'config_from_json'
    pass


def test_config_to_json(random_point_location_int, random_point_location_float, random_image):
    # 'PointLocation' object has no attribute 'config_to_json'
    pass


def test_allowed_transforms(random_point_location_int, random_point_location_float, random_image):
    # 'PointLocation' object has no attribute 'allowed_transforms'
    pass


def test_convert(random_point_location_int, random_point_location_float, random_image):
    # 'PointLocation' object has no attribute 'convert'
    pass


if __name__ == "__main__":
    pytest.main([__file__])

import random
from typing import Tuple, Union

import numpy as np
import pytest

import supervisely_lib as sly
from supervisely import PointLocation
from supervisely.geometry.image_rotator import ImageRotator


@pytest.fixture
def random_point_location_int() -> Tuple[PointLocation, Union[int, float], Union[int, float]]:
    row = random.randint(0, 1000)
    col = random.randint(0, 1000)
    loc = PointLocation(row, col)
    return loc, row, col


@pytest.fixture
def random_point_location_float() -> Tuple[PointLocation, Union[int, float], Union[int, float]]:
    row = round(random.uniform(0, 1000), 6)
    col = round(random.uniform(0, 1000), 6)
    loc = PointLocation(row, col)
    return loc, row, col


def test_row_property(
    random_point_location_int: PointLocation, random_point_location_float: PointLocation
):
    for point_location in [random_point_location_int, random_point_location_float]:
        loc, row, _ = point_location
        assert loc.row == row


def test_col_property(
    random_point_location_int: PointLocation, random_point_location_float: PointLocation
):
    for point_location in [random_point_location_int, random_point_location_float]:
        loc, _, col = point_location
        assert loc.col == col


def test_to_json_method(
    random_point_location_int: PointLocation, random_point_location_float: PointLocation
):
    for point_location in [random_point_location_int, random_point_location_float]:
        loc, row, col = point_location
        expected_json = {"points": {"exterior": [[col, row]], "interior": []}}
        assert loc.to_json() == expected_json


def test_from_json():
    loc_json_int = {"points": {"exterior": [[200, 100]], "interior": []}}
    loc_int = PointLocation.from_json(loc_json_int)
    assert isinstance(loc_int, PointLocation)
    assert loc_int.row == 100
    assert loc_int.col == 200

    loc_json_float = {"points": {"exterior": [[200.548765, 100.213548]], "interior": []}}
    loc_float = PointLocation.from_json(loc_json_float)
    assert isinstance(loc_float, PointLocation)
    assert loc_float.col == 200.548765
    assert loc_float.row == 100.213548


def test_scale(
    random_point_location_int: PointLocation, random_point_location_float: PointLocation
):
    for point_location in [random_point_location_int, random_point_location_float]:
        loc, row, col = point_location
        factor = 0.75
        scale_loc = loc.scale(factor)
        assert isinstance(scale_loc, PointLocation)
        assert scale_loc.row == round(row * factor)
        assert scale_loc.col == round(col * factor)


def test_scale_frow_fcol(
    random_point_location_int: PointLocation, random_point_location_float: PointLocation
):
    for point_location in [random_point_location_int, random_point_location_float]:
        loc, row, col = point_location
        frow, fcol = 0.1, 2.7
        loc_scale_rc = loc.scale_frow_fcol(frow, fcol)
        assert isinstance(loc_scale_rc, PointLocation)
        assert loc_scale_rc.row == round(row * frow)
        assert loc_scale_rc.col == round(col * fcol)


def test_translate(
    random_point_location_int: PointLocation, random_point_location_float: PointLocation
):
    for point_location in [random_point_location_int, random_point_location_float]:
        loc, row, col = point_location
        drow, dcol = 150, 350
        translate_loc = loc.translate(drow, dcol)
        assert isinstance(translate_loc, PointLocation)
        assert translate_loc.row == row + drow
        assert translate_loc.col == col + dcol


def test_rotate(
    random_point_location_int: PointLocation, random_point_location_float: PointLocation
):
    for point_location in [random_point_location_int, random_point_location_float]:
        loc, row, col = point_location
        height, width = 300, 400
        rotator = ImageRotator((height, width), 25)
        rotate_loc = loc.rotate(rotator)
        assert isinstance(rotate_loc, PointLocation)

        point_np_uniform = np.array([row, col, 1])
        transformed_np = rotator.affine_matrix.dot(point_np_uniform)
        expected_row = round(transformed_np[0].item())
        expected_col = round(transformed_np[1].item())

        assert rotate_loc.row == expected_row
        assert rotate_loc.col == expected_col


def test_resize(
    random_point_location_int: PointLocation, random_point_location_float: PointLocation
):
    for point_location in [random_point_location_int, random_point_location_float]:
        loc, _, _ = point_location
        in_size = (300, 400)
        out_size = (600, 800)
        resize_loc = loc.resize(in_size, out_size)
        assert isinstance(resize_loc, PointLocation)
        assert resize_loc.row == round(loc.row * out_size[0] / in_size[0])
        assert resize_loc.col == round(loc.col * out_size[1] / in_size[1])


def test_fliplr(
    random_point_location_int: PointLocation, random_point_location_float: PointLocation
):
    for point_location in [random_point_location_int, random_point_location_float]:
        loc, _, _ = point_location
        img_size = (300, 400)
        fliplr_loc = loc.fliplr(img_size)
        assert isinstance(fliplr_loc, PointLocation)
        assert fliplr_loc.row == loc.row
        assert fliplr_loc.col == img_size[1] - loc.col


def test_flipud(
    random_point_location_int: PointLocation, random_point_location_float: PointLocation
):
    for point_location in [random_point_location_int, random_point_location_float]:
        loc, _, _ = point_location
        img_size = (300, 400)
        flipud_loc = loc.flipud(img_size)
        assert isinstance(flipud_loc, PointLocation)
        assert flipud_loc.row == img_size[0] - loc.row
        assert flipud_loc.col == loc.col


def test_clone(
    random_point_location_int: PointLocation, random_point_location_float: PointLocation
):
    for point_location in [random_point_location_int, random_point_location_float]:
        loc, _, _ = point_location
        clone_loc = loc.clone()
        assert isinstance(clone_loc, PointLocation)
        assert clone_loc.row == loc.row
        assert clone_loc.col == loc.col


if __name__ == "__main__":
    pytest.main([__file__])
if __name__ == "__main__":
    pytest.main([__file__])

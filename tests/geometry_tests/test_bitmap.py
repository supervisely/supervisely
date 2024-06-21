import random
from typing import List, Tuple, Union

import numpy as np
import pytest

from supervisely import Bitmap, Rectangle
from supervisely.geometry.image_rotator import ImageRotator

@pytest.fixture
def random_bitmap_int() -> Tuple[
    Bitmap,
    List[Tuple[Union[int, float], Union[int, float]]],
    List[Tuple[Union[int, float], Union[int, float]]],
]:
    origin = (random.randint(0, 1000), random.randint(0, 1000))
    data = np.zeros((1000, 1000), dtype=np.uint8)
    bitmap = Bitmap(data=data, origin=origin)
    return bitmap, origin


@pytest.fixture
def random_bitmap_float() -> Tuple[
    Bitmap,
    List[Tuple[Union[int, float], Union[int, float]]],
    List[Tuple[Union[int, float], Union[int, float]]],
]:
    origin = (
        round(random.uniform(0, 1000), 6), round(random.uniform(0, 1000), 6)
    )
    data = np.zeros((1000, 1000), dtype=np.uint8)
    bitmap = Bitmap(data=data, origin=origin)
    return bitmap, origin


def check_origin(bitmap: Bitmap, origin: Tuple[int, int]):
    assert bitmap.origin == origin


def test_origin_property(random_bitmap_int: Bitmap, random_bitmap_float: Bitmap):
    for bitmap in [random_bitmap_int, random_bitmap_float]:
        bmp, origin = bitmap
        check_origin(bmp, origin)


def test_to_json_method(random_bitmap_int: Bitmap, random_bitmap_float: Bitmap):
    for bitmap in [random_bitmap_int, random_bitmap_float]:
        bmp, origin = bitmap
        expected_json = {
            "origin": origin,
            "shape": "bitmap",
            "geometryType": "bitmap",
        }
        assert bmp.to_json() == expected_json


def test_from_json():
    bitmap_json = {
        "origin": [200, 100],
        "shape": "bitmap",
        "geometryType": "bitmap",
    }
    bmp = Bitmap.from_json(bitmap_json)
    assert isinstance(bmp, Bitmap)
    check_origin(bmp, (100, 200))


# Add more tests for other methods like scale, translate, rotate, resize, fliplr, flipud, clone, crop, area, approx_dp
# similar to the Polygon tests, adjusting as necessary for the Bitmap class.

if __name__ == "__main__":
    pytest.main([__file__])
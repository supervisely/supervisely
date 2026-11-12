"""
AlphaMask -> Bitmap conversion.

The conversion drew the alpha mask on a boolean canvas through a uint8 temporary
filled with color 1, so the alpha blend `alpha * 1` truncated to 0 for every pixel
with alpha below 255. A mask with no fully opaque pixel became an empty Bitmap and
`Bitmap.__init__` raised "Creating a bitmap with an empty mask", which took down the
whole Convert Class Shape run. A mask with some opaque pixels silently lost the
semi-transparent ones. Every pixel with non-zero alpha is part of the object, as
AlphaMask.area, to_bbox() and the Polygon conversion already count it.
"""

import numpy as np
import pytest

import supervisely as sly
from supervisely.geometry.helpers import geometry_to_alpha_mask, geometry_to_bitmap


def _alpha(values, shape=(12, 10), top=2, left=3):
    data = np.zeros(shape, dtype=np.uint8)
    data[top : top + values.shape[0], left : left + values.shape[1]] = values
    return sly.AlphaMask(data, origin=sly.PointLocation(row=5, col=7))


def test_no_opaque_pixel_converts_to_bitmap():
    mask = _alpha(np.full((3, 4), 128, dtype=np.uint8))

    result = mask.convert(sly.Bitmap)

    assert len(result) == 1
    bitmap = result[0]
    assert isinstance(bitmap, sly.Bitmap)
    assert bitmap.data.shape == (3, 4) and bitmap.data.all()
    # tight bbox of the alpha mask: origin (5, 7) + offset (2, 3)
    assert (bitmap.origin.row, bitmap.origin.col) == (7, 10)
    assert bitmap.to_bbox().to_json() == mask.to_bbox().to_json()
    assert bitmap.area == mask.area == 12


def test_partly_transparent_pixels_are_kept():
    values = np.array([[255, 1, 0], [0, 200, 0], [64, 0, 255]], dtype=np.uint8)
    mask = _alpha(values)

    (bitmap,) = mask.convert(sly.Bitmap)

    assert np.array_equal(bitmap.data, values > 0)
    assert bitmap.area == mask.area == 5


def test_label_convert_does_not_raise():
    """The Convert Class Shape path: Label.convert() with a Bitmap class."""
    src = sly.ObjClass("soft", sly.AlphaMask)
    dst = sly.ObjClass("soft", sly.Bitmap)
    label = sly.Label(_alpha(np.full((2, 2), 30, dtype=np.uint8)), src)

    (converted,) = label.convert(dst)

    assert converted.obj_class == dst
    assert converted.geometry.area == 4


@pytest.mark.parametrize("target", [sly.Polygon, sly.Rectangle, sly.Multipolygon])
def test_other_targets_unchanged(target):
    mask = _alpha(np.full((3, 4), 128, dtype=np.uint8))

    result = mask.convert(target)

    assert len(result) == 1 and isinstance(result[0], target)


def test_geometry_to_alpha_mask_keeps_alpha_values():
    values = np.array([[10, 0], [255, 128]], dtype=np.uint8)
    mask = _alpha(values)

    (copy,) = geometry_to_alpha_mask(mask)

    assert np.array_equal(copy.data, values)
    assert copy.origin.to_json() == mask.origin.to_json()


def test_geometry_to_bitmap_with_crop():
    mask = _alpha(np.full((3, 4), 50, dtype=np.uint8))  # rows 7..9, cols 10..13

    (cropped,) = geometry_to_bitmap(mask, crop_image_shape=(9, 12))

    assert cropped.data.shape == (2, 2) and cropped.data.all()
    assert geometry_to_bitmap(mask, crop_image_shape=(5, 5)) == []


def test_vector_shapes_still_convert_to_bitmap():
    polygon = sly.Polygon(
        exterior=[sly.PointLocation(0, 0), sly.PointLocation(0, 9), sly.PointLocation(9, 9)]
    )
    (bitmap,) = polygon.convert(sly.Bitmap)
    assert bitmap.area > 0

    (thick,) = sly.Polyline([sly.PointLocation(5, 5), sly.PointLocation(5, 15)]).convert(
        sly.Bitmap, contour_radius=2
    )
    assert thick.data.shape[0] >= 5

    (back,) = bitmap.convert(sly.AlphaMask)
    assert back.data.max() == 255 and back.area == bitmap.area

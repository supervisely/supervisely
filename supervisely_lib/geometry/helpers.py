# coding: utf-8

import numpy as np

from supervisely_lib.geometry.point_location import PointLocation
from supervisely_lib.geometry.rectangle import Rectangle
from supervisely_lib.geometry.bitmap import Bitmap


def geometry_to_bitmap(geometry, radius: int = 0, crop_image_shape: tuple = None) -> list:
    """
    Args:
        geometry: Geometry type which implemented 'draw', 'translate' and 'to_bbox` methods
        radius: half of thickness of drawed vector elements
        crop_image_shape: if not None - crop bitmap object by this shape (HxW)
    Returns:
        Bitmap (geometry) object
    """

    thickness = radius + 1

    bbox = geometry.to_bbox()
    extended_bbox = Rectangle(top=bbox.top - radius,
                              left=bbox.left - radius,
                              bottom=bbox.bottom + radius,
                              right=bbox.right + radius)
    bitmap_data = np.full(shape=(extended_bbox.height, extended_bbox.width), fill_value=False)
    geometry = geometry.translate(-extended_bbox.top, -extended_bbox.left)
    geometry.draw(bitmap_data, color=True, thickness=thickness)

    origin = PointLocation(extended_bbox.top, extended_bbox.left)
    bitmap_geometry = Bitmap(data=bitmap_data, origin=origin)
    if crop_image_shape is not None:
        crop_rect = Rectangle.from_size(*crop_image_shape)
        return bitmap_geometry.crop(crop_rect)
    return [bitmap_geometry]

# coding: utf-8

from copy import deepcopy
from supervisely._utils import take_with_default, validate_img_size
from supervisely.video_annotation.frame_collection import FrameCollection
from supervisely.volume_annotation.slice import Slice
from supervisely.volume_annotation.plane_info import (
    PlaneName,
    get_normal,
    get_img_size_from_volume_meta,
    get_slices_count_from_volume_meta,
)

# aka SliceCollection by analogy with FrameCollection
class Plane(FrameCollection):
    """
    Collection that stores Frame instances.
    """

    item_type = Slice

    def __init__(
        self, plane_name, img_size=None, slices_count=None, items=None, volume_meta=None
    ):
        PlaneName.validate(plane_name)
        self._plane_name = plane_name

        if img_size is None and volume_meta is None:
            raise ValueError(
                "Both img_size and volume_meta are None, only one of them is allowed to be a None"
            )
        if slices_count is None and volume_meta is None:
            raise ValueError(
                "Both slices_count and volume_meta are None, only one of them is allowed to be a None"
            )
        self._img_size = take_with_default(
            img_size, get_img_size_from_volume_meta(plane_name, volume_meta)
        )
        self._img_size = validate_img_size(self._img_size)

        self._slices_count = take_with_default(
            slices_count, get_slices_count_from_volume_meta(plane_name, volume_meta)
        )

        super.__init__(items=items)

    @property
    def plane_name(self):
        return self._plane_name

    @property
    def slices_count(self):
        return self._slices_count

    @property
    def img_size(self):
        return deepcopy(self._img_size)

    @property
    def normal(self):
        return get_normal(self.plane_name)

    def __str__(self):
        return super().__str__().replace("Frames", "Slices")

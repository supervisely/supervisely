# coding: utf-8

from copy import deepcopy
from supervisely.video_annotation.frame_collection import FrameCollection
from supervisely.volume_annotation.slice import Slice
from supervisely.volume_annotation.plane_info import PlaneName, get_normal

# aka SliceCollection by analogy with FrameCollection
class Plane(FrameCollection):
    """
    Collection that stores Frame instances.
    """

    item_type = Slice

    def __init__(self, plane_name, img_size, items=None):
        PlaneName.validate(plane_name)
        self._plane_name = plane_name

        if not isinstance(img_size, (tuple, list)):
            raise TypeError(
                '{!r} has to be a tuple or a list. Given type "{}".'.format(
                    "img_size", type(img_size)
                )
            )
        self._img_size = tuple(img_size)

        super.__init__(items=items)

    @property
    def plane_name(self):
        return self._plane_name

    @property
    def img_size(self):
        return deepcopy(self._img_size)

    @property
    def normal(self):
        return get_normal(self.plane_name)

    def __str__(self):
        return super().__str__().replace("Frames", "Slices")

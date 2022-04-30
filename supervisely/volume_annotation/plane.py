# coding: utf-8

from supervisely.video_annotation.frame_collection import FrameCollection
from supervisely.volume_annotation.slice import Slice
from supervisely.volume_annotation.plane_info import PlaneName, get_normal

# aka SliceCollection by analogy with FrameCollection
class Plane(FrameCollection):
    """
    Collection that stores Frame instances.
    """

    item_type = Slice

    def __init__(self, plane_name, items=None):
        PlaneName.validate(plane_name)
        self._plane_name = plane_name
        super.__init__(items=items)

    @property
    def plane_name(self):
        return self._plane_name

    @property
    def normal(self):
        return get_normal(self.plane_name)

    def __str__(self):
        return super().__str__().replace("Frames", "Slices")

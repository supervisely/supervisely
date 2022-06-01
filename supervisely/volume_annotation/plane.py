# coding: utf-8

from copy import deepcopy
from supervisely._utils import take_with_default, validate_img_size
from supervisely.video_annotation.frame_collection import FrameCollection
from supervisely.volume_annotation import constants
from supervisely.volume_annotation.slice import Slice
from supervisely.sly_logger import logger

# example:
# "volumeMeta": {
#     "ACS": "RAS",
#     "intensity": { "max": 3071, "min": -3024 },
#     "windowWidth": 6095,
#     "rescaleSlope": 1,
#     "windowCenter": 23.5,
#     "channelsCount": 1,
#     "dimensionsIJK": { "x": 512, "y": 512, "z": 139 },
#     "IJK2WorldMatrix": [
#         0.7617189884185793, 0, 0, -194.238403081894, 0, 0.7617189884185793, 0,
#         -217.5384061336518, 0, 0, 2.5, -347.7500000000001, 0, 0, 0, 1
#     ],
#     "rescaleIntercept": 0
# },


class Plane(FrameCollection):
    item_type = Slice

    SAGITTAL = "sagittal"
    CORONAL = "coronal"
    AXIAL = "axial"
    _valid_names = [SAGITTAL, CORONAL, AXIAL, None]  # None for spatial figures

    @staticmethod
    def validate_name(name):
        if name not in Plane._valid_names:
            raise ValueError(
                f"Unknown plane {name}, valid names are {Plane._valid_names}"
            )

    def __init__(
        self, plane_name, img_size=None, slices_count=None, items=None, volume_meta=None
    ):
        Plane.validate_name(plane_name)
        self._name = plane_name

        if img_size is None and volume_meta is None:
            raise ValueError(
                "Both img_size and volume_meta are None, only one of them is allowed to be a None"
            )
        if slices_count is None and volume_meta is None:
            raise ValueError(
                "Both slices_count and volume_meta are None, only one of them is allowed to be a None"
            )
        self._img_size = take_with_default(
            img_size, Plane.get_img_size(self._name, volume_meta)
        )
        self._img_size = validate_img_size(self._img_size)

        self._slices_count = take_with_default(
            slices_count, Plane.get_slices_count(self._name, volume_meta)
        )

        super().__init__(items=items)

    @property
    def name(self):
        return self._name

    @property
    def slices_count(self):
        return self._slices_count

    @property
    def img_size(self):
        return deepcopy(self._img_size)

    @property
    def normal(self):
        return Plane.get_normal(self.name)

    def __str__(self):
        return super().__str__().replace("Frames", "Slices")

    @staticmethod
    def get_normal(name):
        Plane.validate_name(name)
        if name == Plane.SAGITTAL:
            return {"x": 1, "y": 0, "z": 0}
        if name == Plane.CORONAL:
            return {"x": 0, "y": 1, "z": 0}
        if name == Plane.AXIAL:
            return {"x": 0, "y": 0, "z": 1}

    @staticmethod
    def get_img_size(name, volume_meta):
        Plane.validate_name(name)
        dimentions = volume_meta["dimensionsIJK"]
        # (height, width)
        height = None
        width = None
        if name == Plane.SAGITTAL:
            width = dimentions["y"]
            height = dimentions["z"]
        elif name == Plane.CORONAL:
            width = dimentions["x"]
            height = dimentions["z"]
        elif name == Plane.AXIAL:
            width = dimentions["x"]
            height = dimentions["y"]
        return [height, width]

    @staticmethod
    def get_slices_count(name, volume_meta):
        Plane.validate_name(name)
        dimentions = volume_meta["dimensionsIJK"]
        if name == Plane.SAGITTAL:
            return dimentions["x"]
        elif name == Plane.CORONAL:
            return dimentions["y"]
        elif name == Plane.AXIAL:
            return dimentions["z"]

    @classmethod
    def from_json(
        cls,
        data,
        plane_name,
        objects,
        img_size=None,
        slices_count=None,
        volume_meta=None,
        key_id_map=None,
    ):
        Plane.validate_name(plane_name)
        if plane_name != data[constants.NAME]:
            raise ValueError(
                f"Plane name {plane_name} differs from the name in json data {data[constants.NAME]}"
            )

        if Plane.get_normal(plane_name) != data[constants.NORMAL]:
            raise ValueError(
                f"Normal json data {data[constants.NORMAL]} is not valid for {plane_name}. It should be {Plane.get_normal(plane_name)}"
            )
        if slices_count is None:
            if volume_meta is None:
                raise ValueError(
                    "Both slices_count and volume_meta are None, only one of them is allowed to be a None"
                )
            else:
                slices_count = Plane.get_slices_count(plane_name, volume_meta)

        slices = []
        for slice_json in data[constants.SLICES]:
            slices.append(
                cls.item_type.from_json(
                    slice_json, objects, plane_name, slices_count, key_id_map
                )
            )

        return cls(plane_name, img_size, slices_count, slices, volume_meta)

    def to_json(self, key_id_map=None):
        json_slices = []
        for slice in self:
            slice: Slice
            json_slices.append(slice.to_json(key_id_map))

        return {
            constants.NAME: self.name,
            constants.NORMAL: self.normal,
            constants.SLICES: json_slices,
        }

    def validate_figures_bounds(self):
        for slice in self:
            slice: Slice
            slice.validate_figures_bounds(self.img_size)

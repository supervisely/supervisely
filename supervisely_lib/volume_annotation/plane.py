from supervisely_lib.video_annotation.frame import Frame
from supervisely_lib.video_annotation.frame_collection import FrameCollection
from supervisely_lib.volume_annotation.constants import PLANE_NAMES, NAME, NORMAL, SLICES


class Plane(FrameCollection):
    item_type = Frame

    def __init__(self, name=None, normal=None, slices=None):
        super(Plane, self).__init__(items=slices)
        self.validate_plane_name(name)
        self._name = name
        self._normal = normal

    def __str__(self):
        return f'Plane name: {self.name}, Normal: {self.normal}, Slices: {super(Plane, self).__str__()}'

    @property
    def name(self):
        return self._name

    @property
    def normal(self):
        return self._normal

    def to_json(self, key_id_map=None):
        return {NAME: self.name,
                NORMAL: self.normal,
                SLICES: [slice.to_json(key_id_map) for slice in self]}

    @classmethod
    def from_json(cls, data, objects, key_id_map=None):
        slices_json = data[SLICES]
        slices = [cls.item_type.from_json(slice_json, objects, key_id_map=key_id_map) for slice_json in slices_json]

        name = data[NAME]
        normal = data[NORMAL]
        return cls(name, normal, slices)

    @staticmethod
    def validate_plane_name(plane_name):
        if plane_name not in PLANE_NAMES and plane_name:
            raise NameError(f'plane name {plane_name} not allowed. Only {PLANE_NAMES}')

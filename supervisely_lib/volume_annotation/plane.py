from supervisely_lib.collection.key_indexed_collection import KeyIndexedCollection, KeyObject
from supervisely_lib.volume_annotation.slice import Slice
from supervisely_lib.volume_annotation.constants import PLANE_NAMES, NAME, NORMAL, SLICES
from supervisely_lib._utils import take_with_default


class Plane(KeyIndexedCollection, KeyObject):
    item_type = Slice

    def __init__(self, name, normal, slices=None):
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

    def key(self):
        return self.name

    def to_json(self, key_id_map=None):
        return {NAME: self.name,
                NORMAL: self.normal,
                SLICES: [slice.to_json(key_id_map) for slice in self]}

    @classmethod
    def from_json(cls, data, objects, key_id_map=None):
        slices_json = data[SLICES]
        name = take_with_default(data[NAME], None)
        normal = data[NORMAL]
        slices = [cls.item_type.from_json(slice_json, objects, normal, key_id_map) for slice_json in slices_json]
        return cls(name, normal, slices)

    @property
    def figures(self):
        figures_array = []
        for slice in self:
            figures_array.extend(slice.figures)
        return figures_array

    @staticmethod
    def validate_plane_name(plane_name):
        if plane_name not in PLANE_NAMES:
            raise NameError(f'plane name {plane_name} not in allowed {PLANE_NAMES}')

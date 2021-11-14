# coding: utf-8

from supervisely_lib.collection.key_indexed_collection import KeyIndexedCollection
from supervisely_lib.volume_annotation.plane import Plane


class PlaneCollection(KeyIndexedCollection):
    item_type = Plane

    def to_json(self, key_id_map=None):
        return [plane.to_json(key_id_map) for plane in self]

    @classmethod
    def from_json(cls, data, objects, key_id_map=None):
        planes = [cls.item_type.from_json(plane_json, objects, key_id_map) for plane_json in data]
        return cls(planes)

    def __str__(self):
        return 'Planes:\n' + super(PlaneCollection, self).__str__()

    @property
    def figures(self):
        figures_array = []
        for plane in self:
            figures_array.extend(plane.figures)
        return figures_array

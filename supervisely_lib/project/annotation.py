# coding: utf-8

from copy import deepcopy

from ..figure.fig_factory import FigureFactory


# @TODO: methods for tags; m/b add ctor-s to create from other ann
class Annotation:
    # will own source internal_data, so copy it if needed
    def __init__(self, internal_data):
        self._data = {'tags': [], 'description': '', 'objects': [], **internal_data}
        for nm in ('width', 'height'):
            val = self._data['size'][nm]
            if val <= 0 or not isinstance(val, int):
                raise RuntimeError('Invalid annotation format.')
        # @TODO: add method for full validation, m/b with strict schema


    # @TODO: m/b disable? provide properties for objects, tags etc
    def __getitem__(self, item):
        return self._data[item]

    # @TODO: m/b disable?
    def __setitem__(self, key, value):
        self._data[key] = value

    @property
    def image_size_wh(self):
        sz = self._data['size']
        img_size_wh = (sz['width'], sz['height'])
        return img_size_wh

    def update_image_size(self, new_img_np):
        shp = new_img_np.shape
        self._data['size'] = {
            'width': shp[1],
            'height': shp[0],
        }

    # fn must return iterable of figures
    def apply_to_figures(self, fn):
        new_objs = []
        for obj in self._data['objects']:
            new_objs.extend(fn(obj))
        self._data['objects'] = new_objs

    def normalize_figures(self):
        self.apply_to_figures(lambda x: x.normalize(self.image_size_wh))

    def pack(self):
        packed_ann = deepcopy(self._data)
        packed_ann['objects'] = self.pack_objects(self._data['objects'])
        return packed_ann

    def add_object_back(self, obj):
        self._data['objects'].insert(0, obj)

    def add_object_front(self, obj):
        self._data['objects'].append(obj)

    @classmethod
    def unpack_objects(cls, in_objects, project_meta):
        unpacked_objects = (FigureFactory.create_from_packed(project_meta, packed_obj)
                            for packed_obj in in_objects)
        unpacked_objects = [x for x in unpacked_objects if x is not None]
        return unpacked_objects

    @classmethod
    def pack_objects(cls, in_objects):
        packed_objects = [obj.pack() for obj in in_objects]
        return packed_objects

    # packed_ann is native python dict which has been read from json or may be jsonized
    @classmethod
    def from_packed(cls, packed_ann, project_meta):
        data_dct = deepcopy(packed_ann)
        # unpack data: objects etc
        data_dct['objects'] = cls.unpack_objects(data_dct['objects'], project_meta)
        return cls(data_dct)

    @classmethod
    def new_with_objects(cls, imsize_wh, objects):
        data_dct = {
            'size': {
                'width': imsize_wh[0],
                'height': imsize_wh[1],
            },
            'objects': list(objects),
        }
        return cls(data_dct)

    @staticmethod
    def get_image_size_wh(ann_packed):
        packed_sz = ann_packed['size']
        return packed_sz['width'], packed_sz['height']

# coding: utf-8

from copy import deepcopy

import numpy as np
from skimage.morphology import skeletonize, medial_axis, thin
from supervisely_lib import FigureBitmap

from Layer import Layer


# processes FigureBitmap
class SkeletonizeLayer(Layer):

    action = 'skeletonize'

    layer_settings = {
        "required": ["settings"],
        "properties": {
            "settings": {
                "type": "object",
                "required": ["classes", "method"],
                "properties": {
                    "classes": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "method": {
                        "type": "string",
                        "enum": [
                            "skeletonization",
                            "medial_axis",
                            "thinning"
                        ]
                    }
                }
            }
        }
    }

    method_mapping = {
        'skeletonization': skeletonize,
        'medial_axis': medial_axis,
        'thinning': thin,
    }

    def __init__(self, config):
        Layer.__init__(self, config)

    def define_classes_mapping(self):
        super().define_classes_mapping()  # don't change

    def process(self, data_el):
        img_desc, ann_orig = data_el
        ann = deepcopy(ann_orig)
        imsize_wh = ann.image_size_wh
        method = self.method_mapping.get(self.settings['method'], None)
        if method is None:
            raise NotImplemented()

        def get_skel(f):
            if f.class_title not in self.settings['classes']:
                return [f]
            if not isinstance(f, FigureBitmap):
                raise RuntimeError('Input class must be a Bitmap in skeletonize layer.')

            origin, mask = f.get_origin_mask()
            mask_u8 = mask.astype(np.uint8)
            res_mask = method(mask_u8).astype(bool)
            res_f = FigureBitmap.from_mask(f.class_title, imsize_wh, origin, res_mask)
            return res_f  # iterable

        ann.apply_to_figures(get_skel)
        yield img_desc, ann

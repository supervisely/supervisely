# coding: utf-8

from copy import deepcopy

import numpy as np
from supervisely_lib import FigureBitmap

from Layer import Layer
from classes_utils import ClassConstants


# converts ALL types to FigureBitmap
class LineToBitmapLayer(Layer):

    action = 'line2bitmap'

    layer_settings = {
        "required": ["settings"],
        "properties": {
            "settings": {
                "type": "object",
                "required": ["classes_mapping", "width"],
                "properties": {
                    "classes_mapping": {
                        "type": "object",
                        "patternProperties": {
                            ".*": {"type": "string"}
                        }
                    },
                    "width": {
                        "description_en": u"Line width in pixels",
                        "description_ru": u"Ширина линии в пикселях",
                        "type": "integer",
                        "minimum": 1
                    }
                }
            }
        }
    }

    def __init__(self, config):
        Layer.__init__(self, config)

    def define_classes_mapping(self):
        for old_class, new_class in self.settings['classes_mapping'].items():
            self.cls_mapping[old_class] = {'title': new_class, 'shape': 'bitmap'}
        self.cls_mapping[ClassConstants.OTHER] = ClassConstants.DEFAULT

    def process(self, data_el):
        img_desc, ann_orig = data_el
        ann = deepcopy(ann_orig)
        imsize_wh = ann.image_size_wh
        shape_hw = imsize_wh[::-1]

        def to_fig_bitmap(f):
            new_title = self.settings['classes_mapping'].get(f.class_title)
            if new_title is None:
                return [f]
            bmp_to_draw = np.zeros(shape_hw, np.uint8)
            f.draw_contour(bmp_to_draw, color=1, thickness=self.settings['width'])
            src_mask = bmp_to_draw.astype(np.bool)
            res = FigureBitmap.from_mask(new_title, imsize_wh, (0, 0), src_mask)
            return res  # iterable

        ann.apply_to_figures(to_fig_bitmap)
        yield img_desc, ann

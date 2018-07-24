# coding: utf-8

from copy import deepcopy

from supervisely_lib import FigureBitmap

from Layer import Layer
from classes_utils import ClassConstants


# converts ALL types to FigureBitmap
class PolyToBitmapLayer(Layer):

    action = 'poly2bitmap'

    layer_settings = {
        "required": ["settings"],
        "properties": {
            "settings": {
                "type": "object",
                "required": ["classes_mapping"],
                "properties": {
                    "classes_mapping": {
                        "type": "object",
                        "patternProperties": {
                            ".*": {"type": "string"}
                        }
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
            src_mask = f.to_bool_mask(shape_hw)
            res = FigureBitmap.from_mask(new_title, imsize_wh, (0, 0), src_mask)
            return res  # iterable

        ann.apply_to_figures(to_fig_bitmap)
        yield img_desc, ann

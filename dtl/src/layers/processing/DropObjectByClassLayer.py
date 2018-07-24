# coding: utf-8

from copy import deepcopy

from Layer import Layer
from classes_utils import ClassConstants


class DropObjByClassLayer(Layer):

    action = 'drop_obj_by_class'

    layer_settings = {
        "required": ["settings"],
        "properties": {
            "settings": {
                "type": "object",
                "required": ["classes"],
                "properties": {
                    "classes": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                }
            }
        }
    }

    def __init__(self, config):
        Layer.__init__(self, config)

    def define_classes_mapping(self):
        for cls in self.settings['classes']:
            self.cls_mapping[cls] = ClassConstants.IGNORE
        self.cls_mapping[ClassConstants.OTHER] = ClassConstants.DEFAULT

    def obj_filter(self, fig):
        curr_class = fig.class_title
        if curr_class in self.settings['classes']:
            return []
        return [fig]

    def process(self, data_el):
        img_desc, ann_orig = data_el

        ann = deepcopy(ann_orig)
        ann.apply_to_figures(self.obj_filter)
        yield img_desc, ann

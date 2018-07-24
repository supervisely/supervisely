# coding: utf-8

from copy import deepcopy

from Layer import Layer
from classes_utils import ClassConstants


class RenameLayer(Layer):

    action = 'rename'

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
        self.cls_mapping = self.settings['classes_mapping']
        self.cls_mapping[ClassConstants.OTHER] = ClassConstants.DEFAULT

    def class_mapper(self, fig):
        curr_class = fig.class_title

        if curr_class in self.cls_mapping:
            new_class = self.cls_mapping[curr_class]
        else:
            raise RuntimeError('Can not find mapping for class: {}'.format(curr_class))

        if new_class == ClassConstants.IGNORE:
            return []  # drop the figure
        elif new_class == ClassConstants.DEFAULT:
            return [fig]  # don't change
        else:
            fig.class_title = new_class  # rename class
            return [fig]

    def process(self, data_el):
        img_desc, ann_orig = data_el
        ann = deepcopy(ann_orig)
        ann.apply_to_figures(self.class_mapper)
        yield img_desc, ann

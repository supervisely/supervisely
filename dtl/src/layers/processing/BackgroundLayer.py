# coding: utf-8

from copy import deepcopy

import supervisely_lib as sly

from Layer import Layer
from classes_utils import ClassConstants


class BackgroundLayer(Layer):

    action = 'background'

    layer_settings = {
        "required": ["settings"],
        "properties": {
            "settings": {
                "type": "object",
                "required": ["class"],
                "properties": {
                    "class": {
                        "type": "string"
                    }
                }
            }
        }
    }

    def __init__(self, config):
        Layer.__init__(self, config)
        self.src_check_mappings = [self.settings['class']]

    def define_classes_mapping(self):
        self.cls_mapping[ClassConstants.NEW] = [{'title': self.settings['class'], 'shape': 'rectangle'}]
        self.cls_mapping[ClassConstants.OTHER] = ClassConstants.DEFAULT

    def process(self, data_el):
        img_desc, ann_orig = data_el
        ann = deepcopy(ann_orig)

        img_wh = ann.image_size_wh
        w = img_wh[0]
        h = img_wh[1]

        rect_bg = sly.Rect(0, 0, w - 1, h - 1)  # @TODO: oh noo, we don't want to add -1 everywhere

        bg_rect_figure = sly.FigureRectangle.from_rect(self.settings['class'], img_wh, rect_bg)[0]
        ann.add_object_back(bg_rect_figure)

        yield img_desc, ann

# coding: utf-8

from copy import deepcopy

from supervisely_lib import SlidingWindows
from supervisely_lib.figure.aux import crop_image_with_rect

from Layer import Layer


class SlidingWindowLayer(Layer):

    action = 'sliding_window'

    layer_settings = {
        "required": ["settings"],
        "properties": {
            "settings": {
                "type": "object",
                "required": ["window", "min_overlap"],
                "properties": {
                    "window": {
                        "type": "object",
                        "required": ["height", "width"],
                        "properties": {
                            "height": {"type": "integer", "minimum": 1},
                            "width": {"type": "integer", "minimum": 1}
                        }
                    },
                    "min_overlap": {
                        "type": "object",
                        "required": ["x", "y"],
                        "properties": {
                            "x": {"type": "integer", "minimum": 0},
                            "y": {"type": "integer", "minimum": 0}
                        }
                    }
                }
            }
        }
    }

    def __init__(self, config):
        Layer.__init__(self, config)
        window_wh = (self.settings['window']['width'], self.settings['window']['height'])
        min_overlap_xy = (self.settings['min_overlap']['x'], self.settings['min_overlap']['y'])
        self.sliding_windows = SlidingWindows(window_wh, min_overlap_xy)  # + some validating

    def requires_image(self):
        return True

    def process(self, data_el):
        img_desc, ann_orig = data_el
        img_wh = ann_orig.image_size_wh
        img_orig = img_desc.read_image()

        for rect_to_crop in self.sliding_windows.get(img_wh):
            img = crop_image_with_rect(img_orig, rect_to_crop)
            new_img_desc = img_desc.clone_with_img(img)

            ann = deepcopy(ann_orig)
            ann.apply_to_figures(lambda x: x.crop(rect_to_crop))

            delta = (-rect_to_crop.left, -rect_to_crop.top)
            for fig in ann['objects']:
                fig.shift(delta)  # to new coords of image
            ann.update_image_size(img)

            yield new_img_desc, ann

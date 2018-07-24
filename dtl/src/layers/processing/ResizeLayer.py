# coding: utf-8

from copy import deepcopy

from Layer import Layer
from supervisely_lib import ImageResizer


class ResizeLayer(Layer):

    action = 'resize'

    layer_settings = {
        "required": ["settings"],
        "properties": {
            "settings": {
                "type": "object",
                "required": ["width", "height", "aspect_ratio"],
                "properties": {
                    "width": {
                        "type": "integer",
                        "minimum": -1
                    },
                    "height": {
                        "type": "integer",
                        "minimum": -1
                    },
                    "aspect_ratio": {
                        "type": "object",
                        "required": ["keep"],
                        "properties": {
                            "keep": {
                                "type": "boolean"
                            }
                        }
                    }
                }
            }
        }
    }

    def __init__(self, config):
        Layer.__init__(self, config)
        if self.settings['height'] * self.settings['width'] == 0:
            raise RuntimeError(self, '"height" and "width" should be != 0.')
        if self.settings['height'] + self.settings['width'] == -2:
            raise RuntimeError(self, '"height" and "width" cannot be both set to -1.')
        if self.settings['height'] * self.settings['width'] < 0:
            if not self.settings['aspect_ratio']['keep']:
                raise RuntimeError(self, '"keep" "aspect_ratio" should be set to "true" '
                                         'when "width" or "height" is -1.')

    def requires_image(self):
        return True

    def process(self, data_el):
        img_desc, ann_orig = data_el
        img_wh = ann_orig.image_size_wh

        keep = self.settings['aspect_ratio']['keep']
        set_size_wh = (self.settings['width'], self.settings['height'])

        resizer = ImageResizer(src_size_wh=img_wh, res_size_wh=set_size_wh, keep=keep)

        img = img_desc.read_image()
        img = resizer.resize_img(img, use_nearest=False)

        ann = deepcopy(ann_orig)

        for fig in ann['objects']:
            fig.resize(resizer)

        ann.update_image_size(img)
        new_img_desc = img_desc.clone_with_img(img)

        yield new_img_desc, ann

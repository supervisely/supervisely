# coding: utf-8

from copy import deepcopy

from Layer import Layer


class FlipLayer(Layer):

    action = 'flip'

    layer_settings = {
        "required": ["settings"],
        "properties": {
            "settings": {
                "type": "object",
                "required": ["axis"],
                "properties": {
                    "axis": {
                        "type": "string",
                        "enum": ["horizontal", "vertical"]
                    }
                }
            }
        }
    }

    def __init__(self, config):
        Layer.__init__(self, config)
        self.horiz = self.settings['axis'] == 'horizontal'

    def requires_image(self):
        return True

    def process(self, data_el):
        img_desc, ann_orig = data_el
        img_wh = ann_orig.image_size_wh
        img = img_desc.read_image()

        if self.horiz:
            img = img[::-1, :, :]
        else:
            img = img[:, ::-1, :]

        new_img_desc = img_desc.clone_with_img(img)
        ann = deepcopy(ann_orig)
        for fig in ann['objects']:
            fig.flip(self.horiz, img_wh)

        yield new_img_desc, ann

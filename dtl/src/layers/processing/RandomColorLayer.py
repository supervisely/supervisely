# coding: utf-8

import numpy as np

from Layer import Layer


class RandomColorLayer(Layer):

    action = 'random_color'

    layer_settings = {
        "required": ["settings"],
        "properties": {
            "settings": {
                "type": "object",
                "properties": {
                    "strength": {
                        "description_en": u"Strength of transform: 0 - no changes, 1 - strong transform",
                        "description_ru": u"Сила преобразования: 0 - без изменений, 1 - сильное преобразование",
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1
                    }
                }
            }
        }
    }

    def __init__(self, config):
        Layer.__init__(self, config)

    def requires_image(self):
        return True

    def process(self, data_el):
        img_desc, ann_orig = data_el

        strength = self.settings.get('strength', 0.25)
        strength /= 5.0

        img = img_desc.read_image()
        img = img.astype(np.float32)

        shape = img.shape
        res_img = img.reshape(-1, 3)
        rand = np.eye(3) + np.random.randn(3, 3) * strength
        res_img = np.dot(res_img, rand).reshape(shape)

        img = np.clip(res_img, 0, 255).astype(np.uint8)
        new_img_desc = img_desc.clone_with_img(img)
        yield new_img_desc, ann_orig

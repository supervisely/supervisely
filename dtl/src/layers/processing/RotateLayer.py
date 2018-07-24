# coding: utf-8

from copy import deepcopy

import numpy as np
from supervisely_lib import logger, ImageRotator, Rect
from supervisely_lib.figure.aux import expand_image_with_rect, crop_image_with_rect

from Layer import Layer


class RotateLayer(Layer):

    action = 'rotate'

    layer_settings = {
        "required": ["settings"],
        "properties": {
            "settings": {
                "type": "object",
                "required": ["rotate_angles", "black_regions"],
                "properties": {
                    "rotate_angles": {
                        "type": "object",
                        "required": ["min_degrees", "max_degrees"],
                        "properties": {
                            "min_degrees": {"type": "number"},
                            "max_degrees": {"type": "number"},
                        }
                    },
                    "black_regions": {
                        "type": "object",
                        "required": ["mode"],
                        "properties": {
                            "mode": {
                                "type": "string",
                                "enum": ["keep", "crop", "preserve_size"]
                            }
                        }
                    }
                }
            }
        }
    }

    def __init__(self, config):
        Layer.__init__(self, config)
        if self.settings['rotate_angles']['min_degrees'] > self.settings['rotate_angles']['max_degrees']:
            raise RuntimeError('"min_degrees" should be <= "max_degrees"')

    def requires_image(self):
        return True

    def process(self, data_el):
        img_desc, ann_orig = data_el
        img_wh = ann_orig.image_size_wh

        angle_dct = self.settings['rotate_angles']
        min_degrees, max_degrees = angle_dct['min_degrees'], angle_dct['max_degrees']
        rotate_degrees = np.random.uniform(min_degrees, max_degrees)

        rotator = ImageRotator(imsize_wh=img_wh, angle_degrees_ccw=rotate_degrees)

        img_orig = img_desc.read_image()
        img = rotator.rotate_img(img_orig, use_inter_nearest=False)

        ann = deepcopy(ann_orig)
        for fig in ann['objects']:
            fig.rotate(rotator)

        black_reg_mode = self.settings['black_regions']['mode']
        if black_reg_mode == 'keep':
            rect_to_crop = None

        elif black_reg_mode == 'crop':
            rect_to_crop = rotator.inner_crop

        elif black_reg_mode == 'preserve_size':
            rect_to_crop = rotator.source_rect
            img, delta = expand_image_with_rect(img, rect_to_crop)

            rect_to_crop = rect_to_crop.move(delta)
            for fig in ann['objects']:
                fig.shift(delta)

        else:
            raise NotImplementedError('Wrong black_regions mode.')

        if rect_to_crop is not None:
            if rect_to_crop.is_empty:
                logger.warning('Rotate layer produced empty crop.')
                return  # no yield

            if not Rect.from_arr(img).contains(rect_to_crop):
                raise RuntimeError('Unable to crop image in Rotate layer.')

            ann.apply_to_figures(lambda x: x.crop(rect_to_crop))

            img = crop_image_with_rect(img, rect_to_crop)
            delta = (-rect_to_crop.left, -rect_to_crop.top)
            for fig in ann['objects']:
                fig.shift(delta)  # to new coords of image

        ann.update_image_size(img)
        new_img_desc = img_desc.clone_with_img(img)

        yield new_img_desc, ann

# coding: utf-8

from copy import deepcopy

from supervisely_lib import logger
from supervisely_lib import Rect, rect_from_bounds, random_rect_from_bounds
from supervisely_lib.figure.aux import crop_image_with_rect

from Layer import Layer


class CropLayer(Layer):

    action = 'crop'

    layer_settings = {
        "required": ["settings"],
        "properties": {
            "settings": {
                "maxItems": 1,
                "oneOf": [
                    {
                        "type": "object",
                        "required": ["random_part"],
                        "properties": {
                            "random_part": {
                                "type": "object",
                                "required": ["height", "width"],
                                "properties": {
                                    "height": {
                                        "type": "object",
                                        "required": ["min_percent", "max_percent"],
                                        "properties": {
                                            "min_percent": {"$ref": "#/definitions/percent"},
                                            "max_percent": {"$ref": "#/definitions/percent"}
                                        }
                                    },
                                    "width": {
                                        "type": "object",
                                        "required": ["min_percent", "max_percent"],
                                        "properties": {
                                            "min_percent": {"$ref": "#/definitions/percent"},
                                            "max_percent": {"$ref": "#/definitions/percent"}
                                        }
                                    },
                                    "keep_aspect_ratio": {
                                        "type": "boolean",
                                        "default": False
                                    }
                                }
                            }
                        }
                    },
                    {
                        "type": "object",
                        "required": ["sides"],
                        "properties": {
                            "sides": {
                                "type": "object",
                                "uniqueItems": True,
                                "items": {
                                    "type": "string",
                                    "patternProperties": {
                                        "(left)|(top)|(bottom)|(right)": {
                                            "type": "string",
                                            "pattern": "^[0-9]+(%)|(px)$"
                                        }
                                    }
                                }
                            }
                        }
                    }
                ]
            }
        }
    }

    def __init__(self, config):
        Layer.__init__(self, config)

        if 'random_part' in self.settings:
            random_part = self.settings['random_part']
            keep_aspect_ratio = random_part.get('keep_aspect_ratio', False)
            if keep_aspect_ratio:
                if random_part['height'] != random_part['width']:
                    raise RuntimeError("When 'keep_aspect_ratio' is 'true', 'height' and 'width' should be equal.")

            def check_min_max(dictionary, text):
                if dictionary['min_percent'] > dictionary['max_percent']:
                    raise RuntimeError("'min_percent' should be <= than 'max_percent' for {}".format(text))

            check_min_max(random_part['height'], 'height')
            check_min_max(random_part['width'], 'width')
        # @TODO: check 'sides' params for percents... or not

    def requires_image(self):
        return True

    def process(self, data_el):
        img_desc, ann_orig = data_el
        img_wh = ann_orig.image_size_wh

        if 'random_part' in self.settings:
            rect_to_crop = random_rect_from_bounds(self.settings['random_part'], *img_wh)
        elif 'sides' in self.settings:
            rect_to_crop = rect_from_bounds(self.settings['sides'], *img_wh)
        else:
            raise NotImplemented('Crop layer: wrong params.')
        # rect_to_crop has 'true' coordinates, e.g. single pixel crop will be defined as (0, 0, 1, 1)

        rect_img = Rect.from_size(img_wh)
        if rect_to_crop.is_empty:
            # tiny image (left >= right etc)
            logger.warning('Crop layer produced empty crop.')
            return  # no yield

        if not rect_img.contains(rect_to_crop):
            # some negative params
            raise RuntimeError('Crop layer: result crop bounds are outside of source image.')

        img = img_desc.read_image()
        img = crop_image_with_rect(img, rect_to_crop)
        new_img_desc = img_desc.clone_with_img(img)

        ann = deepcopy(ann_orig)
        ann.apply_to_figures(lambda x: x.crop(rect_to_crop))

        delta = (-rect_to_crop.left, -rect_to_crop.top)
        for fig in ann['objects']:
            fig.shift(delta)  # to new coords of image
        ann.update_image_size(img)

        yield new_img_desc, ann

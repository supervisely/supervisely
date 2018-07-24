# coding: utf-8

from copy import deepcopy

from skimage.morphology import remove_small_objects
from supervisely_lib import FigureBitmap

from Layer import Layer


class DropNoiseFromBitmap(Layer):

    action = 'drop_noise'

    layer_settings = {
            "required": ["settings"],
            "properties": {
                "settings": {
                    "type": "object",
                    "required": ["classes", "min_area", "src_type"],
                    "properties": {
                        "classes": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "min_area": {
                            "type": "string",
                            "pattern": "^[0-9]+(\.[0-9][0-9]?)?(%)|(px)$"
                        },
                        "src_type": {
                            "type": "string",
                            "enum": ["image", "bbox"]
                        }

                    }
                }
            }
        }

    def __init__(self, config):
        Layer.__init__(self, config)

    def process(self, data_el):
        img_desc, ann_orig = data_el
        ann = deepcopy(ann_orig)
        imsize_wh = ann.image_size_wh

        img_area = float(imsize_wh[0] * imsize_wh[1])
        area_str = self.settings['min_area']

        def drop_noise(fig):
            if fig.class_title not in self.settings['classes']:
                return [fig]

            if not isinstance(fig, FigureBitmap):
                raise RuntimeError('Input class must be a Bitmap in drop_noise layer.')

            if area_str.endswith('%'):
                if self.settings['src_type'] == 'image':
                    refer_area = img_area
                else:
                    refer_area = fig.get_bbox().area

                area_part = float(area_str[:-len('%')]) / 100.0
                req_area = int(refer_area * area_part)
            else:
                req_area = int(area_str[:-len('px')])

            old_origin, old_mask = fig.get_origin_mask()
            res_mask = remove_small_objects(old_mask, req_area)
            res = FigureBitmap.from_mask(fig.class_title, imsize_wh, old_origin, res_mask)
            return res  # iterable

        ann.apply_to_figures(drop_noise)
        yield img_desc, ann

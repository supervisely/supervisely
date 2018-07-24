# coding: utf-8

from copy import deepcopy

from cv2 import connectedComponents
from supervisely_lib import FigureBitmap

from Layer import Layer


class SplitMasksLayer(Layer):

    action = 'split_masks'

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

    def process(self, data_el):
        img_desc, ann_orig = data_el
        ann = deepcopy(ann_orig)
        imsize_wh = ann.image_size_wh

        def split_mask(fig):
            if fig.class_title not in self.settings['classes']:
                return [fig]

            if not isinstance(fig, FigureBitmap):
                raise RuntimeError('Input class must be a Bitmap in split_masks layer.')

            old_origin, old_mask = fig.get_origin_mask()
            ret, label = connectedComponents(old_mask.astype('uint8'), connectivity=8)

            res_figures = []
            for i in range(1, ret):
                obj_mask = label == i
                res_figures.extend(FigureBitmap.from_mask(fig.class_title, imsize_wh, old_origin, obj_mask))
            return res_figures

        ann.apply_to_figures(split_mask)
        yield img_desc, ann

# coding: utf-8

from copy import deepcopy

import cv2
import numpy as np
from legacy_supervisely_lib.figure.figure_bitmap import FigureBitmap
from legacy_supervisely_lib.figure.figure_polygon import FigurePolygon

from Layer import Layer
from classes_utils import ClassConstants


# FigureBitmap to FigurePolygon
class FindContoursLayer(Layer):

    action = 'find_contours'

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
                    },
                    "approx_epsilon": {
                        "type": "number",
                        "minimum": 0
                    }
                }
            }
        }
    }

    def __init__(self, config):
        Layer.__init__(self, config)

    def define_classes_mapping(self):
        for old_class, new_class in self.settings['classes_mapping'].items():
            self.cls_mapping[old_class] = {'title': new_class, 'shape': 'polygon'}
        self.cls_mapping[ClassConstants.OTHER] = ClassConstants.DEFAULT

    def process(self, data_el):
        img_desc, ann_orig = data_el
        ann = deepcopy(ann_orig)
        img_wh = ann_orig.image_size_wh
        approx_epsilon = self.settings.get('approx_epsilon')


        def to_contours(f):
            new_title = self.settings['classes_mapping'].get(f.class_title)
            if new_title is None:
                return [f]
            if not isinstance(f, FigureBitmap):
                raise RuntimeError('Input class must be a Bitmap in find_contours layer.')

            origin, mask = f.get_origin_mask()
            contours, hier = cv2.findContours(
                mask.astype(np.uint8),
                mode=cv2.RETR_CCOMP,  # two-level hierarchy, to get polygons with holes
                method=cv2.CHAIN_APPROX_SIMPLE
            )
            if (hier is None) or (contours is None):
                return []

            res = []
            for idx, hier_pos in enumerate(hier[0]):
                next_idx, prev_idx, child_idx, parent_idx = hier_pos
                if parent_idx < 0:
                    external = contours[idx][:, 0]
                    internals = []
                    while child_idx >= 0:
                        internals.append(contours[child_idx][:, 0])
                        child_idx = hier[0][child_idx][0]
                    res.extend(FigurePolygon.from_np_points(new_title, img_wh, external, internals))

            offset = (origin[0] + .5, origin[1] + .5)
            for x in res:
                x.shift(offset)

            if approx_epsilon is not None:
                for obj in res:
                    obj.approx_dp(approx_epsilon)

            return res  # iterable

        ann.apply_to_figures(to_contours)
        yield img_desc, ann

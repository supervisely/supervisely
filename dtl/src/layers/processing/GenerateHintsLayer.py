# coding: utf-8

from copy import deepcopy

import numpy as np
from supervisely_lib import FigurePoint

from Layer import Layer
from classes_utils import ClassConstants


class GenerateHintsLayer(Layer):

    action = 'generate_hints'

    layer_settings = {
        "required": ["settings"],
        "properties": {
            "settings": {
                "type": "object",
                "required": ["class", "positive_class", "negative_class", "min_points_number"],
                "properties": {
                    "class": {
                        "type": "string"
                    },
                    "positive_class": {
                        "type": "string"
                    },
                    "negative_class": {
                        "type": "string"
                    },
                    "min_points_number": {
                        "type": "integer"
                    }
                }
            }
        }
    }

    def __init__(self, config):
        Layer.__init__(self, config)
        if self.settings['min_points_number'] < 0:
            raise ValueError("GenerateHintsLayer: min_points_number must not be less than zero")

    def define_classes_mapping(self):
        self.cls_mapping[ClassConstants.NEW] = [{'title': self.settings['positive_class'], 'shape': 'point'},
                                                {'title': self.settings['negative_class'], 'shape': 'point'}]
        self.cls_mapping[ClassConstants.OTHER] = ClassConstants.DEFAULT

    def generate_points(self, mask, color=1):
        h, w = mask.shape[:2]
        pos_area = mask.sum()

        def pt_num(cnt=2):
            cs = [int(np.random.exponential(2)) + self.settings['min_points_number'] for _ in range(cnt)]
            return cs

        n_pos, n_neg = pt_num()
        n_pos = min(n_pos, pos_area)
        n_neg = min(n_neg, h*w - pos_area)
        positive_points, negative_points = [], []
        if n_pos > 0:
            # @TODO: speed up (np.argwhere, mm); what if pos or neg is missing?
            points = np.argwhere(mask == color)[:, [1, 0]]  # to xy
            positive_points = points[np.random.choice(points.shape[0], n_pos, replace=False), :]
        if n_neg > 0:
            points = np.argwhere(mask != color)[:, [1, 0]]  # to xy
            negative_points = points[np.random.choice(points.shape[0], n_neg, replace=False), :]
        return positive_points, negative_points

    def process(self, data_el):
        img_desc, ann_orig = data_el
        ann = deepcopy(ann_orig)
        imsize_wh = ann.image_size_wh
        shape_hw = imsize_wh[::-1]

        mask = np.zeros(shape_hw, dtype=np.uint8)
        for fig in ann['objects']:
            if fig.class_title == self.settings['class']:
                fig.draw(mask, 1)

        def add_pt_figures(pts, cls_title):
            for point in pts:
                new_fig = FigurePoint.from_pt(cls_title, tuple(point))
                ann['objects'].append(new_fig)

        positive_points, negative_points = self.generate_points(mask)
        add_pt_figures(positive_points, self.settings['positive_class'])
        add_pt_figures(negative_points, self.settings['negative_class'])

        yield img_desc, ann

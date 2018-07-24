# coding: utf-8

from copy import deepcopy

from supervisely_lib import FigureRectangle, FigurePolygon

from Layer import Layer
from classes_utils import ClassConstants


class BBoxToPolyLayer(Layer):
    action = "bbox2poly"

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
        imsize_wh = ann.image_size_wh

        def to_fig_rect(f):
            new_title = self.settings['classes_mapping'].get(f.class_title)
            if new_title is None:
                return [f]
            if not isinstance(f, FigureRectangle):
                raise RuntimeError('Input class must be a Rectangle in bbox2poly layer.')
            ring_pts = f.get_bbox().to_np_points()
            res = FigurePolygon.from_np_points(new_title, imsize_wh, exterior=ring_pts, interiors=[])
            return res  # iterable

        ann.apply_to_figures(to_fig_rect)
        yield img_desc, ann

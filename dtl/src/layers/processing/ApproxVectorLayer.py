# coding: utf-8

from copy import deepcopy

from supervisely_lib import FigurePolygon, FigureLine

from Layer import Layer


# processes FigurePolygon or FigureLine
class ApproxVectorLayer(Layer):

    action = 'approx_vector'

    layer_settings = {
        "required": ["settings"],
        "properties": {
            "settings": {
                "type": "object",
                "required": ["classes", "epsilon"],
                "properties": {
                    "classes": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "epsilon": {
                        "type": "number",
                        "minimum": 0,
                        "exclusiveMinimum": True
                    }
                }
            }
        }
    }

    def __init__(self, config):
        Layer.__init__(self, config)

    def define_classes_mapping(self):
        super().define_classes_mapping()  # don't change

    def process(self, data_el):
        img_desc, ann_orig = data_el
        ann = deepcopy(ann_orig)
        epsilon = self.settings['epsilon']

        def approx_contours(f):
            if f.class_title not in self.settings['classes']:
                return [f]
            if (not isinstance(f, FigurePolygon)) and (not isinstance(f, FigureLine)):
                raise RuntimeError('Input class must be a Polygon or a Line in approx_vector layer.')
            f.approx_dp(epsilon)
            return [f]

        ann.apply_to_figures(approx_contours)
        yield img_desc, ann

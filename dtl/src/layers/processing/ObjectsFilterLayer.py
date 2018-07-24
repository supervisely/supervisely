# coding: utf-8

from copy import deepcopy

from Layer import Layer


class ObjectsFilterLayer(Layer):

    action = 'objects_filter'

    layer_settings = {
        "required": ["settings"],
        "properties": {
            "settings": {
                "type": "object",
                "required": ["filter_by"],
                "properties": {
                    "filter_by": {
                        "maxItems": 1,
                        "oneOf": [
                            {
                                "type": "object",
                                "required": ["polygon_sizes"],
                                "properties": {
                                    "polygon_sizes": {
                                        "type": "object",
                                        "required": ["filtering_classes", "area_size", "action", "comparator"],
                                        "properties": {
                                            "filtering_classes": {
                                                "type": "array",
                                                "items": {"type": "string"}
                                            },
                                            "area_size": {
                                                "oneOf": [
                                                    {
                                                        "type": "object",
                                                        "required": ["percent"],
                                                        "properties": {
                                                            "percent": {
                                                                "$ref": "#/definitions/percent"
                                                            }
                                                        }

                                                    },
                                                    {
                                                        "type": "object",
                                                        "required": ["height", "width"],
                                                        "properties":{
                                                            "width": {
                                                                "type": "integer"
                                                            },
                                                            "height": {
                                                                "type": "integer"
                                                            },
                                                        }
                                                    }

                                                ]
                                            },
                                            "action": {
                                                "oneOf": [
                                                    {
                                                        "type": "object",
                                                        "required": ["remap_class"],
                                                        "properties": {
                                                            "remap_class": {"type": "string"},
                                                        }
                                                    },
                                                    {
                                                        "type": "string",
                                                        "enum": ["delete"]
                                                    }
                                                ]
                                            },
                                            "comparator": {
                                                "type": "string",
                                                "enum": ["less", "greater"]
                                            }
                                        }
                                    }
                                }
                            }
                        ]
                    }
                }
            }
        }
    }

    def __init__(self, config):
        Layer.__init__(self, config)
        self.params = self.settings['filter_by']['polygon_sizes']
        if self.params['action'] != 'delete':
            raise NotImplementedError('Class remapping is NIY here.')

    def process(self, data_el):
        img_desc, ann_orig = data_el
        ann = deepcopy(ann_orig)
        imsize_wh = ann.image_size_wh
        img_area = float(imsize_wh[0] * imsize_wh[1])
        area_set = self.params['area_size']

        # @TODO: use operator.lt / operator.gt

        def filter_delete_percent(fig):
            if self.params['comparator'] == 'less':
                compar = lambda x: x < area_set['percent']
            else:
                compar = lambda x: x > area_set['percent']

            if fig.class_title in self.params['filtering_classes']:
                fig_area = fig.get_area()
                area_percent = 100.0 * fig_area / img_area
                if compar(area_percent):  # satisfied condition
                    return []  # action 'delete'
            return [fig]

        def filter_delete_size(fig):
            if self.params['comparator'] == 'less':
                compar = lambda x: x.width < area_set['width'] or x.height < area_set['height']
            else:
                compar = lambda x: x.width > area_set['width'] or x.height > area_set['height']

            if fig.class_title in self.params['filtering_classes']:
                fig_rect = fig.get_bbox()
                if compar(fig_rect):  # satisfied condition
                    return []  # action 'delete'
            return [fig]

        if 'percent' in area_set:
            ann.apply_to_figures(filter_delete_percent)
        else:
            ann.apply_to_figures(filter_delete_size)

        yield img_desc, ann

# coding: utf-8

from copy import deepcopy

from Layer import Layer


class DatasetLayer(Layer):

    action = 'dataset'

    layer_settings = {
        "required": ["settings"],
        "properties": {
            "settings": {
                "oneOf": [
                    {
                        "type": "object",
                        "required": ["name"],
                        "properties": {
                            "name": {
                                "type": "string",
                                "pattern": "^[0-9a-zA-Z \\-_]+$"  # @TODO: backslash?
                            }
                        }
                    },
                    {
                        "type": "object",
                        "required": ["rule"],
                        "properties": {
                            "rule": {
                                "type": "string",
                                "enum": ["save_original"]
                            }
                        }
                    }
                ]
            }
        }
    }

    def __init__(self, config):
        Layer.__init__(self, config)
        if len(self.settings.get('name', '')) > 2048:
            raise RuntimeError('Dataset name is too long, huh?')

    def process(self, data_el):
        img_desc, ann_orig = data_el
        new_img_desc = deepcopy(img_desc)

        if 'name' in self.settings:
            new_img_desc.res_ds_name = self.settings['name']
        elif self.settings.get('rule', '') == 'save_original':
            new_img_desc.res_ds_name = new_img_desc.get_ds_name()
        else:
            raise NotImplementedError()

        yield new_img_desc, ann_orig

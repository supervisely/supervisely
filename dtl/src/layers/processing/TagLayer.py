# coding: utf-8

from copy import deepcopy

from Layer import Layer


class TagLayer(Layer):

    action = 'tag'

    layer_settings = {
        "required": ["settings"],
        "properties": {
            "settings": {
                "type": "object",
                "required": ["tag", "action"],
                "properties": {
                    "tag": {"type": "string"},
                    "action": {
                        "type": "string",
                        "enum": ["add", "delete"]
                    }
                }
            }
        }
    }

    def __init__(self, config):
        Layer.__init__(self, config)

    def get_added_tags(self):
        if self.settings['action'] != 'add':
            return None
        return {self.settings['tag']}

    def get_removed_tags(self):
        if self.settings['action'] != 'delete':
            return None
        return {self.settings['tag']}

    def process(self, data_el):
        img_desc, ann_orig = data_el
        ann = deepcopy(ann_orig)
        if self.settings['action'] == 'add':
            ann['tags'].append(self.settings['tag'])
        elif self.settings['tag'] in ann['tags']:
            ann['tags'].remove(self.settings['tag'])
        else:
            pass

        yield img_desc, ann

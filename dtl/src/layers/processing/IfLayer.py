# coding: utf-8

import random
import re

from Layer import Layer


class IfLayer(Layer):

    action = 'if'

    layer_settings = {
        "required": ["settings"],
        "properties": {
            "dst": {
                "type": "array",
                "minItems": 2,
                "maxItems": 2,
                "items": {"type": "string"}
            },
            "settings": {
                "type": "object",
                "required": ["condition"],
                "properties": {
                    "condition": {
                        "maxItems": 1,
                        "oneOf": [
                            {
                                "type": "object",
                                "required": ["probability"],
                                "properties": {
                                    "probability": {
                                        "type": "number",
                                        "minimum": 0,
                                        "maximum": 1
                                    }
                                }
                            },
                            {
                                "type": "object",
                                "required": ["regex_names"],
                                "properties": {
                                    "regex_names": {
                                        "type": "array",
                                        "items": {"type": "string"}
                                    }
                                }
                            },
                            {
                                "type": "object",
                                "required": ["project_datasets"],
                                "properties": {
                                    "project_datasets": {
                                        "type": "array",
                                        "items": {"type": "string"}
                                    }
                                }
                            },
                            {
                                "type": "object",
                                "required": ["min_objects_count"],
                                "properties": {
                                    "min_objects_count": {
                                        "type": "integer",
                                        "minimum": 0
                                    }
                                    # comparator is greater or equal
                                }
                            },
                            {
                                "type": "object",
                                "required": ["include_classes"],
                                "properties": {
                                    "include_classes": {
                                        "type": "array",
                                        "items": {
                                            "type": "string"
                                        }
                                    }
                                }
                            },
                            {
                                "type": "object",
                                "required": ["tags"],
                                "properties": {
                                    "tags": {
                                        "type": "array",
                                        "items": {"type": "string"}
                                    }
                                }
                            },
                            {
                                "type": "object",
                                "required": ["min_height"],
                                "properties": {
                                    "min_height": {
                                        "type": "integer"
                                    }
                                }
                            },
                            {
                                "type": "object",
                                "required": ["min_width"],
                                "properties": {
                                    "min_width": {
                                        "type": "integer"
                                    }
                                }
                            },
                            {
                                "type": "object",
                                "required": ["min_area"],
                                "properties": {
                                    "min_area": {
                                        "type": "integer"
                                    }
                                }
                            },
                            {
                                "type": "object",
                                "required": ["sum_object_area", "classes"],
                                "properties": {
                                    "sum_object_area": {
                                        "type": "number"
                                    },
                                    "classes": {
                                        "type": "array",
                                        "items": {
                                            "type": "string"
                                        }
                                    }
                                }
                            }  # comparator is greater or equal
                        ]
                    }
                }
            }
        }
    }

    def __init__(self, config):
        Layer.__init__(self, config)

        condition = list(self.settings['condition'].keys())[0]
        if condition == 'project_datasets':
            project_datasets = self.settings['condition'][condition]
            for project_dataset_str in project_datasets:
                self._check_project_dataset_str(project_dataset_str)

    # for compatibility; format "project/dataset"
    @classmethod
    def _check_project_dataset_str(cls, s):
        src_regex = r'^[^/]+/[^/]+$'
        if not re.search(src_regex, s):
            raise RuntimeError('Project/dataset string "{}" does not match regex "{}".'.format(s, src_regex))

    @classmethod
    def _matches_ast(cls, pattern, s):
        res = (pattern == '*') or (pattern == s)
        return res

    @classmethod
    def _pr_ds_matches(cls, project_dataset_str, img_desc):
        splitted = project_dataset_str.split('/')
        if len(splitted) != 2:
            raise RuntimeError('Wrong project_dataset string "{}"'.format(project_dataset_str))
        pr_name, ds_name = splitted
        res = cls._matches_ast(pr_name, img_desc.get_pr_name()) and cls._matches_ast(ds_name, img_desc.get_ds_name())
        return res

    def process(self, data_el):
        img_desc, ann_orig = data_el
        w, h = ann_orig.image_size_wh

        condition = self.settings['condition']
        satisfies_cond = False

        if 'probability' in condition:
            prob = condition['probability']
            satisfies_cond = random.random() < prob

        elif 'regex_names' in condition:
            regex_names = condition['regex_names']
            satisfies_cond = any(re.search(regex_name, img_desc.get_img_name())
                                 for regex_name in regex_names)
            # @TODO: is it safe?

        elif 'project_datasets' in condition:
            project_datasets = condition['project_datasets']
            satisfies_cond = any(self._pr_ds_matches(project_dataset_str, img_desc)
                                 for project_dataset_str in project_datasets)

        elif 'min_objects_count' in condition:
            thresh = condition['min_objects_count']
            value = len(ann_orig['objects'])
            satisfies_cond = value >= thresh

        elif 'include_classes' in condition:
            req_classes = condition['include_classes']
            satisfies_cond = any(fig.class_title in req_classes
                                 for fig in ann_orig['objects'])

        elif 'tags' in condition:
            req_tags = condition['tags']
            satisfies_cond = len(set(ann_orig['tags']) & set(req_tags)) > 0

        elif 'min_height' in condition:
            satisfies_cond = h >= condition['min_height']

        elif 'min_width' in condition:
            satisfies_cond = w >= condition['min_width']

        elif 'min_area' in condition:
            satisfies_cond = (h * w) >= condition['min_area']

        elif 'sum_object_area' in condition:
            thresh = condition['sum_object_area']
            req_classes = condition['classes']
            sum_area = sum(fig.get_area() for fig in ann_orig['objects']
                           if fig.class_title in req_classes)
            satisfies_cond = sum_area >= thresh

        if satisfies_cond:
            yield data_el + tuple([0])  # branch 0
        else:
            yield data_el + tuple([1])  # branch 1

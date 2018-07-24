# coding: utf-8

from copy import deepcopy

from supervisely_lib.project import Annotation

from Layer import Layer
from classes_utils import ClassConstants


class DataLayer(Layer):

    action = 'data'

    layer_settings = {
        "required": ["settings"],
        "properties": {
            "settings": {
                "type": "object",
                "required": ["classes_mapping"],
                "properties": {
                    "classes_mapping": {
                        "oneOf": [
                            {
                                "type": "object",
                                "patternProperties": {
                                    ".*": {"type": "string"}
                                }
                            },
                            {
                                "type": "string",
                                "enum": ["default"]
                            }
                        ]
                    }
                }
            }
        }
    }

    def __init__(self, config, input_project_metas):
        Layer.__init__(self, config)
        self._define_layer_project()

        in_project_meta = input_project_metas.get(self.project_name, None)
        if in_project_meta is None:
            raise ValueError('Data Layer can not init corresponding project meta. '
                             'Project name ({}) not found'.format(self.project_name))
        self.in_project_meta = deepcopy(in_project_meta)

    @classmethod
    def _split_data_src(cls, src):
        return src.split('/')

    def _define_layer_project(self):
        all_projects = {}
        for src in self.srcs:
            cur_project_name, dataset_name = self._split_data_src(src)
            all_projects[cur_project_name] = {'dataset': dataset_name}
        if len(all_projects) != 1:
            raise ValueError('data layer can work only with one project')
        self.project_name = list(all_projects.keys())[0]
        self.dataset_name = all_projects[self.project_name]['dataset']

    def define_classes_mapping(self):
        if self.settings['classes_mapping'] != "default":
            self.cls_mapping = self.settings['classes_mapping']
        else:
            Layer.define_classes_mapping(self)

    def class_mapper(self, fig):
        curr_class = fig.class_title

        if curr_class in self.cls_mapping:
            new_class = self.cls_mapping[curr_class]
        else:
            raise RuntimeError('Can not find mapping for class: {}'.format(curr_class))

        if new_class == ClassConstants.IGNORE:
            return []  # drop the figure
        elif new_class != ClassConstants.DEFAULT:
            fig.class_title = new_class  # rename class
        else:
            pass  # don't change
        return [fig]

    def process(self, data_el):
        img_desc, packed_ann = data_el
        ann = Annotation.from_packed(packed_ann, self.in_project_meta)
        ann.normalize_figures()
        ann.apply_to_figures(self.class_mapper)
        yield (img_desc, ann)

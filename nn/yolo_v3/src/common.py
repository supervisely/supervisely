# coding: utf-8

import os.path as osp

import supervisely_lib as sly
from supervisely_lib.utils.jsonschema import MultiTypeValidator
from supervisely_lib import FigureRectangle, Rect, FigClasses


class SettingsValidator:
    validator = MultiTypeValidator('/workdir/src/schemas.json')

    @classmethod
    def validate_train_cfg(cls, config):
        # store all possible requirements in schema, including size % 16 etc
        cls.validator.val('training_config', config)

        sp_classes = config['special_classes']
        if len(set(sp_classes.values())) != len(sp_classes):
            raise RuntimeError('Non-unique special classes in train config.')

    @classmethod
    def validate_inference_cfg(cls, config):
        # store all possible requirements in schema
        cls.validator.val('inference_config', config)


class TrainConfigRW:
    def __init__(self, model_dir):
        self.model_dir = model_dir

    @property
    def train_config_fpath(self):
        res = osp.join(self.model_dir, 'config.json')
        return res

    @property
    def train_config_exists(self):
        res = osp.isfile(self.train_config_fpath)
        return res

    def save(self, config):
        sly.json_dump(config, self.train_config_fpath)

    def load(self):
        res = sly.json_load(self.train_config_fpath)
        return res


def construct_detection_classes(names_list):
    name_shape_list = [{'title': name, 'shape': 'rectangle'} for name in names_list]
    return name_shape_list


def yolo_preds_to_sly_rects(detections, img_wh, cls_names):
    out_figures = []
    for classId, _, box in detections:
            xmin = box[0] - box[2] / 2
            ymin = box[1] - box[3] / 2
            xmax = box[0] + box[2] / 2
            ymax = box[1] + box[3] / 2
            rect = Rect(xmin, ymin, xmax, ymax)
            new_objs = FigureRectangle.from_rect(cls_names[classId], img_wh, rect)
            out_figures.extend(new_objs)
    return out_figures


def create_detection_classes(in_project_classes):
    in_project_titles = sorted((x['title'] for x in in_project_classes))

    class_title_to_idx = {}
    for i, title in enumerate(in_project_titles):
        class_title_to_idx[title] = i  # usually bkg_color is 0

    if len(set(class_title_to_idx.values())) != len(class_title_to_idx):
        raise RuntimeError('Unable to construct internal color mapping for classes.')

    # determine out classes
    out_classes = FigClasses()

    for in_class in in_project_classes:
        title = in_class['title']
        out_classes.add({
            'title': title,
            'shape': 'rectangle',
            'color': in_class['color'],
        })

    return class_title_to_idx, out_classes

# coding: utf-8

import os.path as osp
import tensorflow as tf

import supervisely_lib as sly
from supervisely_lib import logger
from supervisely_lib.utils.jsonschema import MultiTypeValidator


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


def create_detection_graph(model_dirpath):
    fpath = osp.join(model_dirpath, 'model.pb')
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(fpath, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    logger.info('Restored model weights from training.')
    return detection_graph


def inverse_mapping(mapping):
    new_map = {}
    for k, v in mapping.items():
        new_map[v] = k
    return new_map


def get_scope_vars(detection_graph):
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    return detection_boxes, detection_scores, detection_classes, num_detections, image_tensor

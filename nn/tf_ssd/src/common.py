# coding: utf-8

import math
import os.path as osp

import tensorflow as tf
from google.protobuf import text_format
import supervisely_lib as sly
from supervisely_lib import logger
from supervisely_lib.utils.jsonschema import MultiTypeValidator

from object_detection import exporter
from object_detection.protos import pipeline_pb2


class SettingsValidator:
    validator = MultiTypeValidator('/workdir/src/schemas.json')

    @classmethod
    def validate_train_cfg(cls, config):
        # store all possible requirements in schema, including size % 16 etc
        cls.validator.val('training_config', config)

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


def freeze_graph(input_type,
                 pipeline_config_path,
                 trained_checkpoint_prefix,
                 output_directory,
                 input_shape=None):
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    with tf.gfile.GFile(pipeline_config_path, 'r') as f:
        text_format.Merge(f.read(), pipeline_config)
    if input_shape:
        input_shape = [
            int(dim) if dim != '-1' else None
            for dim in input_shape.split(',')
        ]
    else:
        input_shape = None
    exporter.export_inference_graph(input_type, pipeline_config,
                                    trained_checkpoint_prefix,
                                    output_directory, input_shape)


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


class EvalPlanner:
    def __init__(self, epochs, val_every):
        self.epochs = epochs
        self.val_every = val_every
        self.total_val_cnt = self.validations_cnt(epochs, val_every)
        self._val_cnt = 0

    @property
    def performed_val_cnt(self):
        return self._val_cnt

    @staticmethod
    def validations_cnt(ep_float, val_every):
        res = math.floor(ep_float / val_every + 1e-9)
        return res

    def validation_performed(self):
        self._val_cnt += 1

    def need_validation(self, epoch_flt):
        req_val_cnt = self.validations_cnt(epoch_flt, self.val_every)
        need_val = req_val_cnt > self._val_cnt
        return need_val

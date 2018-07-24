# coding: utf-8

import math
import os.path as osp
import tensorflow as tf
import numpy as np

import supervisely_lib as sly
from supervisely_lib import FigureBitmap, FigClasses
from supervisely_lib import logger
from supervisely_lib.utils.jsonschema import MultiTypeValidator
# from object_detection.utils import ops as utils_ops


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

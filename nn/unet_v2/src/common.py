# coding: utf-8

import os.path as osp

import torch
from torch.nn import DataParallel
import supervisely_lib as sly
from supervisely_lib import logger
from supervisely_lib.utils.pytorch import upgraded_load_state_dict
from supervisely_lib.utils.jsonschema import MultiTypeValidator

from unet import construct_unet


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


class WeightsRW:
    def __init__(self, model_dir):
        self.model_dir = model_dir

    @property
    def weights_fpath(self):
        res = osp.join(self.model_dir, 'model.pt')
        return res

    def save(self, model):
        torch.save(model.state_dict(), self.weights_fpath)

    def load_for_transfer_learning(self, model):
        loaded_model = torch.load(self.weights_fpath)

        # rm last conv (with output to classes); predefined name for the unet
        to_delete = [el for el in loaded_model.keys() if 'last_conv' in el]
        for el in to_delete:
            del loaded_model[el]
        if len(to_delete) > 0:
            logger.info('Skip weight init for output layers.', extra={'layer_names': to_delete})

        upgraded_load_state_dict(model, loaded_model, strict=False)
        return model

    def load_strictly(self, model):
        loaded_model = torch.load(self.weights_fpath)
        upgraded_load_state_dict(model, loaded_model, strict=True)
        return model


def create_model(n_cls, device_ids):
    logger.info('Will construct model.')
    model = construct_unet(n_cls=n_cls)
    logger.info('Model has been constructed (w/out weights).')
    model = DataParallel(model, device_ids=device_ids).cuda()
    logger.info('Model has been loaded into GPU(s).', extra={'remapped_device_ids': device_ids})
    return model

# coding: utf-8

import os
import os.path as osp
from copy import deepcopy

import cv2
import numpy as np
import supervisely_lib as sly
from supervisely_lib import logger

from tf_config_converter import load_sample_config, remake_mask_rcnn_config, save_config
from common import SettingsValidator, TrainConfigRW, create_output_classes

from custom_train import train


class MaskRCNNTrainer:
    default_settings = {
        'dataset_tags': {
            'train': 'train',
            'val': 'val',
        },
        'batch_size': {
            'train': 1,
            'val': 1,
        },
        'input_size': {
            'width': 1364,
            'height': 800,
        },
        'epochs': 2,
        'val_every': 1,
        'lr': 0.0001,
        'weights_init_type': 'transfer_learning',  # 'continue_training',
        'gpu_devices': [0],
        'validate_with_model_eval': True
    }

    base_mask_config_path = '/workdir/src/mask_incv2.config'

    def _determine_settings(self):
        input_config = self.helper.task_settings
        logger.info('Input config', extra={'config': input_config})
        config = deepcopy(self.default_settings)
        sly.update_recursively(config, input_config)
        logger.info('Full config', extra={'config': config})
        SettingsValidator.validate_train_cfg(config)
        self.config = config

    def _determine_model_classes(self):
        self.class_title_to_idx, self.out_classes = create_output_classes(
            in_project_classes=self.helper.in_project_meta.classes
        )
        logger.info('Determined model internal class mapping', extra={'class_mapping': self.class_title_to_idx})
        logger.info('Determined model out classes', extra={'classes': self.out_classes.py_container})

    def _determine_out_config(self):
        self.out_config = {
            'settings': self.config,
            'classes': self.out_classes.py_container,
            'mapping': self.class_title_to_idx,
        }

    # for 'continue_training', requires exact match
    def _check_prev_model_config(self):
        prev_model_dir = self.helper.paths.model_dir
        prev_config_rw = TrainConfigRW(prev_model_dir)
        if not prev_config_rw.train_config_exists:
            raise RuntimeError('Unable to continue_training, config for previous training wasn\'t found.')
        prev_config = prev_config_rw.load()

        old_class_mapping = prev_config.get('class_title_to_idx', {})
        if self.class_title_to_idx != old_class_mapping:
            raise RuntimeError('Unable to continue training, class mapping is inconsistent with previous model.')

    def _construct_data_dicts(self):
        logger.info('Will collect samples (img/ann pairs).')

        name_to_tag = self.config['dataset_tags']
        project_fs = sly.ProjectFS.from_disk_dir_project(self.helper.paths.project_dir)
        logger.info('Project structure has been read. Samples: {}.'.format(project_fs.pr_structure.image_cnt))

        samples_dct = sly.samples_by_tags(
            tags=list(name_to_tag.values()), project_fs=project_fs, project_meta=self.helper.in_project_meta
        )

        self.tf_data_dicts = {}
        self.iters_cnt = {}
        for the_name, the_tag in name_to_tag.items():
            samples_lst = samples_dct[the_tag]
            sly.ensure_samples_nonempty(samples_lst, the_tag)
            dataset_dict = {
                "samples": samples_lst,
                "classes_mapping": self.class_title_to_idx,
                "project_meta": self.helper.in_project_meta,
                "sample_cnt": len(samples_lst)
            }
            self.tf_data_dicts[the_name] = dataset_dict
            self.iters_cnt[the_name] = np.ceil(float(len(samples_lst)) /
                                       (self.config['batch_size'][the_name] * len(self.config['gpu_devices']))).astype('int')
            logger.info('Prepared dataset.', extra={
                'dataset_purpose': the_name, 'dataset_tag': the_tag, 'sample_cnt': len(samples_lst)
            })

    def _make_tf_train_config(self):
        self.train_iters = self.tf_data_dicts['train']['sample_cnt'] // self.config['batch_size']['train']
        total_steps = self.config['epochs'] * self.train_iters
        src_size = self.config['input_size']
        input_size_wh = (src_size['width'], src_size['height'])

        tf_config = load_sample_config(self.base_mask_config_path)

        if self.helper.model_dir_is_empty():
            checkpoint = None
            logger.info('Weights will not be inited.')
        else:
            checkpoint = osp.join(self.helper.paths.model_dir, 'model_weights', 'model.ckpt')
            logger.info('Weights will be loaded from previous train.')

        self.tf_config = remake_mask_rcnn_config(tf_config,
                                                 'SUPERVISELY_FORMAT',
                                                 total_steps,
                                                 len(self.out_classes),
                                                 input_size_wh,
                                                 self.config['batch_size']['train'],
                                                 self.config['lr'],
                                                 checkpoint)
        logger.info('Model config created.')

    def __init__(self):
        logger.info('Will init all required to train.')
        self.helper = sly.TaskHelperTrain()

        self._determine_settings()
        self._determine_model_classes()
        self._determine_out_config()

        self._construct_data_dicts()
        self._make_tf_train_config()

        logger.info('Model is ready to train.')

    def train(self):
        device_ids = sly.remap_gpu_devices(self.config['gpu_devices'])

        def dump_model(saver, sess, is_best, opt_data):
            out_dir = self.helper.checkpoints_saver.get_dir_to_write()
            TrainConfigRW(out_dir).save(self.out_config)
            save_config(osp.join(out_dir, 'model.config'), self.tf_config)
            model_fpath = os.path.join(out_dir, 'model_weights', 'model.ckpt')
            saver.save(sess, model_fpath)

            self.helper.checkpoints_saver.saved(is_best, opt_data)

        train(self.tf_data_dicts,
              self.config['epochs'],
              self.config['val_every'],
              self.iters_cnt,
              self.config['validate_with_model_eval'],
              pipeline_config=self.tf_config,
              num_clones=len(device_ids),
              save_cback=dump_model)


def main():
    cv2.setNumThreads(0)
    x = MaskRCNNTrainer()  # load model & prepare all
    x.train()


if __name__ == '__main__':
    if os.getenv('DEBUG_LOG_TO_FILE', None):
        sly.add_default_logging_into_file(logger, sly.TaskPaths().debug_dir)
    sly.main_wrapper('MASK_RCNN_TRAIN', main)

# coding: utf-8

import os
from copy import deepcopy
from os.path import join

import cv2
import numpy as np
import tensorflow as tf
import supervisely_lib as sly
from supervisely_lib import logger

import deeplab.model as model
from common import SettingsValidator, TrainConfigRW
from custom_train import train


slim = tf.contrib.slim


class DeepLabTrainer:
    # @TODO: do smth with settings
    default_settings = {
        'dataset_tags': {
            'train': 'train',
            'val': 'val',
        },
        'batch_size': {
            'train': 1,
            'val': 1,
        },
        'data_workers': {
            'train': 0,
            'val': 0,
        },
        'input_size': {
            'width': 513,
            'height': 513,
        },

        'special_classes': {
            'background': 'bg',
            'neutral': 'neutral',
        },
        'epochs': 2,
        'val_every': 1,
        'lr': 0.0001,
        'weights_init_type': 'transfer_learning',  # 'continue_training',
        'gpu_devices': [0],
    }

    bkg_input_idx = 0
    neutral_input_idx = 255

    def _determine_settings(self):
        input_config = self.helper.task_settings
        logger.info('Input config', extra={'config': input_config})
        config = deepcopy(self.default_settings)
        sly.update_recursively(config, input_config)
        logger.info('Full config', extra={'config': config})
        SettingsValidator.validate_train_cfg(config)
        self.config = config

    def _determine_model_classes(self):
        spec_cls = self.config['special_classes']
        self.class_title_to_idx, self.out_classes = sly.create_segmentation_classes(
            in_project_classes=self.helper.in_project_meta.classes,
            bkg_title=spec_cls['background'],
            neutral_title=spec_cls['neutral'],
            bkg_color=self.bkg_input_idx,
            neutral_color=self.neutral_input_idx,
        )
        logger.info('Determined model internal class mapping', extra={'class_title_to_idx': self.class_title_to_idx})
        logger.info('Determined model out classes', extra={'out_classes': self.out_classes.py_container})

    def _determine_out_config(self):
        self.out_config = {
            'settings': self.config,
            'out_classes': self.out_classes.py_container,
            'class_title_to_idx': self.class_title_to_idx,
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
        src_size = self.config['input_size']
        self.input_size_wh = (src_size['width'], src_size['height'])
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
                "sample_cnt": len(samples_lst),
                "batch_size": self.config['batch_size'][the_name]
            }
            self.tf_data_dicts[the_name] = dataset_dict
            self.iters_cnt[the_name] = np.ceil(float(len(samples_lst)) /
                                       (self.config['batch_size'][the_name] * len(self.config['gpu_devices']))).astype('int')
            logger.info('Prepared dataset.', extra={
                'dataset_purpose': the_name, 'dataset_tag': the_tag, 'sample_cnt': len(samples_lst)
            })

    def __init__(self):
        logger.info('Will init all required to train.')
        self.helper = sly.TaskHelperTrain()

        self._determine_settings()
        self._determine_model_classes()
        self._determine_out_config()
        self._construct_data_dicts()

    def train(self):
        device_ids = sly.remap_gpu_devices(self.config['gpu_devices'])

        def dump_model(saver, sess, is_best, opt_data):
            out_dir = self.helper.checkpoints_saver.get_dir_to_write()
            TrainConfigRW(out_dir).save(self.out_config)
            model_fpath = os.path.join(out_dir, 'model_weights', 'model.ckpt')
            saver.save(sess, model_fpath)

            self.helper.checkpoints_saver.saved(is_best, opt_data)

        def init_model_fn(sess):
            if self.helper.model_dir_is_empty():
                logger.info('Weights will be inited randomly.')
            else:
                exclude_list = ['global_step']
                wi_type = self.config['weights_init_type']
                ewit = {'weights_init_type': wi_type}
                logger.info('Weights will be inited from given model.', extra=ewit)
                if wi_type == 'transfer_learning':
                    last_layers = model.get_extra_layer_scopes(False)
                    exclude_list.extend(last_layers)
                    ignore_missing_vars = True
                elif wi_type == 'continue_training':
                    self._check_prev_model_config()
                    ignore_missing_vars = False
                else:
                    raise NotImplementedError('Only transfer_learning and continue_training modes are available.')
                logger.info('Weights are loaded.', extra=ewit)

                variables_to_restore = slim.get_variables_to_restore(exclude=exclude_list)
                init_fn = slim.assign_from_checkpoint_fn(join(self.helper.paths.model_dir, 'model_weights', 'model.ckpt'),
                                                         variables_to_restore, ignore_missing_vars=ignore_missing_vars)
                init_fn(sess)

        input_shape_hw = (self.input_size_wh[1], self.input_size_wh[0])
        train(
            data_dicts=self.tf_data_dicts,
            class_num=len(self.out_classes),
            input_size=input_shape_hw,
            lr=self.config['lr'],
            n_epochs=self.config['epochs'],
            num_clones=len(device_ids),
            iters_cnt=self.iters_cnt,
            val_every=self.config['val_every'],
            model_init_fn=init_model_fn,
            save_cback=dump_model

        )


def main():
    cv2.setNumThreads(0)
    x = DeepLabTrainer()
    x.train()


if __name__ == '__main__':
    if os.getenv('DEBUG_LOG_TO_FILE', None):
        sly.add_default_logging_into_file(logger, sly.TaskPaths().debug_dir)
    sly.main_wrapper('DEEPLAB_TRAIN', main)

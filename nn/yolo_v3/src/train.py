# coding: utf-8

import os
import os.path as osp
from copy import deepcopy

import numpy as np
import supervisely_lib as sly
from supervisely_lib import logger

from common import SettingsValidator, TrainConfigRW, create_detection_classes
from dataset_utils import load_dataset
from yolo_config_utils import load_config, refact_yolo_config, save_config
from ctypes_utils import train_yolo, int1D_to_p_int, float2D_to_pp_float, string_list_pp_char


class YOLOTrainer:
    default_settings = {
        'dataset_tags': {
            'train': 'train'
        },
        'batch_size': {
            'train': 1
        },
        'subdivisions': {
            'train': 1
        },
        'data_workers': {
            'train': 0,
        },
        'input_size': {
            'width': 416,
            'height': 416,
        },
        'enable_augmentations': False,
        'bn_momentum': 0.01,
        'print_every_iter': 10,
        'epochs': 2,
        'lr': 0.0001,
        'weights_init_type': 'transfer_learning',  # 'continue_training',
        'gpu_devices': [0],
    }

    base_yolo_config_path = '/workdir/src/yolov3_base.cfg'

    def _determine_settings(self):
        input_config = self.helper.task_settings
        logger.info('Input config', extra={'config': input_config})
        config = deepcopy(self.default_settings)
        sly.update_recursively(config, input_config)
        logger.info('Full config', extra={'config': config})
        # SettingsValidator.validate_train_cfg(config)
        self.config = config

    def _determine_model_classes(self):
        self.class_title_to_idx, self.out_classes = create_detection_classes(
            in_project_classes=self.helper.in_project_meta.classes
        )
        logger.info('Determined model internal class mapping', extra={'class_mapping': self.class_title_to_idx})
        logger.info('Determined model out classes', extra={'classes': self.out_classes.py_container})

    def _determine_out_config(self):
        self.out_config = {
            'settings': self.config,
            'out_classes': self.out_classes.py_container,
            'class_title_to_idx': self.class_title_to_idx,
        }

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

        self.data_dicts = {}
        self.iters_cnt = {}
        for the_name, the_tag in name_to_tag.items():
            samples_lst = samples_dct[the_tag]
            if len(samples_lst) < 1:
                raise RuntimeError('Dataset %s should contain at least 1 element.' % the_name)

            img_paths, labels, num_boxes = load_dataset(samples_lst, self.class_title_to_idx, self.helper.in_project_meta)
            dataset_dict = {
                'img_paths': img_paths,
                'labels': labels,
                'num_boxes': num_boxes,
                'sample_cnt': len(samples_lst)
            }
            self.data_dicts[the_name] = dataset_dict
            self.iters_cnt[the_name] = np.ceil(float(len(samples_lst)) /
                                       (self.config['batch_size'][the_name] * len(self.config['gpu_devices']))).astype('int')
            logger.info('Prepared dataset.', extra={
                'dataset_purpose': the_name, 'dataset_tag': the_tag, 'sample_cnt': len(samples_lst)
            })

    def _make_yolo_train_config(self):
        src_size = self.config['input_size']
        input_size_wh = (src_size['width'], src_size['height'])

        yolo_config_base = load_config(self.base_yolo_config_path)

        self.yolo_config = refact_yolo_config(yolo_config_base,
                                              input_size_wh,
                                              self.config['batch_size']['train'],
                                              self.config['subdivisions']['train'],
                                              len(self.out_classes),
                                              self.config['lr'])
        save_config(self.yolo_config, '/tmp/model.cfg')
        logger.info('Model config created.')

    def _define_initializing_params(self):
        if self.helper.model_dir_is_empty():
            self.weights_path = ''
            self.layer_cutoff = 0
            logger.info('Weights will not be inited.')
        else:
            self.weights_path = osp.join(self.helper.paths.model_dir, 'model.weights')
            if not osp.exists(self.weights_path):
                raise RuntimeError("Unable to find file with model weights.")

            wi_type = self.config['weights_init_type']
            ewit = {'weights_init_type': wi_type}
            logger.info('Weights will be inited from given model.', extra=ewit)
            if wi_type == 'transfer_learning':
                self.layer_cutoff = 80  # fixed for the yolo_v3
            elif wi_type == 'continue_training':
                self.layer_cutoff = 0   # load weights for all given layers
                self._check_prev_model_config()

    def _create_checkpoints_dir(self):
        for epoch in range(self.config['epochs']):
            ckpt_dir = os.path.join(self.helper.paths.results_dir, '{:08}'.format(epoch))
            sly.mkdir(ckpt_dir)
            save_config(self.yolo_config, os.path.join(ckpt_dir, 'model.cfg'))
            TrainConfigRW(ckpt_dir).save(self.out_config)

    def __init__(self):
        logger.info('Will init all required to train.')
        self.helper = sly.TaskHelperTrain()

        self._determine_settings()
        self._determine_model_classes()
        self._determine_out_config()

        self._construct_data_dicts()
        self._make_yolo_train_config()
        self._define_initializing_params()
        self._create_checkpoints_dir()

    def train(self):
        device_ids = self.config['gpu_devices']
        data_dict = self.data_dicts['train']

        c_img_paths = string_list_pp_char(data_dict['img_paths'])
        c_boxes = float2D_to_pp_float(data_dict['labels'])
        c_num_gt_boxes = int1D_to_p_int(data_dict['num_boxes'])

        train_steps = int(np.ceil(data_dict['sample_cnt'] / self.config['batch_size']['train']))

        train_yolo(
            '/tmp/model.cfg'.encode('utf-8'),
            self.weights_path.encode('utf-8'),
            c_img_paths,
            c_num_gt_boxes,
            c_boxes,
            data_dict['sample_cnt'],
            int1D_to_p_int(device_ids),
            len(device_ids),
            self.config['data_workers']['train'],
            self.config['epochs'],
            train_steps,
            self.layer_cutoff,
            1 if self.config['enable_augmentations'] else 0,  # with aug
            int(self.config['print_every_iter']),
            float(self.config['bn_momentum'])
        )


def main():
    x = YOLOTrainer()
    x.train()


if __name__ == '__main__':
    if os.getenv('DEBUG_LOG_TO_FILE', None):
        sly.add_default_logging_into_file(logger, sly.TaskPaths().debug_dir)
    sly.main_wrapper('YOLO_V3_TRAIN', main)

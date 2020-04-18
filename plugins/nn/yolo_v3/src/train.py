# coding: utf-8

import os
import json
import math
import numpy as np
import random

import cv2

import supervisely_lib as sly
import supervisely_lib.nn.dataset
from supervisely_lib import logger
from supervisely_lib import sly_logger
from supervisely_lib.nn.config import JsonConfigValidator
from supervisely_lib.nn.hosted.class_indexing import CONTINUE_TRAINING, TRANSFER_LEARNING

import common
from dataset_utils import load_dataset
from yolo_config_utils import find_data_item, read_config, replace_config_section_values, write_config, MODEL_CFG, \
    CONVOLUTIONAL_SECTION, NET_SECTION, YOLO_SECTION
from ctypes_utils import train_yolo, int1D_to_p_int, float2D_to_pp_float, string_list_pp_char

from supervisely_lib.nn.hosted.trainer import SuperviselyModelTrainer, DATASET_TAGS, TRAIN


# Computes anchors in relative (to image size) units.
# To get the values to write to the network config, multiply by input size in pixels.
def compute_anchors_relative(project, num_anchors, train_tag, max_iters=100, num_attempts=1):
    all_box_shapes_list = []
    for ds in project:
        for item in ds:
            ann_path = ds.get_ann_path(item)
            ann = sly.Annotation.load_json_file(ann_path, project.meta)
            if train_tag is None or ann.img_tags.has_key(train_tag):
                for label in ann.labels:
                    raw_bbox = label.geometry.to_bbox()
                    w = raw_bbox.width / float(ann.img_size[1])
                    h = raw_bbox.height / float(ann.img_size[0])
                    all_box_shapes_list.append((w, h))
    return cv2.kmeans(
        data=np.array(all_box_shapes_list, dtype=np.float32),
        K=num_anchors,
        bestLabels=None,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iters, 0),
        attempts=num_attempts,
        flags=cv2.KMEANS_PP_CENTERS)


class YOLOTrainer(SuperviselyModelTrainer):
    @staticmethod
    def get_default_config():
        return {
            'batch_size': {
                'train': 1,
                'val': 1
            },
            'bn_momentum': 0.01,
            'dataset_tags': {
                'train': 'train',
                'val': 'val'
            },
            'data_workers': {
                'train': 1,
                'val': 1
            },
            'enable_augmentations': False,
            'epochs': 2,
            'gpu_devices': [0],
            'input_size': {
                'width': 416,
                'height': 416,
            },
            'lr': 0.0001,
            'print_every_iter': 10,
            'subdivisions': {
                'train': 1,
                'val': 1
            },
            'weights_init_type': TRANSFER_LEARNING,  # CONTINUE_TRAINING,
        }

    def __init__(self):
        super().__init__(default_config=YOLOTrainer.get_default_config())
        logger.info('Model is ready to train.')

    @property
    def class_title_to_idx_key(self):
        return common.class_to_idx_config_key()

    @property
    def train_classes_key(self):
        return common.train_classes_key()

    def get_start_class_id(self):
        return 0

    def _validate_train_cfg(self, config):
        JsonConfigValidator().validate_train_cfg(config)

    def _construct_and_fill_model(self):
        self._make_yolo_train_config()
        self._define_initializing_params()
        self._create_checkpoints_dir()

    def _construct_loss(self):
        pass

    def _determine_model_classes(self):
        self._determine_model_classes_detection()

    def _construct_data_loaders(self):
        self.data_dicts = {}
        self.iters_cnt = {}
        for the_name, the_tag in self.name_to_tag.items():
            samples_lst = self._deprecated_samples_by_tag[the_tag]
            supervisely_lib.nn.dataset.ensure_samples_nonempty(samples_lst, the_tag, self.project.meta)

            img_paths, labels, num_boxes = load_dataset(samples_lst, self.class_title_to_idx, self.project.meta)
            dataset_dict = {
                'img_paths': img_paths,
                'labels': labels,
                'num_boxes': num_boxes,
                'sample_cnt': len(samples_lst)
            }
            self.data_dicts[the_name] = dataset_dict

            samples_per_iter = self.config['batch_size'][the_name] * len(self.config['gpu_devices'])
            self.iters_cnt[the_name] = math.ceil(float(len(samples_lst)) / samples_per_iter)
            logger.info('Prepared dataset.', extra={
                'dataset_purpose': the_name, 'dataset_tag': the_tag, 'sample_cnt': len(samples_lst)
            })

    def _maybe_recomputed_anchors_str(self, width, height, source_yolo_config):
        if not self.config.get('recompute_anchors', False):
            return None
        else:
            logger.info('Started recomputing anchors based on training data.')

            # Check the source config file to find out the number of anchors.
            num_anchors = -1
            for section_idx, section in enumerate(source_yolo_config):
                if section.name == YOLO_SECTION:
                    anchors_item = find_data_item(section, 'anchors')
                    num_anchor_coords = len(anchors_item[1].split(','))
                    if num_anchor_coords % 2 != 0:
                        raise ValueError(
                            'Source Yolo config has an odd-length list in the anchors config. This value is invalid, '
                            'the anchors value must be a list of (w, h) pairs. Got {!r}'.format(anchors_item[1]))
                    num_anchors = num_anchor_coords // 2
                    break

            if num_anchors <= 0:
                raise ValueError('Source Yolo config has no anchors specified. Unable to determine the correct number '
                                 'of anchors to be recomputed.')

            logger.info('Source Yolo config uses {} anchors.'.format(num_anchors))

            _, _, relative_anchors = compute_anchors_relative(
                self.project,
                num_anchors=num_anchors,
                train_tag=self.config[DATASET_TAGS][TRAIN],
                max_iters=200,
                num_attempts=10)
            logger.info('Finished recomputing anchors.')
            # Converting relative to pixel sizes.
            # Yolo anchors are (w, h), NOT our usual (h,w).
            anchors = np.round(relative_anchors * np.array([[width, height]])).astype(np.int32)
            # Sort by area and convert to string.
            anchors = sorted(anchors.tolist(), key=lambda x: x[0] * x[1])
            anchors_str = ',  '.join(','.join(str(x) for x in anchor) for anchor in anchors)
            logger.info('Resulting recomputed anchors: {}'.format(anchors_str))
            return anchors_str

    def _make_yolo_train_config(self):
        src_size = self.config['input_size']
        height = src_size['height']
        width = src_size['width']

        yolo_config = read_config(os.path.join(sly.TaskPaths.MODEL_DIR, MODEL_CFG))
        [net_config] = [section for section in yolo_config if section.name == NET_SECTION]
        net_overrides = {
            'height': height,
            'width': width,
            'batch': self.config['batch_size']['train'],
            'subdivisions': self.config['subdivisions']['train'],
            'learning_rate': self.config['lr']
        }

        # Optionally recompute the anchors
        recomputed_anchors_str = self._maybe_recomputed_anchors_str(width, height, yolo_config)

        replace_config_section_values(net_config, net_overrides)

        for section_idx, section in enumerate(yolo_config):
            if section.name == YOLO_SECTION:
                no_preceding_conv_section_msg = ('Unexpectedly found {!r} section in the config without immediately '
                                                 'preceding {!r} section.'.format(YOLO_SECTION, CONVOLUTIONAL_SECTION))
                if section_idx == 0:
                    raise ValueError(no_preceding_conv_section_msg)
                last_convolution = yolo_config[section_idx - 1]
                if last_convolution.name != CONVOLUTIONAL_SECTION:
                    raise ValueError(no_preceding_conv_section_msg)

                classes_item = find_data_item(section, 'classes')
                classes_item[1] = len(self.out_classes)

                filters_item = find_data_item(last_convolution, 'filters')
                filters_item[1] = (len(self.out_classes) + 5) * 3

                if recomputed_anchors_str is not None:
                    anchors_item = find_data_item(section, 'anchors')
                    anchors_item[1] = recomputed_anchors_str

        first_yolo_section_idx = next(idx for idx, section in enumerate(yolo_config) if section.name == YOLO_SECTION)
        # Compute the index of the last layer to be loaded for transfer learning.
        # This will be the last layer before any layers that depend on the number
        # of predicted classes start.
        # The layers that depend on the number of classes are YOLO layers, and their
        # immediately preceding convolutional layers (since those have number of filters
        # depending on the number of classes).
        # So we find the first YOLO layer, subtract 2 to skip YOLO and preceding convolutional layer,
        # and subtract another 2 to account for the first [net] and root sections of the config.
        self._last_transfer_learning_load_layer_idx = first_yolo_section_idx - 4

        self._effective_model_cfg_path = os.path.join('/tmp', MODEL_CFG)
        write_config(yolo_config, self._effective_model_cfg_path)
        logger.info('Model config created.')

    def _define_initializing_params(self):
        if sly.fs.dir_empty(sly.TaskPaths.MODEL_DIR):
            self.weights_path = ''
            self.layer_cutoff = 0
            logger.info('Weights will not be inited.')
        else:
            self.weights_path = os.path.join(sly.TaskPaths.MODEL_DIR, 'model.weights')
            if not os.path.exists(self.weights_path):
                raise RuntimeError("Unable to find file with model weights.")

            wi_type = self.config['weights_init_type']
            ewit = {'weights_init_type': wi_type}
            logger.info('Weights will be inited from given model.', extra=ewit)
            if wi_type == TRANSFER_LEARNING:
                self.layer_cutoff = self._last_transfer_learning_load_layer_idx
            elif wi_type == CONTINUE_TRAINING:
                self.layer_cutoff = 0  # load weights for all given layers

    def _create_checkpoints_dir(self):
        for epoch in range(self.config['epochs']):
            checkpoint_dir = os.path.join(sly.TaskPaths.RESULTS_DIR, '{:08}'.format(epoch))
            sly.fs.mkdir(checkpoint_dir)
            sly.fs.copy_file(self._effective_model_cfg_path, os.path.join(checkpoint_dir, MODEL_CFG))
            sly.io.json.dump_json_file(self.out_config, os.path.join(checkpoint_dir, sly.TaskPaths.MODEL_CONFIG_NAME))

    def train(self):
        device_ids = self.config['gpu_devices']
        data_dict = self.data_dicts['train']
        c_img_paths = string_list_pp_char(data_dict['img_paths'])
        c_boxes = float2D_to_pp_float(data_dict['labels'])
        c_num_gt_boxes = int1D_to_p_int(data_dict['num_boxes'])

        vdata_dict = self.data_dicts['val']
        vc_img_paths = string_list_pp_char(vdata_dict['img_paths'])
        vc_boxes = float2D_to_pp_float(vdata_dict['labels'])
        vc_num_gt_boxes = int1D_to_p_int(vdata_dict['num_boxes'])

        train_steps = int(np.ceil(data_dict['sample_cnt'] / self.config['batch_size']['train']))

        progress_dummy = sly.Progress('Building model:', 1)
        progress_dummy.iter_done_report()

        logger.info('Will load model layers up to index {!r}.'.format(self.layer_cutoff))

        train_yolo(
            '/tmp/model.cfg'.encode('utf-8'),
            self.weights_path.encode('utf-8'),
            c_img_paths,
            c_num_gt_boxes,
            c_boxes,
            data_dict['sample_cnt'],

            vc_img_paths,
            vc_num_gt_boxes,
            vc_boxes,
            vdata_dict['sample_cnt'],

            int1D_to_p_int(device_ids),
            len(device_ids),
            self.config['data_workers']['train'],
            self.config['epochs'],
            train_steps,
            self.config.get('checkpoint_every', 1),
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
        sly_logger.add_default_logging_into_file(logger, sly.TaskPaths.DEBUG_DIR)
    sly.main_wrapper('YOLO_V3_TRAIN', main)

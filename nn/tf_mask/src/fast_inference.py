# coding: utf-8

import os

import tensorflow as tf
import numpy as np
import supervisely_lib as sly
from supervisely_lib import logger

from common import create_detection_graph, freeze_graph, inverse_mapping, get_output_dict, \
    TrainConfigRW, masks_detection_to_sly_bitmaps


class MaskRCNNFastApplier:
    def _load_train_config(self):
        model_dir = sly.TaskPaths(determine_in_project=False).model_dir

        train_config_rw = TrainConfigRW(model_dir)
        if not train_config_rw.train_config_exists:
            raise RuntimeError('Unable to run inference, config from training wasn\'t found.')
        self.train_config = train_config_rw.load()
        input_size = self.train_config['settings']['input_size']
        w, h = input_size['width'], input_size['height']
        logger.info('Model input size is read (for auto-rescale).', extra={'input_size': {
            'width': w, 'height': h
        }})

        self.class_title_to_idx = self.train_config['mapping']
        self.train_classes = sly.FigClasses(self.train_config['classes'])
        logger.info('Read model internal class mapping', extra={'class_mapping': self.class_title_to_idx})
        logger.info('Read model out classes', extra={'classes': self.train_classes.py_container})

        out_class_mapping = {x: self.class_title_to_idx[x] for x in
                             (x['title'] for x in self.train_classes)}
        self.inv_mapping = inverse_mapping(out_class_mapping)

    def _construct_and_fill_model(self):
        model_dir = sly.TaskPaths(determine_in_project=False).model_dir
        self.device_ids = sly.remap_gpu_devices([self.source_gpu_device])
        if 'model.pb' not in os.listdir(model_dir):
            logger.info('Freezing training checkpoint!')
            freeze_graph('image_tensor',
                         model_dir + '/model.config',
                         model_dir + '/model_weights/model.ckpt',
                         model_dir)
        self.detection_graph = create_detection_graph(model_dir)
        self.session = tf.Session(graph=self.detection_graph)
        logger.info('Weights are loaded.')

    def __init__(self, settings):
        logger.info('Will init all required to inference.')

        self.source_gpu_device = settings['device_id']
        self.score_thresh = settings['min_score_threshold']
        self._load_train_config()
        self._construct_and_fill_model()
        logger.info('Model is ready to inference.')

    def inference(self, img):
        h, w = img.shape[:2]
        image_np_expanded = np.expand_dims(img, axis=0)
        out_dict = get_output_dict(image_np_expanded, self.detection_graph, self.session)

        res_figures = masks_detection_to_sly_bitmaps(self.inv_mapping, out_dict, img.shape, self.score_thresh)
        res_ann = sly.Annotation.new_with_objects((w, h), res_figures)
        return res_ann

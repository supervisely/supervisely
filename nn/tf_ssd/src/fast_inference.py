# coding: utf-8

import os

import tensorflow as tf
import numpy as np
import supervisely_lib as sly
from supervisely_lib import logger

from common import create_detection_graph, freeze_graph, inverse_mapping, get_scope_vars, \
    TrainConfigRW


class SSDFastApplier:
    def _load_train_config(self):
        model_dir = sly.TaskPaths(determine_in_project=False).model_dir

        train_config_rw = TrainConfigRW(model_dir)
        if not train_config_rw.train_config_exists:
            raise RuntimeError('Unable to run inference, config from training wasn\'t found.')
        self.train_config = train_config_rw.load()

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
        detection_boxes, detection_scores, detection_classes, num_detections, image_tensor = \
            get_scope_vars(self.detection_graph)

        net_out = self.session.run([detection_boxes, detection_scores, detection_classes, num_detections],
                                   feed_dict={image_tensor: image_np_expanded})

        res_figures = sly.detection_preds_to_sly_rects(self.inv_mapping, net_out, img.shape, self.score_thresh)

        res_ann = sly.Annotation.new_with_objects((w, h), res_figures)
        return res_ann

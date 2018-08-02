# coding: utf-8

from collections import namedtuple

import tensorflow as tf
import numpy as np
import supervisely_lib as sly
from supervisely_lib import logger

from common import TrainConfigRW
from deeplab.model_utils import ModelOptions
import deeplab.model as model
import deeplab.input_preprocess as input_preprocess


slim = tf.contrib.slim


class DeeplabFastApplier:
    def _load_train_config(self):
        model_dir = sly.TaskPaths(determine_in_project=False).model_dir

        train_config_rw = TrainConfigRW(model_dir)
        if not train_config_rw.train_config_exists:
            raise RuntimeError('Unable to run inference, config from training wasn\'t found.')
        self.train_config = train_config_rw.load()

        src_size = self.train_config['settings']['input_size']
        self.input_size_wh = (src_size['width'], src_size['height'])
        logger.info('Model input size is read (for auto-rescale).', extra={'input_size': {
            'width': self.input_size_wh[0], 'height': self.input_size_wh[1]
        }})

        self.class_title_to_idx = self.train_config['class_title_to_idx']
        self.train_classes = sly.FigClasses(self.train_config['out_classes'])
        logger.info('Read model internal class mapping', extra={'class_title_to_idx': self.class_title_to_idx})
        logger.info('Read model out classes', extra={'out_classes': self.train_classes.py_container})

        self.out_class_mapping = {x: self.class_title_to_idx[x] for x in
                                  (x['title'] for x in self.train_classes)}

    def _construct_and_fill_model(self):
        model_dir = sly.TaskPaths(determine_in_project=False).model_dir
        self.device_ids = sly.remap_gpu_devices([self.source_gpu_device])
        initialized_model = namedtuple('Model',
                                       ['input_images', 'predictions', 'sess'])

        with tf.get_default_graph().as_default():
            img = tf.placeholder(tf.float32, shape=(None, None, 3))
            input_w, input_h = self.input_size_wh
            img_re = tf.image.resize_images(img, (input_h, input_w))
            original_image, image, label = input_preprocess.preprocess_image_and_label(
                img_re,
                None,
                crop_height=input_h,
                crop_width=input_w,
                is_training=False,
                model_variant="xception_65")

            image = tf.expand_dims(image, 0)

            model_options = ModelOptions(
                outputs_to_num_classes={'semantic': len(self.train_classes)},
                crop_size=(input_h, input_w),
                atrous_rates=[6, 12, 18],
                output_stride=16)

            predictions = model.predict_logits(
                tf.shape(img)[0:2],
                image,
                model_options=model_options,
                image_pyramid=None)

            predictions = predictions['semantic']
            saver = tf.train.Saver(slim.get_variables_to_restore())
            sess = tf.train.MonitoredTrainingSession(master='')
            saver.restore(sess, model_dir + '/model_weights/model.ckpt')

            initialized_model.input_images = img
            initialized_model.predictions = predictions
            initialized_model.sess = sess
        self.initialized_model = initialized_model
        logger.info('Weights are loaded.')

    def __init__(self, settings):
        logger.info('Will init all required to inference.')

        self.source_gpu_device = settings['device_id']
        self._load_train_config()
        self._construct_and_fill_model()
        logger.info('Model is ready to inference.')

    def inference(self, img):
        h, w = img.shape[:2]
        img_var = img.astype(np.float32)

        semantic_predictions = self.initialized_model.sess.run(self.initialized_model.predictions,
                                                               feed_dict={self.initialized_model.input_images: img_var})
        pred = np.squeeze(semantic_predictions[0])
        pred_cls_idx = np.argmax(pred, axis=2)
        res_figures = sly.prediction_to_sly_bitmaps(self.out_class_mapping, pred_cls_idx)
        res_ann = sly.Annotation.new_with_objects((w, h), res_figures)
        return res_ann

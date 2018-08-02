# coding: utf-8

import os.path as osp

import numpy as np
import tensorflow as tf
import supervisely_lib as sly
from supervisely_lib import logger

from model import ICNet_BN
from tools import preprocess, TrainConfigRW


class ICNetFastApplier:
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
        logger.info('Read model internal class mapping', extra={'class_mapping': self.class_title_to_idx})
        logger.info('Read model out classes', extra={'classes': self.train_classes.py_container})

        self.out_class_mapping = {x: self.class_title_to_idx[x] for x in
                                  (x['title'] for x in self.train_classes)}

    def _construct_and_fill_model(self):
        model_dir = sly.TaskPaths(determine_in_project=False).model_dir
        self.device_ids = sly.remap_gpu_devices([self.source_gpu_device])

        logger.info('Will create model.')
        with tf.get_default_graph().as_default():
            img_np = tf.placeholder(tf.float32, shape=(None, None, 3))
            img_shape = tf.shape(img_np)

            w, h = self.input_size_wh
            img_np_4d = tf.expand_dims(img_np, axis=0)
            image_rs_4d = tf.image.resize_bilinear(img_np_4d, (h, w), align_corners=True)
            image_rs = tf.squeeze(image_rs_4d, axis=0)
            img = preprocess(image_rs, h, w)

            net = ICNet_BN({'data': img},
                           is_training=False, num_classes=len(self.train_classes),
                           filter_scale=self.train_config['settings']['filter_scale'])
            raw_output = net.layers['conv6_cls']  # 4d

            # Predictions.
            # !!!!!!!!!  align_corners=False
            raw_output_up = tf.image.resize_bilinear(raw_output,
                                                     size=[img_shape[0], img_shape[1]], align_corners=False)
            # raw_output_up = tf.argmax(raw_output_up, dimension=3)

            logger.info('Will load weights from trained model.')
            # Init tf Session
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            sess = tf.Session(config=config)
            init = tf.global_variables_initializer()
            sess.run(init)
            loader = tf.train.Saver(var_list=tf.global_variables())

            # last_checkpoint = tf_saver.latest_checkpoint(output_train_dir)
            last_checkpoint = osp.join(model_dir, 'model.ckpt')
            loader.restore(sess, last_checkpoint)

            self.input_images = img_np
            self.predictions = raw_output_up
            self.sess = sess
        logger.info('Model has been created & weights are loaded.')

    def __init__(self, settings):
        logger.info('Will init all required to inference.')

        self.source_gpu_device = settings['device_id']
        self._load_train_config()
        self._construct_and_fill_model()
        logger.info('Model is ready to inference.')

    def inference(self, img):
        h, w = img.shape[:2]
        img_var = img.astype(np.float32)
        semantic_predictions = self.sess.run(self.predictions,
                                             feed_dict={self.input_images: img_var})
        pred = np.squeeze(semantic_predictions[0])
        pred_cls_idx = np.argmax(pred, axis=2)
        res_figures = sly.prediction_to_sly_bitmaps(self.out_class_mapping, pred_cls_idx)
        res_ann = sly.Annotation.new_with_objects((w, h), res_figures)
        return res_ann

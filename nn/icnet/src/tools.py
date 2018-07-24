import numpy as np
import tensorflow as tf
import os
import os.path as osp
import sys

import supervisely_lib as sly

from prepare_data import get_label_colours

IMG_MEAN = np.array((103.939, 116.779, 123.68), dtype=np.float32)


def decode_labels(mask, img_shape, num_classes, classes_json=None, labels_mapping=None):
    color_table = get_label_colours(classes_json, labels_mapping)

    color_mat = tf.constant(color_table, dtype=tf.float32)
    mask = tf.subtract(mask, tf.constant(1, dtype=tf.int64))
    onehot_output = tf.one_hot(mask, depth=num_classes)
    onehot_output = tf.reshape(onehot_output, (-1, num_classes))
    pred = tf.matmul(onehot_output, color_mat)
    pred = tf.reshape(pred, (1, img_shape[0], img_shape[1], 3))

    return pred


def prepare_label(input_batch, new_size, num_classes, one_hot=True):
    with tf.name_scope('label_encode'):
        input_batch = tf.image.resize_nearest_neighbor(input_batch, new_size) # as labels are integer numbers, need to use NN interp.
        input_batch = tf.squeeze(input_batch, squeeze_dims=[3]) # reducing the channel dimension.
        if one_hot:
            input_batch = tf.one_hot(input_batch, depth=num_classes)
            
    return input_batch


def load_img(img_path):
    if os.path.isfile(img_path):
        print('successful load img: {0}'.format(img_path))
    else:
        print('not found file: {0}'.format(img_path))
        sys.exit(0)

    filename = img_path.split('/')[-1]
    ext = filename.split('.')[-1]

    if ext.lower() == 'png':
        img = tf.image.decode_png(tf.read_file(img_path), channels=3)
    elif ext.lower() == 'jpg':
        img = tf.image.decode_jpeg(tf.read_file(img_path), channels=3)
    else:
        print('cannot process {0} file.'.format(ext.lower()))

    return img, filename


def preprocess(img, h, w):
    # Convert RGB to BGR
    img_r, img_g, img_b = tf.split(axis=2, num_or_size_splits=3, value=img)
    img = tf.cast(tf.concat(axis=2, values=[img_b, img_g, img_r]), dtype=tf.float32)
    # Extract mean.
    img -= IMG_MEAN

    pad_img = tf.image.pad_to_bounding_box(img, 0, 0, h, w)
    pad_img = tf.expand_dims(pad_img, dim=0)

    return pad_img


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

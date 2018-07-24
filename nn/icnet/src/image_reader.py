from copy import deepcopy

import tensorflow as tf

from prepare_data import load_ann


def image_mirroring(img, label):
    distort_left_right_random = tf.random_uniform([1], 0, 1.0, dtype=tf.float32)[0]
    mirror = tf.less(tf.stack([1.0, distort_left_right_random, 1.0]), 0.5)
    mirror = tf.boolean_mask([0, 1, 2], mirror)
    img = tf.reverse(img, mirror)
    label = tf.reverse(label, mirror)
    
    return img, label


def image_scaling(img, label):
    scale = tf.random_uniform([1], minval=0.5, maxval=2.0, dtype=tf.float32, seed=None)
    h_new = tf.to_int32(tf.multiply(tf.to_float(tf.shape(img)[0]), scale))
    w_new = tf.to_int32(tf.multiply(tf.to_float(tf.shape(img)[1]), scale))
    new_shape = tf.squeeze(tf.stack([h_new, w_new]), squeeze_dims=[1])
    img = tf.image.resize_images(img, new_shape)
    label = tf.image.resize_nearest_neighbor(tf.expand_dims(label, 0), new_shape)
    label = tf.squeeze(label, squeeze_dims=[0])

    return img, label


def random_crop_and_pad_image_and_labels(image, label, crop_h, crop_w, ignore_label):
    label = tf.cast(label, dtype=tf.float32)
    label = label - ignore_label # Needs to be subtracted and later added due to 0 padding.
    combined = tf.concat(axis=2, values=[image, label])
    image_shape = tf.shape(image)
    combined_pad = tf.image.pad_to_bounding_box(combined, 0, 0, tf.maximum(crop_h, image_shape[0]), tf.maximum(crop_w, image_shape[1]))

    last_image_dim = tf.shape(image)[-1]
    last_label_dim = tf.shape(label)[-1]
    combined_crop = tf.random_crop(combined_pad, [crop_h,crop_w,4])  # random crop !!
    img_crop = combined_crop[:, :, :last_image_dim]
    label_crop = combined_crop[:, :, last_image_dim:]
    label_crop = label_crop + ignore_label
    label_crop = tf.cast(label_crop, dtype=tf.uint8)

    # Set static shape so that tensorflow knows shape at compile time.
    img_crop.set_shape((crop_h, crop_w, 3))
    label_crop.set_shape((crop_h,crop_w, 1))

    return img_crop, label_crop


def resize_to_fixed(image, label, target_size):
    # @TODO: align_corners??
    image = tf.expand_dims(image, axis=0)
    label = tf.expand_dims(label, axis=0)
    image_rs = tf.image.resize_bilinear(image, target_size, align_corners=True)
    label_rs = tf.image.resize_nearest_neighbor(label, target_size, align_corners=True)
    image_rs = tf.squeeze(image_rs, axis=0)
    label_rs = tf.squeeze(label_rs, axis=0)

    image_rs.set_shape((target_size[0], target_size[1], 3))
    label_rs.set_shape((target_size[0], target_size[1], 1))

    return image_rs, label_rs


def unzip_elements_fpaths(ia_descrs):
    tmp = [(descr.img_path, descr.ann_path) for descr in ia_descrs]
    images, masks = zip(*tmp)
    return images, masks


def read_images_from_disk(input_queue, input_size_wh, random_scale, random_mirror, img_mean,
                          labels_mapping, in_pr_meta):  # optional pre-processing arguments
    img_contents = tf.read_file(input_queue[0])

    img = tf.image.decode_jpeg(img_contents, channels=3)
    # @TODO!!! decode_png etc
    img_r, img_g, img_b = tf.split(axis=2, num_or_size_splits=3, value=img)
    img = tf.cast(tf.concat(axis=2, values=[img_b, img_g, img_r]), dtype=tf.float32)
    # Extract mean.
    img -= img_mean

    label_part = tf.identity(input_queue[1])
    load_ann_cur = lambda x: load_ann(x, deepcopy(labels_mapping), deepcopy(in_pr_meta))
    label = tf.py_func(load_ann_cur, [label_part], tf.float32, stateful=False)
    label.set_shape(())
    label = tf.reshape(label, tf.shape(img)[:2])
    label = tf.cast(label, tf.float32)

    if label.shape.ndims == 2:
        label = tf.expand_dims(label, 2)
    label.set_shape([None, None, 1])

    if input_size_wh is not None:
        w, h = input_size_wh

        if random_scale:
            img, label = image_scaling(img, label)

        if random_mirror:
            img, label = image_mirroring(img, label)

        # img, label = random_crop_and_pad_image_and_labels(img, label, h, w, ignore_label)
        img, label = resize_to_fixed(img, label, (h, w))

    return img, label


class ImageReader(object):
    def __init__(self, ia_descrs, input_size_wh,
                 random_scale, random_mirror, img_mean, coord, in_pr_meta, class_to_idx, shuffle):

        self.input_size_wh = input_size_wh
        self.coord = coord

        self.image_list, self.label_list = unzip_elements_fpaths(ia_descrs)
        self.images = tf.convert_to_tensor(self.image_list, dtype=tf.string)
        self.labels = tf.convert_to_tensor(self.label_list, dtype=tf.string)
        self.queue = tf.train.slice_input_producer([self.images, self.labels], shuffle=shuffle)
        self.in_pr_meta = in_pr_meta
        self.class_to_idx = class_to_idx
        self.image, self.label = read_images_from_disk(self.queue, self.input_size_wh,
                                                       random_scale, random_mirror,
                                                       img_mean,
                                                       labels_mapping=self.class_to_idx,
                                                       in_pr_meta=self.in_pr_meta)

    def dequeue(self, num_elements):
        image_batch, label_batch = tf.train.batch([self.image, self.label],  # @TODO: explore num_threads & capacity
                                                  num_elements,
                                                  allow_smaller_final_batch=True)
        return image_batch, label_batch

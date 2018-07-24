import tensorflow as tf
import PIL.Image
import io
import numpy as np

from tensorflow.python.framework import dtypes
import supervisely_lib as sly

from object_detection.core import standard_fields as fields
from object_detection.utils import dataset_util


def read_images_from_disk(img_path):
    file_contents = tf.read_file(img_path)
    example = tf.image.decode_png(file_contents, channels=3)  # @TODO: test decode_image
    return example


def get_bbox(mask):
    mask_points = np.where(mask == 1)
    return np.min(mask_points[1]), np.min(mask_points[0]), np.max(mask_points[1]), np.max(mask_points[0])


def load_ann(ann_fpath, classes_mapping, project_meta):
    ann_packed = sly.json_load(ann_fpath)
    ann = sly.Annotation.from_packed(ann_packed, project_meta)
    # ann.normalize_figures()  # @TODO: enaaaable!
    (w, h) = ann.image_size_wh

    gt_boxes, classes_text, classes, masks = [], [], [], []
    for fig in ann['objects']:
        gt = np.zeros((h, w), dtype=np.uint8)  # default bkg
        gt_idx = classes_mapping.get(fig.class_title, None)
        if gt_idx is None:
            raise RuntimeError('Missing class mapping (title to index). Class {}.'.format(fig.class_title))
        fig.draw(gt, 1)
        if np.sum(gt) > 0:
            masks.append(gt)
            xmin, ymin, xmax, ymax = get_bbox(gt)
            gt_boxes.append([ymin / h, xmin / w, ymax / h, xmax / w])
            classes_text.append(fig.class_title.encode('utf8'))  # List of string class name of bounding box (1 per box)
            classes.append(gt_idx)  # List of integer class id of bounding box (1 per box)

    num_boxes = len(gt_boxes)
    if len(masks) > 0:
        masks_np = np.stack(masks)
    else:
        masks_np = np.zeros((0, h, w), dtype=np.float32)
        gt_boxes = np.zeros((0, 4), dtype=np.float32)

    out_gt_boxes = np.array(gt_boxes).astype('float32')
    out_classes = np.array(classes).astype('int64')
    out_num_boxes = np.array([num_boxes]).astype('int32')[0]
    out_masks_np = np.array(masks_np).astype('float32')

    # for x in (out_gt_boxes, out_classes, out_num_boxes, out_masks_np):
    #     print('{} {}'.format(x.shape, x.dtype))

    return out_gt_boxes, out_classes, out_num_boxes, out_masks_np

    # ops.convert_to_tensor(len(gt_boxes), dtype=dtypes.int32)
    # gt_boxes = ops.convert_to_tensor(gt_boxes, dtype=dtypes.float32)
    # classes = ops.convert_to_tensor(classes, dtype=dtypes.int64)
    # return np.array(gt_boxes).astype('float32'), np.array(classes).astype('int64'), \
    #        np.array([num_boxes]).astype('int32')[0], np.array(masks_np).astype('float32')


def read_supervisely_data(sample, classes_mapping, project_meta):
    img_filepath, ann_filepath = sample[0], sample[1]

    image = read_images_from_disk(img_filepath)
    train_tensor = dict()

    def load_ann_fn(x):
        return load_ann(x, classes_mapping=classes_mapping, project_meta=project_meta)
    gt_boxes, classes, num_boxes, masks = tf.py_func(load_ann_fn, [ann_filepath],
                                              (dtypes.float32, dtypes.int64, dtypes.int32, dtypes.float32), stateful=False)
    train_tensor[fields.InputDataFields.image] = image
    train_tensor[fields.InputDataFields.source_id] = img_filepath
    train_tensor[fields.InputDataFields.key] = img_filepath
    train_tensor[fields.InputDataFields.filename] = img_filepath
    train_tensor[fields.InputDataFields.groundtruth_boxes] = gt_boxes
    train_tensor[fields.InputDataFields.num_groundtruth_boxes] = num_boxes
    masks.set_shape([None, None, None])
    train_tensor[fields.InputDataFields.groundtruth_instance_masks] = masks
    train_tensor[fields.InputDataFields.groundtruth_classes] = classes
    train_tensor[fields.InputDataFields.image].set_shape([None, None, 3])
    return train_tensor


def build_dataset(data_dict):
    samples = [[descr.img_path, descr.ann_path] for descr in data_dict['samples']]
    samples_dataset = tf.data.Dataset.from_tensor_slices(samples).repeat()

    def sup_decod_fn(x):
        return read_supervisely_data(x, classes_mapping=data_dict['classes_mapping'],
                                     project_meta=data_dict['project_meta'])
    # tensor_dataset = samples_dataset.apply(tf.contrib.data.parallel_interleave(sup_decod_fn, cycle_length=1, sloppy=True))
    tensor_dataset = samples_dataset.map(sup_decod_fn, num_parallel_calls=1)
    return tensor_dataset.prefetch(1)

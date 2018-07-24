# coding: utf-8

import math
import os.path as osp
import tensorflow as tf
import numpy as np

import supervisely_lib as sly
from supervisely_lib import FigureBitmap, FigClasses
from supervisely_lib import logger
from supervisely_lib.utils.jsonschema import MultiTypeValidator
from object_detection.utils import ops as utils_ops


class SettingsValidator:
    validator = MultiTypeValidator('/workdir/src/schemas.json')

    @classmethod
    def validate_train_cfg(cls, config):
        # store all possible requirements in schema, including size % 16 etc
        cls.validator.val('training_config', config)

    @classmethod
    def validate_inference_cfg(cls, config):
        # store all possible requirements in schema
        cls.validator.val('inference_config', config)


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


class EvalPlanner:
    def __init__(self, epochs, val_every):
        self.epochs = epochs
        self.val_every = val_every
        self.total_val_cnt = self.validations_cnt(epochs, val_every)
        self._val_cnt = 0

    @property
    def performed_val_cnt(self):
        return self._val_cnt

    @staticmethod
    def validations_cnt(ep_float, val_every):
        res = math.floor(ep_float / val_every + 1e-9)
        return res

    def validation_performed(self):
        self._val_cnt += 1

    def need_validation(self, epoch_flt):
        req_val_cnt = self.validations_cnt(epoch_flt, self.val_every)
        need_val = req_val_cnt > self._val_cnt
        return need_val


def create_output_classes(in_project_classes):
    in_project_titles = sorted((x['title'] for x in in_project_classes))

    class_title_to_idx = {}
    for i, title in enumerate(in_project_titles):
        class_title_to_idx[title] = i + 1  # usually bkg_color is 0

    if len(set(class_title_to_idx.values())) != len(class_title_to_idx):
        raise RuntimeError('Unable to construct internal color mapping for classes.')

    # determine out classes
    out_classes = FigClasses()

    for in_class in in_project_classes:
        title = in_class['title']
        out_classes.add({
            'title': title,
            'shape': 'bitmap',
            'color': in_class['color'],
        })

    return class_title_to_idx, out_classes


def create_detection_graph(model_dirpath):
    fpath = osp.join(model_dirpath, 'model.pb')
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(fpath, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    logger.info('Restored model weights from training.')
    return detection_graph


def inverse_mapping(mapping):
    new_map = {}
    for k, v in mapping.items():
        new_map[v] = k
    return new_map


def get_output_dict(image, detection_graph, sess):
    with detection_graph.as_default():
        ops = tf.get_default_graph().get_operations()
        all_tensor_names = {output.name for op in ops for output in op.outputs}
        tensor_dict = {}
        for key in [
            'num_detections', 'detection_boxes', 'detection_scores',
            'detection_classes', 'detection_masks'
        ]:
            tensor_name = key + ':0'
            if tensor_name in all_tensor_names:
                tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                    tensor_name)
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        # real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        real_num_detection = tf.cast(100, tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_scores = tf.slice(tensor_dict['detection_scores'],  [0, 0], [-1, real_num_detection])
        detection_classes = tf.slice(tensor_dict['detection_classes'],  [0, 0], [-1, real_num_detection])
        tensor_dict['detection_scores'] = detection_scores
        tensor_dict['detection_classes'] = detection_classes
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[1], image.shape[2])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
        image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
        # Run inference
        output_dict = sess.run(tensor_dict,
                               feed_dict={image_tensor: image})

        # all outputs are float32 numpy arrays, so convert types as appropriate
        output_dict['num_detections'] = int(output_dict['num_detections'][0])
        output_dict['detection_classes'] = output_dict[
            'detection_classes'][0].astype(np.uint8)
        output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
        output_dict['detection_scores'] = output_dict['detection_scores'][0]
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict



def masks_detection_to_sly_bitmaps(inverse_mapping, net_out, img_shape, min_score_thresold):
    out_figures = []
    img_wh = img_shape[1::-1]
    classes = net_out['detection_classes']
    scores = net_out['detection_scores']
    masks = net_out['detection_masks']
    for mask, class_id in zip(masks[scores > min_score_thresold], classes[scores > min_score_thresold]):
        class_pred_mask = mask == 1
        cls_name = inverse_mapping[int(class_id)]
        new_objs = FigureBitmap.from_mask(cls_name, img_wh, origin=(0, 0), mask=class_pred_mask)
        out_figures.extend(new_objs)
    return out_figures

# coding: utf-8

import cv2
import numpy as np
import torch
import torch.nn.functional as functional
import supervisely_lib as sly
from supervisely_lib import logger
from supervisely_lib.utils.pytorch import cuda_variable

from common import create_model, TrainConfigRW, WeightsRW
from dataset import input_image_normalizer


class UnetV2FastApplier:
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
        self.model = create_model(n_cls=len(self.train_classes), device_ids=self.device_ids)

        self.model = WeightsRW(model_dir).load_strictly(self.model)
        self.model.eval()
        logger.info('Weights are loaded.')

    def __init__(self, settings):
        logger.info('Will init all required to inference.')

        self.source_gpu_device = settings['device_id']
        self._load_train_config()
        self._construct_and_fill_model()
        logger.info('Model is ready to inference.')

    def inference(self, img):
        h, w = img.shape[:2]
        x = cv2.resize(img, tuple(self.input_size_wh))
        x = input_image_normalizer(x)
        x = torch.stack([x], 0)  # add dim #0 (batch size 1)
        x = cuda_variable(x, volatile=True)

        output = self.model(x)
        output = functional.softmax(output, dim=1)
        output = output.data.cpu().numpy()[0]  # from batch to 3d

        pred = np.transpose(output, (1, 2, 0))
        pred = cv2.resize(pred, (w, h), cv2.INTER_LINEAR)

        pred_cls_idx = np.argmax(pred, axis=2)
        res_figures = sly.prediction_to_sly_bitmaps(self.out_class_mapping, pred_cls_idx)

        res_ann = sly.Annotation.new_with_objects((w, h), res_figures)
        return res_ann

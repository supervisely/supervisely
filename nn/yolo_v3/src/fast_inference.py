# coding: utf-8

import os
from os.path import join

import cv2
import supervisely_lib as sly
from supervisely_lib import logger

from common import TrainConfigRW, yolo_preds_to_sly_rects
from darknet_utils import load_net, detect


class YOLOFastApplier:
    def _load_train_config(self):
        model_dir = sly.TaskPaths(determine_in_project=False).model_dir
        train_config_rw = TrainConfigRW(model_dir)
        if not train_config_rw.train_config_exists:
            raise RuntimeError('Unable to run inference, config from training wasn\'t found.')
        train_config = train_config_rw.load()

        self.train_classes = sly.FigClasses(train_config['out_classes'])
        tr_class_mapping = train_config['class_title_to_idx']

        # create
        rev_mapping = {v: k for k, v in tr_class_mapping.items()}
        self.train_names = [rev_mapping[i] for i in range(len(rev_mapping))]  # ordered

        logger.info('Read model internal class mapping', extra={'class_mapping': tr_class_mapping})
        logger.info('Read model out classes', extra={'classes': self.train_classes.py_container})

    def _construct_and_fill_model(self):
        model_dir = sly.TaskPaths(determine_in_project=False).model_dir
        self.device_ids = sly.remap_gpu_devices([self.source_gpu_device])

        src_train_cfg_path = join(model_dir, 'model.cfg')
        with open(src_train_cfg_path) as f:
            src_config = f.readlines()

        def repl_batch(row):
            if 'batch=' in row:
                return 'batch=1\n'
            if 'subdivisions=' in row:
                return 'subdivisions=1\n'
            return row

        changed_config = [repl_batch(x) for x in src_config]

        inf_cfg_path = join(model_dir, 'inf_model.cfg')
        if not os.path.exists(inf_cfg_path):
            with open(inf_cfg_path, 'w') as f:
                f.writelines(changed_config)

        self.net = load_net(inf_cfg_path.encode('utf-8'),
                            join(model_dir, 'model.weights').encode('utf-8'),
                            0)
        logger.info('Weights are loaded.')

    def __init__(self, settings):
        logger.info('Will init all required to inference.')

        self.source_gpu_device = settings['device_id']
        self.score_thresh = settings['min_score_threshold']
        self.image_path = join(sly.TaskPaths.task_dir, 'last.png')

        self._load_train_config()
        self._construct_and_fill_model()

        logger.info('Model is ready to inference.')

    def inference(self, img):
        h, w = img.shape[:2]

        cv2.imwrite(self.image_path, img[:, :, ::-1])  # ok for sequential calls
        res_detections = detect(
            self.net,
            len(self.train_names),
            self.image_path.encode('utf-8'),
            thresh=self.score_thresh
        )
        res_figures = yolo_preds_to_sly_rects(res_detections, (w, h), self.train_names)
        res_ann = sly.Annotation.new_with_objects((w, h), res_figures)
        return res_ann

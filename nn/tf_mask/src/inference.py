# coding: utf-8

import os
from copy import copy, deepcopy
import shutil

import tensorflow as tf
import cv2
import numpy as np
import supervisely_lib as sly
from supervisely_lib import logger

from common import create_detection_graph, freeze_graph, inverse_mapping, get_output_dict, SettingsValidator, \
    TrainConfigRW, masks_detection_to_sly_bitmaps


class MaskRCNNApplier:
    default_settings = {
        'gpu_devices': [0],
        'min_score_threshold': 0.5,
        'model_classes': {
            'save_classes': '__all__',
            'add_suffix': '_mask_rcnn',
        },
        'existing_objects': {
            'save_classes': [],
            'add_suffix': '',
        },
        'mode': {
            'source': 'full_image'
        },
    }
    default_settings_for_modes = {
        'full_image': {
            'source': 'full_image',
        },
        'roi': {
            'source': 'roi',
            'bounds': {
                'left': '0px',
                'top': '0px',
                'right': '0px',
                'bottom': '0px',
            },
            'save': False,
            'class_name': 'inference_roi'
        },
        'bboxes': {
            'source': 'bboxes',
            'from_classes': '__all__',
            'padding': {
                'left': '0px',
                'top': '0px',
                'right': '0px',
                'bottom': '0px',
            },
            'save': False,
            'add_suffix': '_input_bbox'
        },
    }

    def _determine_settings(self):
        input_config = self.helper.task_settings
        logger.info('Input config', extra={'config': input_config})

        config = deepcopy(self.default_settings)
        if 'mode' in input_config and 'source' in input_config['mode']:
            mode_name = input_config['mode']['source']
            default_mode_settings = self.default_settings_for_modes.get(mode_name, None)
            if default_mode_settings is not None:
                sly.update_recursively(config['mode'], deepcopy(default_mode_settings))
            # config now contains default values for selected mode

        sly.update_recursively(config, input_config)
        logger.info('Full config', extra={'config': config})
        SettingsValidator.validate_inference_cfg(config)
        self.score_thresh = config['min_score_threshold']
        self.config = config

    def _load_train_config(self):
        train_config_rw = TrainConfigRW(self.helper.paths.model_dir)
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
        model_dir = self.helper.paths.model_dir
        self.device_ids = sly.remap_gpu_devices(self.config['gpu_devices'])
        if 'model.pb' not in os.listdir(model_dir):
            logger.info('Freezing training checkpoint!')
            freeze_graph('image_tensor',
                         model_dir + '/model.config',
                         model_dir + '/model_weights/model.ckpt',
                         model_dir)
        self.detection_graph = create_detection_graph(model_dir)
        self.session = tf.Session(graph=self.detection_graph)
        logger.info('Weights are loaded.')

    def _determine_input_data(self):
        project_fs = sly.ProjectFS.from_disk_dir_project(self.helper.paths.project_dir)
        logger.info('Project structure has been read. Samples: {}.'.format(project_fs.pr_structure.image_cnt))
        self.in_project_fs = project_fs

    def __init__(self):
        logger.info('Will init all required to inference.')
        self.helper = sly.TaskHelperInference()

        self._determine_settings()
        self._load_train_config()
        self._construct_and_fill_model()
        self._determine_input_data()

        self.debug_copy_images = os.getenv('DEBUG_COPY_IMAGES') is not None

        logger.info('Model is ready to inference.')

    def _infer_on_img(self, img, _):
        image_np_expanded = np.expand_dims(img, axis=0)
        out_dict = get_output_dict(image_np_expanded, self.detection_graph, self.session)

        res_figures = masks_detection_to_sly_bitmaps(self.inv_mapping, out_dict, img.shape, self.score_thresh)
        return res_figures

    def run_inference(self):
        out_project_fs = copy(self.in_project_fs)
        out_project_fs.root_path = self.helper.paths.results_dir
        out_project_fs.make_dirs()

        inf_feeder = sly.InferenceFeederFactory.create(self.config, self.helper.in_project_meta, self.train_classes)
        out_pr_meta = inf_feeder.out_meta
        out_pr_meta.to_dir(out_project_fs.project_path)

        ia_cnt = out_project_fs.pr_structure.image_cnt
        progress = sly.progress_counter_inference(cnt_imgs=ia_cnt)

        for sample in self.in_project_fs:
            logger.info('Will process image',
                        extra={'dataset_name': sample.ds_name, 'image_name': sample.image_name})
            ann_packed = sly.json_load(sample.ann_path)
            ann = sly.Annotation.from_packed(ann_packed, self.helper.in_project_meta)

            img = cv2.imread(sample.img_path)[:, :, ::-1]
            res_ann = inf_feeder.feed(img, ann, self._infer_on_img)

            out_ann_fpath = out_project_fs.ann_path(sample.ds_name, sample.image_name)
            res_ann_packed = res_ann.pack()
            sly.json_dump(res_ann_packed, out_ann_fpath)

            if self.debug_copy_images:
                out_img_fpath = out_project_fs.img_path(sample.ds_name, sample.image_name)
                sly.ensure_base_path(out_img_fpath)
                shutil.copy(sample.img_path, out_img_fpath)

            progress.iter_done_report()

        sly.report_inference_finished()


def main():
    x = MaskRCNNApplier()
    x.run_inference()


if __name__ == '__main__':
    if os.getenv('DEBUG_LOG_TO_FILE', None):
        sly.add_default_logging_into_file(logger, sly.TaskPaths().debug_dir)
    sly.main_wrapper('MASK_RCNN_INFERENCE', main)

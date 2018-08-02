# coding: utf-8

import os
from os.path import join
from copy import copy, deepcopy
import shutil

import supervisely_lib as sly
from supervisely_lib import logger

from common import SettingsValidator, TrainConfigRW, yolo_preds_to_sly_rects
from darknet_utils import load_net, detect


class YOLOApplier:
    default_settings = {
        'gpu_devices': [0],
        'min_score_threshold': 0.5,
        'model_classes': {
            'save_classes': '__all__',
            'add_suffix': '_yolo',
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
        model_dir = self.helper.paths.model_dir
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
        model_dir = self.helper.paths.model_dir
        self.device_ids = sly.remap_gpu_devices(self.config['gpu_devices'])

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
        # self.meta = load_meta(join(self.helper.paths.model_dir, 'model.names').encode('utf-8'))
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

    def _infer_on_img(self, img, ann):
        res_detections = detect(self.net, len(self.train_names), img.encode('utf-8'), thresh=self.score_thresh)
        res_figures = yolo_preds_to_sly_rects(res_detections, ann.image_size_wh, self.train_names)
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
            logger.trace('Will process image',
                         extra={'dataset_name': sample.ds_name, 'image_name': sample.image_name})
            ann_packed = sly.json_load(sample.ann_path)
            ann = sly.Annotation.from_packed(ann_packed, self.helper.in_project_meta)

            img = sample.img_path
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
    x = YOLOApplier()
    x.run_inference()


if __name__ == '__main__':
    if os.getenv('DEBUG_LOG_TO_FILE', None):
        sly.add_default_logging_into_file(logger, sly.TaskPaths().debug_dir)
    sly.main_wrapper('YOLO_V3_INFERENCE', main)

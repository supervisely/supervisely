import os
from copy import copy, deepcopy
import shutil

import cv2
import numpy as np
import torch
import torch.nn.functional as functional
import supervisely_lib as sly
from supervisely_lib import logger
from supervisely_lib.utils.pytorch import cuda_variable

from common import create_model, SettingsValidator, TrainConfigRW, WeightsRW
from dataset import input_image_normalizer


class UnetV2Applier:
    default_settings = {
        'gpu_devices': [0],
        'model_classes': {
            'save_classes': '__all__',
            'add_suffix': '_unet',
        },
        'existing_objects': {
            'save_classes': ['crack_net'],
            'add_suffix': '',
        },
        'mode': {
            'source': 'full_image'
        },
    }
    default_settings_for_modes = {
        'gpu_devices': [0],
        'model_classes': {
            'save_classes': '__all__',
            'add_suffix': '_unet',
        },
        'existing_objects': {
            'save_classes': [],
            'add_suffix': '',
        },
        'mode': {
            'source': 'full_image'
        },
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
        'mode': {
            'source': 'sliding_window',
            'window': {
                'width': 300,
                'height': 150,
            },
            'min_overlap': {
                'x': 100,
                'y': 100,
            },
            'save': False,
            'class_name': 'sliding_window',
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
        self.config = config

    def _load_train_config(self):
        train_config_rw = TrainConfigRW(self.helper.paths.model_dir)
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
        self.device_ids = sly.remap_gpu_devices(self.config['gpu_devices'])
        self.model = create_model(n_cls=len(self.train_classes), device_ids=self.device_ids)

        self.model = WeightsRW(self.helper.paths.model_dir).load_strictly(self.model)
        self.model.eval()
        logger.info('Weights are loaded.')

    def _determine_input_data(self):
        project_fs = sly.ProjectFS.from_disk_dir_project(self.helper.paths.project_dir)
        logger.info('Project structure has been read. Samples: {}.'.format(project_fs.pr_structure.image_cnt))
        self.in_project_fs = project_fs

        self.inf_feeder = sly.InferenceFeederFactory.create(
            self.config, self.helper.in_project_meta, self.train_classes
        )
        self._postproc = lambda a, b: (a, b)

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

        res = self._postproc(self.out_class_mapping, pred)
        return res
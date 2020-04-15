# coding: utf-8

import os
import json
import shutil

import supervisely_lib as sly
from supervisely_lib.task.paths import TaskPaths
from supervisely_lib.io.json import dump_json_file

from worker.task_dockerized import TaskDockerized, TaskStep
from worker import constants


class TaskInferenceRPC(TaskDockerized):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.docker_runtime = 'nvidia'

        #@TODO: need refatoring
        self.entrypoint = "/workdir/src/deploy.py"
        if self.info['task_type'] == 'smarttool':
            self.entrypoint = "/workdir/src/deploy_smart.py"

        self.dir_model = os.path.join(self.dir_task, os.path.basename(TaskPaths.MODEL_DIR))
        self.config_path1 = os.path.join(self.dir_task, os.path.basename(TaskPaths.SETTINGS_PATH))
        self.config_path2 = os.path.join(self.dir_task, os.path.basename(TaskPaths.TASK_CONFIG_PATH))

    def init_additional(self):
        super().init_additional()
        sly.fs.mkdir(self.dir_model)

    def download_step(self):
        if self.info.get('nn_model', None) is None:
            self.logger.critical('TASK_NN_EMPTY')
            raise ValueError('TASK_NN_EMPTY')

        self.data_mgr.download_nn(self.info['nn_model']['title'], self.dir_model)

        #@TODO: only for compatibility with old models
        shutil.move(self.dir_model, self.dir_model + '_delme')
        shutil.move(os.path.join(self.dir_model + '_delme', self.info['nn_model']['title']), self.dir_model)
        sly.fs.remove_dir(self.dir_model + '_delme')

        out_cfg = {
            **self.info['task_settings'],  # settings from server
            'connection': {
                'server_address': constants.SERVER_ADDRESS(),
                'token': constants.TOKEN(),
                'task_id': str(self.info['task_id']),
            },
            'model_settings': self.info['task_settings']
        }

        dump_json_file(out_cfg, self.config_path1)  # Deprecated 'task_settings.json'
        dump_json_file(out_cfg, self.config_path2)  # New style task_config.json
        self.report_step_done(TaskStep.DOWNLOAD)

    def before_main_step(self):
        pass

    def upload_step(self):
        self.report_step_done(TaskStep.UPLOAD)  # stub

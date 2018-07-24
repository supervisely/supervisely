# coding: utf-8

import os.path as osp

import supervisely_lib as sly

from .task_dockerized import TaskDockerized, TaskStep
from . import constants


class TaskInferenceRPC(TaskDockerized):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.docker_runtime = 'nvidia'

        self.dir_model = osp.join(self.dir_task, 'model')
        self.config_path = osp.join(self.dir_task, 'task_settings.json')

    def init_additional(self):
        super().init_additional()
        sly.mkdir(self.dir_model)

    def download_step(self):
        if self.info.get('nn_model', None) is None:
            self.logger.critical('TASK_NN_EMPTY')
            raise ValueError('TASK_NN_EMPTY')

        nn_id = self.info['nn_model']['id']
        nn_hash = self.info['nn_model']['hash']
        self.logger.info('DOWNLOAD_NN', extra={'nn_id': nn_id, 'nn_hash': nn_hash})
        self.data_mgr.download_nn(nn_id, nn_hash, self.dir_model)

        out_cfg = {
            **self.info,
            'connection': {
                'server_address': constants.SERVER_ADDRESS,
                'token': constants.TOKEN,
                'task_id': str(self.info['task_id']),
            },
        }
        sly.json_dump(out_cfg, self.config_path)
        self.report_step_done(TaskStep.DOWNLOAD)

    def before_main_step(self):
        pass

    def upload_step(self):
        self.report_step_done(TaskStep.UPLOAD)  # stub

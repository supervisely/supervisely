# coding: utf-8

import os.path as osp

import supervisely_lib as sly
from supervisely_lib import EventType
import supervisely_lib.worker_proto as api_proto

from .task_dockerized import TaskDockerized, TaskStep


class TaskInference(TaskDockerized):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.action_map = {
            str(EventType.MODEL_APPLIED): self.upload_anns
        }
        self.docker_runtime = 'nvidia'

        self.dir_data = osp.join(self.dir_task, 'data')
        self.dir_results = osp.join(self.dir_task, 'results')
        self.dir_model = osp.join(self.dir_task, 'model')
        self.config_path = osp.join(self.dir_task, 'task_settings.json')

    def init_additional(self):
        super().init_additional()
        sly.mkdir(self.dir_data)
        sly.mkdir(self.dir_results)

    def download_step(self):
        if self.info.get('nn_model', None) is None:
            self.logger.critical('TASK_NN_EMPTY')
            raise ValueError('TASK_NN_EMPTY')

        nn_id = self.info['nn_model']['id']
        nn_hash = self.info['nn_model']['hash']
        self.logger.info('DOWNLOAD_NN', extra={'nn_id': nn_id, 'nn_hash': nn_hash})
        self.data_mgr.download_nn(nn_id, nn_hash, self.dir_model)

        self.logger.info("DOWNLOAD_DATA")
        sly.json_dump(self.info['config'], self.config_path)

        pr_info = self.info['project']
        project = api_proto.Project(id=pr_info['id'], title=pr_info['title'])
        datasets = [api_proto.Dataset(id=ds['id'], title=ds['title']) for ds in pr_info['datasets']]
        self.data_mgr.download_project(self.dir_data, project, datasets)

        self.report_step_done(TaskStep.DOWNLOAD)

    def before_main_step(self):
        sly.clean_dir(self.dir_results)

    def upload_step(self):
        self.upload_anns({})

    def upload_anns(self, _):
        self.report_step_done(TaskStep.MAIN)
        res_project_path = osp.join(self.dir_results, self.info['project']['title'])

        pr_id = self.data_mgr.upload_project(res_project_path, self.info['new_title'], no_image_files=True)
        self.logger.info('PROJECT_CREATED', extra={'event_type': EventType.PROJECT_CREATED, 'project_id': pr_id})

        self.report_step_done(TaskStep.UPLOAD)
        return {}

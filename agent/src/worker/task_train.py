# coding: utf-8

import os
import os.path as osp

import supervisely_lib as sly
from supervisely_lib import EventType
import supervisely_lib.worker_proto as api_proto

from .task_dockerized import TaskDockerized, TaskStep


class TaskTrain(TaskDockerized):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.action_map = {
            str(EventType.CHECKPOINT): self.upload_model
        }
        self.docker_runtime = 'nvidia'

        self.dir_data = osp.join(self.dir_task, 'data')
        self.dir_results = osp.join(self.dir_task, 'results')
        self.dir_model = osp.join(self.dir_task, 'model')
        self.dir_tmp = osp.join(self.dir_task, 'tmp')
        self.config_path = osp.join(self.dir_task, 'task_settings.json')

    def init_additional(self):
        super().init_additional()
        sly.mkdir(self.dir_data)
        sly.mkdir(self.dir_tmp)
        sly.mkdir(self.dir_results)

    def download_step(self):
        self.logger.info("DOWNLOAD_DATA")
        sly.json_dump(self.info['config'], self.config_path)

        pr_info = self.info['project']
        project = api_proto.Project(id=pr_info['id'], title=pr_info['title'])
        datasets = [api_proto.Dataset(id=ds['id'], title=ds['title']) for ds in pr_info['datasets']]
        self.data_mgr.download_project(self.dir_data, project, datasets)

        if self.info.get('nn_model', None) is not None:
            nn_id = self.info['nn_model']['id']
            nn_hash = self.info['nn_model']['hash']
            self.logger.info('DOWNLOAD_NN', extra={'nn_id': nn_id, 'nn_hash': nn_hash})
            self.data_mgr.download_nn(nn_id, nn_hash, self.dir_model)
        else:
            self.logger.info('Initializing task without source NN.')

        self.report_step_done(TaskStep.DOWNLOAD)

    def before_main_step(self):
        sly.clean_dir(self.dir_results)

    def upload_step(self):
        self.report_step_done(TaskStep.UPLOAD)

    def upload_model(self, extra):
        model_desc = self.api.simple_request('GenerateNewModelId', api_proto.ModelDescription, api_proto.Empty())
        local_extra = {'model_id': model_desc.id, 'model_hash': model_desc.hash}
        self.logger.trace('NEW_MODEL_ID', extra=local_extra)

        cur_checkpoint_dir = os.path.join(self.dir_results, extra['subdir'])

        storage = self.data_mgr.storage.nns
        if storage.check_storage_object(model_desc.hash):
            self.logger.critical('CHECKPOINT_ALREADY_EXISTS', extra=local_extra)
            raise RuntimeError()
        storage.write_object(cur_checkpoint_dir, model_desc.hash)
        model_config_path = osp.join(cur_checkpoint_dir, 'config.json')
        if osp.isfile(model_config_path):
            local_extra['model_config'] = sly.json_load(model_config_path)

        self.logger.info('MODEL_SAVED', extra=local_extra)
        # don't report TaskStep.UPLOAD because there should be multiple uploads

        res = {'model_id': model_desc.id, 'model_hash': model_desc.hash,
               'model_config': local_extra.get('model_config', {})}
        return res

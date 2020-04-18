# coding: utf-8

import os
import os.path as osp
import json
import supervisely_lib as sly
from supervisely_lib.task.paths import TaskPaths

from .task_dockerized import TaskDockerized, TaskStep
from worker import agent_utils
import shutil


class TaskTrain(TaskDockerized):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.action_map = {
            str(sly.EventType.CHECKPOINT): self.upload_model
        }
        self.docker_runtime = 'nvidia'
        self.entrypoint = "/workdir/src/train.py"

        self.dir_data = os.path.join(self.dir_task, os.path.basename(TaskPaths.DATA_DIR))
        self.dir_results = os.path.join(self.dir_task, os.path.basename(TaskPaths.RESULTS_DIR))
        self.dir_model = os.path.join(self.dir_task, os.path.basename(TaskPaths.MODEL_DIR))
        self.config_path1 = os.path.join(self.dir_task, os.path.basename(TaskPaths.SETTINGS_PATH))
        self.config_path2 = os.path.join(self.dir_task, os.path.basename(TaskPaths.TASK_CONFIG_PATH))
        self.dir_tmp = osp.join(self.dir_task, 'tmp')

        self.last_checkpoint = {}

    def init_additional(self):
        super().init_additional()

        if not self.data_mgr.has_nn_storage():
            raise ValueError('Host agent has local neural networks storage disabled. Training without local storage ' +
                             'is not supported because there is no space to keep checkpoints.')

        sly.fs.mkdir(self.dir_data)
        sly.fs.mkdir(self.dir_tmp)
        sly.fs.mkdir(self.dir_results)

    def download_step(self):
        self.logger.info("DOWNLOAD_DATA")

        sly.io.json.dump_json_file(self.info['config'], self.config_path1)  # Deprecated 'task_settings.json'
        sly.io.json.dump_json_file(self.info['config'], self.config_path2)  # New style task_config.json

        if len(self.info['projects']) != 1:
            raise ValueError("Config contains {} projects. Training works only with single project.".format(len(self.info['projects'])))

        project_name = agent_utils.get_single_item_or_die(self.info, 'projects', 'config')['title']
        self.data_mgr.download_project(self.dir_data, project_name)

        model = agent_utils.get_single_item_or_die(self.info, 'models', 'config')
        self.data_mgr.download_nn(model['title'], self.dir_model)

        # @TODO: only for compatibility with old models
        shutil.move(self.dir_model, self.dir_model + '_delme')
        shutil.move(os.path.join(self.dir_model + '_delme', model['title']), self.dir_model)
        sly.fs.remove_dir(self.dir_model + '_delme')

        self.report_step_done(TaskStep.DOWNLOAD)

    def before_main_step(self):
        sly.fs.clean_dir(self.dir_results)

    def upload_step(self):
        if 'id' not in self.last_checkpoint or 'hash' not in self.last_checkpoint:
            raise RuntimeError('No checkpoints produced during training')
        self.data_mgr.upload_nn(self.last_checkpoint['id'], self.last_checkpoint['hash'])
        self.report_step_done(TaskStep.UPLOAD)

    def upload_model(self, extra):
        model = self.public_api.model.generate_hash(self.info['task_id'])
        model_id = model['id']
        model_hash = model['hash']

        log_extra = {'model_id': model_id, 'model_hash': model_hash}
        self.logger.trace('NEW_MODEL_ID', extra=log_extra)

        cur_checkpoint_dir = os.path.join(self.dir_results, extra['subdir'])

        storage = self.data_mgr.storage.nns
        if storage.check_storage_object(model_hash):
            self.logger.critical('CHECKPOINT_HASH_ALREADY_EXISTS')
            raise RuntimeError()
        storage.write_object(cur_checkpoint_dir, model_hash)
        model_config_path = osp.join(cur_checkpoint_dir, 'config.json')
        if osp.isfile(model_config_path):
            log_extra['model_config'] = json.load(open(model_config_path, 'r'))
        else:
            log_extra['model_config'] = {}

        self.logger.info('MODEL_SAVED', extra=log_extra)
        # don't report TaskStep.UPLOAD because there should be multiple uploads

        self.last_checkpoint["id"] = model_id
        self.last_checkpoint["hash"] = model_hash
        return log_extra

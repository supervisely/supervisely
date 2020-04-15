# coding: utf-8

import os
import json
import shutil

import supervisely_lib as sly
from supervisely_lib.task.paths import TaskPaths
from supervisely_lib.io.json import dump_json_file

from worker.task_dockerized import TaskDockerized, TaskStep
from worker import agent_utils


class TaskInference(TaskDockerized):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.action_map = {
            str(sly.EventType.MODEL_APPLIED): self.upload_anns
        }
        self.docker_runtime = 'nvidia'
        self.entrypoint = "/workdir/src/inference.py"

        self.dir_data = os.path.join(self.dir_task, os.path.basename(TaskPaths.DATA_DIR))
        self.dir_results = os.path.join(self.dir_task, os.path.basename(TaskPaths.RESULTS_DIR))
        self.dir_model = os.path.join(self.dir_task, os.path.basename(TaskPaths.MODEL_DIR))
        self.config_path1 = os.path.join(self.dir_task, os.path.basename(TaskPaths.SETTINGS_PATH))
        self.config_path2 = os.path.join(self.dir_task, os.path.basename(TaskPaths.TASK_CONFIG_PATH))

    def init_additional(self):
        super().init_additional()
        sly.fs.mkdir(self.dir_data)
        sly.fs.mkdir(self.dir_model)
        sly.fs.mkdir(self.dir_results)

    def download_step(self):
        self.logger.info("DOWNLOAD_DATA")
        dump_json_file(self.info['config'], self.config_path1)
        dump_json_file(self.info['config'], self.config_path2)

        model = agent_utils.get_single_item_or_die(self.info, 'models', 'config')
        self.data_mgr.download_nn(model['title'], self.dir_model)

        project_name = agent_utils.get_single_item_or_die(self.info, 'projects', 'config')['title']
        self.data_mgr.download_project(self.dir_data, project_name)

        #@TODO: only for compatibility with old models
        shutil.move(self.dir_model, self.dir_model + '_delme')
        shutil.move(os.path.join(self.dir_model + '_delme', model['title']), self.dir_model)
        sly.fs.remove_dir(self.dir_model + '_delme')

        self.report_step_done(TaskStep.DOWNLOAD)

    def before_main_step(self):
        sly.fs.clean_dir(self.dir_results)

    def upload_step(self):
        self.upload_anns({})

    def upload_anns(self, _):
        self.report_step_done(TaskStep.MAIN)
        self.data_mgr.upload_project(self.dir_results, self.info['projects'][0]['title'], self.info['new_title'])
        self.report_step_done(TaskStep.UPLOAD)
        return {}

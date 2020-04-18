# coding: utf-8

import os
import json
import supervisely_lib as sly
from supervisely_lib.task.paths import TaskPaths
from .task_dockerized import TaskDockerized, TaskStep
from supervisely_lib.io.json import dump_json_file


class TaskImport(TaskDockerized):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.action_map = {
            str(sly.EventType.IMPORT_APPLIED): self.upload_result_project
        }

        self.dir_data = os.path.join(self.dir_task, os.path.basename(TaskPaths.DATA_DIR))
        self.dir_results = os.path.join(self.dir_task, os.path.basename(TaskPaths.RESULTS_DIR))
        self.config_path1 = os.path.join(self.dir_task, os.path.basename(TaskPaths.SETTINGS_PATH))
        self.config_path2 = os.path.join(self.dir_task, os.path.basename(TaskPaths.TASK_CONFIG_PATH))

    def init_additional(self):
        super().init_additional()
        sly.fs.mkdir(self.dir_data)
        sly.fs.mkdir(self.dir_results)

    def make_human_config(self):
        return {
            'res_names': {'project': self.info['project_name']},
            'preset': self.info['preset'],
            'options': self.info['options'],
            'append_to_existing_project': self.info['append_to_existing_project'],
            'task_id': self.info['task_id'],
            'server_address': self.info['server_address'],
            'api_token': self.info['api_token'],
            'workspace_id': self.public_api_context['workspace']['id']
        }

    def download_step(self):
        self.logger.info("DOWNLOAD_DATA")
        human_config = self.make_human_config()
        dump_json_file(human_config, self.config_path1)
        dump_json_file(human_config, self.config_path2)
        self.data_mgr.download_import_files(self.info['task_id'], self.dir_data)
        self.report_step_done(TaskStep.DOWNLOAD)

    def before_main_step(self):
        sly.fs.clean_dir(self.dir_results)

    def upload_step(self):
        self.upload_result_project({})

    def upload_result_project(self, _):
        self.report_step_done(TaskStep.MAIN)

        project_info_file = os.path.join(self.dir_results, 'project_info.json')
        if os.path.isfile(project_info_file):
            with open(project_info_file) as json_file:  
                pr_id = json.load(json_file)['project_id']
            self.logger.info('PROJECT_CREATED', extra={'event_type': sly.EventType.PROJECT_CREATED, 'project_id': pr_id})
        else:
            add_to_existing = self.info.get('append_to_existing_project', False)
            self.data_mgr.upload_project(self.dir_results, self.info['project_name'], self.info['project_name'],
                                         add_to_existing=add_to_existing)
        self.report_step_done(TaskStep.UPLOAD)
        return {}

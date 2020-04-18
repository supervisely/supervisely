# coding: utf-8

import os
import json
import supervisely_lib as sly
from supervisely_lib.task.paths import TaskPaths
from .task_dockerized import TaskDockerized, TaskStep


class TaskPlugin(TaskDockerized):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.action_map = {}

        self.dir_data = os.path.join(self.dir_task, os.path.basename(TaskPaths.DATA_DIR))
        self.config_path = os.path.join(self.dir_task, os.path.basename(TaskPaths.TASK_CONFIG_PATH))

    def init_additional(self):
        super().init_additional()
        sly.fs.mkdir(self.dir_data)

    def download_step(self):
        self.logger.info("DOWNLOAD_DATA")
        human_config = self.info
        sly.io.json.dump_json_file(human_config, self.config_path)
        self.report_step_done(TaskStep.DOWNLOAD)

    def before_main_step(self):
        pass

    def upload_step(self):
        self.upload_result_project({})

    def upload_result_project(self, _):
        self.report_step_done(TaskStep.MAIN)
        self.report_step_done(TaskStep.UPLOAD)
        return {}

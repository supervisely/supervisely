# coding: utf-8

from distutils.dir_util import copy_tree

import supervisely_lib as sly

from . import constants
from .task_dockerized import TaskStep
from .task_import import TaskImport


class TaskImportLocal(TaskImport):
    def download_step(self):
        self.logger.info("DOWNLOAD_DATA")
        sly.json_dump(self.human_config, self.config_path)
        copy_tree(constants.AGENT_IMPORT_DIR, self.dir_data)
        self.report_step_done(TaskStep.DOWNLOAD)

# coding: utf-8

from distutils.dir_util import copy_tree
import json

from worker import constants
from worker.task_dockerized import TaskStep
from worker.task_import import TaskImport


class TaskImportLocal(TaskImport):
    def download_step(self):
        self.logger.info("DOWNLOAD_DATA")
        json.dump(self.human_config, open(self.config_path1, 'w'))
        json.dump(self.human_config, open(self.config_path2, 'w'))
        copy_tree(constants.AGENT_IMPORT_DIR(), self.dir_data)
        self.report_step_done(TaskStep.DOWNLOAD)

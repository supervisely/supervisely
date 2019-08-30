# coding: utf-8

import json

from worker import constants
from worker.task_dockerized import TaskStep
from worker.task_import import TaskImport

from supervisely_lib.io.fs import hardlink_or_copy_tree


class TaskImportLocal(TaskImport):
    def download_step(self):
        self.logger.info("DOWNLOAD_DATA")
        human_config = self.make_human_config()
        json.dump(human_config, open(self.config_path1, 'w'))
        json.dump(human_config, open(self.config_path2, 'w'))
        hardlink_or_copy_tree(constants.AGENT_IMPORT_DIR(), self.dir_data)
        self.report_step_done(TaskStep.DOWNLOAD)

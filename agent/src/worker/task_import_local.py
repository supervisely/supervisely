# coding: utf-8

import json

from worker import constants
from worker.task_dockerized import TaskStep
from worker.task_import import TaskImport

from supervisely_lib.io.fs import hardlink_or_copy_tree
from supervisely_lib.io.json import dump_json_file


class TaskImportLocal(TaskImport):
    def download_step(self):
        self.logger.info("DOWNLOAD_DATA")
        human_config = self.make_human_config()
        dump_json_file(human_config, self.config_path1)
        dump_json_file(human_config, self.config_path2)
        hardlink_or_copy_tree(constants.AGENT_IMPORT_DIR(), self.dir_data)
        self.report_step_done(TaskStep.DOWNLOAD)

# coding: utf-8

import os
import json
import supervisely_lib as sly
from worker import constants
from supervisely_lib.task.paths import TaskPaths
from worker.task_plugin import TaskPlugin


class TaskPluginImportLocal(TaskPlugin):
    def download_step(self):
        super().download_step()

        # upload local files to server
        sly.fs.log_tree(constants.AGENT_IMPORT_DIR(), self.logger)
        absolute_paths = sly.fs.list_files_recursively(constants.AGENT_IMPORT_DIR())

        relative_paths = [path.replace(constants.AGENT_IMPORT_DIR().rstrip("/"), "") for path in absolute_paths]

        progress = sly.Progress('Upload files', len(absolute_paths), self.logger)
        self.data_mgr.public_api.task.upload_files(self.info['task_id'],
                                                   absolute_paths,
                                                   relative_paths,
                                                   progress.iters_done_report)


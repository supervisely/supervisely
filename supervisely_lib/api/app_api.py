# coding: utf-8

import os
from supervisely_lib.api.module_api import ApiField
from supervisely_lib.api.task_api import TaskApi
from supervisely_lib._utils import take_with_default
from supervisely_lib.app.constants import DATA, STATE, CONTEXT, TEMPLATE
from supervisely_lib.io.fs import ensure_base_path
from supervisely_lib.task.progress import Progress
from supervisely_lib._utils import sizeof_fmt
from supervisely_lib import logger


class AppApi(TaskApi):
    def run_dtl(self, workspace_id, dtl_graph, agent_id=None):
        raise RuntimeError("Method is unavailable")

    def _run_plugin_task(self, task_type, agent_id, plugin_id, version, config, input_projects, input_models,
                         result_name):
        raise RuntimeError("Method is unavailable")

    def run_train(self, agent_id, input_project_id, input_model_id, result_nn_name, train_config=None):
        raise RuntimeError("Method is unavailable")

    def run_inference(self, agent_id, input_project_id, input_model_id, result_project_name, inference_config=None):
        raise RuntimeError("Method is unavailable")

    def get_training_metrics(self, task_id):
        raise RuntimeError("Method is unavailable")

    def deploy_model(self, agent_id, model_id):
        raise RuntimeError("Method is unavailable")

    def get_import_files_list(self, id):
        raise RuntimeError("Method is unavailable")

    def download_import_file(self, id, file_path, save_path):
        raise RuntimeError("Method is unavailable")

    def create_task_detached(self, workspace_id, task_type: str=None):
        raise RuntimeError("Method is unavailable")

    def upload_files(self, task_id, abs_paths, names, progress_cb=None):
        raise RuntimeError("Method is unavailable")

    def initialize(self, task_id, template, data=None, state=None):
        d = take_with_default(data, {})
        s = take_with_default(state, {})
        fields = [{"field": TEMPLATE, "payload": template}, {"field": DATA, "payload": d}, {"field": STATE, "payload": s}]
        resp = self._api.task.set_fields(task_id, fields)
        return resp

    def get_url(self, task_id):
        return os.path.join(self._api.server_address, "apps/sessions", str(task_id))

    def download_git_file(self, app_id, version, file_path, save_path):
        raise NotImplementedError()

    def download_git_archive(self, ecosystem_item_id, app_id, version, save_path, log_progress=True, ext_logger=None):
        payload = {
            ApiField.ECOSYSTEM_ITEM_ID: ecosystem_item_id,
            ApiField.VERSION: version,
            "isArchive": True
        }
        if app_id is not None:
            payload[ApiField.APP_ID] = app_id

        response = self._api.post('ecosystem.file.download', payload, stream=True)
        if log_progress:
            if ext_logger is None:
                ext_logger = logger

            length = -1
            # Content-Length
            if "Content-Length" in response.headers:
                length = int(response.headers['Content-Length'])
            progress = Progress("Downloading: ", length, ext_logger=ext_logger, is_size=True)

        mb1 = 1024 * 1024
        ensure_base_path(save_path)
        with open(save_path, 'wb') as fd:
            log_size = 0
            for chunk in response.iter_content(chunk_size=mb1):
                fd.write(chunk)
                log_size += len(chunk)
                if log_progress and log_size > mb1:
                    progress.iters_done_report(log_size)
                    log_size = 0

    def get_info(self, module_id):
        response = self._api.post('ecosystem.info', {ApiField.ID: module_id})
        return response.json()
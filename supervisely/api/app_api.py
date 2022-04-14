# coding: utf-8

import os
from supervisely.api.module_api import ApiField
from supervisely.api.task_api import TaskApi
from supervisely._utils import take_with_default

# from supervisely.app.constants import DATA, STATE, CONTEXT, TEMPLATE
STATE = "state"
DATA = "data"
TEMPLATE = "template"

from supervisely.io.fs import ensure_base_path
from supervisely.task.progress import Progress
from supervisely._utils import sizeof_fmt
from supervisely import logger


class AppApi(TaskApi):
    @staticmethod
    def info_sequence():
        return [ApiField.ID,
                ApiField.CREATED_BY_ID,
                ApiField.MODULE_ID,
                ApiField.DISABLED,
                ApiField.USER_LOGIN,
                ApiField.CONFIG,
                ApiField.NAME,
                ApiField.SLUG,
                ApiField.IS_SHARED,
                ApiField.TASKS,
                ApiField.REPO,
                ApiField.TEAM_ID]

    @staticmethod
    def info_tuple_name():
        return 'AppInfo'

    def _convert_json_info(self, info: dict, skip_missing=True):
        return super(TaskApi, self)._convert_json_info(info, skip_missing=skip_missing)

    def get_info_by_id(self, id):
        '''
        :param id: int
        :return: application info by numeric id
        '''
        return self._get_info_by_id(id, 'apps.info')

    def get_list(self,
                 team_id,
                 filter=None,
                 context=None,
                 repository_key=None,
                 show_disabled=False,
                 integrated_into=None,
                 session_tags=None,
                 only_running=False,
                 with_shared=True):

        return self.get_list_all_pages(method='apps.list', data={
            "teamId": team_id,
            "filter": take_with_default(filter, []),  # for example [{"field": "id", "operator": "=", "value": None}]
            "context": take_with_default(context, []),  # for example ["images_project"]
            "repositoryKey": repository_key,
            "integratedInto": take_with_default(integrated_into, []),  # for example ["image_annotation_tool"]
            "sessionTags": take_with_default(session_tags, []),  # for example ["string"]
            "onlyRunning": only_running,
            "showDisabled": show_disabled,
            "withShared": with_shared
        })

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

    def create_task_detached(self, workspace_id, task_type: str = None):
        raise RuntimeError("Method is unavailable")

    def upload_files(self, task_id, abs_paths, names, progress_cb=None):
        raise RuntimeError("Method is unavailable")

    def initialize(self, task_id, template, data=None, state=None):
        d = take_with_default(data, {})
        if "notifyDialog" not in d:
            d["notifyDialog"] = None
        if "scrollIntoView" not in d:
            d["scrollIntoView"] = None

        s = take_with_default(state, {})
        fields = [{"field": TEMPLATE, "payload": template}, {"field": DATA, "payload": d},
                  {"field": STATE, "payload": s}]
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

    def get_info(self, module_id, version=None):
        data = {ApiField.ID: module_id}
        if version is not None:
            data[ApiField.VERSION] = version
        response = self._api.post('ecosystem.info', data)
        return response.json()

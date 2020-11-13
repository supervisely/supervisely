# coding: utf-8

import os
from worker import constants
import requests
import tarfile
import shutil
import json
from pathlib import Path

import supervisely_lib as sly
from .task_dockerized import TaskDockerized
from supervisely_lib.io.json import dump_json_file
from supervisely_lib.io.json import flatten_json, modify_keys
from supervisely_lib.api.api import SUPERVISELY_TASK_ID
from supervisely_lib.api.api import Api
from supervisely_lib.io.fs import ensure_base_path, silent_remove, get_file_name, remove_dir, get_subdirs

_ISOLATE = "isolate"


class TaskApp(TaskDockerized):
    def __init__(self, *args, **kwargs):
        self.app_config = None
        self.dir_task_src = None
        self.dir_task_container = None
        self.dir_task_src_container = None
        self._exec_id = None
        self.app_info = None
        super().__init__(*args, **kwargs)

    def init_task_dir(self):
        # agent container paths
        self.dir_task = os.path.join(constants.AGENT_APP_SESSIONS_DIR(), str(self.info['task_id']))
        self.dir_task_src = os.path.join(self.dir_task, 'repo')
        # host paths
        self.dir_task_host = os.path.join(constants.AGENT_APP_SESSIONS_DIR_HOST(), str(self.info['task_id']))
        # task container path
        self.dir_task_container = os.path.join("/sessions", str(self.info['task_id']))
        self.dir_task_src_container = os.path.join(self.dir_task_container, 'repo')
        self.app_info = self.info["appInfo"]

    def download_or_get_repo(self):
        git_url = self.app_info["githubUrl"]
        version = self.app_info.get("version", "master")

        already_downloaded = False
        path_cache = None
        if version != "master":
            path_cache = os.path.join(constants.APPS_STORAGE_DIR(), *Path(git_url.replace(".git", "")).parts[1:],
                                      version)
            already_downloaded = sly.fs.dir_exists(path_cache)

        if already_downloaded is False:
            self.logger.info("Git repo will be downloaded")

            api = Api(self.info['server_address'], self.info['api_token'])
            tar_path = os.path.join(self.dir_task_src, 'repo.tar.gz')
            api.app.download_git_archive(self.app_info["moduleId"],
                                         self.app_info["id"],
                                         version,
                                         tar_path,
                                         log_progress=True,
                                         ext_logger=self.logger)
            with tarfile.open(tar_path) as archive:
                archive.extractall(self.dir_task_src)

            subdirs = get_subdirs(self.dir_task_src)
            if len(subdirs) != 1:
                raise RuntimeError("Repo is downloaded and extracted, but resulting directory not found")
            extracted_path = os.path.join(self.dir_task_src, subdirs[0])

            for filename in os.listdir(extracted_path):
                shutil.move(os.path.join(extracted_path, filename), os.path.join(self.dir_task_src, filename))
            remove_dir(extracted_path)
            silent_remove(tar_path)

            #git.download(git_url, self.dir_task_src, github_token, version)
            if path_cache is not None:
                shutil.copytree(self.dir_task_src, path_cache)
        else:
            self.logger.info("Git repo already exists")
            shutil.copytree(path_cache, self.dir_task_src)

    def init_docker_image(self):
        self.download_or_get_repo()
        api = Api(self.info['server_address'], self.info['api_token'])
        module_id = self.info["appInfo"]["moduleId"]
        self.logger.info("APP moduleId == {} in ecosystem".format(module_id))
        self.app_config = api.app.get_info(module_id)["config"]
        #self.app_config = sly.io.json.load_json_file(os.path.join(self.dir_task_src, 'config.json'))
        self.read_dockerimage_from_config()
        super().init_docker_image()

    def read_dockerimage_from_config(self):
        self.info['app_info'] = self.app_config
        try:
            self.info['docker_image'] = self.app_config['docker_image']
            if constants.APP_DEBUG_DOCKER_IMAGE() is not None:
                self.logger.info("APP DEBUG MODE: docker image {!r} is replaced to {!r}"
                                 .format(self.info['docker_image'], constants.APP_DEBUG_DOCKER_IMAGE()))
                self.info['docker_image'] = constants.APP_DEBUG_DOCKER_IMAGE()

        except KeyError as e:
            self.logger.critical('File \"config.json\" does not contain \"docker_image\" field')

    def is_isolate(self):
        if self.app_config is None:
            raise RuntimeError("App config is not initialized")
        return self.app_config.get(_ISOLATE, True)

    def _get_task_volumes(self):
        res = {}
        if self.is_isolate():
            res[self.dir_task_host] = {'bind': self.dir_task_container, 'mode': 'rw'}
        else:
            res[constants.AGENT_APP_SESSIONS_DIR_HOST()] = {'bind': sly.app.SHARED_DATA, 'mode': 'rw'}

        #@TODO: cache ecoystem item in agent (remove copypaste download-extract logic)
        # state = self.info.get('state', {})
        # if "slyEcosystemItemGitUrl" in state and "slyEcosystemItemId" in state and "slyEcosystemItemVersion" in state:
        #     path_cache = None
        #     if state["slyEcosystemItemVersion"] != "master":
        #         path_cache = os.path.join(constants.APPS_STORAGE_DIR(),
        #                                   *Path(state["slyEcosystemItemGitUrl"].replace(".git", "")).parts[1:],
        #                                   state["slyEcosystemItemVersion"])
        #         already_downloaded = sly.fs.dir_exists(path_cache)
        #         if already_downloaded:
        #             constants.APPS_STORAGE_DIR
        #             res["/sly_ecosystem_item"] = {'bind': self.dir_task_container, 'mode': 'rw'}
        #     pass

        return res

    def download_step(self):
        pass

    def find_or_run_container(self):
        add_labels = {"sly_app": "1", "app_session_id": str(self.info['task_id'])}
        if self.is_isolate():
            # spawn app session in new container
            add_labels["isolate"] = "1"
            sly.docker_utils.docker_pull_if_needed(self._docker_api, self.docker_image_name, constants.PULL_POLICY(), self.logger)
            self.spawn_container(add_labels=add_labels)
        else:
            # spawn app session in existing container
            add_labels["isolate"] = "0"
            filter = {"ancestor": self.info['docker_image'], "label": ["sly_app=1", "isolate=0"]}
            containers = self.docker_api.containers.list(all=True, filters=filter)
            if len(containers) == 0:
                sly.docker_utils.docker_pull_if_needed(self._docker_api, self.docker_image_name, constants.PULL_POLICY(), self.logger)
                self.spawn_container(add_labels=add_labels)
            else:
                self._container = containers[0]

    def get_spawn_entrypoint(self):
        return ["sh", "-c", "while true; do sleep 30; done;"]

    def exec_command(self, add_envs=None):
        add_envs = sly.take_with_default(add_envs, {})
        main_script_path = os.path.join(self.dir_task_src_container, self.app_config.get('main_script', 'src/main.py'))
        command = "python {}".format(main_script_path)

        self._exec_id = self._docker_api.api.exec_create(self._container.id,
                                                         cmd=command,
                                                         environment={
                                                             'LOG_LEVEL': 'DEBUG',
                                                             'LANG': 'C.UTF-8',
                                                             'PYTHONUNBUFFERED': '1',
                                                             constants._HTTP_PROXY: constants.HTTP_PROXY(),
                                                             constants._HTTPS_PROXY: constants.HTTPS_PROXY(),
                                                             'HOST_TASK_DIR': self.dir_task_host,
                                                             'TASK_ID': self.info['task_id'],
                                                             'SERVER_ADDRESS': self.info['server_address'],
                                                             'API_TOKEN': self.info['api_token'],
                                                             'AGENT_TOKEN': constants.TOKEN(),
                                                             **add_envs
                                                         })
        self._logs_output = self._docker_api.api.exec_start(self._exec_id, stream=True, demux=False)

        #change pulling progress to app progress
        progress_dummy = sly.Progress('Application is started ...', 1, ext_logger=self.logger)
        progress_dummy.iter_done_report()

        self.logger.info("command is running", extra={"command": command})

    def main_step(self):
        self.find_or_run_container()
        self.exec_command(add_envs= self.main_step_envs())
        self.process_logs()
        self.drop_container_and_check_status()

    def upload_step(self):
        pass

    def main_step_envs(self):
        context = self.info.get('context', {})

        context_envs = {}
        if len(context) > 0:
            context_envs = flatten_json(context)
            context_envs = modify_keys(context_envs, prefix="context.")

        modal_state = self.info.get('state', {})
        modal_envs = {}
        if len(modal_state) > 0:
            modal_envs = flatten_json(modal_state)
            modal_envs = modify_keys(modal_envs, prefix="modal.state.")

        envs = {
            "CONTEXT": json.dumps(context),
            "MODAL_STATE": json.dumps(modal_state),
            **modal_envs,
            "USER_ID": context.get("userId"),
            "USER_LOGIN": context.get("userLogin"),
            "TEAM_ID": context.get("teamId"),
            "API_TOKEN": context.get("apiToken"),
            "CONFIG_DIR": self.info["appInfo"].get("configDir", ""),
            **context_envs,
            SUPERVISELY_TASK_ID: str(self.info['task_id']),
        }
        return envs

    def process_logs(self):
        logs_found = False

        def _process_line(log_line):
            #log_line = log_line.decode("utf-8")
            msg, res_log, lvl = self.parse_log_line(log_line)
            output = self.call_event_function(res_log)

            lvl_description = sly.LOGGING_LEVELS.get(lvl, None)
            if lvl_description is not None:
                lvl_int = lvl_description.int
            else:
                lvl_int = sly.LOGGING_LEVELS['INFO'].int
            self.logger.log(lvl_int, msg, extra=res_log)

        #@TODO: parse multiline logs correctly (including exceptions)
        log_line = ""
        for log_line_arr in self._logs_output:
            for log_part in log_line_arr.decode("utf-8").splitlines():
                logs_found = True
                _process_line(log_part)

        if not logs_found:
            self.logger.warn('No logs obtained from container.')  # check if bug occurred

    def _stop_wait_container(self):
        if self.is_isolate():
            return super()._stop_wait_container()
        else:
            return self.exec_stop()

    def exec_stop(self):
        exec_info = self._docker_api.api.exec_inspect(self._exec_id)
        if exec_info["Running"] == True:
            pid = exec_info["Pid"]
            self._container.exec_run(cmd="kill {}".format(pid))
        else:
            return

    def _drop_container(self):
        if self.is_isolate():
           super()._drop_container()
        else:
           self.exec_stop()

    def drop_container_and_check_status(self):
        status = self._docker_api.api.exec_inspect(self._exec_id)['ExitCode']
        if self.is_isolate():
            self._drop_container()
        self.logger.debug('Task container finished with status: {}'.format(str(status)))
        if status != 0:
            raise RuntimeError('Task container finished with non-zero status: {}'.format(str(status)))

# coding: utf-8

import os
from worker import constants
import requests
import tarfile
import shutil
import json
from pathlib import Path
from packaging import version
from version_parser import Version

import supervisely_lib as sly
from .task_dockerized import TaskDockerized
from supervisely_lib.io.json import dump_json_file
from supervisely_lib.io.json import flatten_json, modify_keys
from supervisely_lib.api.api import SUPERVISELY_TASK_ID
from supervisely_lib.api.api import Api
from supervisely_lib.io.fs import ensure_base_path, silent_remove, get_file_name, remove_dir, get_subdirs, file_exists

_ISOLATE = "isolate"
_LINUX_DEFAULT_PIP_CACHE_DIR = "/root/.cache/pip"


class TaskApp(TaskDockerized):
    def __init__(self, *args, **kwargs):
        self.app_config = None
        self.dir_task_src = None
        self.dir_task_container = None
        self.dir_task_src_container = None
        self.dir_apps_cache_host = None
        self.dir_apps_cache_container = None
        self._exec_id = None
        self.app_info = None
        self._path_cache_host = None
        self._need_sync_pip_cache = False
        self._requirements_path_relative = None
        super().__init__(*args, **kwargs)

    def init_logger(self, loglevel=None):
        app_loglevel = self.info["appInfo"].get("logLevel")
        super().init_logger(loglevel=app_loglevel)

    def init_task_dir(self):
        # agent container paths
        self.dir_task = os.path.join(constants.AGENT_APP_SESSIONS_DIR(), str(self.info['task_id']))
        self.dir_task_src = os.path.join(self.dir_task, 'repo')
        # host paths
        self.dir_task_host = os.path.join(constants.AGENT_APP_SESSIONS_DIR_HOST(), str(self.info['task_id']))

        team_id = self.info.get("context", {}).get("teamId", "unknown")
        if team_id == "unknown":
            self.logger.warn("teamId not found in context")
        self.dir_apps_cache_host = os.path.join(constants.AGENT_APPS_CACHE_DIR_HOST(), str(team_id))
        sly.fs.ensure_base_path(self.dir_apps_cache_host)

        # task container path
        self.dir_task_container = os.path.join("/sessions", str(self.info['task_id']))
        self.dir_task_src_container = os.path.join(self.dir_task_container, 'repo')
        self.dir_apps_cache_container = "/apps_cache"
        self.app_info = self.info["appInfo"]

    def download_or_get_repo(self):
        def is_fixed_version(name):
            try:
                v = Version(name)
                return True
            except ValueError as e:
                return False

        git_url = self.app_info["githubUrl"]
        version = self.app_info.get("version", "master")

        already_downloaded = False
        path_cache = None
        if version != "master" and is_fixed_version(version):
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
        version = self.app_info.get("version", "master")
        self.logger.info("App moduleId == {} [v={}] in ecosystem".format(module_id, version))
        self.app_config = api.app.get_info(module_id, version)["config"]
        self.logger.info("App config", extra={"config": self.app_config})

        need_gpu = self.app_config.get('needGPU', False)
        if need_gpu:
            self.docker_runtime = 'nvidia'

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
        return True #self.app_config.get(_ISOLATE, True)

    def _get_task_volumes(self):
        res = {}
        res[self.dir_task_host] = {'bind': self.dir_task_container, 'mode': 'rw'}
        res[self.dir_apps_cache_host] = {'bind': self.dir_apps_cache_container, 'mode': 'rw'}

        if self._need_sync_pip_cache is True:
            res[self._path_cache_host] = {'bind': _LINUX_DEFAULT_PIP_CACHE_DIR, 'mode': 'rw'}

        if constants.HOST_REQUESTS_CA_BUNDLE() is not None:
            res[constants.HOST_REQUESTS_CA_BUNDLE()] = {'bind': constants.REQUESTS_CA_BUNDLE(), 'mode': 'ro'}

        return res

    def download_step(self):
        pass

    def sync_pip_cache(self):
        version = self.app_info.get("version", "master")
        module_id = self.app_info.get("moduleId")

        requirements_path = os.path.join(self.dir_task_src, self.app_info.get("configDir"), "requirements.txt")
        if file_exists(requirements_path) is False:
            requirements_path = os.path.join(self.dir_task_src, "requirements.txt")
            if file_exists(requirements_path) is True:
                self._requirements_path_relative = "requirements.txt"
        else:
            self._requirements_path_relative = os.path.join(self.app_info.get("configDir"), "requirements.txt")

        self.logger.info(f"Relative path to requirements: {self._requirements_path_relative}")

        path_cache = os.path.join(constants.APPS_PIP_CACHE_DIR(), str(module_id), version)  # in agent container
        self._path_cache_host = constants._agent_to_host_path(os.path.join(path_cache, "pip"))

        if sly.fs.file_exists(requirements_path):

            self._need_sync_pip_cache = True

            self.logger.info("requirements.txt:")
            with open(requirements_path, 'r') as f:
                self.logger.info(f.read())

            if sly.fs.dir_exists(path_cache) is False or version == "master":
                sly.fs.mkdir(path_cache)
                archive_destination = os.path.join(path_cache, "archive.tar")
                self.spawn_container(add_labels={"pip_cache": "1", "app_session_id": str(self.info['task_id'])})
                self.install_pip_requirements(container_id=self._container.id)

                #@TODO: handle 404 not found
                bits, stat = self._container.get_archive(_LINUX_DEFAULT_PIP_CACHE_DIR)
                self.logger.info("Download initial pip cache from dockerimage",
                                 extra={
                                     "dockerimage": self.docker_image_name,
                                     "module_id": module_id,
                                     "version": version,
                                     "save_path": path_cache,
                                     "stats": stat,
                                     "default_pip_cache": _LINUX_DEFAULT_PIP_CACHE_DIR,
                                     "archive_destination": archive_destination
                                 })

                with open(archive_destination, 'wb') as archive:
                    for chunk in bits:
                        archive.write(chunk)

                with tarfile.open(archive_destination) as archive:
                    archive.extractall(path_cache)
                sly.fs.silent_remove(archive_destination)
            else:
                self.logger.info("Use existing pip cache")

    def find_or_run_container(self):
        add_labels = {"sly_app": "1", "app_session_id": str(self.info['task_id'])}
        sly.docker_utils.docker_pull_if_needed(self._docker_api, self.docker_image_name, constants.PULL_POLICY(), self.logger)
        self.sync_pip_cache()
        if self._container is None:
            self.spawn_container(add_labels=add_labels)
            self.logger.info("Double check pip cache for old agents")
            self.install_pip_requirements(container_id=self._container.id)
            self.logger.info("pip second install for old agents is finished")

    def get_spawn_entrypoint(self):
        return ["sh", "-c", "while true; do sleep 30; done;"]

    def _exec_command(self, command, add_envs=None, container_id=None):
        add_envs = sly.take_with_default(add_envs, {})
        self._exec_id = self._docker_api.api.exec_create(self._container.id if container_id is None else container_id,
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
                                                             constants._REQUESTS_CA_BUNDLE: constants.REQUESTS_CA_BUNDLE(),
                                                             **add_envs
                                                         })
        self._logs_output = self._docker_api.api.exec_start(self._exec_id, stream=True, demux=False)

    def exec_command(self, add_envs=None, command=None):
        add_envs = sly.take_with_default(add_envs, {})
        main_script_path = os.path.join(self.dir_task_src_container, self.app_config.get('main_script', 'src/main.py'))

        if command is None:
            command = "python {}".format(main_script_path)
        self.logger.info("command to run", extra={"command": command})

        self._exec_command(command, add_envs)

        #change pulling progress to app progress
        progress_dummy = sly.Progress('Application is started ...', 1, ext_logger=self.logger)
        progress_dummy.iter_done_report()

        self.logger.info("command is running", extra={"command": command})

    def install_pip_requirements(self, container_id=None):
        if self._need_sync_pip_cache is True:
            self.logger.info("Installing app requirements")
            progress_dummy = sly.Progress('Installing app requirements...', 1, ext_logger=self.logger)
            progress_dummy.iter_done_report()
            command = "pip3 install -r " + os.path.join(self.dir_task_src_container, self._requirements_path_relative)
            self._exec_command(command, add_envs=self.main_step_envs(), container_id=container_id)
            self.process_logs()
            self.logger.info("Requirements are installed")

    def main_step(self):
        self.find_or_run_container()
        self.exec_command(add_envs=self.main_step_envs())
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
            'LOG_LEVEL': str(self.app_info.get('logLevel', 'INFO')),
            'LOGLEVEL': str(self.app_info.get('logLevel', 'INFO')),
            'PYTHONUNBUFFERED': 1
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

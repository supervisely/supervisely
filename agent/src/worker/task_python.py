# coding: utf-8

import os
import os.path as osp
import tarfile

import supervisely_lib as sly
from supervisely_lib.task import paths as sly_paths
from supervisely_lib.api.api import SUPERVISELY_TASK_ID

from worker.task_dockerized import TaskDockerized, TaskStep
from worker import constants as worker_constants

PROJECTS = 'projects'
TITLE = 'title'

TAR_EXTN = '.tar'


class TaskPython(TaskDockerized):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.dir_data = osp.join(self.dir_task, sly_paths.DATA)
        self.dir_code = osp.join(self.dir_task, sly_paths.CODE)
        self.script_body = osp.join(self.dir_code, 'script.py')
        self.dir_results = osp.join(self.dir_task, sly_paths.RESULTS)
        self._dir_results_artifacts = osp.join(self.dir_results, sly_paths.ARTIFACTS)
        self._dir_results_projects = osp.join(self.dir_results, sly_paths.PROJECTS)

    def init_additional(self):
        super().init_additional()
        sly.fs.mkdir(self.dir_data)
        sly.fs.mkdir(self.dir_code)
        sly.fs.mkdir(self.dir_results)
        sly.fs.mkdir(self._dir_results_artifacts)
        sly.fs.mkdir(self._dir_results_projects)

        with open(self.script_body, 'w') as fout:
            fout.write(self.info['script'])

    def download_step(self):
        self.report_step_done(TaskStep.DOWNLOAD)

    def before_main_step(self):
        pass

    def main_step_envs(self):
        envs = super().main_step_envs()
        envs.update({
            'SERVER_ADDRESS': worker_constants.SERVER_ADDRESS(), 
            'API_TOKEN': self._user_api_key,
            SUPERVISELY_TASK_ID: str(self.info['task_id'])})
        return envs

    def _get_task_volumes(self):
        volumes = super()._get_task_volumes()
        volumes[worker_constants.AGENT_TASK_SHARED_DIR_HOST()] = {
            'bind': sly_paths.TaskPaths.TASK_SHARED_DIR, 'mode': 'rw'}
        return volumes

    def upload_step(self):
        self.report_step_done(TaskStep.MAIN)

        # Upload the resulting projects.
        for project_name in sly.fs.get_subdirs(self._dir_results_projects):
            self.data_mgr.upload_project(self._dir_results_projects, project_name, project_name)

        # Archive the non-project artifacts for uploading.
        for artifact_name in sly.fs.get_subdirs(self._dir_results_artifacts):
            artifact_tar_path = os.path.join(self._dir_results_artifacts, artifact_name + TAR_EXTN)
            with tarfile.open(artifact_tar_path, 'w', encoding='utf-8') as tar:
                tar.add(os.path.join(self._dir_results_artifacts, artifact_name), arcname=artifact_name)

        # Upload all the tar files as artifacts.
        # For now, only one artifact per task is supported
        final_artifacts = [x.name for x in os.scandir(self._dir_results_artifacts)
                           if x.is_file() and x.name.endswith(TAR_EXTN)]
        if len(final_artifacts) > 0:
            if len(final_artifacts) > 1:
                raise RuntimeError(
                    'Uploading multiple artifacts is not supported yet. Found multiple {!r} files in '
                    '{!r} directory: {!r}.'.format(TAR_EXTN, self._dir_results_artifacts, final_artifacts))
            [artifact_name] = final_artifacts
            self.data_mgr.upload_tar_file(
                self.info['task_id'], os.path.join(self._dir_results_artifacts, artifact_name))

        self.report_step_done(TaskStep.UPLOAD)

# coding: utf-8

import os
import os.path as osp
import tarfile

import supervisely_lib as sly

from worker.task_dockerized import TaskDockerized, TaskStep
from worker import constants as worker_constants

ARTIFACTS = 'artifacts'
CODE = 'code'
DATASETS = 'datasets'
PROJECTS = 'projects'

TITLE = 'title'

TAR_EXTN = '.tar'

_API_INIT_PREAMBLE = """
import supervisely_lib as sly

WORKSPACE_ID = {workspace_id}
IN_PROJECTS = {in_projects!r}
RESULT_DATASETS_DIR = '{result_datasets_dir}'
RESULT_ARTIFACTS_DIR = '{result_artifacts_dir}'

api = sly.Api(server_address='{address}', token='{token}')

"""

class TaskPython(TaskDockerized):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.dir_data = osp.join(self.dir_task, sly.task.paths.DATA)
        self.dir_code = osp.join(self.dir_task, CODE)
        self.script_main = osp.join(self.dir_code, 'main.py')
        self.entrypoint = self._agent_path_to_task_path(self.script_main)
        self.dir_results = osp.join(self.dir_task, sly.task.paths.RESULTS)
        self._dir_results_artifacts = osp.join(self.dir_results, ARTIFACTS)
        self._dir_results_datasets = osp.join(self.dir_results, DATASETS)

    def init_additional(self):
        super().init_additional()
        sly.fs.mkdir(self.dir_data)
        sly.fs.mkdir(self.dir_code)
        sly.fs.mkdir(self.dir_results)
        sly.fs.mkdir(self._dir_results_artifacts)
        sly.fs.mkdir(self._dir_results_datasets)

        with open(self.script_main, 'w') as fout:
            fout.write(_API_INIT_PREAMBLE.format(
                **{'address': worker_constants.SERVER_ADDRESS(),
                   'token': self._user_api_key,
                   'workspace_id': self.data_mgr.workspace_id,
                   'in_projects': [x[TITLE] for x in self.info.get(PROJECTS, [])],
                   'result_artifacts_dir': self._agent_path_to_task_path(self._dir_results_artifacts),
                   'result_datasets_dir': self._agent_path_to_task_path(self._dir_results_datasets)}))
            fout.write(self.info['script'])

    def _agent_path_to_task_path(self, path):
        return osp.join(sly.TaskPaths.TASK_DIR, os.path.relpath(path, start=self.dir_task))

    def download_step(self):
        for project_info in self.info.get(PROJECTS, []):
            self.data_mgr.download_project(self.dir_data, project_info[TITLE])
        self.report_step_done(TaskStep.DOWNLOAD)

    def before_main_step(self):
        pass

    def upload_step(self):
        self.report_step_done(TaskStep.MAIN)

        # Upload the resulting projects.
        datasets_root = os.path.join(self.dir_results, DATASETS)
        for project_name in sly.fs.get_subdirs(datasets_root):
            self.data_mgr.upload_project(datasets_root, project_name, project_name)

        # Archive the non-project artifacts for uploading.
        artifacts_root = os.path.join(self.dir_results, ARTIFACTS)
        for artifact_name in sly.fs.get_subdirs(artifacts_root):
            artifact_tar_path = os.path.join(artifacts_root, artifact_name + TAR_EXTN)
            with tarfile.open(artifact_tar_path, 'w', encoding='utf-8') as tar:
                tar.add(os.path.join(artifacts_root, artifact_name), arcname=artifact_name)

        # Upload all the tar files as artifacts.
        # For now, only one artifact per task is supported
        final_artifacts = [x.name for x in os.scandir(artifacts_root) if x.is_file() and x.name.endswith(TAR_EXTN)]
        if len(final_artifacts) > 0:
            if len(final_artifacts) > 1:
                raise RuntimeError('Uploading multiple artifacts is not supported yet. Found multiple {!r} files in '
                                   '{!r} directory: {!r}.'.format(TAR_EXTN, artifacts_root, final_artifacts))
            [artifact_name] = final_artifacts
            self.data_mgr.upload_tar_file(self.info['task_id'], os.path.join(artifacts_root, artifact_name))

        self.report_step_done(TaskStep.UPLOAD)

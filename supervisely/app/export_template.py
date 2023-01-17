import os
from os.path import basename, isdir, isfile, join
from typing import NamedTuple

import supervisely.io.env as env
from supervisely import Progress
from supervisely._utils import is_production
from supervisely.api.api import Api
from supervisely.api.project_api import ProjectInfo
from supervisely.app import get_data_dir
from supervisely.io.fs import get_file_name_with_ext, remove_dir, silent_remove
from supervisely.project.project_type import ProjectType
from supervisely.sly_logger import logger
from supervisely.task.progress import Progress
from supervisely.team_files import RECOMMENDED_EXPORT_PATH


class Export:
    class Context:
        def __init__(self, project: ProjectInfo, datasets: list):
            self._project = project
            self._datasets = datasets
            self._work_dir = join(get_data_dir(), "work_dir")

        def __str__(self):
            return (
                f"Project: {self._project}\n"
                f"Dataset: {self._datasets}\n"
                f"Working directory: {self._work_dir}\n"
            )

        @property
        def project(self) -> ProjectInfo:
            return self._project

        @property
        def datasets(self) -> list:
            return self._datasets

        @property
        def work_dir(self) -> bool:
            return self._work_dir

    def process(self, context) -> str:
        raise NotImplementedError()  # implement your own method when inherit

    def prepare(
        self,
        api: Api,
        project: ProjectInfo,
        dataset_id: int = None,
    ):

        if dataset_id is None:
            datasets = api.dataset.get_list(project_id=project.id)
        else:
            datasets = [api.dataset.get_info_by_id(id=dataset_id)]

        return self.Context(project=project, datasets=datasets)

    def run(self):
        api = Api.from_env()
        task_id = env.task_id()

        team_id = env.team_id()
        workspace_id = env.workspace_id()

        app_name = "test-export-app"

        if is_production():
            module_id = os.environ["modal.state.slyEcosystemItemId"]
            app_info = api.app.get_info(module_id)
            app_name = app_info["name"].lower().replace(" ", "-")

        project_id = env.project_id(raise_not_found=False)
        dataset_id = env.dataset_id(raise_not_found=False)

        project = api.project.get_info_by_id(id=project_id)
        if project is None:
            raise ValueError(
                f"Project with ID: {project_id} either archived or you don't have access to it"
            )
        logger.info(f"Exporting Project: id={project.id}, name={project.name}, type={project.type}")

        context = self.prepare(
            api=api,
            project=project,
            dataset_id=dataset_id,
        )

        local_path = self.process(context=context)

        if type(local_path) is not str:
            raise ValueError("Path must be a 'string'")

        upload_progress = []

        def _print_progress(monitor, upload_progress):
            if len(upload_progress) == 0:
                upload_progress.append(
                    Progress(
                        message=f"Uploading '{task_id}_{basename(local_path)}'",
                        total_cnt=monitor.len,
                        ext_logger=logger,
                        is_size=True,
                    )
                )
            upload_progress[0].set_current_value(monitor.bytes_read)

        if isfile(local_path):
            remote_path = join(
                RECOMMENDED_EXPORT_PATH,
                app_name,
                f"{task_id}_{get_file_name_with_ext(local_path)}",
            )
            file_info = api.file.upload(
                team_id=team_id,
                src=local_path,
                dst=remote_path,
                progress_cb=lambda m: _print_progress(m, upload_progress),
            )
            api.task.set_output_archive(
                task_id=task_id, file_id=file_info.id, file_name=file_info.name
            )
            logger.info(f"Remote file: id={file_info.id}, name={file_info.name}")
            silent_remove(local_path)
        elif isdir(local_path):
            remote_path = join(
                RECOMMENDED_EXPORT_PATH,
                app_name,
                f"{task_id}_{basename(local_path)}",
            )
            remote_path = api.file.upload_directory(
                team_id=team_id,
                local_dir=local_path,
                remote_dir=remote_path,
                change_name_if_conflict=True,
                progress_size_cb=lambda m: _print_progress(m, upload_progress),
            )
            remote_dir_files = api.file.listdir(team_id, remote_path)
            for curr_file in remote_dir_files:
                file_info = api.file.get_info_by_path(team_id, curr_file)
                if file_info is not None:
                    break
            api.task.set_output_directory(
                task_id=task_id, file_id=file_info.id, directory_path=remote_path
            )
            logger.info(f"Remote directory: id={file_info.id}, name={remote_path}")
            remove_dir(local_path)

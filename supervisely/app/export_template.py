from os.path import basename, isdir, join

import supervisely.io.env as env
from supervisely import Progress
from supervisely.api.api import Api
from supervisely.api.api import Api
from supervisely.io.fs import get_file_name_with_ext, silent_remove
from supervisely.sly_logger import logger
from supervisely.task.progress import Progress
from supervisely.team_files import RECOMMENDED_EXPORT_PATH
from supervisely.io.fs import archive_directory
from typing import Optional


class Export:
    class Context:
        def __init__(
            self, team_id: int, workspace_id: int, project_id: int, dataset_id: Optional[int] = None
        ):
            self._team_id = team_id
            self._workspace_id = workspace_id
            self._project_id = project_id
            self._dataset_id = dataset_id

        def __str__(self):
            return (
                f"Team ID: {self._team_id}\n"
                f"Workspace ID: {self._workspace_id}\n"
                f"Project ID: {self._project_id}\n"
                f"Dataset ID: {self._dataset_id}\n"
            )

        @property
        def team_id(self) -> int:
            return self._team_id

        @property
        def workspace_id(self) -> int:
            return self._workspace_id

        @property
        def project_id(self) -> int:
            return self._project_id

        @property
        def dataset_id(self) -> int:
            return self._dataset_id

    def process(self, context: Context) -> tuple:
        raise NotImplementedError()  # implement your own method when inherit

    def run(self):
        api = Api.from_env()
        task_id = env.task_id()

        team_id = env.team_id()
        workspace_id = env.workspace_id()
        project_id = env.project_id(raise_not_found=False)
        dataset_id = env.dataset_id(raise_not_found=False)

        project = api.project.get_info_by_id(id=project_id)
        if project is None:
            raise ValueError(
                f"Project with ID: {project_id} either archived or you don't have access to it"
            )
        logger.info(f"Exporting Project: id={project.id}, name={project.name}, type={project.type}")

        if dataset_id is not None:
            dataset = api.dataset.get_info_by_id(id=dataset_id)
            if dataset is None:
                raise ValueError(
                    f"Dataset with ID: {dataset_id} either archived or you don't have access to it"
                )

        context = self.Context(
            team_id=team_id, workspace_id=workspace_id, project_id=project_id, dataset_id=dataset_id
        )

        local_path, app_name = self.process(context=context)

        if type(local_path) is not str:
            raise ValueError("Path must be a 'string'")

        upload_progress = []

        def _print_progress(monitor, upload_progress):
            if len(upload_progress) == 0:
                upload_progress.append(
                    Progress(
                        message=f"Uploading '{basename(local_path)}'",
                        total_cnt=monitor.len,
                        ext_logger=logger,
                        is_size=True,
                    )
                )
            upload_progress[0].set_current_value(monitor.bytes_read)

        if isdir(local_path):
            archive_path = local_path + ".tar"
            archive_directory(local_path, archive_path)
            local_path = archive_path

        remote_path = join(
            RECOMMENDED_EXPORT_PATH,
            app_name,
            str(task_id),
            f"{get_file_name_with_ext(local_path)}",
        )
        file_info = api.file.upload(
            team_id=team_id,
            src=local_path,
            dst=remote_path,
            progress_cb=lambda m: _print_progress(m, upload_progress),
        )
        api.task.set_output_archive(task_id=task_id, file_id=file_info.id, file_name=file_info.name)
        logger.info(f"Remote file: id={file_info.id}, name={file_info.name}")
        silent_remove(local_path)

from supervisely._utils import is_production
import supervisely.io.env as env
from supervisely.api.api import Api
from supervisely.app import get_data_dir
from os.path import join, isdir, isfile, basename
from supervisely.io.fs import get_file_name_with_ext
from supervisely import Progress
import supervisely as sly
from typing import NamedTuple
from supervisely.project.project_type import ProjectType
from supervisely.io.fs import silent_remove, remove_dir
from supervisely.api.project_api import ProjectInfo
import os


class Export:
    class Context:
        def __init__(
            self,
            project: ProjectInfo,
            datasets: list,
            items: dict,
            anns: dict,
        ):
            self._project = project
            self._datasets = datasets
            self._items = items
            self._anns = anns
            self._work_dir = join(get_data_dir(), "work_dir")

        def __str__(self):
            return (
                f"Project: {self._project}\n"
                f"Dataset: {self._datasets}\n"
                f"Items: {self._items}\n"
                f"Anns: {self._anns}\n"
                f"Working directory: {self._work_dir}\n"
            )

        @property
        def project(self) -> ProjectInfo:
            return self._project

        @property
        def datasets(self) -> list:
            return self._datasets

        @property
        def items(self) -> dict:
            return self._items

        @property
        def anns(self) -> dict:
            return self._anns

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

        items = {}
        anns = {}
        for dataset in datasets:
            if project.type == ProjectType.IMAGES.value:
                items[dataset.name] = api.image.get_list(dataset_id=dataset.id)
                entity_ids = [item_info.id for item_info in items[dataset.name]]
                anns[dataset.name] = api.annotation.download_batch(
                    dataset_id=dataset.id, image_ids=entity_ids
                )
            if project.type == ProjectType.VIDEOS.value:
                items[dataset.name] = api.video.get_list(dataset_id=dataset.id)
                entity_ids = [item_info.id for item_info in items[dataset.name]]
                anns[dataset.name] = api.video.annotation.download_bulk(
                    dataset_id=dataset.id, entity_ids=entity_ids
                )
            if project.type == ProjectType.VOLUMES.value:
                items[dataset.name] = api.volume.get_list(dataset_id=dataset.id)
                entity_ids = [item_info.id for item_info in items[dataset.name]]
                anns[dataset.name] = api.volume.annotation.download_bulk(
                    dataset_id=dataset.id, entity_ids=entity_ids
                )
            if project.type == ProjectType.POINT_CLOUDS.value:
                items[dataset.name] = api.pointcloud.get_list(dataset_id=dataset.id)
                entity_ids = [item_info.id for item_info in items[dataset.name]]
                anns[dataset.name] = api.pointcloud.annotation.download_bulk(
                    dataset_id=dataset.id, entity_ids=entity_ids
                )
            if project.type == ProjectType.POINT_CLOUD_EPISODES.value:
                items[dataset.name] = api.pointcloud_episode.get_list(dataset_id=dataset.id)
                entity_ids = [item_info.id for item_info in items[dataset.name]]
                anns[dataset.name] = api.pointcloud_episode.annotation.download_bulk(
                    dataset_id=dataset.id, entity_ids=entity_ids
                )

        return self.Context(
            project=project,
            datasets=datasets,
            items=items,
            anns=anns,
        )

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
        sly.logger.info(
            f"Exporting Project: id={project.id}, name={project.name}, type={project.type}"
        )

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
                    sly.Progress(
                        message=f"Uploading '{task_id}_{basename(local_path)}'",
                        total_cnt=monitor.len,
                        ext_logger=sly.logger,
                        is_size=True,
                    )
                )
            upload_progress[0].set_current_value(monitor.bytes_read)

        if isfile(local_path):
            remote_path = join(
                sly.team_files.RECOMMENDED_EXPORT_PATH,
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
            sly.logger.info(f"Remote file: id={file_info.id}, name={file_info.name}")
            silent_remove(local_path)
        elif isdir(local_path):
            remote_path = join(
                sly.team_files.RECOMMENDED_EXPORT_PATH,
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
            sly.logger.info(f"Remote directory: id={file_info.id}, name={remote_path}")
            remove_dir(local_path)

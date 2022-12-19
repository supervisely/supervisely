from typing import Optional, Union
from supervisely._utils import is_production
import supervisely.io.env as env
from supervisely.api.api import Api
from supervisely.app import get_data_dir
from os.path import join, basename
from pathlib import Path

from supervisely.project.project import download_project
from supervisely.project.video_project import download_video_project
from supervisely.project.volume_project import download_volume_project
from supervisely.project.pointcloud_project import download_pointcloud_project
from supervisely.project.pointcloud_episode_project import download_pointcloud_episode_project
from supervisely.project.project_type import ProjectType
from supervisely.project.project import Project, Dataset, OpenMode

try:
    from typing import Literal
except ImportError:
    # for compatibility with python 3.7
    from typing_extensions import Literal


class Export:
    class Context:
        def __init__(
            self,
            team_id: int,
            workspace_id: int,
            project_id: int,
            dataset_id: int,
            project: Project,
            dataset: Dataset,
        ):
            self._team_id = team_id
            if self._team_id is None:
                raise ValueError(f"Team ID is not specified: {self._team_id}")
            if type(self._team_id) is not int:
                raise ValueError(f"Team ID must be 'int': {self._team_id}")

            self._workspace_id = workspace_id
            if self._workspace_id is None:
                raise ValueError(f"Workspace ID is not specified: {self._workspace_id}")
            if type(self._workspace_id) is not int:
                raise ValueError(f"Workspace ID must be 'int': {self._workspace_id}")

            self._project_id = project_id
            if self._project_id is None:
                raise ValueError(f"Project ID is not specified: {self._project_id}")
            if type(self._project_id) is not int:
                raise ValueError(f"Project ID must be 'int': {self._project_id}")

            self._dataset_id = dataset_id
            if self._dataset_id is None:
                raise ValueError(f"Dataset ID is not specified: {self._dataset_id}")
            if type(self._dataset_id) is not int:
                raise ValueError(f"Dataset ID must be 'int': {self._dataset_id}")

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

    def process(self, project: Project) -> str:
        raise NotImplementedError()  # implement your own method when inherit

    def prepare(
        self,
        api: Api,
        project_id: int,
        project_type: ProjectType,
        dataset_id: int,
        local_project_path: str,
        log_progress: bool,
    ):
        if project_type == ProjectType.IMAGES.value:
            download_project(
                api=api,
                project_id=project_id,
                dest_dir=local_project_path,
                dataset_ids=(dataset_id),
                log_progress=log_progress,
            )
        elif project_type == ProjectType.VIDEOS.value:
            download_video_project(
                api=api,
                project_id=project_id,
                dest_dir=local_project_path,
                dataset_ids=(dataset_id),
                log_progress=log_progress,
            )
        elif project_type == ProjectType.VOLUMES.value:
            download_volume_project(
                api=api,
                project_id=project_id,
                dest_dir=local_project_path,
                dataset_ids=(dataset_id),
                log_progress=log_progress,
            )
        elif project_type == ProjectType.POINT_CLOUDS.value:
            download_pointcloud_project(
                api=api,
                project_id=project_id,
                dest_dir=local_project_path,
                dataset_ids=(dataset_id),
                log_progress=log_progress,
            )
        elif project_type == ProjectType.POINT_CLOUD_EPISODES.value:
            download_pointcloud_episode_project(
                api=api,
                project_id=project_id,
                dest_dir=local_project_path,
                dataset_ids=(dataset_id),
                log_progress=True,
            )

        local_project = Project(directory=local_project_path, mode=OpenMode.READ)
        return local_project

    def run(self):
        api = Api.from_env()
        task_id = None
        if is_production():
            task_id = env.task_id()

        team_id = env.team_id()
        workspace_id = env.workspace_id()

        project_id = env.project_id(raise_not_found=False)
        dataset_id = env.dataset_id(raise_not_found=False)

        # get or create project with the same name as input file and empty dataset in it
        project = api.project.get_info_by_id(id=project_id)
        print(f"Exporting Project: id={project.id}, name={project.name}, type={project.type}")

        if dataset_id is not None:
            dataset = api.dataset.get_info_by_id(id=dataset_id)
            print(f"Exporting Dataset: id={dataset.id}, name={dataset.name}")

        local_project_path = join(get_data_dir(), project.name)

        local_project = self.prepare(
            api=api,
            project_id=project.id,
            project_type=project.type,
            dataset_id=[dataset_id],
            local_project_path=local_project_path,
            log_progress=True,
        )

        path_to_upload = self.process(project=local_project)
        if type(project_id) is int and is_production():
            info = api.project.get_info_by_id(project_id)
            api.task.set_output_project(task_id=task_id, project_id=info.id, project_name=info.name)
            print(f"Result project: id={info.id}, name={info.name}")

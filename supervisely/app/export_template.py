from typing import Optional, Union
from supervisely._utils import is_production
import supervisely.io.env as env
from supervisely.api.api import Api
from supervisely.app import get_data_dir
from os.path import join, isdir, isfile, basename
from supervisely.io.fs import get_file_name_with_ext
from supervisely import Progress

from supervisely.project.project import download_project
from supervisely.project.video_project import download_video_project
from supervisely.project.volume_project import download_volume_project
from supervisely.project.pointcloud_project import download_pointcloud_project
from supervisely.project.pointcloud_episode_project import download_pointcloud_episode_project
from supervisely.project.project_type import ProjectType
from supervisely.project.project import Project, OpenMode

try:
    from typing import Literal
except ImportError:
    # for compatibility with python 3.7
    from typing_extensions import Literal

ARCHIVE_EXT = ["zip", "tar", "rar", "tar.gz", "7z"]


class Export:

    # context
    #     project_info, dataset_info, items_infos, anns_infos, work_dir

    def process(self, project: Project, work_dir: str) -> str:
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
        work_dir = join(get_data_dir(), "work_dir")

        local_project = self.prepare(
            api=api,
            project_id=project.id,
            project_type=project.type,
            dataset_id=[dataset_id],
            local_project_path=local_project_path,
            log_progress=True,
        )

        local_path = self.process(project=local_project, work_dir=work_dir)

        if type(local_path) is str:
            if isfile():
                progress = Progress(message="Uploading file", total_cnt=1)
                remote_path = join("template-export-app", get_file_name_with_ext(local_path))
                file_info = api.file.upload(
                    team_id=team_id, src=local_path, dst=remote_path, progress_cb=progress
                )
                api.task.set_output_archive(
                    task_id=task_id, file_id=file_info.id, file_name=file_info.name
                )
                print(f"Remote file: id={file_info.id}, name={file_info.name}")
            elif isdir():
                progress = Progress(message="Uploading file", total_cnt=1)
                remote_path = join("template-export-app", basename(local_path))
                file_info = api.file.upload_directory(
                    team_id=team_id,
                    local_dir=local_path,
                    remote_dir=remote_path,
                    change_name_if_conflict=True,
                    progress_size_cb=progress,
                )
                api.task.set_output_directory(
                    task_id=task_id, file_id=file_info.id, directory_path=file_info.name
                )
                print(f"Remote directory: id={file_info.id}, name={file_info.name}")
        else:
            raise ValueError("Path must be a 'string'")

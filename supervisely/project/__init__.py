from typing import Optional, Union

from supervisely.io.fs import get_file_name_with_ext, list_files
from supervisely.io.json import load_json_file
from supervisely.project.pointcloud_episode_project import PointcloudEpisodeProject
from supervisely.project.pointcloud_project import PointcloudProject
from supervisely.project.project import Project, read_single_project
from supervisely.project.project_type import ProjectType
from supervisely.project.video_project import VideoProject
from supervisely.project.volume_project import VolumeProject


def read_any_single_project(
    dir: str,
) -> Optional[
    Union[Project, VideoProject, VolumeProject, PointcloudProject, PointcloudEpisodeProject]
]:
    paths = list_files(dir)
    for path in paths:
        if get_file_name_with_ext(path) == "meta.json":
            project_type: str = load_json_file(path)["projectType"]

    project_class = None
    if project_type == ProjectType.IMAGES.value:
        project_class = Project
    elif project_type == ProjectType.VIDEOS.value:
        project_class = VideoProject
    elif project_type == ProjectType.VOLUMES.value:
        project_class = VolumeProject
    elif project_type == ProjectType.POINT_CLOUDS.value:
        project_class = PointcloudProject
    elif project_type == ProjectType.POINT_CLOUD_EPISODES.value:
        project_class = PointcloudEpisodeProject

    return read_single_project(dir, project_class)

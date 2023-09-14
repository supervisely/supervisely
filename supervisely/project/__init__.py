from typing import Optional, Union

from supervisely.io.fs import get_file_name_with_ext, list_files
from supervisely.io.json import load_json_file
from supervisely.project.pointcloud_episode_project import PointcloudEpisodeProject
from supervisely.project.pointcloud_project import PointcloudProject
from supervisely.project.project import Project, read_single_project
from supervisely.project.project_type import ProjectType
from supervisely.project.video_project import VideoProject
from supervisely.project.volume_project import VolumeProject


def read_project(
    dir: str,
) -> Optional[
    Union[Project, VideoProject, VolumeProject, PointcloudProject, PointcloudEpisodeProject]
]:
    """
    Read project of arbitrary modality from given directory.

    :param dir: Path to directory, which contains project folder.
    :type dir: :class: str

    :return: Project class object of specific modality
    :rtype: :class: Project or VideoProject or VolumeProject or PointcloudProject or PointcloudEpisodeProject

    :Usage example:
     .. code-block:: python
        import supervisely as sly

        proj_dir = "/path/to/your/source/project"
        project_fs = sly.read_project(proj_dir)
    """
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

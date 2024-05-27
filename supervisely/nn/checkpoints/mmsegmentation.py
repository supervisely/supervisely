from os.path import join
from typing import List, Literal

from supervisely.api.api import Api
from supervisely._utils import abs_url, is_development
from supervisely.nn.checkpoints.checkpoint import CheckpointInfo


def get_list(api: Api, team_id: int, sort: Literal["desc", "asc"] = "desc") -> List[CheckpointInfo]:
    """
    Parse the TeamFiles directory with the checkpoints trained
    in Supervisely of the MMSegmentation model
    and return the list of CheckpointInfo objects.

    :param api: Supervisely API object
    :type api: Api
    :param team_id: Team ID
    :type team_id: int
    :param sort: Sorting order, defaults to "desc", which means new models first
    :type sort: Literal["desc", "asc"], optional

    :return: List of CheckpointInfo objects
    :rtype: List[CheckpointInfo]
    """
    if sort not in ["desc", "asc"]:
        raise ValueError(f"Invalid sort value: {sort}")

    checkpoints = []
    weights_dir_name = "checkpoints"
    weights_subdir_name = "data"
    training_app_directory = "/mmsegmentation/"
    task_type = "instance segmentation"
    if not api.file.dir_exists(team_id, training_app_directory):
        return []
    task_files_infos = api.file.list(
        team_id, training_app_directory, recursive=False, return_type="fileinfo"
    )
    for task_file_info in task_files_infos:
        task_id = task_file_info.name.split("_")[0]
        project_name = task_file_info.name.split("_")[1]
        if is_development():
            session_link = abs_url(f"/apps/sessions/{task_id}")
        else:
            session_link = f"/apps/sessions/{task_id}"
        path_to_checkpoints = join(task_file_info.path, weights_dir_name, weights_subdir_name)
        checkpoints_infos = [
            file
            for file in api.file.list(
                team_id, path_to_checkpoints, recursive=False, return_type="fileinfo"
            )
            if file.name.endswith(".pth")
        ]
        config_url = join(path_to_checkpoints, "config.py")
        if not api.file.exists(team_id, config_url):
            continue
        if len(checkpoints_infos) == 0:
            continue
        checkpoint_info = CheckpointInfo(
            app_name="Train MMSegmentation",
            session_id=task_id,
            session_path=task_file_info.path,
            session_link=session_link,
            task_type=task_type,
            training_project_name=project_name,
            checkpoints=checkpoints_infos,
            config=config_url,
        )
        checkpoints.append(checkpoint_info)

    if sort == "desc":
        checkpoints = sorted(checkpoints, key=lambda x: x.session_id, reverse=True)
    elif sort == "asc":
        checkpoints = sorted(checkpoints, key=lambda x: x.session_id)
    return checkpoints

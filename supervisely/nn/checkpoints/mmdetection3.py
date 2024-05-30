from os.path import join
from typing import List, Literal

from supervisely.api.api import Api
from supervisely.io.fs import silent_remove
from supervisely._utils import abs_url, is_development
from supervisely.nn.checkpoints.checkpoint import CheckpointInfo


def get_list(api: Api, team_id: int, sort: Literal["desc", "asc"] = "desc") -> List[CheckpointInfo]:
    """
    Parse the TeamFiles directory with the checkpoints trained
    in Supervisely of the MMDetection 3.0 model
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
    training_app_directory = "/mmdetection-3/"
    if not api.file.dir_exists(team_id, training_app_directory):
        return []
    task_files_infos = api.file.list(
        team_id, training_app_directory, recursive=False, return_type="fileinfo"
    )
    for task_file_info in task_files_infos:
        task_id = task_file_info.name.split("_")[0]
        session_dir_files = api.file.list(
            team_id, task_file_info.path, recursive=False, return_type="fileinfo"
        )
        if is_development():
            session_link = abs_url(f"/apps/sessions/{task_id}")
        else:
            session_link = f"/apps/sessions/{task_id}"
        checkpoints_infos = [file for file in session_dir_files if file.name.endswith(".pth")]
        if not api.file.exists(team_id, join(task_file_info.path, "config.py")):
            continue
        config_url = join(task_file_info.path, "config.py")
        api.file.download(team_id, join(task_file_info.path, "config.py"), "model_config.txt")
        with open("model_config.txt", "r") as f:
            lines = f.readlines()
            project_line = lines[-1] if lines else None
            start = project_line.find("'") + 1
            end = project_line.find("'", start)
            project_name = project_line[start:end]
            task_type_line = lines[-3] if lines else None
            start = task_type_line.find("'") + 1
            end = task_type_line.find("'", start)
            task_type = task_type_line[start:end].replace("_", " ")
            f.close()
        silent_remove("model_config.txt")
        if len(checkpoints_infos) == 0:
            continue
        checkpoint_info = CheckpointInfo(
            app_name="Train MMDetection 3.0",
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

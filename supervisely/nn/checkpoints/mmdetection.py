from os.path import join
from typing import List

from supervisely.api.api import Api
from supervisely.io.fs import silent_remove
from supervisely.io.json import load_json_file
from supervisely.nn.checkpoints.checkpoint import CheckpointInfo


def get_list(api: Api, team_id: int) -> List[CheckpointInfo]:
    checkpoints = []
    weights_dir_name = "checkpoints"
    weights_subdir_name = "data"
    info_dir_name = "info"
    training_app_directory = "/mmdetection/"
    if not api.file.dir_exists(team_id, training_app_directory):
        return []
    task_files_infos = api.file.list(team_id, training_app_directory, recursive=False)
    for task_file_info in task_files_infos:
        task_id = task_file_info["name"].split("_")[0]
        project_name = task_file_info["name"].split("_")[1]
        session_link = f"{api.server_address}/apps/sessions/{task_id}"
        path_to_info = join(task_file_info["path"], info_dir_name, "ui_state.json")
        if api.file.exists(team_id, path_to_info):
            api.file.download(team_id, path_to_info, "model_config.json")
            model_config = load_json_file("model_config.json")
            task_type = model_config["task"].replace("_", " ")
            silent_remove("model_config.json")
        else:
            continue
        paths_to_checkpoints = join(task_file_info["path"], weights_dir_name, weights_subdir_name)
        checkpoints_infos = [
            file
            for file in api.file.list(team_id, paths_to_checkpoints, recursive=False)
            if file["name"].endswith(".pth")
        ]
        if len(checkpoints_infos) == 0:
            continue
        checkpoint_info = CheckpointInfo(
            app_name="Train MMDetection",
            session_id=task_id,
            session_path=task_file_info["path"],
            session_link=session_link,
            task_type=task_type,
            training_project_name=project_name,
            checkpoints=checkpoints_infos,
        )
        checkpoints.append(checkpoint_info)
    return checkpoints

from os.path import join
from typing import List

from supervisely.api.api import Api
from supervisely.nn.checkpoints.checkpoint import CheckpointInfo


def get_list(api: Api, team_id: int) -> List[CheckpointInfo]:
    checkpoints = []
    weights_dir_name = "detectron_data"
    weights_subdir_name = "inference/main_validation"  # contain instance_predictions.pth
    training_app_directory = "/detectron2/"
    task_type = "instance segmentation"
    if not api.file.dir_exists(team_id, training_app_directory):
        return []
    task_files_infos = api.file.list(team_id, training_app_directory, recursive=False)
    for task_file_info in task_files_infos:
        task_id = task_file_info["name"].split("_")[0]
        project_name = task_file_info["name"].split("_")[1]

        session_link = f"{api.server_address}/apps/sessions/{task_id}"

        paths_to_checkpoints = join(task_file_info["path"], weights_dir_name)
        checkpoints_infos = [
            file
            for file in api.file.list(team_id, paths_to_checkpoints, recursive=False)
            if file["name"].endswith(".pth")
        ]
        if len(checkpoints_infos) == 0:
            continue
        checkpoint_info = CheckpointInfo(
            app_name="Train Detectron2",
            session_id=task_id,
            session_path=task_file_info["path"],
            session_link=session_link,
            task_type=task_type,
            training_project_name=project_name,
            checkpoints=checkpoints_infos,
        )
        checkpoints.append(checkpoint_info)
    return checkpoints

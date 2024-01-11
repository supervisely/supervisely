from os.path import join
from typing import List

from supervisely.api.api import Api
from supervisely.nn.checkpoints.checkpoint import CheckpointInfo


def get_list(api: Api, team_id: int) -> List[CheckpointInfo]:
    checkpoints = []
    weights_dir_name = "weights"
    training_app_directory = "/yolov5_train/"
    task_type = "object detection"
    if not api.file.dir_exists(team_id, training_app_directory):
        return []
    project_files_infos = api.file.list(team_id, training_app_directory, recursive=False)
    for project_file_info in project_files_infos:
        project_name = project_file_info["name"]
        task_files_infos = api.file.list(team_id, project_file_info["path"], recursive=False)
        for task_file_info in task_files_infos:
            if task_file_info["name"] == "images":
                continue
            task_id = task_file_info["name"]
            session_link = f"{api.server_address}/apps/sessions/{task_id}"
            paths_to_checkpoints = join(task_file_info["path"], weights_dir_name)
            checkpoints_infos = api.file.list(team_id, paths_to_checkpoints, recursive=False)
            if len(checkpoints_infos) == 0:
                continue
            checkpoint_info = CheckpointInfo(
                app_name="Train YOLOv5",
                session_id=task_id,
                session_path=task_file_info["path"],
                session_link=session_link,
                task_type=task_type,
                training_project_name=project_name,
                checkpoints=checkpoints_infos,
            )
            checkpoints.append(checkpoint_info)
    return checkpoints

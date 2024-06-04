from os.path import basename, join
from typing import List, Literal

from supervisely._utils import abs_url, is_development
from supervisely.api.api import Api
from supervisely.nn.checkpoints.checkpoint import CheckpointInfo, Checkpoint


class YOLOv8Checkpoint(Checkpoint):
    def __init__(self, team_id: int):
        super().__init__(team_id)
        
        self._training_app = "Train YOLOv8"
        self._model_dir = "/yolov8_train"
        self._weights_dir = "weights"
        self._task_type = None
        self._weights_ext = ".pt"
        self._config_file = None
        
    def get_model_dir(self):
        return self._model_dir
        
    def get_list(self, sort: Literal["desc", "asc"] = "desc") -> List[CheckpointInfo]:
        if sort not in ["desc", "asc"]:
            raise ValueError(f"Invalid sort value: {sort}")
        if not self._api.file.dir_exists(self._team_id, self._model_dir):
            return []
        
        checkpoints = []
        object_detection_projects_dir = join(self._model_dir, "object detection")
        instance_segmentation_projects_dir = join(self._model_dir, "instance segmentation")
        pose_estimation_projects_dir = join(self._model_dir, "pose estimation")
        paths_to_projects = [
            object_detection_projects_dir,
            instance_segmentation_projects_dir,
            pose_estimation_projects_dir,
        ]
        for path_to_projects in paths_to_projects:
            if not self._api.file.dir_exists(self._team_id, path_to_projects):
                continue
            task_type = basename(path_to_projects).strip()
            project_files_infos = self._api.file.list(
                self._team_id, path_to_projects, recursive=False, return_type="fileinfo"
            )
            for project_file_info in project_files_infos:
                project_name = project_file_info.name
                task_files_infos = self._api.file.list(
                    self._team_id, project_file_info.path, recursive=False, return_type="fileinfo"
                )
                for task_file_info in task_files_infos:
                    if task_file_info.name == "images":
                        continue
                    
                    json_data = self._fetch_json_from_url(
                        f"{task_file_info.path}{self._metadata_file_name}"
                    )
                    if json_data is None:
                        json_data = self._generate_sly_metadata(
                            task_file_info.path,
                            self._weights_dir,
                            self._weights_ext,
                            self._training_app,
                            task_type,
                            project_name=project_name,
                        )
                        if json_data is None:
                            continue
                    checkpoint_info = CheckpointInfo(**json_data)
                    checkpoints.append(checkpoint_info)

        if sort == "desc":
            checkpoints = sorted(checkpoints, key=lambda x: x.session_id, reverse=True)
        elif sort == "asc":
            checkpoints = sorted(checkpoints, key=lambda x: x.session_id)
        return checkpoints

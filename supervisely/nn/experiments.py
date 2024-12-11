from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, fields
from json import JSONDecodeError
from os.path import dirname, join
from typing import List
from pathlib import Path

import requests

from supervisely import logger
from supervisely.api.api import Api, ApiField
from supervisely.nn.artifacts.artifacts import TrainInfo, BaseTrainArtifacts


@dataclass
class ExperimentInfo:
    experiment_name: str
    """Name of the experiment. Defined by the user in the training app"""
    framework_name: str
    """Name of the framework used in the experiment"""
    model_name: str
    """Name of the model used in the experiment. Defined by the user in the training app"""
    task_type: str
    """Task type of the experiment"""
    project_id: int
    """Project ID in Supervisely"""
    task_id: int
    """Task ID in Supervisely"""
    model_files: dict
    """Dictionary with paths to model files that needs to be downloaded for training"""
    checkpoints: List[str]
    """List of relative paths to checkpoints"""
    best_checkpoint: str
    """Name of the best checkpoint. Defined by the user in the training app"""
    export: dict
    """Dictionary with exported weights in different formats"""
    app_state: str
    """Path to file with settings that were used in the app"""
    model_meta: str
    """Path to file with model metadata such as model name, project id, project name and classes used for training"""
    train_val_split: str
    """Path to train and validation splits, which contains IDs of the images used in each split"""
    hyperparameters: str
    """Path to .yaml file with hyperparameters used in the experiment"""
    artifacts_dir: str
    """Path to the directory with artifacts"""
    datetime: str
    """Date and time when the experiment was started"""
    evaluation_report_id: int
    """ID of the evaluation report"""
    evaluation_metrics: dict
    """Evaluation metrics"""


def get_experiment_infos(api: Api, team_id: int, framework_name: str) -> List[ExperimentInfo]:
    """
    Get experiments from the specified framework folder for Train v2

    :param api: Supervisely API client
    :type api: Api
    :param team_id: Team ID
    :type team_id: int
    :param framework_name: Name of the framework
    :type framework_name: str
    :return: List of ExperimentInfo objects
    :rtype: List[ExperimentInfo]
    :Usage example:

     .. code-block:: python

        import supervisely as sly

        api = sly.Api.from_env()
        team_id = sly.env.team_id()
        framework_name = "rt-detr"
        experiment_infos = sly.nn.training.experiments.get_experiment_infos(api, team_id, framework_name)
    """
    metadata_name = "experiment_info.json"
    experiments_folder = "/experiments"
    experiment_infos = []

    file_infos = api.file.list(team_id, experiments_folder, recursive=True, return_type="fileinfo")
    sorted_experiment_paths = []
    for file_info in file_infos:
        if not file_info.path.endswith(metadata_name):
            continue

        experiment_dir = dirname(file_info.path)
        if experiment_dir.endswith(framework_name):
            experiment_path = join(experiment_dir, metadata_name)
            sorted_experiment_paths.append(experiment_path)

    def fetch_experiment_data(experiment_path: str):
        try:
            response = api.post(
                "file-storage.download",
                {ApiField.TEAM_ID: team_id, ApiField.PATH: experiment_path},
                stream=True,
            )
            response.raise_for_status()
            response_json = response.json()
            required_fields = {field.name for field in fields(ExperimentInfo)}
            missing_fields = required_fields - response_json.keys()
            if missing_fields:
                logger.debug(f"Missing fields: {missing_fields} for '{experiment_path}'")
                return None
            return ExperimentInfo(**response_json)
        except requests.exceptions.RequestException as e:
            logger.debug(f"Request failed for '{experiment_path}': {e}")
        except JSONDecodeError as e:
            logger.debug(f"JSON decode failed for '{experiment_path}': {e}")
        except TypeError as e:
            logger.error(f"TypeError for '{experiment_path}': {e}")
        return None

    with ThreadPoolExecutor() as executor:
        experiment_infos = list(executor.map(fetch_experiment_data, sorted_experiment_paths))

    experiment_infos = [info for info in experiment_infos if info is not None]
    return experiment_infos


def build_experiment_info_list_from_train_infos(
    api: Api, framework_cls: BaseTrainArtifacts, train_infos: List[TrainInfo]
) -> List[ExperimentInfo]:

    def build_experiment_info_from_train_info(
        api: Api, framework_cls: BaseTrainArtifacts, train_info: TrainInfo
    ) -> ExperimentInfo:

        # Convert checkpoint files into absolute paths
        checkpoints = [
            join(framework_cls.weights_folder, chk.name) for chk in train_info.checkpoints
        ]

        # Identify best checkpoint (if any), otherwise fallback to the last checkpoint
        best_checkpoint = next(
            (chk.name for chk in train_info.checkpoints if "best" in chk.name), None
        )
        if not best_checkpoint and checkpoints:
            best_checkpoint = Path(checkpoints[-1]).name

        # Retrieve task info and workspace
        task_info = api.task.get_info_by_id(train_info.task_id)
        workspace_id = task_info["workspaceId"]

        # Retrieve project info (if available)
        project = api.project.get_info_by_name(workspace_id, train_info.project_name)
        project_id = project.id if project else None

        # Prepare model files dictionary
        model_files = {}
        if train_info.config_path:
            model_files["config"] = Path(train_info.config_path).name

        # Basic experiment info data
        experiment_info_data = {
            "experiment_name": f"Unknown {framework_cls.framework_name} experiment",
            "framework_name": framework_cls.framework_name,
            "model_name": f"Unknown {framework_cls.framework_name} model",
            "task_type": train_info.task_type,
            "project_id": project_id,
            "task_id": train_info.task_id,
            "model_files": model_files,
            "checkpoints": checkpoints,
            "best_checkpoint": best_checkpoint,
            "export": None,
            "app_state": None,
            "model_meta": None,
            "train_val_split": None,
            "hyperparameters": None,
            "artifacts_dir": train_info.artifacts_folder,
            "datetime": task_info["startedAt"],
            "evaluation_report_id": None,
            "evaluation_metrics": {},
        }

        # Ensure all fields of ExperimentInfo are present, set missing to None
        experiment_info_fields = {
            field.name for field in ExperimentInfo.__dataclass_fields__.values()
        }
        for field in experiment_info_fields:
            if field not in experiment_info_data:
                experiment_info_data[field] = None

        return ExperimentInfo(**experiment_info_data)

    # Async version
    with ThreadPoolExecutor() as executor:
        experiment_infos = list(
            executor.map(
                lambda t: build_experiment_info_from_train_info(api, framework_cls, t), train_infos
            )
        )
    return [info for info in experiment_infos if info is not None]

    # Sync version
    # Uncomment for debug
    # experiment_infos = [
    #     convert_traininfo_to_experimentinfo(api, framework_cls, train_info)
    #     for train_info in train_infos
    # ]
    # return experiment_infos

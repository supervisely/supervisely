from concurrent.futures import ThreadPoolExecutor
from dataclasses import MISSING, dataclass, fields
from json import JSONDecodeError
from os.path import dirname, join
from typing import Dict, List, Optional, Union

import requests

from supervisely import logger
from supervisely.api.api import Api, ApiField

EXPERIMENT_INFO_FILENAME = "experiment_info.json"


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
    model_meta: str
    """Path to file with model metadata such as model name, project id, project name and classes used for training"""
    checkpoints: List[str]
    """List of relative paths to checkpoints"""
    best_checkpoint: str
    """Name of the best checkpoint. Defined by the user in the training app"""
    hyperparameters: str
    """Path to .yaml file with hyperparameters used in the experiment"""
    artifacts_dir: str
    """Path to the directory with artifacts"""
    base_checkpoint: Optional[str] = None
    """Name of the base checkpoint used for training"""
    base_checkpoint_link: Optional[str] = None
    """Link to the base checkpoint used for training. URL in case of pretrained model, or Team Files path in case of custom model."""
    export: Optional[dict] = None
    """Dictionary with exported weights in different formats"""
    app_state: Optional[str] = None
    """Path to file with settings that were used in the app"""
    train_val_split: Optional[str] = None
    """Path to train and validation splits, which contains IDs of the images used in each split"""
    train_size: Optional[int] = None
    """Number of images in the training set"""
    val_size: Optional[int] = None
    """Number of images in the validation set"""
    datetime: Optional[str] = None
    """Date and time when the experiment was started"""
    experiment_report_id: Optional[int] = None
    """ID of the experiment report"""
    evaluation_report_id: Optional[int] = None
    """ID of the model benchmark evaluation report"""
    evaluation_report_link: Optional[str] = None
    """Link to the model benchmark evaluation report"""
    evaluation_metrics: Optional[dict] = None
    """Evaluation metrics"""
    logs: Optional[dict] = None
    """Dictionary with link and type of logger"""
    train_collection_id: Optional[int] = None
    """ID of the collection with train images"""
    val_collection_id: Optional[int] = None
    """ID of the collection with validation images"""
    project_version: Optional[int] = None
    """Version of the project"""

    def __init__(self, **kwargs):
        required_fieds = {
            field.name for field in fields(self.__class__) if field.default is MISSING
        }
        missing_fields = required_fieds - set(kwargs.keys())
        if missing_fields:
            raise ValueError(
                f"ExperimentInfo missing required arguments: '{', '.join(missing_fields)}'"
            )
        field_names = set(f.name for f in fields(self.__class__))
        kwargs = {k: v for k, v in kwargs.items() if k in field_names}
        for key, value in kwargs.items():
            setattr(self, key, value)

    def to_json(self) -> Dict:
        data = {}
        for field in fields(self.__class__):
            value = getattr(self, field.name)
            data[field.name] = value
        return data


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
        framework_name = "RT-DETRv2"
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
            required_fields = {
                field.name for field in fields(ExperimentInfo) if field.default is not None
            }
            optional_fields = {
                field.name for field in fields(ExperimentInfo) if field.default is None
            }

            missing_optional_fields = optional_fields - response_json.keys()
            if missing_optional_fields:
                logger.debug(
                    f"Missing optional fields: {missing_optional_fields} for '{experiment_path}'"
                )
                for field in missing_optional_fields:
                    response_json[field] = None

            missing_required_fields = required_fields - response_json.keys()
            if missing_required_fields:
                logger.debug(
                    f"Missing required fields: {missing_required_fields} for '{experiment_path}'. Skipping."
                )
                return None
            field_names = {field.name for field in fields(ExperimentInfo)}
            return ExperimentInfo(**{k: v for k, v in response_json.items() if k in field_names})
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


def _fetch_experiment_data(api, team_id: int, experiment_path: str) -> Union[ExperimentInfo, None]:
    """
    Fetch experiment data from the specified path in Supervisely Team Files

    :param api: Supervisely API client
    :type api: Api
    :param team_id: Team ID
    :type team_id: int
    :param experiment_path: Path to the experiment data
    :type experiment_path: str
    :return: ExperimentInfo object
    :rtype: Union[ExperimentInfo, None]
    """
    try:
        response = api.post(
            "file-storage.download",
            {ApiField.TEAM_ID: team_id, ApiField.PATH: experiment_path},
            stream=True,
        )
        response.raise_for_status()
        response_json = response.json()
        required_fields = {
            field.name for field in fields(ExperimentInfo) if field.default is MISSING
        }
        optional_fields = {
            field.name for field in fields(ExperimentInfo) if field.default is not MISSING
        }

        missing_optional_fields = optional_fields - response_json.keys()
        if missing_optional_fields:
            logger.debug(
                f"Missing optional fields: {missing_optional_fields} for '{experiment_path}'"
            )
            for field in missing_optional_fields:
                response_json[field] = None

        missing_required_fields = required_fields - response_json.keys()
        if missing_required_fields:
            logger.debug(
                f"Missing required fields: {missing_required_fields} for '{experiment_path}'. Skipping."
            )
            return None
        all_fields = required_fields | optional_fields
        return ExperimentInfo(**{k: v for k, v in response_json.items() if k in all_fields})
    except requests.exceptions.RequestException as e:
        logger.debug(f"Request failed for '{experiment_path}': {e}")
    except JSONDecodeError as e:
        logger.debug(f"JSON decode failed for '{experiment_path}': {e}")
    except TypeError as e:
        logger.error(f"TypeError for '{experiment_path}': {e}")
    return None


def get_experiment_info_by_artifacts_dir(
    api: Api, team_id: int, artifacts_dir: str
) -> Union[ExperimentInfo, None]:
    """
    Get experiment info by artifacts directory

    :param api: Supervisely API client
    :type api: Api
    :param team_id: Team ID
    :type team_id: int
    :param artifacts_dir: Path to the directory with artifacts
    :type artifacts_dir: str
    :return: ExperimentInfo object
    :rtype: Optional[ExperimentInfo]
    :Usage example:

     .. code-block:: python

        import supervisely as sly

        api = sly.Api.from_env()
        team_id = sly.env.team_id()
        artifacts_dir = "/experiments/27_Lemons (Rectangle)/265_RT-DETRv2/"
        experiment_info = sly.nn.training.experiments.get_experiment_info_by_artifacts_dir(api, team_id, artifacts_dir)
    """
    if not artifacts_dir.startswith("/experiments"):
        raise ValueError("Artifacts directory should start with '/experiments'")
    experiment_path = join(artifacts_dir, EXPERIMENT_INFO_FILENAME)
    return _fetch_experiment_data(api, team_id, experiment_path)

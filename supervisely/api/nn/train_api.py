from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, List, Literal, Union

import supervisely.io.env as env
from supervisely._utils import logger
from supervisely.api.api import Api
from supervisely.api.nn.utils import (
    find_apps_by_framework,
    find_team_by_path,
    get_artifacts_dir_and_checkpoint_name,
    run_app,
)
from supervisely.nn.experiments import get_experiment_info_by_artifacts_dir


class Model:
    def __init__(
        self,
        api: "Api",
        source: str,
        framework: str = None,
        model_name: str = None,
        team_id: int = None,
        artifacts_dir: str = None,
        checkpoint_name: str = None
    ):
        self.api = api
        self.source: str = source
        self.framework: str = framework
        self.model_name: str = model_name
        self.team_id: int = team_id
        self.artifacts_dir: str = artifacts_dir
        self.checkpoint_name: str = checkpoint_name
        self._init()

    def _init(self):
        if self.source == "Custom models":
            if self.team_id is None:
                self.team_id = find_team_by_path(self.api, self.artifacts_dir, team_id=self.team_id)
            experiment_info = get_experiment_info_by_artifacts_dir(self.api, self.team_id, self.artifacts_dir)
            if not experiment_info:
                raise ValueError(
                    f"Failed to retrieve experiment info for artifacts_dir: '{self.artifacts_dir}'"
                )
            self.task_id = experiment_info.task_id
            self.framework = experiment_info.framework_name


    @classmethod
    def parse(cls, api: "Api", model: str) -> Model:
        checkpoint = None
        team_id = None
        if model.startswith("/"):
            checkpoint = model
        else:
            found_team_id = find_team_by_path(api,
                f"/{model}", team_id=team_id, raise_not_found=False
            )
            if found_team_id is not None:
                checkpoint = f"/{model}"
                team_id = found_team_id
                logger.debug(f"Found checkpoint in team {team_id}")
            else:
                pretrained = model

        if checkpoint is not None:
            artifacts_dir, checkpoint_name = get_artifacts_dir_and_checkpoint_name(checkpoint)
            return cls(
                source="Custom models",
                artifacts_dir=artifacts_dir,
                checkpoint_name=checkpoint_name,
                team_id=team_id
            )
        else:
            framework, model_name = pretrained.split("/", 1)
            return cls(
                source="Pretrained models",
                framework=framework,
                model_name=model_name
            )
    
    def app_state(self):
        if self.source == "Pretrained models":
            return {
                "source": self.source,
                "model_name": self.model_name,
            }
        else:
            return {
                "source": self.source,
                "task_id": self.task_id,
                "checkpoint": self.checkpoint_name
            }


class _TrainValSplit:
    @abstractmethod
    def app_state(self) -> Dict[str, Any]:
        raise NotImplementedError()
    

class RandomSplit(_TrainValSplit):
    def __init__(self, percent: int = 80, split: str = "train"):
        self.percent = percent
        self.split = split
    
    def app_state(self):
        return {
            "method": "random",
            "split": self.split,
            "percent": self.percent
        }

class DatasetsSplit(_TrainValSplit):
    def __init__(self, train_datasets: List[int], val_datasets: List[int]):
        self.train_datasets = train_datasets
        self.val_datasets = val_datasets
    
    def app_state(self):
        return {
            "method": "datasets",
            "train_datasets": self.train_datasets,
            "val_datasets": self.val_datasets
        }

class TagsSplit(_TrainValSplit):
    def __init__(self, train_tag: str, val_tag: str, untagged_action: Literal["train", "val", "ignore"]):
        self.train_tag = train_tag
        self.val_tag = val_tag
        self.untagged_action = untagged_action

    def app_state(self):
        return {
            "method": "tags",
            "train_tag": self.train_tag,
            "val_tag": self.val_tag,
            "untagged_action": self.untagged_action
        }

class CollectionsSplit(_TrainValSplit):
    def __init__(self, train_collections: List[int], val_collecitons: List[int]):
        self.train_collections = train_collections
        self.val_collections = val_collecitons

    def app_state(self):
        return {
            "method": "collections",
            "train_collections": self.train_collections,
            "val_collections": self.val_collections
        }

class TrainApi:
    """ """

    def __init__(self, api: "Api"):
        self._api = api
    
    def _get_app_state(
        self,
        project_id: int,
        model: Model,
        classes: List[str],
        train_val_split: _TrainValSplit,
        experiment_name: str,
        hyperparameters: str,
        convert_class_shapes: bool = True,
        enable_benchmark: bool = True,
        enable_speedtest: bool = True,
        cache_project: bool = True,
        export_onnx: bool = False,
        export_tensorrt: bool = False,
        autostart: bool = True,
    ):
        app_state = {
            # 1. Project
            "slyProjectId": project_id,
            "guiState": {
                # 2. Model
                "model": model.app_state(),
                # 3. Classes
                "classes": classes,
                # 4. Train/Val Split
                "train_val_split": train_val_split.app_state(), # or latest train_xxx val_xxx collections
                # 5. Hyperparameters
                "hyperparameters": hyperparameters,  # yaml string
                # 6. Options
                "options": {
                    "convert_class_shapes": convert_class_shapes,
                    "model_benchmark": {
                        "enable": enable_benchmark or enable_speedtest,
                        "speed_test": enable_speedtest,
                    },
                    "cache_project": cache_project,
                    "export": {
                        "enable": export_onnx or export_tensorrt,
                        "ONNXRuntime": export_onnx,
                        "TensorRT": export_tensorrt,
                    },
                },
                # 7. Experiment Name
                "experiment_name": experiment_name,
                # 8. Start Training
                "start_training": autostart,  # Starts training automatically
            },
        }
        return app_state
    
    def run(
        self,
        project_id: int,
        model: str,
        hyperparameters: str = None,
        experiment_name: str = None,
        classes: List[str] = None,
        train_val_split: Union[RandomSplit, DatasetsSplit, TagsSplit, CollectionsSplit] = None,
        convert_class_shapes: bool = True,
        enable_benchmark: bool = True,
        enable_speedtest: bool = True,
        cache_project: bool = True,
        export_onnx: bool = False,
        export_tensorrt: bool = False,
        autostart: bool = True,
        **kwargs
    ):
        """
        Docstring for run
        :param model: Either a path to a model checkpoint in team files or model name in format `framework/model_name` (e.g., "RT-DETRv2/RT-DETRv2-M").
        :type model: str
        """
        project_info = self._api.project.get_info_by_id(project_id)
        workspace_id = project_info.workspace_id

        model: Model = Model.parse(self._api, model)

        project_meta_json = self._api.project.get_meta(project_id)
        project_meta = sly.ProjectMeta.from_json(project_meta_json)
        if classes: # todo: warning if passed class not in project meta
            classes = [obj_class.name for obj_class in project_meta.obj_classes if obj_class.name in classes]
        else:
            classes = [obj_class.name for obj_class in project_meta.obj_classes]

        if train_val_split is None:
            train_val_split = RandomSplit()
        
        module = self.find_train_app_by_framework(model.framework)
        if module is None:
            raise ValueError(f"Failed to detect train app by framework: '{model.framework}'")
        module_id = module["id"]

        if hyperparameters is None:
            pass
        
        app_state = self._get_app_state(
            project_id=project_id,
            model=model,
            classes=classes,
            train_val_split=train_val_split,
            experiment_name=experiment_name,
            hyperparameters=hyperparameters,
            convert_class_shapes = convert_class_shapes,
            enable_benchmark = enable_benchmark,
            enable_speedtest = enable_speedtest,
            cache_project = cache_project,
            export_onnx = export_onnx,
            export_tensorrt = export_tensorrt,
            autostart = autostart,
        )
        task_info = run_app(
            api=self._api,
            agent_id=agent_id,
            module_id=module_id,
            workspace_id=workspace_id,
            params=app_state,
            **kwargs,
        )
        return task_info
    
    def find_train_app_by_framework(self, framework: str):
        modules = find_apps_by_framework(self._api, framework, ["train"])
        if not modules:
            return None
        return modules[0]
    
    def finetune(
        self,
        task_id: int,
        project_id: int = None,
        hyperparameters: str = None,
        experiment_name: str = None,
        classes: List[str] = None,
        train_val_split: Union[RandomSplit, DatasetsSplit, TagsSplit, CollectionsSplit] = None,
        convert_class_shapes: bool = True,
        enable_benchmark: bool = True,
        enable_speedtest: bool = True,
        cache_project: bool = True,
        export_onnx: bool = False,
        export_tensorrt: bool = False,
        autostart: bool = True
    ):
        # on best checkpoint
        experiment_info = get_experiment_info_by_task_id()
        self.run(
            experiment_info.project_id,
        )

    def run_new(
        self,
        task_id: int,
        project_id: int = None,
        hyperparameters: str = None,
        experiment_name: str = None,
        classes: List[str] = None,
        train_val_split: Union[RandomSplit, DatasetsSplit, TagsSplit, CollectionsSplit] = None,
        convert_class_shapes: bool = True,
        enable_benchmark: bool = True,
        enable_speedtest: bool = True,
        cache_project: bool = True,
        export_onnx: bool = False,
        export_tensorrt: bool = False,
        autostart: bool = True
    ):
        # checkpoint that was used in this task
        pass



import os

from dotenv import load_dotenv

import supervisely as sly

load_dotenv("local.env")
load_dotenv(os.path.expanduser("~/supervisely.env"))


api = sly.Api.from_env()
agent_id = sly.env.agent_id()  # take from env or enter ID
workspace_id = sly.env.workspace_id()  # take from env or enter ID
project_id = sly.env.project_id()  # take from env or enter ID
experiment_name = "My Experiment"

# Read from yaml file for convenience
# Example: https://github.com/supervisely-ecosystem/yolo/blob/master/supervisely_integration/train/hyperparameters.yaml
hyperparameters_path = "hyperparameters.yaml"
with open(hyperparameters_path, "r") as file:
    hyperparameters = file.read()

train = TrainApi(api)
train.run(
    project_id=project_id,
    model="model",
    experiment_name=experiment_name,
    train_val_split=RandomSplit(),
    hyperparameters=hyperparameters,
)


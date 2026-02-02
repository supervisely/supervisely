"""API for training neural network models in Supervisely."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, List, Literal, Optional, Union

import supervisely.io.env as env
from supervisely import ProjectMeta
from supervisely._utils import logger
from supervisely.api.api import Api
from supervisely.api.nn.utils import (
    find_agent,
    find_apps_by_framework,
    find_team_by_path,
    get_artifacts_dir_and_checkpoint_name,
    get_experiment_info_by_task_id,
    run_train_app,
)
from supervisely.nn.experiments import get_experiment_info_by_artifacts_dir


class Model:
    """
    This class normalizes the user input into a structure that can be embedded into an
    application state (`guiState["model"]`) for a training app.

    The training UI supports two model sources:
    - **Pretrained models**: referenced by `"framework/model_name"` (e.g. `"RT-DETRv2/RT-DETRv2-M"`).
    - **Custom models**: referenced by a path to a checkpoint in Team Files (starts with `"/"`),
      or by a path without leading slash if it exists in Team Files.
    """

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
        """
        Initialize a :class:`~supervisely.api.nn.train_api.Model` instance.

        :param api: Supervisely API client.
        :type api: :class:`~supervisely.api.api.Api`
        :param source: Source of the model.
        :type source: str
        :param framework: Framework of the model.
        :type framework: str
        :param model_name: Name of the model.
        :type model_name: str
        :param team_id: Team id of the model.
        :type team_id: int
        :param artifacts_dir: Artifacts directory of the model.
        :type artifacts_dir: str
        :param checkpoint_name: Checkpoint name of the model.
        :type checkpoint_name: str
        """
        self.api = api
        self.source: str = source
        self.framework: str = framework
        self.model_name: str = model_name
        self.team_id: int = team_id
        self.artifacts_dir: str = artifacts_dir
        self.checkpoint_name: str = checkpoint_name
        self._init()

    def _init(self):
        """
        Finalize initialization for custom models.

        For custom checkpoints, we resolve experiment metadata to determine:
        - `task_id` of the experiment that produced artifacts.
        - `framework` name used by that experiment.
        """
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
        """
        Parse a user-provided model identifier into a `Model`.

        :param api: Supervisely API client.
        :type api: :class:`~supervisely.api.api.Api`
        :param model: Either a checkpoint path starting with "/" (Team Files), a checkpoint path
            without leading slash (will be checked in Team Files), or a pretrained model name in
            format "framework/model_name" (e.g. "RT-DETRv2/RT-DETRv2-M", "YOLO/YOLO26s-det").
        :type model: str
        :returns: Parsed model reference ready to be embedded into training app state.
        :rtype: :class:`~supervisely.api.nn.train_api.Model`
        :raises ValueError: If the string cannot be parsed or required metadata cannot be resolved.
        """
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
                api=api,
                source="Custom models",
                artifacts_dir=artifacts_dir,
                checkpoint_name=checkpoint_name,
                team_id=team_id,
            )
        else:
            framework, model_name = pretrained.split("/", 1)
            return cls(
                api=api, source="Pretrained models", framework=framework, model_name=model_name
            )

    def app_state(self):
        """Return a JSON-serializable dict for `guiState["model"]`."""
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
    """Base class for Train/Val split strategies used by training UI."""

    @abstractmethod
    def app_state(self) -> Dict[str, Any]:
        """Return a JSON-serializable dict for `guiState["train_val_split"]`.

        :returns: Dict for training app UI state.
        :rtype: Dict[str, Any]
        """
        raise NotImplementedError()


class RandomSplit(_TrainValSplit):
    """Split dataset randomly by percent.

    :param percent: Percent of the dataset to split into train.
    :type percent: int
    :param split: Split method: "train" or "val".
    :type split: str
    """

    def __init__(self, percent: int = 80, split: str = "train"):
        self.percent = percent
        self.split = split

    def app_state(self):
        """Return a JSON-serializable dict for `guiState["train_val_split"]`."""
        return {
            "method": "random",
            "split": self.split,
            "percent": self.percent
        }

class DatasetsSplit(_TrainValSplit):
    """Split by explicit train/val dataset ids inside the project.

    :param train_datasets: List of dataset ids to split into train.
    :type train_datasets: List[int]
    :param val_datasets: List of dataset ids to split into val.
    :type val_datasets: List[int]
    """

    def __init__(self, train_datasets: List[int], val_datasets: List[int]):
        self.train_datasets = train_datasets
        self.val_datasets = val_datasets

    def app_state(self):
        """Serialize split settings for training UI."""
        return {
            "method": "datasets",
            "train_datasets": self.train_datasets,
            "val_datasets": self.val_datasets
        }

class TagsSplit(_TrainValSplit):
    """
    Split by tags: items with `train_tag` go to train, `val_tag` to val.

    :param train_tag: Tag to split into train.
    :type train_tag: str
    :param val_tag: Tag to split into val.
    :type val_tag: str
    :param untagged_action: Action to take for untagged items: "train", "val", "ignore".
    :type untagged_action: Literal["train", "val", "ignore"]
    """

    def __init__(self, train_tag: str, val_tag: str, untagged_action: Literal["train", "val", "ignore"]):
        self.train_tag = train_tag
        self.val_tag = val_tag
        self.untagged_action = untagged_action

    def app_state(self):
        """Serialize split settings for training UI."""
        return {
            "method": "tags",
            "train_tag": self.train_tag,
            "val_tag": self.val_tag,
            "untagged_action": self.untagged_action
        }

class CollectionsSplit(_TrainValSplit):
    """
    Split by entity collections (train collections vs val collections).

    :param train_collections: List of collection ids to split into train.
    :type train_collections: List[int]
    :param val_collections: List of collection ids to split into val.
    :type val_collections: List[int]
    """

    def __init__(self, train_collections: List[int], val_collecitons: List[int]):
        self.train_collections = train_collections
        self.val_collections = val_collecitons

    def app_state(self):
        """Serialize split settings for training UI."""
        return {
            "method": "collections",
            "train_collections": self.train_collections,
            "val_collections": self.val_collections
        }

class TrainApi:
    """High-level API to start a training application.

    You can read more about the training API in the [Training API documentation](https://developer.supervisely.com/advanced-user-guide/automate-with-python-sdk-and-api/training-api).

    This wrapper prepares the `params`/`state` payload expected by the training UI app
    and starts an app task on a given agent.

    Typical usage:

    - Choose a model (pretrained or custom checkpoint)
    - Provide training settings (model, classes, train/val split, hyperparameters, etc.)
    - Start the training app
    """

    def __init__(self, api: "Api"):
        """
        Create a :class:`~supervisely.api.nn.train_api.TrainApi` instance.

        :param api: Supervisely API client.
        :type api: :class:`~supervisely.api.api.Api`
        """
        self._api = api

    def _get_app_state(
        self,
        project_id: int,
        model: Model,
        classes: List[str],
        train_val_split: _TrainValSplit,
        experiment_name: str = None,
        hyperparameters: str = None,
        convert_class_shapes: bool = True,
        enable_benchmark: bool = True,
        enable_speedtest: bool = False,
        cache_project: bool = True,
        export_onnx: bool = False,
        export_tensorrt: bool = False,
        autostart: bool = True,
    ):
        """
        Build training app state payload.

        The resulting structure is passed to the training app as `params` (task arguments).
        It follows the `TrainApp`/GUI expected schema:

        - `state.slyProjectId`: project id
        - `state.guiState`: UI state (model/classes/split/hyperparameters/options/experiment_name/start_training)

        Notes:
            - `hyperparameters` is expected to be a YAML string (or `None` to keep defaults from training app).

        :param project_id: Project id to train on.
        :type project_id: int
        :param model: Parsed model reference.
        :type model: Model
        :param classes: Class names to train on (filtered to project meta upstream).
        :type classes: List[str]
        :param train_val_split: Train/Val split strategy for the GUI.
        :type train_val_split: :class:`~supervisely.api.nn.train_api._TrainValSplit`
        :param experiment_name: Optional experiment name shown in UI and used for artifacts.
        :type experiment_name: str, optional
        :param hyperparameters: Hyperparameters YAML string for the training app. If None, GUI keeps defaults.
        :type hyperparameters: str, optional
        :param convert_class_shapes: Whether to auto-convert shapes to framework requirements.
        :type convert_class_shapes: bool
        :param enable_benchmark: Enable post-training evaluation (Model Benchmark).
        :type enable_benchmark: bool
        :param enable_speedtest: Enable speed test as part of benchmark.
        :type enable_speedtest: bool
        :param cache_project: Cache project on agent before training.
        :type cache_project: bool
        :param export_onnx: Enable export to ONNXRuntime (if supported by the app/framework).
        :type export_onnx: bool
        :param export_tensorrt: Enable export to TensorRT engine (if supported by the app/framework).
        :type export_tensorrt: bool
        :param autostart: If True, training is started automatically after UI state is applied.
        :type autostart: bool
        :returns: Task params payload for the training app.
        :rtype: Dict[str, Any]
        """
        app_state = {
            "state": {
                # 1. Project
                "slyProjectId": project_id,
                "guiState": {
                    # 2. Model
                    "model": model.app_state(),
                    # 3. Classes
                    "classes": classes,
                    # 4. Train/Val Split
                    "train_val_split": train_val_split.app_state(),
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
        enable_speedtest: bool = False,
        cache_project: bool = True,
        export_onnx: bool = False,
        export_tensorrt: bool = False,
        autostart: bool = True,
        agent_id: int = None,
        **kwargs,
    ):
        """
        Start a training application task for a project.

        :param agent_id: Agent ID where the app task will run.
        :type agent_id: int
        :param project_id: Project ID to train on.
        :type project_id: int
        :param model: Either a checkpoint path in Team Files (e.g. "/experiments/.../checkpoints/best.pth"),
            or a pretrained model name in format "framework/model_name" (e.g. "RT-DETRv2/RT-DETRv2-M", "YOLO/YOLO26s-det").
        :type model: str
        :param hyperparameters: Hyperparameters YAML string for the training app. If None, uses defaults from training app.
        :type hyperparameters: str, optional
        :param experiment_name: Optional experiment name used in training app. Will be auto-generated if not provided.
        :type experiment_name: str, optional
        :param classes: Optional subset of class names to train on. If provided, names not present in project meta are ignored.
        :type classes: List[str], optional
        :param train_val_split: Optional split strategy; defaults to :class:`~supervisely.api.nn.train_api.RandomSplit`.
        :type train_val_split: Union[:class:`~supervisely.api.nn.train_api.RandomSplit`, :class:`~supervisely.api.nn.train_api.DatasetsSplit`, :class:`~supervisely.api.nn.train_api.TagsSplit`, :class:`~supervisely.api.nn.train_api.CollectionsSplit`], optional
        :param convert_class_shapes: Whether to convert class shapes to framework requirements automatically.
        :type convert_class_shapes: bool, optional
        :param enable_benchmark: Enable post-training evaluation (Model Benchmark) after training.
        :type enable_benchmark: bool, optional
        :param enable_speedtest: Enable speed test as part of benchmark after training.
        :type enable_speedtest: bool, optional
        :param cache_project: Cache project on agent before training to save time on downloading project next time.
        :type cache_project: bool, optional
        :param export_onnx: Enable export to ONNXRuntime (if supported by the training app/framework).
        :type export_onnx: bool, optional
        :param export_tensorrt: Enable export to TensorRT engine (if supported by the training app/framework).
        :type export_tensorrt: bool, optional
        :param autostart: If True, training is started automatically after all settings are applied. If False, training must be started manually from the training app UI by clicking the "Start Training" button.
        :type autostart: bool, optional
        :returns: Task information dict for the created app task.
        :rtype: Dict[str, Any]
        :raises ValueError: If a suitable training app cannot be found for the detected framework.

        :Usage Example:

            .. code-block:: python

                import os
                from dotenv import load_dotenv

                import supervisely as sly
                from supervisely.api.nn.train_api import TrainApi

                if sly.is_development():
                    load_dotenv("local.env")
                    load_dotenv(os.path.expanduser("~/supervisely.env"))

                api = sly.Api.from_env()

                agent_id = sly.env.agent_id()
                project_id = sly.env.project_id()

                train = TrainApi(api)
                train.run(agent_id, project_id, model="YOLO/YOLO26s-det")
        """
        model: Model = Model.parse(self._api, model)
        return self._run(
            project_id=project_id,
            model=model,
            hyperparameters=hyperparameters,
            experiment_name=experiment_name,
            classes=classes,
            train_val_split=train_val_split,
            convert_class_shapes=convert_class_shapes,
            enable_benchmark=enable_benchmark,
            enable_speedtest=enable_speedtest,
            cache_project=cache_project,
            export_onnx=export_onnx,
            export_tensorrt=export_tensorrt,
            autostart=autostart,
            agent_id=agent_id,
            **kwargs
        )
    
    def _run(
        self,
        project_id: int,
        model: Model,
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
        agent_id: int = None,
        **kwargs
    ) -> Dict:
        project_info = self._api.project.get_info_by_id(project_id)
        workspace_id = project_info.workspace_id
        team_id = project_info.team_id

        project_meta_json = self._api.project.get_meta(project_id)
        project_meta = ProjectMeta.from_json(project_meta_json)
        if classes:
            classes = [obj_class.name for obj_class in project_meta.obj_classes if obj_class.name in classes]
        else:
            classes = [obj_class.name for obj_class in project_meta.obj_classes]

        if train_val_split is None:
            train_val_split = RandomSplit()

        module = self.find_train_app_by_framework(model.framework)
        if module is None:
            raise ValueError(f"Failed to detect train app by framework: '{model.framework}'")
        module_id = module["id"]

        if agent_id is None:
            agent_id = find_agent(self._api, team_id)

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
        task_info = run_train_app(
            api=self._api,
            agent_id=agent_id,
            module_id=module_id,
            workspace_id=workspace_id,
            app_state=app_state,
            timeout=100,
            **kwargs
        )
        return task_info

    def find_train_app_by_framework(self, framework: str):
        """Find a training app module for the given framework.

        :param framework: Framework name (e.g. "RT-DETRv2", "YOLO", "DEIM").
        :type framework: str
        :returns: Ecosystem module dict (as returned by the API) or None if not found.
        :rtype: Union[dict, None]
        """
        modules = find_apps_by_framework(self._api, framework, ["train"])
        if not modules:
            return None
        return modules[0]

    def finetune(
        self,
        task_id: int,
        project_id: int = None,
        agent_id: int = None,
        **kwargs
    ) -> Dict:
        experiment_info = get_experiment_info_by_task_id(self._api, task_id)
        if experiment_info is None:
            raise ValueError(f"Not found experiment data for task {task_id}")
        module = self.find_train_app_by_framework(experiment_info.framework_name)
        if module is None:
            raise ValueError(f"Failed to detect train app by framework: '{experiment_info.framework_name}'")
        if project_id is None:
            project_id = experiment_info.project_id
        project_info = self._api.project.get_info_by_id(project_id)
        team_id = project_info.team_id
        if agent_id is None:
            agent_id = find_agent(self._api, team_id)
        module_id = module["id"]
        app_state = {"state": {"trainTaskId": task_id, "trainMode": "continue", "slyProjectId": project_id}}
        task_info = run_train_app(
            api=self._api,
            agent_id=agent_id,
            module_id=module_id,
            timeout=100,
            workspace_id=project_info.workspace_id,
            app_state=app_state,
            **kwargs
        )
        return task_info

    def run_new(
        self,
        task_id: int,
        project_id: int = None,
        agent_id: int = None,
        **kwargs
    ):
        experiment_info = get_experiment_info_by_task_id(self._api, task_id)
        if experiment_info is None:
            raise ValueError(f"Not found experiment data for task {task_id}")
        module = self.find_train_app_by_framework(experiment_info.framework_name)
        if module is None:
            raise ValueError(f"Failed to detect train app by framework: '{experiment_info.framework_name}'")
        if project_id is None:
            project_id = experiment_info.project_id
        project_info = self._api.project.get_info_by_id(project_id)
        team_id = project_info.team_id
        if agent_id is None:
            agent_id = find_agent(self._api, team_id)
        module_id = module["id"]
        app_state = {"state": {"trainTaskId": task_id, "trainMode": "new", "slyProjectId": project_id}}
        task_info = run_train_app(
            api=self._api,
            agent_id=agent_id,
            module_id=module_id,
            timeout=100,
            workspace_id=project_info.workspace_id,
            app_state=app_state,
            **kwargs
        )
        return task_info

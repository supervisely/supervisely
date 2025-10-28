"""
TrainApp module.

This module contains the `TrainApp` class and related functionality to facilitate
training workflows in a Supervisely application.
"""

import shutil
import subprocess
import time
from datetime import datetime
from os import getcwd, listdir, walk
from os.path import basename, dirname, exists, expanduser, isdir, isfile, join
from typing import Any, Dict, List, Literal, Optional, Union
from urllib.request import urlopen

import httpx
import yaml
from fastapi import Request, Response
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask

import supervisely.io.env as sly_env
import supervisely.io.fs as sly_fs
import supervisely.io.json as sly_json
from supervisely import (
    Api,
    Application,
    Dataset,
    DatasetInfo,
    OpenMode,
    Project,
    ProjectInfo,
    ProjectMeta,
    ProjectType,
    VideoProject,
    WorkflowMeta,
    WorkflowSettings,
    batched,
    download_project,
    is_development,
    is_production,
    logger,
)
from supervisely._utils import abs_url, get_filename_from_headers
from supervisely.api.file_api import FileInfo
from supervisely.app import get_synced_data_dir, show_dialog
from supervisely.app.widgets import Progress
from supervisely.decorators.profile import timeit_with_result
from supervisely.nn.benchmark import (
    InstanceSegmentationBenchmark,
    InstanceSegmentationEvaluator,
    ObjectDetectionBenchmark,
    ObjectDetectionEvaluator,
    SemanticSegmentationBenchmark,
    SemanticSegmentationEvaluator,
)
from supervisely.nn.inference import RuntimeType, SessionJSON
from supervisely.nn.inference.inference import Inference, torch_load_safe
from supervisely.nn.task_type import TaskType
from supervisely.nn.training.gui.gui import TrainGUI
from supervisely.nn.training.gui.utils import generate_task_check_function_js
from supervisely.nn.training.loggers import setup_train_logger, train_logger
from supervisely.nn.utils import ModelSource, _get_model_name
from supervisely.output import set_directory
from supervisely.project.download import (
    copy_from_cache,
    download_fast,
    download_to_cache,
    get_cache_size,
    get_dataset_path,
    is_cached,
)
from supervisely.template.experiment.experiment_generator import ExperimentGenerator
from supervisely.api.entities_collection_api import EntitiesCollectionInfo


class TrainApp:
    """
    A class representing the training application.

    This class initializes and manages the training workflow, including
    handling inputs, hyperparameters, project management, and output artifacts.

    :param framework_name: Name of the ML framework used.
    :type framework_name: str
    :param models: List of model configurations.
    :type models: Union[str, List[Dict[str, Any]]]
    :param hyperparameters: Path or string content of hyperparameters in YAML format.
    :type hyperparameters: str
    :param app_options: Options for the application layout and behavior.
    :type app_options: Optional[Union[str, Dict[str, Any]]]
    :param work_dir: Path to the working directory for storing intermediate files.
    :type work_dir: Optional[str]
    """

    def __init__(
        self,
        framework_name: str,
        models: Union[str, List[Dict[str, Any]]],
        hyperparameters: str,
        app_options: Optional[Union[str, Dict[str, Any]]] = None,
        work_dir: Optional[str] = None,
    ):

        # Init
        self._api = Api.from_env()

        # Constants
        self._experiment_json_file = "experiment_info.json"
        self._app_state_file = "app_state.json"
        self._train_val_split_file = "train_val_split.json"
        self._hyperparameters_file = "hyperparameters.yaml"
        self._model_meta_file = "model_meta.json"

        self._sly_project_dir_name = "sly_project"
        self._model_dir_name = "model"
        self._log_dir_name = "logs"
        self._output_dir_name = "output"
        self._output_checkpoints_dir_name = "checkpoints"
        self._remote_checkpoints_dir_name = "checkpoints"
        self._experiments_dir_name = "experiments"
        self._default_work_dir_name = "work_dir"
        self._export_dir_name = "export"
        self._tensorboard_port = 6006

        if is_production():
            self._app_name = sly_env.app_name()
            self.task_id = sly_env.task_id()
        else:
            self._app_name = sly_env.app_name(raise_not_found=False)
            if self._app_name is None:
                self._app_name = "custom-app"
            self.task_id = sly_env.task_id(raise_not_found=False)
            if self.task_id is None:
                self.task_id = -1
            logger.info("TrainApp is running in debug mode")

        self.framework_name = framework_name
        self._tensorboard_process = None

        self._models = self._load_models(models)
        self._hyperparameters = self._load_hyperparameters(hyperparameters)
        self._app_options = self._load_app_options(app_options)
        self._inference_class = None
        self._is_inference_class_regirested = False
        # ----------------------------------------- #

        # Directories
        if work_dir is not None:
            self.work_dir = work_dir
        else:
            self.work_dir = join(get_synced_data_dir(), self._default_work_dir_name)
        self.output_dir = join(self.work_dir, self._output_dir_name)
        self._output_checkpoints_dir = join(self.output_dir, self._output_checkpoints_dir_name)
        self.project_dir = join(self.work_dir, self._sly_project_dir_name)
        self.train_dataset_dir = join(self.project_dir, "train")
        self.val_dataset_dir = join(self.project_dir, "val")
        self._model_cache_dir = join(expanduser("~"), ".cache", "supervisely", "checkpoints")
        self.sly_project = None
        # -------------------------- #

        # Train Val Splits
        self._train_split = []
        self._train_split_item_ids = set()
        self._train_collection_id = None

        self._val_split = []
        self._val_split_item_ids = set()
        self._val_collection_id = None
        # -------------------------- #

        # Input
        # ----------------------------------------- #

        # Classes
        # ----------------------------------------- #

        # Model
        self.model_files = {}
        self.model_dir = join(self.work_dir, self._model_dir_name)
        self.log_dir = join(self.work_dir, self._log_dir_name)
        # ----------------------------------------- #

        # Hyperparameters
        # ----------------------------------------- #

        # Layout
        self.gui: TrainGUI = TrainGUI(
            self.framework_name, self._models, self._hyperparameters, self._app_options
        )

        self.app = Application(layout=self.gui.layout)
        self._server = self.app.get_server()
        self._train_func = None
        self._training_duration = None

        self._onnx_supported = self._app_options.get("export_onnx_supported", False)
        self._tensorrt_supported = self._app_options.get("export_tensorrt_supported", False)
        if self._onnx_supported:
            self._convert_onnx_func = None
        if self._tensorrt_supported:
            self._convert_tensorrt_func = None

        # Benchmark parameters
        if self.is_model_benchmark_enabled:
            self._benchmark_params = {
                "model_files": {},
                "model_source": ModelSource.CUSTOM,
                "model_info": {},
                "device": None,
                "runtime": RuntimeType.PYTORCH,
            }
        # -------------------------- #

        # Train endpoints
        @self._server.post("/train_from_api")
        def _train_from_api(response: Response, request: Request):
            try:
                state = request.state.state
                wait = state.get("wait", True)
                app_state = state["app_state"]
                self.gui.load_from_app_state(app_state)

                if wait:
                    self._wrapped_start_training()
                else:
                    import threading

                    training_thread = threading.Thread(
                        target=self._wrapped_start_training,
                        daemon=True,
                    )
                    training_thread.start()
                    return {"result": "model training started"}

                return {"result": "model was successfully trained"}
            except Exception as e:
                self.gui.training_process.start_button.loading = False
                raise e

        @self._server.post("/train_status")
        def _train_status(response: Response, request: Request):
            """Returns the current training status."""
            status = self.gui.training_process.validator_text.get_value()
            if status == "Training is in progress...":
                try:
                    total_epochs = getattr(self.progress_bar_main, "total", None)
                    current_epoch = getattr(self.progress_bar_main, "current", None)
                    if total_epochs is not None and current_epoch is not None:
                        status += f" (Epoch {current_epoch}/{total_epochs})"
                except Exception:
                    pass
            return {"status": status}

        # Read GUI State when launched from experiment modal
        state = self.gui._extract_state_from_env()
        logger.debug(f"State: {state}")
        gui_state_raw = state.get("guiState")
        if gui_state_raw is not None:
            logger.info("Loading GUI from state")
            logger.debug(f"GUI State: {gui_state_raw}")
            try:
                self.gui.load_from_app_state(gui_state_raw)
                logger.info("Successfully loaded GUI from state")
            except Exception as e:
                raise e
        # ----------------------------------------- #

    def _register_routes(self):
        """
        Registers API routes for TensorBoard and training endpoints.

        These routes enable communication with the application for training
        and visualizing logs in TensorBoard.
        """
        client = httpx.AsyncClient(base_url=f"http://127.0.0.1:{self._tensorboard_port}/")

        @self._server.post("/tensorboard/{path:path}")
        @self._server.get("/tensorboard/{path:path}")
        async def _proxy_tensorboard(path: str, request: Request):
            url = httpx.URL(path=path, query=request.url.query.encode("utf-8"))
            headers = [(k, v) for k, v in request.headers.raw if k != b"host"]
            req = client.build_request(
                request.method, url, headers=headers, content=request.stream()
            )
            r = await client.send(req, stream=True)
            return StreamingResponse(
                r.aiter_raw(),
                status_code=r.status_code,
                headers=r.headers,
                background=BackgroundTask(r.aclose),
            )

    def _prepare_working_dir(self):
        """
        Prepares the working directory by creating required subdirectories.
        """
        sly_fs.mkdir(self.work_dir, True)
        sly_fs.mkdir(self.output_dir, True)
        sly_fs.mkdir(self._output_checkpoints_dir, True)
        sly_fs.mkdir(self.project_dir, True)
        sly_fs.mkdir(self.model_dir, True)
        sly_fs.mkdir(self.log_dir, True)

    # Properties
    # General
    @property
    def auto_start(self) -> bool:
        """
        If True, the training will start automatically after the GUI is loaded and train server is started.
        """
        return self.gui._start_training
    # ----------------------------------------- #

    # Input Data
    @property
    def team_id(self) -> int:
        """
        Returns the ID of the team.

        :return: Team ID.
        :rtype: int
        """
        return self.gui.team_id

    @property
    def workspace_id(self) -> int:
        """
        Returns the ID of the workspace.

        :return: Workspace ID.
        :rtype: int
        """
        return self.gui.workspace_id

    @property
    def project_id(self) -> int:
        """
        Returns the ID of the project.

        :return: Project ID.
        :rtype: int
        """
        return self.gui.project_id

    @property
    def project_name(self) -> str:
        """
        Returns the name of the project.

        :return: Project name.
        :rtype: str
        """
        return self.gui.project_info.name

    @property
    def project_info(self) -> ProjectInfo:
        """
        Returns ProjectInfo object, which contains information about the project.

        :return: Project name.
        :rtype: str
        """
        return self.gui.project_info

    @property
    def project_meta(self) -> ProjectMeta:
        """
        Returns the project metadata.

        :return: Project metadata.
        :rtype: ProjectMeta
        """
        return self.gui.project_meta

    # ----------------------------------------- #

    # Model
    @property
    def model_source(self) -> str:
        """
        Return whether the model is pretrained or custom.

        :return: Model source.
        :rtype: str
        """
        return self.gui.model_selector.get_model_source()

    @property
    def model_name(self) -> str:
        """
        Returns the name of the model.

        :return: Model name.
        :rtype: str
        """
        return self.gui.model_selector.get_model_name()

    @property
    def model_info(self) -> dict:
        """
        Returns a selected row in dict format from the models table.

        :return: Model name.
        :rtype: str
        """
        return self.gui.model_selector.get_model_info()

    @property
    def task_type(self) -> TaskType:
        """
        Returns the task type of the model.

        :return: Task type.
        :rtype: TaskType
        """
        return self.gui.model_selector.get_selected_task_type()

    @property
    def device(self) -> str:
        """
        Returns the selected device for training.

        :return: Device name.
        :rtype: str
        """
        return self.gui.training_process.get_device()

    @property
    def base_checkpoint(self) -> str:
        """
        Returns the name of the base checkpoint.
        """
        return self.gui.model_selector.get_checkpoint_name()

    @property
    def base_checkpoint_link(self) -> str:
        """
        Returns the link to the base checkpoint.
        """
        return self.gui.model_selector.get_checkpoint_link()

    # Classes
    @property
    def classes(self) -> List[str]:
        """
        Returns the selected classes names for training.

        :return: List of selected classes names.
        :rtype: List[str]
        """
        if not self._has_classes_selector:
            return []
        selected_classes = set(self.gui.classes_selector.get_selected_classes())
        # remap classes with project_meta order
        return [x for x in self.project_meta.obj_classes.keys() if x in selected_classes]

    @property
    def num_classes(self) -> int:
        """
        Returns the number of selected classes for training.

        :return: Number of selected classes.
        :rtype: int
        """
        if not self._has_classes_selector:
            return 0
        return len(self.gui.classes_selector.get_selected_classes())

    @property
    def tags(self) -> List[str]:
        """
        Returns the selected tags for training.
        """
        if not self._has_tags_selector:
            return []
        selected_tags = set(self.gui.tags_selector.get_selected_tags())
        return [x for x in self.project_meta.tag_metas.keys() if x in selected_tags]

    @property
    def num_tags(self) -> int:
        """
        Returns the number of selected tags for training.
        """
        if not self._has_tags_selector:
            return 0
        return len(self.gui.tags_selector.get_selected_tags())

    # Hyperparameters
    @property
    def hyperparameters(self) -> Dict[str, Any]:
        """
        Returns the selected hyperparameters for training in dict format.

        :return: Hyperparameters in dict format.
        :rtype: Dict[str, Any]
        """
        return yaml.safe_load(self.hyperparameters_yaml)

    @property
    def hyperparameters_yaml(self) -> str:
        """
        Returns the selected hyperparameters for training in raw format as a string.

        :return: Hyperparameters in raw format.
        :rtype: str
        """
        return self.gui.hyperparameters_selector.get_hyperparameters()

    # Train Process
    @property
    def progress_bar_main(self) -> Progress:
        """
        Returns the main progress bar widget.

        :return: Main progress bar widget.
        :rtype: Progress
        """
        return self.gui.training_logs.progress_bar_main

    @property
    def progress_bar_secondary(self) -> Progress:
        """
        Returns the secondary progress bar widget.

        :return: Secondary progress bar widget.
        :rtype: Progress
        """
        return self.gui.training_logs.progress_bar_secondary

    @property
    def is_model_benchmark_enabled(self) -> bool:
        """
        Checks if model benchmarking is enabled based on application options and GUI settings.

        :return: True if model benchmarking is enabled, False otherwise.
        :rtype: bool
        """
        return (
            self._app_options.get("model_benchmark", True)
            and self.gui.hyperparameters_selector.get_model_benchmark_checkbox_value()
        )

    # Output
    # ----------------------------------------- #

    # Helper properties
    @property
    def _has_splits_selector(self) -> bool:
        """Return True if Train/Val splits selector is enabled in GUI."""
        return self.gui.train_val_splits_selector is not None

    @property
    def _has_classes_selector(self) -> bool:
        """Return True if Classes selector is enabled in GUI."""
        return self.gui.classes_selector is not None

    @property
    def _has_tags_selector(self) -> bool:
        """Return True if Tags selector is enabled in GUI."""
        return self.gui.tags_selector is not None

    # ----------------------------------------- #

    # Wrappers
    @property
    def start(self):
        """
        Decorator for the training function defined by user.
        It wraps user-defined training function and prepares and finalizes the training process.
        """

        def decorator(func):
            self._train_func = timeit_with_result(func)
            self.gui.training_process.start_button.click(self._wrapped_start_training)
            return func

        return decorator

    @property
    def export_onnx(self):
        """
        Decorator for the export to ONNX function defined by user.
        It wraps user-defined export function and prepares and finalizes the training process.
        """

        def decorator(func):
            self._convert_onnx_func = func
            return func

        return decorator

    @property
    def export_tensorrt(self):
        """
        Decorator for the export to TensorRT function defined by user.
        It wraps user-defined export function and prepares and finalizes the training process.
        """

        def decorator(func):
            self._convert_tensorrt_func = func
            return func

        return decorator

    # ----------------------------------------- #

    def _prepare(self) -> None:
        """
        Prepares the environment for training by setting up directories,
        downloading project and model data.
        """
        logger.info("Preparing for training")

        # Step 1. Workflow Input
        if is_production():
            self._workflow_input()
        # Step 2. Download Project
        self._download_project()
        # Step 3. Convert Project to Task
        if self.gui.need_convert_shapes:
            if self.gui.classes_selector is not None:
                if self.gui.classes_selector.is_convert_class_shapes_enabled():
                    self._convert_project_to_model_task()

        # Step 4. Split Project
        self._split_project()
        # Step 5. Remove classes except selected
        if self.sly_project.type == ProjectType.IMAGES.value:
            self.sly_project.remove_classes_except(self.project_dir, self.classes, True)
            self._read_project()

        # Step 6. Create collections
        self._create_collection_splits()

        # Step 7. Download Model files
        self._download_model()

    def _finalize(self, experiment_info: dict) -> None:
        """
        Finalizes the training process by validating outputs, uploading artifacts,
        and updating the UI.

        :param experiment_info: Information about the experiment results that should be returned in user's training function.
        :type experiment_info: dict
        """
        logger.info("Finalizing training")
        # Step 1. Validate experiment TaskType
        experiment_info = self._validate_experiment_task_type(experiment_info)

        # Step 2. Validate experiment_info
        success, reason = self._validate_experiment_info(experiment_info)
        if not success:
            raise ValueError(f"{reason}. Failed to upload artifacts")

        # Step 3. Create model meta according to model CV task type
        model_meta = self.create_model_meta(experiment_info["task_type"])

        # Step 4. Preprocess artifacts
        experiment_info = self._preprocess_artifacts(experiment_info, model_meta)

        # Step 5. Postprocess splits
        train_splits_data = self._postprocess_splits()

        # Step 6. Upload artifacts
        self._set_text_status("uploading")
        remote_dir, session_link_file_info = self._upload_artifacts()

        # Step 7. [Optional] Run Model Benchmark
        mb_eval_lnk_file_info, mb_eval_report = None, None
        mb_eval_report_id, eval_metrics = None, {}
        evaluation_report_link, primary_metric_name = None, None
        if self.is_model_benchmark_enabled:
            try:
                # Convert GT project
                gt_project_id, bm_splits_data = None, train_splits_data
                # @TODO: check with anyshape classes
                if self.gui.need_convert_shapes:
                    if self.gui.hyperparameters_selector.get_model_benchmark_checkbox_value():
                        self._set_text_status("convert_gt_project")
                        gt_project_id, bm_splits_data = self._convert_and_split_gt_project(
                            experiment_info["task_type"]
                        )

                self._set_text_status("benchmark")
                (
                    mb_eval_lnk_file_info,
                    mb_eval_report,
                    mb_eval_report_id,
                    eval_metrics,
                    primary_metric_name,
                ) = self._run_model_benchmark(
                    self.output_dir,
                    remote_dir,
                    experiment_info,
                    bm_splits_data,
                    model_meta,
                    gt_project_id,
                )
                if mb_eval_report_id is not None:
                    evaluation_report_link = abs_url(
                        f"/model-benchmark?id={str(mb_eval_report_id)}"
                    )
            except Exception as e:
                logger.error(f"Model benchmark failed: {e}")

        # Step 8. [Optional] Convert weights
        export_weights = {}
        if self.gui.hyperparameters_selector.is_export_required():
            try:
                export_weights, export_classes_path = self._export_weights(experiment_info)
                export_weights = self._upload_export_weights(
                    export_weights, export_classes_path, remote_dir
                )
            except Exception as e:
                logger.error(f"Export weights failed: {e}")

        # Step 9. Generate and upload additional files
        self._set_text_status("metadata")
        experiment_info = self._generate_experiment_info(
            remote_dir,
            experiment_info,
            eval_metrics,
            mb_eval_report_id,
            evaluation_report_link,
            primary_metric_name,
            export_weights,
        )
        self._generate_app_state(remote_dir, experiment_info)
        experiment_info = self._generate_hyperparameters(remote_dir, experiment_info)
        self._generate_train_val_splits(remote_dir, train_splits_data)
        self._generate_model_meta(remote_dir, model_meta)
        self._upload_demo_files(remote_dir)

        # Step 10. Generate training output
        output_file_info, experiment_info = self._generate_experiment_output(experiment_info, model_meta, session_link_file_info)

        # Step 11. Set output widgets
        self._set_text_status("reset")
        self._set_training_output(
            experiment_info,
            remote_dir,
            output_file_info,
            mb_eval_report,
        )
        self._set_ws_progress_status("completed")

        # Step 12. Workflow output
        if is_production():
            best_checkpoint_file_info = self._get_best_checkpoint_info(experiment_info, remote_dir)
            self._workflow_output(
                remote_dir,
                best_checkpoint_file_info,
                mb_eval_lnk_file_info,
                mb_eval_report_id,
            )

    def _get_best_checkpoint_info(self, experiment_info: dict, remote_dir: str) -> FileInfo:
        """
        Returns the best checkpoint info.

        :param experiment_info: Experiment info.
        :type experiment_info: dict
        :param remote_dir: Remote directory.
        :type remote_dir: str
        :return: Best checkpoint info.
        :rtype: FileInfo
        """
        best_checkpoint_name = experiment_info.get("best_checkpoint")
        remote_best_checkpoint_path = join(remote_dir, "checkpoints", best_checkpoint_name)
        best_checkpoint_file_info = self._api.file.get_info_by_path(
            self.team_id, remote_best_checkpoint_path
        )
        return best_checkpoint_file_info

    def register_inference_class(
        self, inference_class: Inference, inference_settings: Union[str, dict] = None
    ) -> None:
        """
        Registers an inference class for the training application to do model benchmarking.

        :param inference_class: Inference class to be registered inherited from `supervisely.nn.inference.Inference`.
        :type inference_class: Any
        :param inference_settings: Settings for the inference class.
        :type inference_settings: dict
        """
        self._is_inference_class_regirested = True
        self._inference_class = inference_class
        self._inference_settings = None
        if isinstance(inference_settings, str):
            with open(inference_settings, "r") as file:
                self._inference_settings = yaml.safe_load(file)
        else:
            self._inference_settings = inference_settings

    def get_app_state(self, experiment_info: dict = None) -> dict:
        """
        Returns the current state of the application.

        :param experiment_info: Experiment info.
        :type experiment_info: dict
        :return: Application state.
        :rtype: dict
        """
        # Prepare optional sections depending on what selectors are enabled in GUI
        train_val_splits = self._get_train_val_splits_for_app_state()
        classes = self.classes
        tags = self.tags

        convert_class_shapes = False
        if self.gui.classes_selector is not None:
            convert_class_shapes = self.gui.classes_selector.is_convert_class_shapes_enabled()

        options = {
            "cache_project": self.gui.input_selector.get_cache_value(),
            "convert_class_shapes": convert_class_shapes,
            "model_benchmark": {
                "enable": self.gui.hyperparameters_selector.get_model_benchmark_checkbox_value(),
                "speed_test": self.gui.hyperparameters_selector.get_speedtest_checkbox_value(),
            },
            "export": {
                "enable": self.gui.hyperparameters_selector.is_export_required(),
                "ONNXRuntime": self.gui.hyperparameters_selector.get_export_onnx_checkbox_value(),
                "TensorRT": self.gui.hyperparameters_selector.get_export_tensorrt_checkbox_value(),
            },
        }

        model = self._get_model_config_for_app_state(experiment_info)
        app_state = {
            "model": model,
            "hyperparameters": self.hyperparameters_yaml,
            "options": options,
        }

        app_state["train_val_split"] = train_val_splits
        app_state["classes"] = classes
        app_state["tags"] = tags
        return app_state

    def load_app_state(self, app_state: Union[str, dict]) -> None:
        """
        Load the GUI state from app state dictionary.

        :param app_state: The state dictionary or path to the state file.
        :type app_state: Union[str, dict]

        app_state example:

            app_state = {
                "train_val_split": {
                    "method": "random",
                    "split": "train",
                    "percent": 90
                },
                "classes": ["apple"],
                # Pretrained model
                "model": {
                    "source": "Pretrained models",
                    "model_name": "rtdetr_r50vd_coco_objects365"
                },
                # Custom model
                # "model": {
                #     "source": "Custom models",
                #     "task_id": 555,
                #     "checkpoint": "checkpoint_10.pth"
                # },
                "hyperparameters": hyperparameters, # yaml string
                "options": {
                    "convert_class_shapes": True,
                    "model_benchmark": {
                        "enable": True,
                        "speed_test": True
                    },
                    "cache_project": True,
                    "export": {
                        "enable": True,
                        "ONNXRuntime": True,
                        "TensorRT": True
                    },
                },
                "experiment_name": "My Experiment",
            }
        """
        self.gui.load_from_app_state(app_state)

    def add_output_files(self, paths: List[str]) -> None:
        """
        Copies files or directories to the output directory, which will be uploaded to the team files upon training completion.
        If path is a file, it will be uploaded to the root artifacts directory.
        If path is a directory, it will be uploded to the root artifacts directory with the same directory name and structure.

        :param paths: List of paths to files or directories to be copied to the output directory.
        :type paths: List[str]
        :return: None
        :rtype: None
        """

        for path in paths:
            if sly_fs.file_exists(path):
                shutil.copyfile(path, join(self.output_dir, sly_fs.get_file_name_with_ext(path)))
            elif sly_fs.dir_exists(path):
                shutil.copytree(path, join(self.output_dir, basename(path)))
            else:
                logger.warning(f"Provided path: '{path}' does not exist. Skipping...")
                continue

    # Loaders
    def _load_models(self, models: Union[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        Loads models from the provided file or list of model configurations.
        """
        if isinstance(models, str):
            if sly_fs.file_exists(models) and sly_fs.get_file_ext(models) == ".json":
                models = sly_json.load_json_file(models)
            else:
                raise ValueError(
                    "Invalid models file. Please provide a valid '.json' file or a list of model configurations."
                )

        if not isinstance(models, list):
            raise ValueError("models parameters must be a list of dicts")
        for item in models:
            if not isinstance(item, dict):
                raise ValueError(f"Each item in models must be a dict.")
            model_meta = item.get("meta")
            if model_meta is None:
                raise ValueError(
                    "Model metadata not found. Please update provided models parameter to include key 'meta'."
                )
            model_files = model_meta.get("model_files")
            if model_files is None:
                raise ValueError(
                    "Model files not found in model metadata. "
                    "Please update provided models parameter to include key 'model_files' in 'meta' key."
                )
        return models

    def _load_hyperparameters(self, hyperparameters: str) -> dict:
        """
        Loads hyperparameters from file path.

        :param hyperparameters: Path to hyperparameters file.
        :type hyperparameters: str
        :return: Hyperparameters in dict format.
        :rtype: dict
        """
        if not isinstance(hyperparameters, str):
            raise ValueError(
                f"Expected a string with config or path for hyperparameters, but got {type(hyperparameters).__name__}"
            )
        if hyperparameters.endswith((".yml", ".yaml")):
            try:
                with open(hyperparameters, "r") as file:
                    return file.read()
            except Exception as e:
                raise ValueError(f"Failed to load YAML file: {hyperparameters}. Error: {e}")
        return hyperparameters

    def _load_app_options(self, app_options: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Loads the app_options parameter to ensure it is in the correct format.
        """
        if app_options is None:
            return {}

        if isinstance(app_options, str):
            if sly_fs.file_exists(app_options) and sly_fs.get_file_ext(app_options) in [
                ".yaml",
                ".yml",
            ]:
                app_options = self._load_yaml(app_options)
            else:
                raise ValueError(
                    "Invalid app_options file provided. Please provide a valid '.yaml' or '.yml' file or a dictionary with app_options."
                )
        if not isinstance(app_options, dict):
            raise ValueError("app_options must be a dict")
        return app_options

    def _load_yaml(self, path: str) -> dict:
        """
        Load a YAML file from the specified path.

        :param path: Path to the YAML file.
        :type path: str
        :return: YAML file contents.
        :rtype: dict
        """
        with open(path, "r") as file:
            return yaml.safe_load(file)

    # ----------------------------------------- #

    # Preprocess
    # Download Project
    def _read_project(self) -> None:
        """
        Reads the project data from Supervisely.
        """
        if self.project_info.type == ProjectType.IMAGES.value:
            self.sly_project = Project(self.project_dir, OpenMode.READ)
        elif self.project_info.type == ProjectType.VIDEOS.value:
            self.sly_project = VideoProject(self.project_dir, OpenMode.READ)
        else:
            raise ValueError(
                f"Unsupported project type: {self.project_info.type}. Only images and videos are supported."
            )

    def _convert_project_to_model_task(self) -> None:
        """
        Converts the project to the appropriate type.
        """
        if not self.project_info.type == ProjectType.IMAGES.value:
            logger.info("Class shape conversion supported only for images projects")
            return

        task_type = self.gui.model_selector.get_selected_task_type()
        if task_type not in [
            TaskType.OBJECT_DETECTION,
            TaskType.INSTANCE_SEGMENTATION,
            TaskType.SEMANTIC_SEGMENTATION,
        ]:
            logger.info(f"Class shape conversion for {task_type} task is not supported")
            return

        logger.info(f"Converting project for {task_type} task")
        with self.progress_bar_main(
            message=f"Converting project to {task_type} task",
            total=len(self.sly_project.datasets),
        ) as pbar:
            if task_type == TaskType.OBJECT_DETECTION:
                self.sly_project.to_detection_task(
                    self.project_dir, inplace=True, progress_cb=pbar.update
                )
            elif task_type == TaskType.INSTANCE_SEGMENTATION:
                self.sly_project.to_segmentation_task(
                    self.project_dir,
                    inplace=True,
                    segmentation_type="instance",
                    progress_cb=pbar.update,
                )
            elif task_type == TaskType.SEMANTIC_SEGMENTATION:
                self.sly_project.to_segmentation_task(
                    self.project_dir,
                    inplace=True,
                    segmentation_type="semantic",
                    progress_cb=pbar.update,
                )
        self.sly_project = Project(self.project_dir, OpenMode.READ)

    def _download_project(self) -> None:
        """
        Downloads the project data from Supervisely.
        If the cache is enabled, it will attempt to retrieve the project from the cache.
        """
        dataset_infos = [dataset for _, dataset in self._api.dataset.tree(self.project_id)]
        if self.gui.train_val_splits_selector is not None:
            if self.gui.train_val_splits_selector.get_split_method() == "Based on datasets":
                selected_ds_ids = (
                    self.gui.train_val_splits_selector.get_train_dataset_ids()
                    + self.gui.train_val_splits_selector.get_val_dataset_ids()
                )
                dataset_infos = [
                    ds_info for ds_info in dataset_infos if ds_info.id in selected_ds_ids
                ]

        total_images = sum(ds_info.images_count for ds_info in dataset_infos)
        if not self.gui.input_selector.get_cache_value():
            self._download_no_cache(dataset_infos, total_images)
            self._read_project()
            return

        try:
            self._download_with_cache(dataset_infos, total_images)
        except Exception:
            logger.warning(
                "Failed to retrieve project from cache. Downloading it",
                exc_info=True,
            )
            if sly_fs.dir_exists(self.project_dir):
                sly_fs.clean_dir(self.project_dir)
            self._download_no_cache(dataset_infos, total_images)
        finally:
            self._read_project()
            logger.info(f"Project downloaded successfully to: '{self.project_dir}'")

    def _download_no_cache(self, dataset_infos: List[DatasetInfo], total_images: int) -> None:
        """
        Downloads the project data from Supervisely without using the cache.

        :param dataset_infos: List of dataset information objects.
        :type dataset_infos: List[DatasetInfo]
        :param total_images: Total number of images to download.
        :type total_images: int
        """
        with self.progress_bar_main(message="Downloading input data", total=total_images) as pbar:
            logger.debug("Downloading project data without cache")
            self.progress_bar_main.show()
            download_fast(
                api=self._api,
                project_id=self.project_id,
                dest_dir=self.project_dir,
                dataset_ids=[ds_info.id for ds_info in dataset_infos],
                save_image_info=True,
                log_progress=True,
                progress_cb=pbar.update,
            )
        self.progress_bar_main.hide()

    def _download_with_cache(
        self,
        dataset_infos: List[DatasetInfo],
        total_images: int,
    ) -> None:
        """
        Downloads the project data from Supervisely using the cache.

        :param dataset_infos: List of dataset information objects.
        :type dataset_infos: List[DatasetInfo]
        :param total_images: Total number of images to download.
        :type total_images: int
        """
        ds_paths = {
            info.id: get_dataset_path(self._api, dataset_infos, info.id) for info in dataset_infos
        }
        to_download = [
            info for info in dataset_infos if not is_cached(self.project_info.id, ds_paths[info.id])
        ]
        cached = [
            info for info in dataset_infos if is_cached(self.project_info.id, ds_paths[info.id])
        ]

        logger.info(self._get_cache_log_message(cached, to_download))
        with self.progress_bar_main(message="Downloading input data", total=total_images) as pbar:
            logger.debug("Downloading project data with cache")
            self.progress_bar_main.show()
            download_to_cache(
                api=self._api,
                project_id=self.project_info.id,
                dataset_infos=dataset_infos,
                log_progress=True,
                progress_cb=pbar.update,
            )

        total_cache_size = sum(
            get_cache_size(self.project_info.id, ds.name) for ds in dataset_infos
        )
        with self.progress_bar_main(
            message="Retrieving data from cache",
            total=total_cache_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            copy_from_cache(
                project_id=self.project_info.id,
                dest_dir=self.project_dir,
                dataset_names=[ds_paths[info.id] for info in dataset_infos],
                progress_cb=pbar.update,
            )
        self.progress_bar_main.hide()

    def _get_cache_log_message(self, cached: bool, to_download: List[DatasetInfo]) -> str:
        """
        Utility method to generate a log message for cache status.
        """
        if not cached:
            log_msg = "No cached datasets found"
        else:
            log_msg = "Using cached datasets: " + ", ".join(
                f"{ds_info.name} ({ds_info.id})" for ds_info in cached
            )

        if not to_download:
            log_msg += ". All datasets are cached. No datasets to download"
        else:
            log_msg += ". Downloading datasets: " + ", ".join(
                f"{ds_info.name} ({ds_info.id})" for ds_info in to_download
            )

        return log_msg

    # Split Project
    def _split_project(self) -> None:
        """
        Split the project into training and validation sets.
        All images and annotations will be renamed and moved to the appropriate directories.
        Assigns self.sly_project to the new project, which contains only 2 datasets: train and val.
        """
        if not self._has_splits_selector:
            # Splits disabled in options, init empty splits
            self.train_dataset_dir = None
            self.val_dataset_dir = None
            self._train_val_split_file = None
            self._train_split = []
            self._val_split = []
            self._train_split_item_ids = set()
            self._val_split_item_ids = set()
            return

        # Load splits
        self.gui.train_val_splits_selector.set_sly_project(self.sly_project)
        self._train_split, self._val_split = self.gui.train_val_splits_selector.train_val_splits.get_splits()
        self._train_split_ids, self._val_split_ids = [], []
        self._train_split_item_ids, self._val_split_item_ids = set(), set()

        # Prepare paths
        project_split_path = join(self.work_dir, "splits")
        paths = {
            "train": {
                "split_path": join(project_split_path, "train"),
                "img_dir": join(project_split_path, "train", "img"),
                "ann_dir": join(project_split_path, "train", "ann"),
                "img_info_dir": join(project_split_path, "train", "img_info"),
            },
            "val": {
                "split_path": join(project_split_path, "val"),
                "img_dir": join(project_split_path, "val", "img"),
                "ann_dir": join(project_split_path, "val", "ann"),
                "img_info_dir": join(project_split_path, "val", "img_info"),
            },
        }

        # Create necessary directories (only once)
        for dataset_paths in paths.values():
            for path in dataset_paths.values():
                sly_fs.mkdir(path, True)

        # Format for image names
        items_count = max(len(self._train_split), len(self._val_split))
        num_digits = len(str(items_count))
        image_name_formats = {
            "train": f"train_img_{{:0{num_digits}d}}",
            "val": f"val_img_{{:0{num_digits}d}}",
        }

        # Utility function to move files
        def move_files(split, split_name, paths, img_name_format, pbar):
            """
            Move files to the appropriate directories.
            """
            for idx, item in enumerate(split, start=1):
                item_name = img_name_format.format(idx) + sly_fs.get_file_ext(item.name)
                ann_name = f"{item_name}.json"
                shutil.copy(item.img_path, join(paths["img_dir"], item_name))
                shutil.copy(item.ann_path, join(paths["ann_dir"], ann_name))

                # Move img_info
                img_info_name = f"{sly_fs.get_file_name_with_ext(item.img_path)}.json"
                img_info_path = join(dirname(dirname(item.img_path)), "img_info", img_info_name)
                # shutil.copy(img_info_path, join(paths["img_info_dir"], ann_name))

                # Write split ids to img_info
                img_info = sly_json.load_json_file(img_info_path)
                if split_name == "train":
                    self._train_split_item_ids.add(img_info["id"])
                else:
                    self._val_split_item_ids.add(img_info["id"])

                pbar.update(1)

        # Main split processing
        with self.progress_bar_main(
            message="Applying train/val splits to project", total=2
        ) as main_pbar:
            self.progress_bar_main.show()
            for dataset in ["train", "val"]:
                split_name = dataset
                if split_name == "train":
                    split = self._train_split
                else:
                    split = self._val_split
                with self.progress_bar_secondary(
                    message=f"Preparing '{dataset}'", total=len(split)
                ) as second_pbar:
                    self.progress_bar_secondary.show()
                    move_files(split, split_name, paths[dataset], image_name_formats[dataset], second_pbar)
                    main_pbar.update(1)
                self.progress_bar_secondary.hide()
            self.progress_bar_main.hide()

        # Clean up project directory
        project_datasets = [
            join(self.project_dir, item)
            for item in listdir(self.project_dir)
            if isdir(join(self.project_dir, item))
        ]
        for dataset in project_datasets:
            sly_fs.remove_dir(dataset)

        # Move processed splits to final destination
        train_ds_path = join(self.project_dir, "train")
        val_ds_path = join(self.project_dir, "val")
        with self.progress_bar_main(message="Processing splits", total=2) as pbar:
            self.progress_bar_main.show()
            for dataset in ["train", "val"]:
                shutil.move(paths[dataset]["split_path"], train_ds_path if dataset == "train" else val_ds_path)
                pbar.update(1)
            self.progress_bar_main.hide()

        # Clean up temporary directory
        sly_fs.remove_dir(project_split_path)
        self._read_project()

    # ----------------------------------------- #

    # ----------------------------------------- #
    # Download Model
    def _download_model(self) -> None:
        """
        Downloads the model data from the selected source.
        - Checkpoint and config keys inside the model_files dict can be provided as local paths.

        For Pretrained models:
            - The files that will be downloaded are specified in the `meta` key under `model_files`.
            - All files listed in the `model_files` key will be downloaded by provided link.
            - If model files are already cached on agent, they will be copied to the model directory without downloading.
            Example of a pretrained model entry:
                [
                    {
                            "Model": "example_model",
                            "dataset": "COCO",
                            "AP_val": 46.4,
                            "Params(M)": 20,
                            "FPS(T4)": 217,
                            "meta": {
                                    "task_type": "object detection",
                                    "model_name": "example_model",
                                    "model_files": {
                                            # For remote files provide as links
                                            "checkpoint": "https://example.com/checkpoint.pth",
                                            "config": "https://example.com/config.yaml"

                                            # For local files provide as paths
                                            # "checkpoint": "/path/to/checkpoint.pth",
                                            # "config": "/path/to/config.yaml"
                                    }
                            }
                    },
                    ...
                ]

        For Custom models:
            - All custom models trained inside Supervisely are managed automatically by this class.
        """
        if self.model_source == ModelSource.PRETRAINED:
            self._download_pretrained_model()

        else:
            self._download_custom_model()
        logger.info(f"Model files have been downloaded successfully to: '{self.model_dir}'")

    def _download_pretrained_model(self):
        """
        Downloads the pretrained model data.
        """
        # General
        self.model_files = {}
        model_meta = self.model_info["meta"]
        model_files = model_meta["model_files"]

        with self.progress_bar_main(
            message="Downloading model files",
            total=len(model_files),
        ) as model_download_main_pbar:
            self.progress_bar_main.show()
            for file in model_files:
                file_url = model_files[file]
                file_path = join(self.model_dir, file)
                file_name = sly_fs.get_file_name_with_ext(file_url)
                if file_url.startswith("http"):
                    with urlopen(file_url) as f:
                        file_size = f.length
                        file_name = get_filename_from_headers(file_url)
                        if file_name is None:
                            file_name = file
                        file_path = join(self.model_dir, file_name)
                        cached_path = join(self._model_cache_dir, file_name)
                        if exists(cached_path):
                            self.model_files[file] = cached_path
                            logger.debug(f"Model: '{file_name}' was found in checkpoint cache")
                            model_download_main_pbar.update(1)
                            continue
                        if exists(file_path):
                            self.model_files[file] = file_path
                            logger.debug(f"Model: '{file_name}' was found in model dir")
                            model_download_main_pbar.update(1)
                            continue

                        with self.progress_bar_secondary(
                            message=f"Downloading '{file_name}' ",
                            total=file_size,
                            unit="bytes",
                            unit_scale=True,
                        ) as model_download_secondary_pbar:
                            self.progress_bar_secondary.show()
                            sly_fs.download(
                                url=file_url,
                                save_path=file_path,
                                progress=model_download_secondary_pbar.update,
                            )
                        self.model_files[file] = file_path
                else:
                    self.model_files[file] = file_url
                model_download_main_pbar.update(1)

        self.progress_bar_main.hide()
        self.progress_bar_secondary.hide()

    def _download_custom_model(self):
        """
        Downloads the custom model data.
        """
        # General
        self.model_files = {}

        # Need to merge file_url with arts dir
        artifacts_dir = self.model_info["artifacts_dir"]
        model_files = self.model_info["model_files"]
        remote_paths = {name: join(artifacts_dir, file) for name, file in model_files.items()}

        # Add selected checkpoint to model_files
        checkpoint = self.gui.model_selector.experiment_selector.get_selected_checkpoint_path()
        remote_paths["checkpoint"] = checkpoint

        with self.progress_bar_main(
            message="Downloading model files",
            total=len(model_files),
        ) as model_download_main_pbar:
            self.progress_bar_main.show()
            for name, remote_path in remote_paths.items():
                file_info = self._api.file.get_info_by_path(self.team_id, remote_path)
                file_name = basename(remote_path)
                local_path = join(self.model_dir, file_name)
                file_size = file_info.sizeb

                with self.progress_bar_secondary(
                    message=f"Downloading {file_name}",
                    total=file_size,
                    unit="bytes",
                    unit_scale=True,
                ) as model_download_secondary_pbar:
                    self.progress_bar_secondary.show()
                    self._api.file.download(
                        self.team_id,
                        remote_path,
                        local_path,
                        progress_cb=model_download_secondary_pbar.update,
                    )
                model_download_main_pbar.update(1)
                self.model_files[name] = local_path

        self.progress_bar_main.hide()
        self.progress_bar_secondary.hide()

    # ----------------------------------------- #

    # Postprocess
    def _validate_experiment_task_type(self, experiment_info: dict) -> dict:
        """
        Checks if the task_type key if returned from the user's training function.
        If not, it will be set to the task type of the model selected in the model selector.

        :param experiment_info: Information about the experiment results.
        :type experiment_info: dict
        :return: Experiment info with task_type key.
        :rtype: dict
        """
        task_type = experiment_info.get("task_type", None)
        if task_type is None:
            logger.debug(
                "Task type not found in experiment_info. Task type from model config will be used."
            )
            task_type = self.gui.model_selector.get_selected_task_type()
            experiment_info["task_type"] = task_type
        return experiment_info

    def _validate_experiment_info(self, experiment_info: dict) -> tuple:
        """
        Validates the experiment_info parameter to ensure it is in the correct format.
        experiment_info is returned by the user's training function.

        experiment_info should contain the following keys:
            - model_name": str
            - task_type": str
            - checkpoints": list
            - best_checkpoint": str
            - model_files": Optional[dict]

        Other keys are generated by the TrainApp class automatically

        :param experiment_info: Information about the experiment results.
        :type experiment_info: dict
        :return: Tuple of success status and reason for failure.
        :rtype: tuple
        """
        if not isinstance(experiment_info, dict):
            reason = f"Validation failed: 'experiment_info' must be a dictionary not '{type(experiment_info)}'"
            return False, reason

        logger.debug("Starting validation of 'experiment_info'")
        required_keys = {
            "model_name": str,
            "task_type": str,
            "checkpoints": (list, str),
            "best_checkpoint": str,
        }

        for key, expected_type in required_keys.items():
            if key not in experiment_info:
                reason = f"Validation failed: Missing required key '{key}'"
                return False, reason

            if not isinstance(experiment_info[key], expected_type):
                reason = (
                    f"Validation failed: Key '{key}' should be of type {expected_type.__name__}"
                )
                return False, reason

        if isinstance(experiment_info["checkpoints"], list):
            for checkpoint in experiment_info["checkpoints"]:
                if not isinstance(checkpoint, str):
                    reason = "Validation failed: All items in 'checkpoints' list must be strings"
                    return False, reason
                if not sly_fs.file_exists(checkpoint):
                    reason = f"Validation failed: Checkpoint file: '{checkpoint}' does not exist"
                    return False, reason

        best_checkpoint = experiment_info["best_checkpoint"]
        checkpoints = experiment_info["checkpoints"]
        if isinstance(checkpoints, list):
            checkpoints = [sly_fs.get_file_name_with_ext(checkpoint) for checkpoint in checkpoints]
            if best_checkpoint not in checkpoints:
                reason = (
                    f"Validation failed: Best checkpoint file: '{best_checkpoint}' does not exist"
                )
                return False, reason
        elif isinstance(checkpoints, str):
            checkpoints = [
                sly_fs.get_file_name_with_ext(checkpoint)
                for checkpoint in listdir(checkpoints)
                if sly_fs.get_file_ext(checkpoint) in [".pt", ".pth"]
            ]
            if best_checkpoint not in checkpoints:
                reason = (
                    f"Validation failed: Best checkpoint file: '{best_checkpoint}' does not exist"
                )
                return False, reason
        else:
            reason = "Validation failed: 'checkpoints' should be a list of paths or a path to directory with checkpoints"
            return False, reason

        logger.debug("Validation successful")
        return True, None

    def _postprocess_splits(self, project_id: Optional[int] = None) -> dict:
        """
        Processes the train and val splits to generate the necessary data for the experiment_info.json file.

        :param project_id: ID of the ground truth project for model benchmark. Provide only when cv task convertion is required.
        :type project_id: Optional[int]
        :return: Splits data.
        :rtype: dict
        """
        val_dataset_ids = None
        val_images_ids = None
        train_dataset_ids = None
        train_images_ids = None

        if not self._has_splits_selector:
            return {}  # splits disabled in options

        split_method = self.gui.train_val_splits_selector.get_split_method()
        train_set, val_set = self._train_split, self._val_split
        if split_method == "Based on datasets":
            if project_id is None:
                val_dataset_ids = self.gui.train_val_splits_selector.get_val_dataset_ids()
                train_dataset_ids = self.gui.train_val_splits_selector.get_train_dataset_ids()
            else:
                src_datasets_map = {
                    dataset.id: dataset
                    for _, dataset in self._api.dataset.tree(self.project_info.id)
                }
                val_dataset_ids = self.gui.train_val_splits_selector.get_val_dataset_ids()
                train_dataset_ids = self.gui.train_val_splits_selector.get_train_dataset_ids()

                train_dataset_names = [src_datasets_map[ds_id].name for ds_id in train_dataset_ids]
                val_dataset_names = [src_datasets_map[ds_id].name for ds_id in val_dataset_ids]

                gt_datasets_map = {
                    dataset.name: dataset.id for _, dataset in self._api.dataset.tree(project_id)
                }
                train_dataset_ids = [gt_datasets_map[ds_name] for ds_name in train_dataset_names]
                val_dataset_ids = [gt_datasets_map[ds_name] for ds_name in val_dataset_names]
        else:
            if project_id is None:
                project_id = self.project_id

            dataset_infos = [dataset for _, dataset in self._api.dataset.tree(project_id)]
            id_to_info = {ds.id: ds for ds in dataset_infos}
            ds_infos_dict = {}
            for dataset in dataset_infos:
                name_parts = [dataset.name]
                parent_id = dataset.parent_id
                while parent_id is not None:
                    parent_ds = id_to_info.get(parent_id)
                    if parent_ds is None:
                        parent_ds = self._api.dataset.get_info_by_id(parent_id)
                    name_parts.append(parent_ds.name)
                    parent_id = parent_ds.parent_id
                dataset_name = "/".join(reversed(name_parts))
                ds_infos_dict[dataset_name] = dataset

            def get_image_infos_by_split(ds_infos_dict: dict, split: list):
                image_names_per_dataset = {}
                for item in split:
                    image_names_per_dataset.setdefault(item.dataset_name, []).append(item.name)
                image_infos = []
                for dataset_name, image_names in image_names_per_dataset.items():
                    ds_info = ds_infos_dict[dataset_name]
                    for names_batch in batched(image_names, 200):
                        image_infos.extend(
                            self._api.image.get_list(
                                ds_info.id,
                                filters=[
                                    {
                                        "field": "name",
                                        "operator": "in",
                                        "value": names_batch,
                                    }
                                ],
                            )
                        )
                return image_infos

            val_image_infos = get_image_infos_by_split(ds_infos_dict, val_set)
            train_image_infos = get_image_infos_by_split(ds_infos_dict, train_set)
            val_images_ids = [img_info.id for img_info in val_image_infos]
            train_images_ids = [img_info.id for img_info in train_image_infos]

        splits_data = {
            "train": {
                "dataset_ids": train_dataset_ids,
                "images_ids": train_images_ids,
            },
            "val": {
                "dataset_ids": val_dataset_ids,
                "images_ids": val_images_ids,
            },
        }
        return splits_data

    def _preprocess_artifacts(self, experiment_info: dict, model_meta: ProjectMeta) -> None:
        """
        Preprocesses and move the artifacts generated by the training process to output directories.

        :param experiment_info: Information about the experiment results.
        :type experiment_info: dict
        """
        # Preprocess artifacts
        logger.debug("Preprocessing artifacts")
        if "model_files" not in experiment_info:
            experiment_info["model_files"] = {}
        else:
            # Move model files to output directory except config, config will be processed next
            files = {k: v for k, v in experiment_info["model_files"].items() if k != "config"}
            for file in files:
                if isfile:
                    shutil.move(
                        file,
                        join(self.output_dir, sly_fs.get_file_name_with_ext(file)),
                    )
                elif isdir:
                    shutil.move(file, join(self.output_dir, basename(file)))

        # Preprocess config
        logger.debug("Preprocessing config")
        config = experiment_info["model_files"].get("config")
        if config is not None:
            config_name = sly_fs.get_file_name_with_ext(experiment_info["model_files"]["config"])
            output_config_path = join(self.output_dir, config_name)
            shutil.move(experiment_info["model_files"]["config"], output_config_path)
            experiment_info["model_files"]["config"] = output_config_path
            if self.is_model_benchmark_enabled:
                self._benchmark_params["model_files"]["config"] = output_config_path

        # Prepare checkpoints
        checkpoints = experiment_info["checkpoints"]
        # If checkpoints returned as directory
        if isinstance(checkpoints, str):
            checkpoint_paths = []
            for checkpoint_path in listdir(checkpoints):
                checkpoint_ext = sly_fs.get_file_ext(checkpoint_path)
                if checkpoint_ext in [".pt", ".pth"]:
                    checkpoint_paths.append(join(checkpoints, checkpoint_path))
        elif isinstance(checkpoints, list):
            checkpoint_paths = checkpoints
        else:
            raise ValueError(
                "Checkpoints should be a list of paths or a path to directory with checkpoints"
            )

        new_checkpoint_paths = []
        best_checkpoints_name = experiment_info["best_checkpoint"]

        # Prepare checkpoint files
        try:
            # need to save original key names
            ckpt_files = {}
            for file in experiment_info["model_files"]:
                file_name = sly_fs.get_file_name_with_ext(experiment_info["model_files"][file])
                with open(experiment_info["model_files"][file], "r") as f:
                    ckpt_files[file] = {"name": file_name, "content": f.read()}
        except Exception as e:
            logger.warning(f"Error loading model files: {e}")
            ckpt_files = {}

        for checkpoint_path in checkpoint_paths:
            checkpoint_name = sly_fs.get_file_name_with_ext(checkpoint_path)
            new_checkpoint_path = join(self._output_checkpoints_dir, checkpoint_name)
            shutil.move(checkpoint_path, new_checkpoint_path)
            try:
                # pylint: disable=import-error
                import torch

                state_dict = torch_load_safe(new_checkpoint_path)
                state_dict["model_info"] = {
                    "task_id": self.task_id,
                    "model_name": experiment_info["model_name"],
                    "framework": self.framework_name,
                    "checkpoint": checkpoint_name,
                    "experiment": self.gui.training_process.get_experiment_name(),
                }
                state_dict["model_meta"] = model_meta.to_json()
                state_dict["model_files"] = ckpt_files
                torch.save(state_dict, new_checkpoint_path)
            except Exception as e:
                logger.warning(f"Error writing info to checkpoint: '{checkpoint_name}'. Error:{e}")
                continue

            new_checkpoint_paths.append(new_checkpoint_path)
            if sly_fs.get_file_name_with_ext(checkpoint_path) == best_checkpoints_name:
                experiment_info["best_checkpoint"] = new_checkpoint_path
                if self.is_model_benchmark_enabled:
                    self._benchmark_params["model_files"]["checkpoint"] = new_checkpoint_path
        experiment_info["checkpoints"] = new_checkpoint_paths

        # Prepare logs
        if sly_fs.dir_exists(self.log_dir):
            logs_dir = join(self.output_dir, "logs")
            logger_type = self._app_options.get("train_logger", None)
            for root, _, files in walk(self.log_dir):
                for file in files:
                    if logger_type is None or logger_type == "tensorboard":
                        if ".tfevents." in file:
                            src_log_path = join(root, file)
                            dst_log_path = join(logs_dir, file)
                            sly_fs.copy_file(src_log_path, dst_log_path)
        return experiment_info

    # Generate experiment_info.json and app_state.json
    def _upload_file_to_team_files(
        self, local_path: str, remote_path: str, message: str
    ) -> Union[FileInfo, None]:
        """Helper function to upload a file with progress."""
        logger.debug(f"Uploading '{local_path}' to Supervisely")
        total_size = sly_fs.get_file_size(local_path)
        with self.progress_bar_main(
            message=message,
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as upload_artifacts_pbar:
            self.progress_bar_main.show()
            file_info = self._api.file.upload(
                self.team_id,
                local_path,
                remote_path,
                progress_cb=upload_artifacts_pbar,
            )
            self.progress_bar_main.hide()
            return file_info

    def _generate_train_val_splits(self, remote_dir: str, splits_data: dict) -> None:
        """
        Generates and uploads the train and val splits to the output directory.

        :param remote_dir: Remote directory path.
        :type remote_dir: str
        """
        if not self._has_splits_selector:
            return  # splits disabled in options

        local_train_val_split_path = join(self.output_dir, self._train_val_split_file)
        remote_train_val_split_path = join(remote_dir, self._train_val_split_file)

        data = {
            "train": splits_data["train"]["images_ids"],
            "val": splits_data["val"]["images_ids"],
        }

        sly_json.dump_json_file(data, local_train_val_split_path)
        self._upload_file_to_team_files(
            local_train_val_split_path,
            remote_train_val_split_path,
            f"Uploading '{self._train_val_split_file}' to Team Files",
        )

    def _generate_model_meta(self, remote_dir: str, model_meta: ProjectMeta) -> None:
        """
        Generates and uploads the model_meta.json file to the output directory.

        :param remote_dir: Remote directory path.
        :type remote_dir: str
        :param experiment_info: Information about the experiment results.
        :type experiment_info: dict
        """
        local_path = join(self.output_dir, self._model_meta_file)
        remote_path = join(remote_dir, self._model_meta_file)

        sly_json.dump_json_file(model_meta.to_json(), local_path)
        self._upload_file_to_team_files(
            local_path,
            remote_path,
            f"Uploading '{self._model_meta_file}' to Team Files",
        )

    def create_model_meta(self, task_type: str):
        """
        Convert project meta according to task type.
        """
        names_to_delete = [
            c.name for c in self.project_meta.obj_classes if c.name not in self.classes
        ]
        model_meta = self.project_meta.delete_obj_classes(names_to_delete)

        if task_type == TaskType.OBJECT_DETECTION:
            model_meta, _ = model_meta.to_detection_task(True)
        elif task_type in [
            TaskType.INSTANCE_SEGMENTATION,
            TaskType.SEMANTIC_SEGMENTATION,
        ]:
            model_meta, _ = model_meta.to_segmentation_task()  # @TODO: check background class
        return model_meta

    def _generate_experiment_info(
        self,
        remote_dir: str,
        experiment_info: Dict,
        eval_metrics: Dict = {},
        evaluation_report_id: Optional[int] = None,
        evaluation_report_link: Optional[str] = None,
        primary_metric_name: str = None,
        export_weights: Dict = {},
    ) -> dict:
        """
        Generates and uploads the experiment_info.json file to the output directory.

        :param remote_dir: Remote directory path.
        :type remote_dir: str
        :param experiment_info: Information about the experiment results.
        :type experiment_info: dict
        :param eval_metrics: Evaluation metrics.
        :type eval_metrics: dict
        :param evaluation_report_id: Evaluation report file ID.
        :type evaluation_report_id: int
        :param evaluation_report_link: Evaluation report file link.
        :type evaluation_report_link: str
        :param export_weights: Export data.
        :type export_weights: dict
        """
        logger.debug("Updating experiment info")
        experiment_info = {
            "experiment_name": self.gui.training_process.get_experiment_name(),
            "framework_name": self.framework_name,
            "model_name": experiment_info["model_name"],
            "base_checkpoint": self.base_checkpoint,
            "base_checkpoint_link": self.base_checkpoint_link,
            "task_type": experiment_info["task_type"],
            "project_id": self.project_info.id,
            "project_version": self.project_info.version,
            "task_id": self.task_id,
            "model_files": experiment_info["model_files"],
            "checkpoints": experiment_info["checkpoints"],
            "best_checkpoint": sly_fs.get_file_name_with_ext(experiment_info["best_checkpoint"]),
            "export": export_weights,
            "app_state": self._app_state_file,
            "model_meta": self._model_meta_file,
            "hyperparameters": self._hyperparameters_file,
            "artifacts_dir": remote_dir,
            "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "evaluation_report_id": evaluation_report_id,
            "evaluation_report_link": evaluation_report_link,
            "evaluation_metrics": eval_metrics,
            "primary_metric": primary_metric_name,
            "logs": {"type": "tensorboard", "link": f"{remote_dir}logs/"},
            "device": self.gui.training_process.get_device_name(),
            "training_duration": self._training_duration,
            "train_collection_id": self._train_collection_id,
            "val_collection_id": self._val_collection_id,
        }

        if self._has_splits_selector:
            experiment_info["train_val_split"] = self._train_val_split_file
            experiment_info["train_size"] = len(self._train_split)
            experiment_info["val_size"] = len(self._val_split)

        remote_checkpoints_dir = join(remote_dir, self._remote_checkpoints_dir_name)
        checkpoint_files = self._api.file.list(
            self.team_id, remote_checkpoints_dir, return_type="fileinfo"
        )
        experiment_info["checkpoints"] = [
            f"checkpoints/{checkpoint.name}" for checkpoint in checkpoint_files
        ]

        experiment_info["best_checkpoint"] = sly_fs.get_file_name_with_ext(
            experiment_info["best_checkpoint"]
        )

        for file in experiment_info["model_files"]:
            experiment_info["model_files"][file] = sly_fs.get_file_name_with_ext(
                experiment_info["model_files"][file]
            )

        local_path = join(self.output_dir, self._experiment_json_file)
        remote_path = join(remote_dir, self._experiment_json_file)
        sly_json.dump_json_file(experiment_info, local_path)
        self._upload_file_to_team_files(
            local_path,
            remote_path,
            f"Uploading '{self._experiment_json_file}' to Team Files",
        )

        # Do not include this fields to uploaded file:
        experiment_info["project_preview"] = self.project_info.image_preview_url
        return experiment_info

    def _generate_experiment_report(
        self, experiment_info: dict, model_meta: ProjectMeta
    ) -> FileInfo:
        """
        Generates and uploads the experiment report to the output directory.
        """
        # @TODO: add report to workflow output
        experiment = ExperimentGenerator(
            api=self._api,
            experiment_info=experiment_info,
            hyperparameters=self.hyperparameters_yaml,
            model_meta=model_meta,
            serving_class=self._inference_class,
            team_id=self.team_id,
            output_dir=join(self.work_dir, "experiment_report"),
            app_options=self._app_options,
        )
        experiment.generate()

        experiment.upload_to_artifacts()
        file_info = experiment.get_report()
        return file_info

    def _generate_hyperparameters(self, remote_dir: str, experiment_info: Dict) -> None:
        """
        Generates and uploads the hyperparameters.yaml file to the output directory.

        :param remote_dir: Remote directory path.
        :type remote_dir: str
        :param experiment_info: Information about the experiment results.
        :type experiment_info: dict
        """
        local_path = join(self.output_dir, self._hyperparameters_file)
        remote_path = join(remote_dir, self._hyperparameters_file)

        with open(local_path, "w") as file:
            file.write(self.hyperparameters_yaml)

        file_info = self._upload_file_to_team_files(
            local_path,
            remote_path,
            f"Uploading '{self._hyperparameters_file}' to Team Files",
        )
        experiment_info["hyperparameters_id"] = file_info.id
        return experiment_info

    def _generate_app_state(self, remote_dir: str, experiment_info: Dict) -> None:
        """
        Generates and uploads the app_state.json file to the output directory.

        :param remote_dir: Remote directory path.
        :type remote_dir: str
        :param experiment_info: Information about the experiment results.
        :type experiment_info: dict
        """
        app_state = self.get_app_state(experiment_info)

        local_path = join(self.output_dir, self._app_state_file)
        remote_path = join(remote_dir, self._app_state_file)
        sly_json.dump_json_file(app_state, local_path)
        self._upload_file_to_team_files(
            local_path, remote_path, f"Uploading '{self._app_state_file}' to Team Files"
        )

    def _upload_demo_files(self, remote_dir: str) -> None:
        if not self.gui.training_artifacts.need_upload_demo:
            return
        demo = self._app_options.get("demo")
        if demo is None:
            return
        demo_path = demo.get("path")
        if demo_path is None:
            return

        local_demo_dir = join(getcwd(), demo_path)
        if not sly_fs.dir_exists(local_demo_dir):
            logger.info(f"Demo directory '{local_demo_dir}' does not exist")
            return

        remote_demo_dir = join(remote_dir, "demo")
        local_files = sly_fs.list_files_recursively(local_demo_dir)
        total_size = sum([sly_fs.get_file_size(file_path) for file_path in local_files])
        logger.debug(
            f"Uploading demo files of total size {total_size} bytes to Supervisely Team Files directory '{remote_demo_dir}'"
        )
        with self.progress_bar_main(
            message="Uploading demo files to Team Files",
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as upload_artifacts_pbar:
            self.progress_bar_main.show()
            remote_dir = self._api.file.upload_directory_fast(
                team_id=self.team_id,
                local_dir=local_demo_dir,
                remote_dir=remote_demo_dir,
                progress_cb=upload_artifacts_pbar.update,
            )

            self.progress_bar_main.hide()

    def _generate_experiment_output(self, experiment_info: dict, model_meta: ProjectMeta, session_link_file_info: FileInfo) -> tuple:
        """
        Generates and uploads the experiment page to the output directory, if report generation is successful.
        Otherwise, artifacts directory link will be used for output.

        :param experiment_info: Information about the experiment results.
        :type experiment_info: dict
        :param model_meta: Model meta with object classes.
        :type model_meta: ProjectMeta
        :param session_link_file_info: Artifacts directory link, used if report is not generated.
        :type session_link_file_info: FileInfo
        :return: Output file info and experiment info.
        :rtype: tuple
        """
        need_generate_report = self._app_options.get("generate_report", False)
        if need_generate_report:  # link to experiment page
            try:
                output_file_info = self._generate_experiment_report(experiment_info, model_meta)
                experiment_info["has_report"] = True
                experiment_info["experiment_report_id"] = output_file_info.id
            except Exception as e:
                logger.error(f"Error generating experiment report: {e}")
                output_file_info = session_link_file_info
                experiment_info["has_report"] = False
                experiment_info["experiment_report_id"] = None
        else:  # link to artifacts directory
            output_file_info = session_link_file_info
            experiment_info["has_report"] = False
            experiment_info["experiment_report_id"] = None
        return output_file_info, experiment_info

    def _get_train_val_splits_for_app_state(self) -> Dict:
        """
        Gets the train and val splits information for app_state.json.

        :return: Train and val splits information based on selected split method.
        :rtype: dict
        """
        if not self._has_splits_selector:
            return {}  # splits disabled in options

        split_method = self.gui.train_val_splits_selector.get_split_method()
        train_val_splits = {"method": split_method.lower()}
        if split_method == "Random":
            train_val_splits.update(
                {
                    "split": "train",
                    "percent": self.gui.train_val_splits_selector.train_val_splits.get_train_split_percent(),
                }
            )
        elif split_method == "Based on tags":
            train_val_splits.update(
                {
                    "train_tag": self.gui.train_val_splits_selector.train_val_splits.get_train_tag(),
                    "val_tag": self.gui.train_val_splits_selector.train_val_splits.get_val_tag(),
                    "untagged_action": self.gui.train_val_splits_selector.train_val_splits.get_untagged_action(),
                }
            )
        elif split_method == "Based on datasets":
            train_val_splits.update(
                {
                    "train_datasets": self.gui.train_val_splits_selector.train_val_splits.get_train_dataset_ids(),
                    "val_datasets": self.gui.train_val_splits_selector.train_val_splits.get_val_dataset_ids(),
                }
            )
        elif split_method == "Based on collections":
            train_val_splits.update(
                {
                    "train_collections": self.gui.train_val_splits_selector.get_train_collection_ids(),
                    "val_collections": self.gui.train_val_splits_selector.get_val_collection_ids(),
                }
            )
        return train_val_splits

    def _get_model_config_for_app_state(self, experiment_info: Dict = None) -> Dict:
        """
        Gets the model configuration information for app_state.json.

        :param experiment_info: Information about the experiment results.
        :type experiment_info: dict
        """
        experiment_info = experiment_info or {}

        if self.model_source == ModelSource.PRETRAINED:
            model_name = experiment_info.get("model_name") or _get_model_name(self.model_info)
            return {
                "source": ModelSource.PRETRAINED,
                "model_name": model_name,
            }
        elif self.model_source == ModelSource.CUSTOM:
            return {
                "source": ModelSource.CUSTOM,
                "task_id": self.task_id,
                "checkpoint": "custom checkpoint",
            }

    # ----------------------------------------- #

    # Upload artifacts
    def _upload_artifacts(self) -> None:
        """
        Uploads the training artifacts to Supervisely.
        Path is generated based on the project ID, task ID, and framework name.

        Path: /experiments/{project_id}_{project_name}/{task_id}_{framework_name}/
        Example path: /experiments/43192_Apples/68271_rt-detr/
        """
        task_id = self.task_id

        remote_artifacts_dir = f"/{self._experiments_dir_name}/{self.project_id}_{self.project_name}/{task_id}_{self.framework_name}/"
        logger.info(
            f"Uploading artifacts directory: '{self.output_dir}' to Supervisely Team Files directory '{remote_artifacts_dir}'"
        )
        # Clean debug directory if exists
        if task_id == -1:
            if self._api.file.dir_exists(self.team_id, f"{remote_artifacts_dir}/", True):
                with self.progress_bar_main(
                    message=f"[Debug] Cleaning train artifacts: '{remote_artifacts_dir}/'",
                    total=1,
                ) as upload_artifacts_pbar:
                    self.progress_bar_main.show()
                    self._api.file.remove_dir(self.team_id, f"{remote_artifacts_dir}", True)
                    upload_artifacts_pbar.update(1)
                    self.progress_bar_main.hide()

        # Generate link file
        if is_production():
            app_url = f"/apps/sessions/{task_id}"
        else:
            app_url = "This is a debug session. No link available."
        app_link_path = join(self.output_dir, "open_app.lnk")
        with open(app_link_path, "w") as text_file:
            print(app_url, file=text_file)

        local_files = sly_fs.list_files_recursively(self.output_dir)
        total_size = sum([sly_fs.get_file_size(file_path) for file_path in local_files])
        with self.progress_bar_main(
            message="Uploading train artifacts to Team Files",
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as upload_artifacts_pbar:
            self.progress_bar_main.show()
            remote_dir = self._api.file.upload_directory_fast(
                team_id=self.team_id,
                local_dir=self.output_dir,
                remote_dir=remote_artifacts_dir,
                progress_cb=upload_artifacts_pbar.update,
            )
            self.progress_bar_main.hide()

        file_info = self._api.file.get_info_by_path(self.team_id, join(remote_dir, "open_app.lnk"))

        # Set offline tensorboard button payload
        if is_production():
            remote_log_dir = join(remote_dir, "logs")
            tb_btn_payload = {"state": {"slyFolder": remote_log_dir}}
            self.gui.training_logs.tensorboard_offline_button.payload = tb_btn_payload
            self.gui.training_logs.tensorboard_offline_button.set_check_existing_task_cb(
                generate_task_check_function_js(remote_log_dir)
            )
            self.gui.training_logs.tensorboard_offline_button.enable()

        return remote_dir, file_info

    def _set_training_output(
        self,
        experiment_info: dict,
        remote_dir: str,
        file_info: FileInfo,
        mb_eval_report: Optional[FileInfo] = None,
    ) -> None:
        """
        Sets the training output in the GUI.
        """
        self.gui.set_next_step()
        logger.info("All training artifacts uploaded successfully")
        self.gui.training_process.start_button.loading = False
        self.gui.training_process.start_button.disable()
        self.gui.training_process.stop_button.disable()

        # Set artifacts to GUI
        if is_production():
            self._api.task.set_output_experiment(self.task_id, experiment_info)
        set_directory(remote_dir)
        # Set artifacts thumbnail to GUI
        self.gui.training_artifacts.artifacts_thumbnail.set(file_info)
        self.gui.training_artifacts.artifacts_thumbnail.show()
        # Set experiment report thumbnail to GUI
        self.gui.training_artifacts.artifacts_field.show()
        # ---------------------------- #

        # Set model benchmark to GUI
        if self._app_options.get("model_benchmark", False):
            if mb_eval_report is not None:
                self.gui.training_artifacts.model_benchmark_report_thumbnail.set(mb_eval_report)
                self.gui.training_artifacts.model_benchmark_report_thumbnail.show()
                self.gui.training_artifacts.model_benchmark_report_field.show()
            else:
                if self.gui.hyperparameters_selector.get_model_benchmark_checkbox_value():
                    self.gui.training_artifacts.model_benchmark_fail_text.show()
                    self.gui.training_artifacts.model_benchmark_report_field.show()
        # ---------------------------- #

        # Set instruction to GUI
        demo_options = self._app_options.get("demo", {})
        demo_path = demo_options.get("path", None)
        if demo_path is not None:
            # Show PyTorch demo if available
            if self.gui.training_artifacts.pytorch_demo_exists(demo_path):
                if self.gui.training_artifacts.pytorch_instruction is not None:
                    self.gui.training_artifacts.pytorch_instruction.show()

            # Show ONNX demo if supported and available
            if (
                self._app_options.get("export_onnx_supported", False)
                and self.gui.hyperparameters_selector.get_export_onnx_checkbox_value()
                and self.gui.training_artifacts.onnx_demo_exists(demo_path)
                and self.gui.training_artifacts.onnx_instruction is not None
            ):
                self.gui.training_artifacts.onnx_instruction.show()

            # Show TensorRT demo if supported and available
            if (
                self._app_options.get("export_tensorrt_supported", False)
                and self.gui.hyperparameters_selector.get_export_tensorrt_checkbox_value()
                and self.gui.training_artifacts.trt_demo_exists(demo_path)
                and self.gui.training_artifacts.trt_instruction is not None
            ):
                self.gui.training_artifacts.trt_instruction.show()

            # Show the inference demo widget if overview or any demo is available
            if self.gui.training_artifacts.inference_demo_field is not None and any(
                [
                    self.gui.training_artifacts.overview_demo_exists(demo_path),
                    self.gui.training_artifacts.pytorch_demo_exists(demo_path),
                    self.gui.training_artifacts.onnx_demo_exists(demo_path),
                    self.gui.training_artifacts.trt_demo_exists(demo_path),
                ]
            ):
                if self.gui.training_artifacts.inference_demo_field is not None:
                    self.gui.training_artifacts.inference_demo_field.show()
        # ---------------------------- #

        # Set status to completed and unlock
        self.gui.training_artifacts.validator_text.set(
            self.gui.training_artifacts.success_message_text, "success"
        )
        self.gui.training_artifacts.validator_text.show()
        self.gui.training_artifacts.card.unlock()
        # ---------------------------- #

    # Model Benchmark
    def _get_eval_results_dir_name(self) -> str:
        """
        Returns the evaluation results path.
        """
        task_dir = f"{self.task_id}_{self._app_name}"
        eval_res_dir = (
            f"/model-benchmark/{self.project_info.id}_{self.project_info.name}/{task_dir}/"
        )
        eval_res_dir = self._api.storage.get_free_dir_name(self.team_id, eval_res_dir)
        return eval_res_dir

    def _run_model_benchmark(
        self,
        local_artifacts_dir: str,
        remote_artifacts_dir: str,
        experiment_info: dict,
        splits_data: dict,
        model_meta: ProjectInfo,
        gt_project_id: int = None,
    ) -> tuple:
        """
        Runs the Model Benchmark evaluation process. Model benchmark runs only in production mode.

        :param local_artifacts_dir: Local directory path.
        :type local_artifacts_dir: str
        :param remote_artifacts_dir: Remote directory path.
        :type remote_artifacts_dir: str
        :param experiment_info: Information about the experiment results.
        :type experiment_info: dict
        :param splits_data: Information about the train and val splits.
        :type splits_data: dict
        :param model_meta: Model meta with object classes.
        :type model_meta: ProjectInfo
        :param gt_project_id: Ground truth project ID with converted shapes.
        :type gt_project_id: int
        :return: Evaluation report, report ID and evaluation metrics.
        :rtype: tuple
        """
        lnk_file_info, report, report_id, eval_metrics, primary_metric_name = (
            None,
            None,
            None,
            {},
            None,
        )
        if self._inference_class is None:
            logger.warning(
                "Inference class is not registered, model benchmark disabled. "
                "Use 'register_inference_class' method to register inference class."
            )
            return lnk_file_info, report, report_id, eval_metrics, primary_metric_name

        # Can't get task type from session. requires before session init
        supported_task_types = [
            TaskType.OBJECT_DETECTION,
            TaskType.INSTANCE_SEGMENTATION,
            TaskType.SEMANTIC_SEGMENTATION,
        ]
        task_type = experiment_info["task_type"]
        if task_type not in supported_task_types:
            logger.warning(
                f"Task type: '{task_type}' is not supported for Model Benchmark. "
                f"Supported tasks: {', '.join(task_type)}"
            )
            return lnk_file_info, report, report_id, eval_metrics, primary_metric_name

        logger.info("Running Model Benchmark evaluation")
        try:
            remote_checkpoints_dir = join(remote_artifacts_dir, "checkpoints")
            best_checkpoint = experiment_info.get("best_checkpoint", None)
            best_filename = sly_fs.get_file_name_with_ext(best_checkpoint)
            remote_best_checkpoint = join(remote_checkpoints_dir, best_filename)

            logger.info(f"Creating the report for the best model: {best_filename!r}")
            self.gui.training_process.validator_text.set(
                f"Creating evaluation report for the best model: {best_filename!r}",
                "info",
            )
            self.progress_bar_main(message="Starting Model Benchmark evaluation", total=1)
            self.progress_bar_main.show()

            # 0. Serve trained model
            m: Inference = self._inference_class(
                model_dir=self.model_dir,
                use_gui=False,
                custom_inference_settings=self._inference_settings,
            )
            if hasattr(m, "in_train"):
                m.in_train = True

            logger.info(f"Using device: {self.device}")

            self._benchmark_params["device"] = self.device
            self._benchmark_params["model_info"] = {
                "artifacts_dir": remote_artifacts_dir,
                "model_name": experiment_info["model_name"],
                "framework_name": self.framework_name,
                "model_meta": model_meta.to_json(),
                "task_type": task_type,
            }
            self._benchmark_params["without_workflow"] = True

            logger.info(f"Deploy parameters: {self._benchmark_params}")

            m._load_model_headless(**self._benchmark_params)
            m.serve()

            port = 8000
            session = SessionJSON(self._api, session_url=f"http://localhost:{port}")
            benchmark_dir = join(local_artifacts_dir, "benchmark")
            sly_fs.mkdir(benchmark_dir, True)

            # 1. Init benchmark
            benchmark_dataset_ids = splits_data["val"]["dataset_ids"]
            benchmark_images_ids = splits_data["val"]["images_ids"]
            train_dataset_ids = splits_data["train"]["dataset_ids"]
            train_images_ids = splits_data["train"]["images_ids"]

            bm = None
            if gt_project_id is None:
                gt_project_id = self.project_info.id

            if task_type == TaskType.OBJECT_DETECTION:
                eval_params = ObjectDetectionEvaluator.load_yaml_evaluation_params()
                eval_params = yaml.safe_load(eval_params)
                bm = ObjectDetectionBenchmark(
                    self._api,
                    gt_project_id,
                    output_dir=benchmark_dir,
                    gt_dataset_ids=benchmark_dataset_ids,
                    gt_images_ids=benchmark_images_ids,
                    progress=self.progress_bar_main,
                    progress_secondary=self.progress_bar_secondary,
                    classes_whitelist=self.classes,
                    evaluation_params=eval_params,
                )
            elif task_type == TaskType.INSTANCE_SEGMENTATION:
                eval_params = InstanceSegmentationEvaluator.load_yaml_evaluation_params()
                eval_params = yaml.safe_load(eval_params)
                bm = InstanceSegmentationBenchmark(
                    self._api,
                    gt_project_id,
                    output_dir=benchmark_dir,
                    gt_dataset_ids=benchmark_dataset_ids,
                    gt_images_ids=benchmark_images_ids,
                    progress=self.progress_bar_main,
                    progress_secondary=self.progress_bar_secondary,
                    classes_whitelist=self.classes,
                    evaluation_params=eval_params,
                )
            elif task_type == TaskType.SEMANTIC_SEGMENTATION:
                eval_params = SemanticSegmentationEvaluator.load_yaml_evaluation_params()
                eval_params = yaml.safe_load(eval_params)
                bm = SemanticSegmentationBenchmark(
                    self._api,
                    gt_project_id,
                    output_dir=benchmark_dir,
                    gt_dataset_ids=benchmark_dataset_ids,
                    gt_images_ids=benchmark_images_ids,
                    progress=self.progress_bar_main,
                    progress_secondary=self.progress_bar_secondary,
                    classes_whitelist=self.classes,
                    evaluation_params=eval_params,
                )
            else:
                raise ValueError(f"Task type: '{task_type}' is not supported for Model Benchmark")

            if self._has_splits_selector:
                app_session_id = self.task_id
                if app_session_id == -1:
                    app_session_id = None
                if self.gui.train_val_splits_selector.get_split_method() == "Based on datasets":
                    train_info = {
                        "app_session_id": app_session_id,
                        "train_dataset_ids": train_dataset_ids,
                        "train_images_ids": None,
                        "images_count": len(self._train_split),
                    }
                else:
                    train_info = {
                        "app_session_id": app_session_id,
                        "train_dataset_ids": None,
                        "train_images_ids": train_images_ids,
                        "images_count": len(self._train_split),
                    }
            else:
                # @TODO: Add train info for apps without splits
                train_info = None

            bm.train_info = train_info

            # 2. Run inference
            bm.run_inference(session)

            # 3. Pull results from the server
            gt_project_path, dt_project_path = bm.download_projects(save_images=False)

            # 4. Evaluate
            bm._evaluate(gt_project_path, dt_project_path)
            bm._dump_eval_inference_info(bm._eval_inference_info)

            # 5. Upload evaluation results
            eval_res_dir = self._get_eval_results_dir_name()
            bm.upload_eval_results(eval_res_dir + "/evaluation/")

            # 6. Speed test
            if self.gui.hyperparameters_selector.get_speedtest_checkbox_value() is True:
                self.progress_bar_secondary.show()
                bm.run_speedtest(session, self.project_info.id)
                self.progress_bar_secondary.hide()
                bm.upload_speedtest_results(eval_res_dir + "/speedtest/")

            # 7. Prepare visualizations, report and upload
            bm.visualize()
            _ = bm.upload_visualizations(eval_res_dir + "/visualizations/")
            lnk_file_info = bm.lnk
            report = bm.report
            report_id = bm.report.id
            eval_metrics = bm.key_metrics
            primary_metric_name = bm.primary_metric_name
            bm.upload_report_link(remote_artifacts_dir, report_id, self.output_dir)

            # 8. UI updates
            self.progress_bar_main.hide()
            self.progress_bar_secondary.hide()
            logger.info("Model benchmark evaluation completed successfully")
            logger.info(
                f"Predictions project name: {bm.dt_project_info.name}. Workspace_id: {bm.dt_project_info.workspace_id}"
            )

        except Exception as e:
            logger.error(f"Model benchmark failed. {repr(e)}", exc_info=True)
            pred_error_message = (
                "Not found any predictions. Please make sure that your model produces predictions."
            )
            if isinstance(e, ValueError) and str(e) == pred_error_message:
                self.gui.training_artifacts.model_benchmark_fail_text.set(
                    "The Model Evaluation report cannot be generated: The model is not making predictions. "
                    "This indicates that your model may not have trained successfully or is underfitted. "
                    "You can try increasing the number of epochs or adjusting the hyperparameters more carefully.",
                    "warning",
                )

            lnk_file_info, report, report_id, eval_metrics, primary_metric_name = (
                None,
                None,
                None,
                {},
                None,
            )

            self._set_text_status("finalizing")
            self.progress_bar_main.hide()
            self.progress_bar_secondary.hide()
            try:
                if bm.dt_project_info:
                    self._api.project.remove(bm.dt_project_info.id)
                diff_project_info = bm.get_diff_project_info()
                if diff_project_info:
                    self._api.project.remove(diff_project_info.id)
            except Exception as e2:
                return (
                    lnk_file_info,
                    report,
                    report_id,
                    eval_metrics,
                    primary_metric_name,
                )
        return lnk_file_info, report, report_id, eval_metrics, primary_metric_name

    # ----------------------------------------- #

    # Workflow
    def _workflow_input(self):
        """
        Adds the input data to the workflow.
        """
        if self.project_info.type == ProjectType.IMAGES.value:
            try:
                project_version_id = self._api.project.version.create(
                    self.project_info,
                    self._app_name,
                    f"This backup was created automatically by Supervisely before the {self._app_name} task with ID: {self._api.task_id}",
                )
            except Exception as e:
                logger.warning(f"Failed to create a project version: {repr(e)}")
                project_version_id = None
        else:
            project_version_id = None

        try:
            if project_version_id is None:
                project_version_id = (
                    self.project_info.version.get("id", None) if self.project_info.version else None
                )
            self._api.app.workflow.add_input_project(
                self.project_info.id, version_id=project_version_id
            )

            file_info = None
            if self.model_source == ModelSource.CUSTOM:
                file_info = self._api.file.get_info_by_path(
                    self.team_id,
                    self.gui.model_selector.experiment_selector.get_selected_checkpoint_path(),
                )
                if file_info is not None:
                    self._api.app.workflow.add_input_file(file_info, model_weight=True)
            logger.debug(
                f"Workflow Input: Project ID - {self.project_info.id}, Project Version ID - {project_version_id}, Input File - {True if file_info else False}"
            )
        except Exception as e:
            logger.debug(f"Failed to add input to the workflow: {repr(e)}")

    def _workflow_output(
        self,
        team_files_dir: str,
        file_info: FileInfo,
        model_benchmark_report: Optional[FileInfo] = None,
        model_benchmark_report_id: Optional[FileInfo] = None,
    ):
        """
        Adds the output data to the workflow.

        :param team_files_dir: Team files directory.
        :type team_files_dir: str
        :param file_info: FileInfo of the best checkpoint.
        :type file_info: FileInfo
        :param model_benchmark_report: FileInfo of the model benchmark report link (.lnk).
        :type model_benchmark_report: Optional[FileInfo]
        :param model_benchmark_report_id: Model benchmark report ID.
        """
        try:
            module_id = (
                self._api.task.get_info_by_id(self._api.task_id)
                .get("meta", {})
                .get("app", {})
                .get("id")
            )
            logger.debug(f"Workflow Output: Model artifacts - '{team_files_dir}'")

            node_settings = WorkflowSettings(
                title=self._app_name,
                url=(
                    f"/apps/{module_id}/sessions/{self._api.task_id}"
                    if module_id
                    else f"apps/sessions/{self._api.task_id}"
                ),
                url_title="Show Results",
            )

            if file_info:
                relation_settings = WorkflowSettings(
                    title="Checkpoints",
                    icon="folder",
                    icon_color="#FFA500",
                    icon_bg_color="#FFE8BE",
                    url=f"/files/{file_info.id}/true",
                    url_title="Open Folder",
                )
                meta = WorkflowMeta(
                    relation_settings=relation_settings, node_settings=node_settings
                )
                logger.debug(f"Workflow Output: meta \n    {meta}")
                # self._api.app.workflow.add_output_file(file_info, model_weight=True, meta=meta)

                remote_checkpoint_dir = dirname(file_info.path)
                self._api.app.workflow.add_output_folder(remote_checkpoint_dir, meta=meta)
            else:
                logger.debug(
                    "File with checkpoints not found in Team Files. Cannot set workflow output."
                )

            if self.is_model_benchmark_enabled:
                if model_benchmark_report:
                    mb_relation_settings = WorkflowSettings(
                        title="Model Benchmark",
                        icon="assignment",
                        icon_color="#674EA7",
                        icon_bg_color="#CCCCFF",
                        url=f"/model-benchmark?id={model_benchmark_report_id}",
                        url_title="Open Report",
                    )

                    meta = WorkflowMeta(
                        relation_settings=mb_relation_settings,
                        node_settings=node_settings,
                    )
                    self._api.app.workflow.add_output_file(model_benchmark_report, meta=meta)
                else:
                    logger.debug(
                        "File with model benchmark report not found in Team Files. Cannot set workflow output."
                    )
        except Exception as e:
            logger.debug(f"Failed to add output to the workflow: {repr(e)}")
        # ----------------------------------------- #

    # Logger
    def _init_logger(self):
        """
        Initialize training logger. Set up Tensorboard and callbacks.
        """
        selected_logger = self._app_options.get("train_logger", "")
        if selected_logger.lower() == "tensorboard":
            setup_train_logger("tensorboard_logger")
            train_logger.set_log_dir(self.log_dir)
            self._init_tensorboard()
        else:
            setup_train_logger("default_logger")
        self._setup_logger_callbacks()

    def _init_tensorboard(self):
        if self._tensorboard_process is not None:
            logger.debug("Tensorboard server is already running")
            return
        self._register_routes()

        args = [
            "tensorboard",
            "--logdir",
            self.log_dir,
            "--host=localhost",
            f"--port={self._tensorboard_port}",
            "--load_fast=auto",
            "--reload_multifile=true",
        ]
        self._tensorboard_process = subprocess.Popen(args)
        self.app.call_before_shutdown(self.stop_tensorboard)
        print(f"Tensorboard server has been started")
        self.gui.training_logs.tensorboard_button.enable()

    def start_tensorboard(self, log_dir: str, port: int = None):
        """
        Method to manually start Tensorboard in the user's training code.
        Tensorboard is started automatically if the 'train_logger' is set to 'tensorboard' in app_options.yaml file.

        :param log_dir: Directory path to the log files.
        :type log_dir: str
        :param port: Port number for Tensorboard, defaults to None
        :type port: int, optional
        """
        if port is not None:
            self._tensorboard_port = port
        self.log_dir = log_dir
        self._init_tensorboard()

    def stop_tensorboard(self):
        """Stop Tensorboard server"""
        if self._tensorboard_process is not None:
            self._tensorboard_process.terminate()
            self._tensorboard_process = None
            print(f"Tensorboard server has been stopped")
        else:
            print("Tensorboard server is not running")

    def _setup_logger_callbacks(self):
        """
        Set up callbacks for the training logger.
        """
        epoch_pbar = None
        step_pbar = None

        def start_training_callback(total_epochs: int):
            """
            Callback function that is called when the training process starts.
            """
            nonlocal epoch_pbar
            logger.info(f"Training started for {total_epochs} epochs")
            pbar_widget = self.progress_bar_main
            pbar_widget.show()
            epoch_pbar = pbar_widget(message=f"Epochs", total=total_epochs)

        def finish_training_callback():
            """
            Callback function that is called when the training process finishes.
            """
            self.progress_bar_main.hide()
            self.progress_bar_secondary.hide()
            train_logger.close()

        def start_epoch_callback(total_steps: int):
            """
            Callback function that is called when a new epoch starts.
            """
            nonlocal step_pbar
            logger.info(f"Epoch started. Total steps: {total_steps}")
            pbar_widget = self.progress_bar_secondary
            pbar_widget.show()
            step_pbar = pbar_widget(message=f"Steps", total=total_steps)

        def finish_epoch_callback():
            """
            Callback function that is called when an epoch finishes.
            """
            epoch_pbar.update(1)

        def step_callback():
            """
            Callback function that is called when a step iteration is completed.
            """
            step_pbar.update(1)

        train_logger.add_on_train_started_callback(start_training_callback)
        train_logger.add_on_train_finish_callback(finish_training_callback)

        train_logger.add_on_epoch_started_callback(start_epoch_callback)
        train_logger.add_on_epoch_finish_callback(finish_epoch_callback)

        train_logger.add_on_step_finished_callback(step_callback)

    # ----------------------------------------- #
    def start_in_thread(self):
        def auto_train():
            import threading
            threading.Thread(target=self._wrapped_start_training, daemon=True).start()
        self._server.add_event_handler("startup", auto_train)

    def _wrapped_start_training(self):
        """
        Wrapper function to wrap the training process.
        """
        experiment_info = None
        check_logs_text = "Please check the logs for more details."

        try:
            self._set_train_widgets_state_on_start()
            if self._train_func is None:
                raise ValueError("Train function is not defined")
            self._prepare_working_dir()
            self._init_logger()
        except Exception as e:
            message = f"Error occurred during training initialization. {check_logs_text}"
            self._show_error(message, e)
            self._set_ws_progress_status("reset")
            self.app.shutdown()
            raise e

        try:
            self._set_text_status("preparing")
            self._set_ws_progress_status("preparing")
            self._prepare()
        except Exception as e:
            message = f"Error occurred during data preparation. {check_logs_text}"
            self._show_error(message, e)
            self._set_ws_progress_status("reset")
            raise e

        try:
            self._set_text_status("training")
            if self._app_options.get("train_logger", None) is None:
                self._set_ws_progress_status("training")

            experiment_info = self._train_func()
            self._training_duration = self._train_func.elapsed
        except ZeroDivisionError as e:
            message = (
                "'ZeroDivisionError' occurred during training. "
                "The error was caused by an insufficient dataset size relative to the specified batch size in hyperparameters. "
                "Please check input data and hyperparameters."
            )
            self._show_error(message, e)
            self._set_ws_progress_status("reset")
            return
        except Exception as e:
            message = f"Error occurred during training. {check_logs_text}"
            self._show_error(message, e)
            self._set_ws_progress_status("reset")
            raise e

        try:
            self._set_text_status("finalizing")
            self._set_ws_progress_status("finalizing")
            self._finalize(experiment_info)
            self.gui.training_process.start_button.loading = False

            if is_production() and self.gui.training_logs.tensorboard_offline_button is not None:
                self.gui.training_logs.tensorboard_button.hide()
                self.gui.training_logs.tensorboard_offline_button.show()

            time.sleep(1)
        except Exception as e:
            message = f"Error occurred during finalizing and uploading training artifacts. {check_logs_text}"
            self._show_error(message, e)
            self._set_ws_progress_status("reset")
            raise e
        finally:
            self.app.shutdown()

    def _show_error(self, message: str, e=None):
        if e is not None:
            logger.error(f"{message}: {repr(e)}", exc_info=True)
        else:
            logger.error(message)
        self.gui.training_process.validator_text.set(message, "error")
        self.gui.training_process.validator_text.show()
        self.gui.training_process.start_button.loading = False
        self._restore_train_widgets_state_on_error()
        show_dialog(title="Error", description=message, status="error")

    def _set_train_widgets_state_on_start(self):
        self.gui.disable_select_buttons()
        self.gui.training_artifacts.validator_text.hide()
        self._validate_experiment_name()
        self.gui.training_process.experiment_name_input.disable()
        if self._app_options.get("device_selector", False):
            self.gui.training_process.select_device._select.disable()
            self.gui.training_process.select_device.disable()

        if self._app_options.get("model_benchmark", False):
            self.gui.training_artifacts.model_benchmark_report_thumbnail.hide()
            self.gui.training_artifacts.model_benchmark_fail_text.hide()
            self.gui.training_artifacts.model_benchmark_report_field.hide()

        self.gui.training_logs.card.unlock()
        self.gui.stepper.set_active_step(7)
        self.gui.training_process.validator_text.set("Training has been started...", "info")
        self.gui.training_process.validator_text.show()
        self.gui.training_process.start_button.loading = True

    def _restore_train_widgets_state_on_error(self):
        self.gui.training_logs.card.lock()
        self.gui.stepper.set_active_step(self.gui.stepper.get_active_step() - 1)
        self.gui.training_process.experiment_name_input.enable()
        if self._app_options.get("device_selector", False):
            self.gui.training_process.select_device._select.enable()
            self.gui.training_process.select_device.enable()
        self.gui.enable_select_buttons()

    def _validate_experiment_name(self) -> bool:
        experiment_name = self.gui.training_process.get_experiment_name()
        if not experiment_name:
            logger.error("Experiment name is empty")
            raise ValueError("Experiment name is empty")
        invalid_chars = r"\/"
        if any(char in experiment_name for char in invalid_chars):
            logger.error(f"Experiment name contains invalid characters: {invalid_chars}")
            raise ValueError(f"Experiment name contains invalid characters: {invalid_chars}")
        return True

    def _set_text_status(
        self,
        status: Literal[
            "reset",
            "completed",
            "training",
            "finalizing",
            "preparing",
            "uploading",
            "benchmark",
            "metadata",
            "export_onnx",
            "export_trt",
            "convert_gt_project",
            "experiment_report",
        ],
    ):
        if status == "reset":
            message = ""
            status = "text"
            self.gui.training_process.validator_text.set(message, status)
            return
        elif status == "completed":
            message = "Training completed"
            status = "success"
        elif status == "training":
            message = "Training is in progress..."
            status = "info"
        elif status == "finalizing":
            message = "Finalizing and preparing training artifacts..."
            status = "info"
        elif status == "preparing":
            message = "Preparing data for training..."
            status = "info"
        elif status == "export_onnx":
            message = f"Converting to {RuntimeType.ONNXRUNTIME}"
            status = "info"
        elif status == "export_trt":
            message = f"Converting to {RuntimeType.TENSORRT}"
            status = "info"
        elif status == "uploading":
            message = "Uploading training artifacts..."
            status = "info"
        elif status == "benchmark":
            message = "Running Model Benchmark evaluation..."
            status = "info"
        elif status == "validating":
            message = "Validating experiment..."
            status = "info"
        elif status == "metadata":
            message = "Generating training metadata..."
            status = "info"
        elif status == "convert_gt_project":
            message = "Converting GT project..."
            status = "info"
        elif status == "experiment_report":
            message = "Generating experiment report..."
            status = "info"

        message = f"Status: {message}"
        self.gui.training_process.validator_text.set(message, status)

    def _set_ws_progress_status(
        self,
        status: Literal["reset", "completed", "training", "finalizing", "preparing"],
    ):
        message = ""
        if status == "reset":
            message = "Ready for training"
        elif status == "completed":
            message = "Training completed"
        elif status == "training":
            message = "Training is in progress..."
        elif status == "finalizing":
            message = "Finalizing and uploading training artifacts..."
        elif status == "preparing":
            message = "Preparing data for training..."

        self.progress_bar_main.hide()
        self.progress_bar_secondary.hide()
        with self.progress_bar_main(message=message, total=1) as pbar:
            pbar.update(1)
        with self.progress_bar_secondary(message=message, total=1) as pbar:
            pbar.update(1)

    def _export_weights(self, experiment_info: dict) -> List[str]:
        export_weights = {}
        if (
            self.gui.hyperparameters_selector.get_export_onnx_checkbox_value() is True
            and self._convert_onnx_func is not None
        ):
            try:
                self._set_text_status("export_onnx")
                onnx_path = self._convert_onnx_func(experiment_info)
                if sly_fs.file_exists(onnx_path):
                    export_weights[RuntimeType.ONNXRUNTIME] = onnx_path
            except Exception as e:
                logger.error(f"Failed to export ONNX model: {e}")

        if (
            self.gui.hyperparameters_selector.get_export_tensorrt_checkbox_value() is True
            and self._convert_tensorrt_func is not None
        ):
            try:
                self._set_text_status("export_trt")
                tensorrt_path = self._convert_tensorrt_func(experiment_info)
                if sly_fs.file_exists(tensorrt_path):
                    export_weights[RuntimeType.TENSORRT] = tensorrt_path
            except Exception as e:
                logger.error(f"Failed to export TensorRT model: {e}")

        export_classes_path = None
        if len(export_weights) > 0:
            export_classes = {idx: cls_name for idx, cls_name in enumerate(self.classes)}
            export_dir = join(self.output_dir, self._export_dir_name)
            sly_fs.mkdir(export_dir)
            export_classes_path = join(export_dir, "classes.json")
            sly_json.dump_json_file(export_classes, export_classes_path)
        return export_weights, export_classes_path

    def _upload_export_weights(
        self, export_weights: Dict[str, str], export_classes_path: str, remote_dir: str
    ) -> Dict[str, str]:
        """Uploads export weights (any other specified formats) to Supervisely Team Files.
        The default export is handled by the `_upload_artifacts` method."""
        file_dest_paths = []
        size = 0
        files_to_upload = list(export_weights.values())
        if export_classes_path is not None:
            files_to_upload.append(export_classes_path)
        for path in files_to_upload:
            file_name = sly_fs.get_file_name_with_ext(path)
            file_dest_paths.append(join(remote_dir, self._export_dir_name, file_name))
            size += sly_fs.get_file_size(path)
        with self.progress_bar_main(
            message=f"Uploading {len(export_weights)} export weights",
            total=size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as export_upload_main_pbar:
            logger.debug(f"Uploading {len(export_weights)} export weights of size {size} bytes")
            logger.debug(f"Destination paths: {file_dest_paths}")
            self.progress_bar_main.show()
            self._api.file.upload_bulk_fast(
                team_id=self.team_id,
                src_paths=files_to_upload,
                dst_paths=file_dest_paths,
                progress_cb=export_upload_main_pbar.update,
            )

        self.progress_bar_main.hide()

        remote_export_weights = {
            runtime: join(self._export_dir_name, sly_fs.get_file_name_with_ext(path))
            for runtime, path in export_weights.items()
        }
        return remote_export_weights

    def _convert_and_split_gt_project(self, task_type: str):
        # 1. Convert GT project to cv task
        Project.download(
            self._api,
            self.project_info.id,
            "tmp_project",
            save_images=False,
            save_image_info=True,
        )
        project = Project("tmp_project", OpenMode.READ)
        project.remove_classes_except(project.directory, self.classes, True)

        pr_prefix = ""
        if task_type == TaskType.OBJECT_DETECTION:
            Project.to_detection_task(project.directory, inplace=True)
            pr_prefix = "[detection]: "
        elif task_type == TaskType.INSTANCE_SEGMENTATION:
            Project.to_segmentation_task(
                project.directory, segmentation_type="instance", inplace=True
            )
            pr_prefix = "[instance segmentation]: "
        elif task_type == TaskType.SEMANTIC_SEGMENTATION:
            Project.to_segmentation_task(
                project.directory, segmentation_type="semantic", inplace=True
            )
            pr_prefix = "[semantic segmentation]: "

        gt_project_info = self._api.project.create(
            self.workspace_id,
            f"{pr_prefix}{self.project_info.name}",
            description=(
                f"Converted ground truth project for trainig session: '{self.task_id}'. "
                f"Original project id: '{self.project_info.id}. "
                "Removing this project will affect model benchmark evaluation report."
            ),
            change_name_if_conflict=True,
        )

        # 3. Upload converted gt project
        project = Project("tmp_project", OpenMode.READ)
        self._api.project.update_meta(gt_project_info.id, project.meta)
        for dataset in project.datasets:
            dataset: Dataset
            ds_info = self._api.dataset.create(
                gt_project_info.id, dataset.name, change_name_if_conflict=True
            )
            for batch_names in batched(dataset.get_items_names(), 100):
                img_infos = [dataset.get_item_info(name) for name in batch_names]
                img_ids = [img_info.id for img_info in img_infos]
                anns = [dataset.get_ann(name, project.meta) for name in batch_names]

                img_infos = self._api.image.copy_batch(ds_info.id, img_ids)
                img_ids = [img_info.id for img_info in img_infos]
                self._api.annotation.upload_anns(img_ids, anns)
        sly_fs.remove_dir(project.directory)

        # 4. Match splits with original project
        gt_split_data = self._postprocess_splits(gt_project_info.id)
        return gt_project_info.id, gt_split_data

    def _create_collection_splits(self):
        def _check_match(current_selected_collection_ids: List[int], all_split_collections: List[EntitiesCollectionInfo]):
            if len(current_selected_collection_ids) > 0:
                if len(current_selected_collection_ids) == 1:
                    current_selected_collection_id = current_selected_collection_ids[0]
                    for collection in all_split_collections:
                        if collection.id == current_selected_collection_id:
                            return True
            return False

        # Case 1: Use existing collections for training. No need to create new collections
        split_method = self.gui.train_val_splits_selector.get_split_method()
        all_train_collections = self.gui.train_val_splits_selector.all_train_collections
        all_val_collections = self.gui.train_val_splits_selector.all_val_collections
        if split_method == "Based on collections":
            current_selected_train_collection_ids = self.gui.train_val_splits_selector.train_val_splits.get_train_collections_ids()
            train_match = _check_match(current_selected_train_collection_ids, all_train_collections)
            if train_match:
                current_selected_val_collection_ids = self.gui.train_val_splits_selector.train_val_splits.get_val_collections_ids()
                val_match = _check_match(current_selected_val_collection_ids, all_val_collections)
                if val_match:
                    self._train_collection_id = current_selected_train_collection_ids[0]
                    self._val_collection_id = current_selected_val_collection_ids[0]
                    self._update_project_custom_data(self._train_collection_id, self._val_collection_id)
                    return
        # ------------------------------------------------------------ #

        # Case 2: Create new collections for selected train val splits. Need to create new collections
        item_type = self.project_info.type
        experiment_name = self.gui.training_process.get_experiment_name()

        train_collection_idx = 1
        val_collection_idx = 1

        def _extract_index_from_col_name(name: str, expected_prefix: str) -> Optional[int]:
            parts = name.split("_")
            if len(parts) == 2 and parts[0] == expected_prefix and parts[1].isdigit():
                return int(parts[1])
            return None

        # Get train collection with max idx
        if len(all_train_collections) > 0:
            train_indices = [_extract_index_from_col_name(collection.name, "train") for collection in all_train_collections]
            train_indices = [idx for idx in train_indices if idx is not None]
            if len(train_indices) > 0:
                train_collection_idx = max(train_indices) + 1

        # Get val collection with max idx
        if len(all_val_collections) > 0:
            val_indices = [_extract_index_from_col_name(collection.name, "val") for collection in all_val_collections]
            val_indices = [idx for idx in val_indices if idx is not None]
            if len(val_indices) > 0:
                val_collection_idx = max(val_indices) + 1
        # -------------------------------- #

        # Create Train Collection
        train_img_ids = list(self._train_split_item_ids)
        train_collection_description = f"Collection with train {item_type} for experiment: {experiment_name}"
        train_collection = self._api.entities_collection.create(self.project_id, f"train_{train_collection_idx:03d}", train_collection_description)
        train_collection_id = getattr(train_collection, "id", None)
        if train_collection_id is None:
            raise AttributeError("Train EntitiesCollectionInfo object does not have 'id' attribute")
        self._api.entities_collection.add_items(train_collection_id, train_img_ids)
        self._train_collection_id = train_collection_id

        # Create Val Collection
        val_img_ids = list(self._val_split_item_ids)
        val_collection_description = f"Collection with val {item_type} for experiment: {experiment_name}"
        val_collection = self._api.entities_collection.create(self.project_id, f"val_{val_collection_idx:03d}", val_collection_description)
        val_collection_id = getattr(val_collection, "id", None)
        if val_collection_id is None:
            raise AttributeError("Val EntitiesCollectionInfo object does not have 'id' attribute")
        self._api.entities_collection.add_items(val_collection_id, val_img_ids)
        self._val_collection_id = val_collection_id

        # Update Project Custom Data
        self._update_project_custom_data(train_collection_id, val_collection_id)

    def _update_project_custom_data(self, train_collection_id: int, val_collection_id: int):
        train_info = {
            "task_id": self.task_id,
            "framework_name": self.framework_name,
            "splits": {"train_collection": train_collection_id, "val_collection": val_collection_id}
        }
        custom_data = self._api.project.get_info_by_id(self.project_id).custom_data
        train_info_list = custom_data.get("train_info", [])
        train_info_list.append(train_info)
        custom_data.update({"train_info": train_info_list})
        self._api.project.update_custom_data(self.project_id, custom_data)

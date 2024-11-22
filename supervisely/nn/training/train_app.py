"""
TrainApp module.

This module contains the `TrainApp` class and related functionality to facilitate
training workflows in a Supervisely application.
"""

import shutil
import time
from datetime import datetime
from os import listdir
from os.path import basename, isdir, isfile, join
from typing import Any, Dict, List, Optional
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
    DatasetInfo,
    OpenMode,
    Project,
    ProjectInfo,
    WorkflowMeta,
    WorkflowSettings,
    download_project,
    is_development,
    is_production,
    logger,
)
from supervisely.api.file_api import FileInfo
from supervisely.app.widgets import Progress
from supervisely.nn.benchmark import (
    InstanceSegmentationBenchmark,
    ObjectDetectionBenchmark,
)
from supervisely.nn.inference import RuntimeType, SessionJSON
from supervisely.nn.task_type import TaskType
from supervisely.nn.training.gui.gui import TrainGUI
from supervisely.nn.training.loggers.tensorboard_logger import tb_logger
from supervisely.nn.utils import ModelSource
from supervisely.output import set_directory
from supervisely.project.download import (
    copy_from_cache,
    download_to_cache,
    get_cache_size,
    is_cached,
)


class StopTrainingException(Exception):
    def __init__(self, message: str):
        super().__init__(message)


class TrainApp:
    """
    A class representing the training application.

    This class initializes and manages the training workflow, including
    handling inputs, hyperparameters, project management, and output artifacts.

    :param framework_name: Name of the ML framework used.
    :type framework_name: str
    :param models: List of model configurations.
    :type models: List[Dict[str, Any]]
    :param hyperparameters: Path or string content of hyperparameters in YAML format.
    :type hyperparameters: str
    :param app_options: Options for the application layout and behavior.
    :type app_options: Optional[Dict[str, Any]]
    :param work_dir: Path to the working directory for storing intermediate files.
    :type work_dir: Optional[str]
    """

    def __init__(
        self,
        framework_name: str,
        models: List[Dict[str, Any]],
        hyperparameters: str,
        app_options: Dict[str, Any] = None,
        work_dir: str = None,
    ):

        # Init
        self._api = Api.from_env()

        # Constants
        self._experiment_json_file = "experiment_info.json"
        self._app_state_file = "app_state.json"
        self._train_split_file = "train_split.json"
        self._val_split_file = "val_split.json"

        if is_production():
            self.task_id = sly_env.task_id()
        else:
            self.task_id = "debug-session"
            logger.info("TrainApp is running in debug mode")

        self.framework_name = framework_name
        self._team_id = sly_env.team_id()
        self._workspace_id = sly_env.workspace_id()
        self.app_name = sly_env.app_name()

        self._models = self._validate_models(models)
        self._hyperparameters = self._load_hyperparameters(hyperparameters)
        self._app_options = self._validate_app_options(app_options)
        self._inference_class = None
        # ----------------------------------------- #

        # Input
        self.work_dir = work_dir
        self.output_dir = join(self.work_dir, "result")
        self.output_weights_dir = join(self.output_dir, "weights")
        self.project_dir = join(self.work_dir, "sly_project")
        self.sly_project = None
        self.train_split, self.val_split = None, None
        # ----------------------------------------- #

        # Classes
        # ----------------------------------------- #

        # Model
        self.model_files = {}
        self.model_dir = join(self.work_dir, "model")
        self.log_dir = join(self.work_dir, "logs")
        # ----------------------------------------- #

        # Hyperparameters
        # ----------------------------------------- #

        # Layout
        self.gui: TrainGUI = TrainGUI(
            self.framework_name, self._models, self._hyperparameters, self._app_options
        )
        self.app = Application(layout=self.gui.layout)
        self._server = self.app.get_server()
        self._register_routes()
        self._train_func = None
        # -------------------------- #

        # Tensorboard debug
        # tb_logger.start_tensorboard()
        # self.gui.training_process.tensorboard_button.enable()

        # Train endpoints
        @self._server.post("/train_from_api")
        def _deploy_from_api(response: Response, request: Request):
            try:
                state = request.state.state
                app_config = state["app_config"]
                self.gui.load_from_config(app_config)

                self._wrapped_start_training()

                return {"result": "model was successfully trained"}
            except Exception as e:
                self.gui.training_process.start_button.loading = False
                raise e

    def _register_routes(self):
        """
        Registers API routes for TensorBoard and training endpoints.

        These routes enable communication with the application for training
        and visualizing logs in TensorBoard.
        """
        client = httpx.AsyncClient(base_url="http://127.0.0.1:8001/")

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
        sly_fs.mkdir(self.output_weights_dir, True)
        sly_fs.mkdir(self.project_dir, True)
        sly_fs.mkdir(self.model_dir, True)
        sly_fs.mkdir(self.log_dir, True)

    # Properties
    # General
    # ----------------------------------------- #

    # Input Data
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
    def device(self) -> str:
        """
        Returns the selected device for training.

        :return: Device name.
        :rtype: str
        """
        return self.gui.training_process.get_device()

    # Classes
    @property
    def classes(self) -> List[str]:
        """
        Returns the selected classes for training.

        :return: List of selected classes.
        :rtype: List[str]
        """
        return self.gui.classes_selector.get_selected_classes()

    @property
    def num_classes(self) -> int:
        """
        Returns the number of selected classes for training.

        :return: Number of selected classes.
        :rtype: int
        """
        return len(self.gui.classes_selector.get_selected_classes())

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
        return self.gui.training_process.progress_bar_main

    @property
    def progress_bar_secondary(self) -> Progress:
        """
        Returns the secondary progress bar widget.

        :return: Secondary progress bar widget.
        :rtype: Progress
        """
        return self.gui.training_process.progress_bar_secondary

    # Output
    # ----------------------------------------- #

    # region TRAIN START
    @property
    def start(self):
        """
        Decorator for the training function defined by user.
        It wraps user-defined training function and prepares and finalizes the training process.
        """

        def decorator(func):
            self._train_func = func
            self.gui.training_process.start_button.click(self._wrapped_start_training)
            return func

        return decorator

    def _prepare(self) -> None:
        """
        Prepares the environment for training by setting up directories,
        downloading project and model data.
        """
        logger.info("Preparing for training")
        self.gui.disable_select_buttons()
        self._process_optional_widgets(self._app_options)

        # Step 1. Workflow Input
        if is_production():
            self._workflow_input()
        # Step 2. Download Project
        self._download_project()
        # Step 3. Split Project
        self._split_project()
        # Step 4. Convert Supervisely to X format
        # Step 5. Download Model files
        self._download_model()

    def _finalize(self, experiment_info: dict) -> None:
        """
        Finalizes the training process by validating outputs, uploading artifacts,
        and updating the UI.

        :param experiment_info: Information about the experiment results that should be returned in user's training function.
        :type experiment_info: dict
        """
        logger.info("Finalizing training")

        # Step 1. Validate experiment_info
        success, reason = self._validate_experiment_info(experiment_info)
        if not success:
            raise ValueError(f"{reason}. Failed to upload artifacts")

        # Step 2. Preprocess artifacts
        self._preprocess_artifacts(experiment_info)

        # Step3. Postprocess splits
        splits_data = self._postprocess_splits()

        # Step 3. Upload artifacts
        remote_dir, file_info = self._upload_artifacts()

        # Step 4. Run Model Benchmark
        mb_eval_report, mb_eval_report_id = None, None
        if self.gui.hyperparameters_selector.get_model_benchmark_checkbox_value() is True:
            if is_production():
                try:
                    mb_eval_report, mb_eval_report_id = self._run_model_benchmark(
                        self.output_dir, remote_dir, experiment_info, splits_data
                    )
                except Exception as e:
                    logger.error(f"Model benchmark failed: {e}")

        # Step 4. Generate and upload additional files
        self._generate_experiment_info(remote_dir, experiment_info, splits_data, mb_eval_report_id)
        self._generate_app_state(remote_dir, experiment_info)
        self._generate_train_val_splits(remote_dir)

        # Step 5. Set output widgets
        self._set_training_output(remote_dir, file_info)

        # Step 6. Workflow output
        if is_production():
            self._workflow_output(remote_dir, file_info, mb_eval_report)
        # Step 7. Shutdown app
        self.app.shutdown()

        # region TRAIN END

    def register_inference_class(self, inference_class: Any, inference_settings: dict = {}) -> None:
        """
        Registers an inference class for the training application to do model benchmarking.

        :param inference_class: Inference class to be registered inherited from `supervisely.nn.inference.Inference`.
        :type inference_class: Any
        :param inference_settings: Settings for the inference class.
        :type inference_settings: dict
        """
        self._inference_class = inference_class
        self._inference_settings = inference_settings

    # Loaders
    def _validate_models(self, models: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Validates the models parameter to ensure it is in the correct format.
        """
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
                    "Please update provided models oarameter to include key 'model_files' in 'meta' key."
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

    def _validate_app_options(self, app_options: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validates the app_options parameter to ensure it is in the correct format.
        """
        if not isinstance(app_options, dict):
            raise ValueError("app_options must be a dict")
        # @TODO: Validate app_options
        return app_options

    # ----------------------------------------- #

    # Preprocess
    # Download Project
    def _download_project(self) -> None:
        """
        Downloads the project data from Supervisely.
        If the cache is enabled, it will attempt to retrieve the project from the cache.
        """
        dataset_infos = [dataset for _, dataset in self._api.dataset.tree(self.project_id)]

        if self.gui.train_val_splits_selector.get_split_method() == "Based on datasets":
            selected_ds_ids = (
                self.gui.train_val_splits_selector.get_train_dataset_ids()
                + self.gui.train_val_splits_selector.get_val_dataset_ids()
            )
            dataset_infos = [ds_info for ds_info in dataset_infos if ds_info.id in selected_ds_ids]

        total_images = sum(ds_info.images_count for ds_info in dataset_infos)
        if not self.gui.input_selector.get_cache_value() or is_development():
            self._download_no_cache(dataset_infos, total_images)
            self.sly_project = Project(self.project_dir, OpenMode.READ)
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
            self.sly_project = Project(self.project_dir, OpenMode.READ)
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
            self.progress_bar_main.show()
            download_project(
                api=self._api,
                project_id=self.project_id,
                dest_dir=self.project_dir,
                dataset_ids=[ds_info.id for ds_info in dataset_infos],
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
        to_download = [
            info for info in dataset_infos if not is_cached(self.project_info.id, info.name)
        ]
        cached = [info for info in dataset_infos if is_cached(self.project_info.id, info.name)]

        logger.info(self._get_cache_log_message(cached, to_download))
        with self.progress_bar_main(message="Downloading input data", total=total_images) as pbar:
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
                dataset_names=[ds_info.name for ds_info in dataset_infos],
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
        # Load splits
        self.gui.train_val_splits_selector.set_sly_project(self.sly_project)
        self.train_split, self.val_split = (
            self.gui.train_val_splits_selector.train_val_splits.get_splits()
        )

        # Prepare paths
        project_split_path = join(self.work_dir, "splits")
        paths = {
            "train": {
                "split_path": join(project_split_path, "train"),
                "img_dir": join(project_split_path, "train", "img"),
                "ann_dir": join(project_split_path, "train", "ann"),
            },
            "val": {
                "split_path": join(project_split_path, "val"),
                "img_dir": join(project_split_path, "val", "img"),
                "ann_dir": join(project_split_path, "val", "ann"),
            },
        }

        # Create necessary directories (only once)
        for dataset_paths in paths.values():
            for path in dataset_paths.values():
                sly_fs.mkdir(path, True)

        # Format for image names
        items_count = max(len(self.train_split), len(self.val_split))
        num_digits = len(str(items_count))
        image_name_formats = {
            "train": f"train_img_{{:0{num_digits}d}}",
            "val": f"val_img_{{:0{num_digits}d}}",
        }

        # Utility function to move files
        def move_files(split, paths, img_name_format, pbar):
            """
            Move files to the appropriate directories.
            """
            for idx, item in enumerate(split, start=1):
                item_name = img_name_format.format(idx) + sly_fs.get_file_ext(item.name)
                ann_name = f"{item_name}.json"
                shutil.copy(item.img_path, join(paths["img_dir"], item_name))
                shutil.copy(item.ann_path, join(paths["ann_dir"], ann_name))
                pbar.update(1)

        # Main split processing
        with self.progress_bar_main(
            message="Applying train/val splits to project", total=2
        ) as main_pbar:
            self.progress_bar_main.show()
            for dataset in ["train", "val"]:
                split = self.train_split if dataset == "train" else self.val_split
                with self.progress_bar_secondary(
                    message=f"Preparing '{dataset}'", total=len(split)
                ) as second_pbar:
                    self.progress_bar_secondary.show()
                    move_files(split, paths[dataset], image_name_formats[dataset], second_pbar)
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
                shutil.move(
                    paths[dataset]["split_path"],
                    train_ds_path if dataset == "train" else val_ds_path,
                )
                pbar.update(1)
            self.progress_bar_main.hide()

        # Clean up temporary directory
        sly_fs.remove_dir(project_split_path)
        self.sly_project = Project(self.project_dir, OpenMode.READ)

    # ----------------------------------------- #

    # ----------------------------------------- #
    # Download Model
    def _download_model(self) -> None:
        """
        Downloads the model data from the selected source.

        For Pretrained models:
            - The files that will be downloaded are specified in the `meta` key under `model_files`.
            - All files listed in the `model_files` key will be downloaded by provided link.
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
                                            "checkpoint": "https://example.com/checkpoint.pth"
                                            "config": "https://example.com/config.yaml"
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

                with urlopen(file_url) as f:
                    weights_size = f.length

                file_path = join(self.model_dir, file)

                with self.progress_bar_secondary(
                    message=f"Downloading '{file}' ",
                    total=weights_size,
                    unit="bytes",
                    unit_scale=True,
                ) as model_download_secondary_pbar:
                    self.progress_bar_secondary.show()
                    sly_fs.download(
                        url=file_url,
                        save_path=file_path,
                        progress=model_download_secondary_pbar.update,
                    )

                model_download_main_pbar.update(1)
                self.model_files[file] = file_path

        self.progress_bar_main.hide()
        self.progress_bar_secondary.hide()

    def _download_custom_model(self):
        """
        Downloads the custom model data.
        """
        # General
        self.model_files = {}

        # Need to merge file_url with arts dir
        model_files = self.model_info["model_files"]
        for file in model_files:
            model_files[file] = join(self.model_info["artifacts_dir"], model_files[file])

        # Add selected checkpoint to model_files
        checkpoint = self.gui.model_selector.custom_models_table.get_selected_checkpoint_path()
        model_files["checkpoint"] = checkpoint

        with self.progress_bar_main(
            message="Downloading model files",
            total=len(model_files),
        ) as model_download_main_pbar:
            self.progress_bar_main.show()
            for file in model_files:
                file_url = model_files[file]

                file_info = self._api.file.get_info_by_path(self._team_id, file_url)
                file_path = join(self.model_dir, file)
                file_size = file_info.sizeb

                with self.progress_bar_secondary(
                    message=f"Downloading {file}",
                    total=file_size,
                    unit="bytes",
                    unit_scale=True,
                ) as model_download_secondary_pbar:
                    self.progress_bar_secondary.show()
                    self._api.file.download(
                        self._team_id,
                        file_url,
                        file_path,
                        progress_cb=model_download_secondary_pbar.update,
                    )
                model_download_main_pbar.update(1)
                self.model_files[file] = file_path

        self.progress_bar_main.hide()
        self.progress_bar_secondary.hide()

    # ----------------------------------------- #

    # Postprocess

    def _validate_experiment_info(self, experiment_info: dict) -> tuple:
        """
        Validates the experiment_info parameter to ensure it is in the correct format.
        experiment_info is returned by the user's training function.

        experiment_info should contain the following keys:
            - experiment_name": str
            - model_name": str
            - task_type": str
            - model_files": dict
            - checkpoints": list
            - best_checkpoint": str

        Other keys are generated by the TrainApp class automatically

        :param experiment_info: Information about the experiment results.
        :type experiment_info: dict
        :return: Tuple of success status and reason for failure.
        :rtype: tuple
        """
        if not isinstance(experiment_info, dict):
            reason = f"Validation failed: 'experiment_info' must be a dictionary not '{type(experiment_info)}'"
            return False, reason

        logger.info("Starting validation of 'experiment_info'")
        required_keys = {
            "model_name": str,
            "task_type": str,
            "model_files": dict,
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

        if not sly_fs.file_exists(experiment_info["best_checkpoint"]):
            reason = f"Validation failed: Best checkpoint file: '{experiment_info['best_checkpoint']}' does not exist"
            return False, reason

        logger.info("Validation successful")
        return True, None

    def _postprocess_splits(self) -> dict:
        """
        Processes the train and val splits to generate the necessary data for the experiment_info.json file.
        """
        val_dataset_ids = None
        val_images_ids = None
        train_dataset_ids = None
        train_images_ids = None

        split_method = self.gui.train_val_splits_selector.get_split_method()
        train_set, val_set = self.train_split, self.val_split
        if split_method == "Based on datasets":
            val_dataset_ids = self.gui.train_val_splits_selector.get_val_dataset_ids()
            train_dataset_ids = self.gui.train_val_splits_selector.get_train_dataset_ids
        else:
            dataset_infos = [dataset for _, dataset in self._api.dataset.tree(self.project_id)]
            ds_infos_dict = {}
            for dataset in dataset_infos:
                if dataset.parent_id is not None:
                    parent_ds = self._api.dataset.get_info_by_id(dataset.parent_id)
                    dataset_name = f"{parent_ds.name}/{dataset.name}"
                else:
                    dataset_name = dataset.name
                ds_infos_dict[dataset_name] = dataset

            def get_image_infos_by_split(ds_infos_dict: dict, split: list):
                image_names_per_dataset = {}
                for item in split:
                    image_names_per_dataset.setdefault(item.dataset_name, []).append(item.name)
                image_infos = []
                for dataset_name, image_names in image_names_per_dataset.items():
                    ds_info = ds_infos_dict[dataset_name]
                    image_infos.extend(
                        self._api.image.get_list(
                            ds_info.id,
                            filters=[
                                {
                                    "field": "name",
                                    "operator": "in",
                                    "value": image_names,
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

    def _preprocess_artifacts(self, experiment_info: dict) -> None:
        """
        Preprocesses and move the artifacts generated by the training process to output directories.

        :param experiment_info: Information about the experiment results.
        :type experiment_info: dict
        """
        logger.info("Preprocessing artifacts")
        if "model_files" not in experiment_info:
            experiment_info["model_files"] = {}
        else:
            # Move model files to output directory except config, config will be processed later
            files = {k: v for k, v in experiment_info["model_files"].items() if k != "config"}
            for file in files:
                if isfile:
                    shutil.move(
                        file,
                        join(self.output_dir, sly_fs.get_file_name_with_ext(file)),
                    )
                elif isdir:
                    shutil.move(file, join(self.output_dir, basename(file)))

        # Prepare or create config
        config = experiment_info["model_files"].get("config")
        if config is None:
            config = "config.yaml"
            experiment_info["model_files"]["config"] = config

        output_config_path = join(
            self.output_dir,
            sly_fs.get_file_name_with_ext(experiment_info["model_files"]["config"]),
        )

        # Add sly_metadata to config
        logger.info("Adding 'sly_metadata' to config file")
        with open(experiment_info["model_files"]["config"], "r") as file:
            custom_config = yaml.safe_load(file)

        custom_config["sly_metadata"] = {
            "classes": self.classes,
            "model": experiment_info["model_name"],
            "project_id": self.project_info.id,
            "project_name": self.project_info.name,
        }

        with open(experiment_info["model_files"]["config"], "w") as f:
            yaml.safe_dump(custom_config, f)
        shutil.move(experiment_info["model_files"]["config"], output_config_path)

        # Prepare checkpoints
        checkpoints = experiment_info["checkpoints"]
        if isinstance(checkpoints, str):
            checkpoint_paths = []
            for checkpoint_path in sly_fs.list_files_recursively(checkpoints, [".pt", ".pth"]):
                checkpoint_paths.append(checkpoint_path)
        else:
            checkpoint_paths = checkpoints

        for checkpoint_path in checkpoint_paths:
            new_checkpoint_path = join(
                self.output_weights_dir, sly_fs.get_file_name_with_ext(checkpoint_path)
            )
            shutil.move(checkpoint_path, new_checkpoint_path)

        # Prepare logs
        if sly_fs.dir_exists(self.log_dir):
            logs_dir = join(self.output_dir, "logs")
            shutil.move(self.log_dir, logs_dir)

    # Generate experiment_info.json and app_state.json
    def _upload_json_file(self, local_path: str, remote_path: str, message: str) -> None:
        """Helper function to upload a JSON file with progress."""
        logger.info(f"Uploading '{local_path}' to Supervisely")
        total_size = sly_fs.get_file_size(local_path)
        with self.progress_bar_main(
            message=message, total=total_size, unit="bytes", unit_scale=True
        ) as upload_artifacts_pbar:
            self.progress_bar_main.show()
            self._api.file.upload(
                self._team_id,
                local_path,
                remote_path,
                progress_cb=upload_artifacts_pbar,
            )
            self.progress_bar_main.hide()

    def _generate_train_val_splits(self, remote_dir: str) -> None:
        """
        Generates and uploads the train and val splits to the output directory.

        :param remote_dir: Remote directory path.
        :type remote_dir: str
        """
        # 1. Process train split
        local_train_split_path = join(self.output_dir, self._train_split_file)
        remote_train_split_path = join(remote_dir, self._train_split_file)

        sly_json.dump_json_file(self.train_split, local_train_split_path)
        self._upload_json_file(
            local_train_split_path,
            remote_train_split_path,
            f"Uploading '{self._train_split_file}' to Team Files",
        )
        # 2. Process val split
        local_val_split_path = join(self.output_dir, self._val_split_file)
        remote_val_split_path = join(remote_dir, self._val_split_file)

        sly_json.dump_json_file(self.val_split, local_val_split_path)
        self._upload_json_file(
            local_val_split_path,
            remote_val_split_path,
            f"Uploading '{self._val_split_file}' to Team Files",
        )

    def _generate_experiment_info(
        self,
        remote_dir: str,
        experiment_info: Dict,
        splits_data: Dict,
        evaluation_report_id: Optional[int] = None,
    ) -> None:
        """
        Generates and uploads the experiment_info.json file to the output directory.

        :param remote_dir: Remote directory path.
        :type remote_dir: str
        :param experiment_info: Information about the experiment results.
        :type experiment_info: dict
        :param splits_data: Information about the train and val splits.
        :type splits_data: dict
        :param evaluation_report_id: Evaluation report file ID.
        :type evaluation_report_id: int
        """
        logger.info("Updating experiment info")
        experiment_info.update(
            {
                "framework_name": self.framework_name,
                "app_state": self._app_state_file,
                "train_val_splits": {
                    "train": {
                        "split": self._train_split_file,
                        "images_ids": splits_data["train"]["images_ids"],
                    },
                    "val": {
                        "split": self._val_split_file,
                        "images_ids": splits_data["val"]["images_ids"],
                    },
                },
                "hyperparameters": self.hyperparameters,
                "artifacts_dir": remote_dir,
                "task_id": self.task_id,
                "project_id": self.project_info.id,
                "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "evaluation_report_id": evaluation_report_id,
                # "eval_metrics": {"mAP": None, "mIoU": None, "f1_conf_threshold": None},
            }
        )

        remote_weights_dir = join(remote_dir, "weights")
        checkpoint_files = self._api.file.list(
            self._team_id, remote_weights_dir, return_type="fileinfo"
        )
        experiment_info["checkpoints"] = [
            f"weights/{checkpoint.name}" for checkpoint in checkpoint_files
        ]

        experiment_info["best_checkpoint"] = sly_fs.get_file_name_with_ext(
            experiment_info["best_checkpoint"]
        )
        experiment_info["model_files"]["config"] = sly_fs.get_file_name_with_ext(
            experiment_info["model_files"]["config"]
        )

        local_path = join(self.output_dir, self._experiment_json_file)
        remote_path = join(remote_dir, self._experiment_json_file)
        sly_json.dump_json_file(experiment_info, local_path)
        self._upload_json_file(
            local_path,
            remote_path,
            f"Uploading '{self._experiment_json_file}' to Team Files",
        )

    def _generate_app_state(self, remote_dir: str, experiment_info: Dict) -> None:
        """
        Generates and uploads the app_state.json file to the output directory.

        :param remote_dir: Remote directory path.
        :type remote_dir: str
        :param experiment_info: Information about the experiment results.
        :type experiment_info: dict
        """
        input_data = {"project_id": self.project_id}
        train_val_splits = self._get_train_val_splits_for_app_state()
        model = self._get_model_config_for_app_state(experiment_info)

        options = {
            "model_benchmark": {
                "enable": self.gui.hyperparameters_selector.get_model_benchmark_checkbox_value(),
                "speed_test": self.gui.hyperparameters_selector.get_speedtest_checkbox_value(),
            },
            "cache_project": self.gui.input_selector.get_cache_value(),
        }

        app_state = {
            "input": input_data,
            "train_val_splits": train_val_splits,
            "classes": self.classes,
            "model": model,
            "hyperparameters": self.hyperparameters_yaml,
            "options": options,
        }

        local_path = join(self.output_dir, self._app_state_file)
        remote_path = join(remote_dir, self._app_state_file)
        sly_json.dump_json_file(app_state, local_path)
        self._upload_json_file(
            local_path, remote_path, f"Uploading '{self._app_state_file}' to Team Files"
        )

    def _get_train_val_splits_for_app_state(self) -> Dict:
        """
        Gets the train and val splits information for app_state.json.

        :return: Train and val splits information based on selected split method.
        :rtype: dict
        """
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
        return train_val_splits

    def _get_model_config_for_app_state(self, experiment_info: Dict) -> Dict:
        """
        Gets the model configuration information for app_state.json.

        :param experiment_info: Information about the experiment results.
        :type experiment_info: dict
        """
        if self.model_source == ModelSource.PRETRAINED:
            return {
                "source": ModelSource.PRETRAINED,
                "model_name": experiment_info["model_name"],
            }
        elif self.model_source == ModelSource.CUSTOM:
            return {
                "source": ModelSource.CUSTOM,
                "task_id": self.task_id,
                "checkpoint": "checkpoint.pth",
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
        logger.info(f"Uploading directory: '{self.output_dir}' to Supervisely")

        experiments_dir = "experiments"
        task_id = self.task_id

        remote_artifacts_dir = f"/{experiments_dir}/{self.project_id}_{self.project_name}/{task_id}_{self.framework_name}/"

        # Clean debug directory if exists
        if task_id == "debug-session":
            if self._api.file.dir_exists(self._team_id, f"{remote_artifacts_dir}/", True):
                with self.progress_bar_main(
                    message=f"[Debug] Cleaning train artifacts: '{remote_artifacts_dir}/'",
                    total=1,
                ) as upload_artifacts_pbar:
                    self.progress_bar_main.show()
                    self._api.file.remove_dir(self._team_id, f"{remote_artifacts_dir}", True)
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
            unit="bytes",
            unit_scale=True,
        ) as upload_artifacts_pbar:
            self.progress_bar_main.show()
            remote_dir = self._api.file.upload_directory(
                self._team_id,
                self.output_dir,
                remote_artifacts_dir,
                progress_size_cb=upload_artifacts_pbar,
            )
            self.progress_bar_main.hide()

        file_info = self._api.file.get_info_by_path(self._team_id, join(remote_dir, "open_app.lnk"))
        return remote_dir, file_info

    def _set_training_output(self, remote_dir: str, file_info: FileInfo) -> None:
        """
        Sets the training output in the GUI.
        """
        logger.info("All training artifacts uploaded successfully")
        self.gui.training_process.start_button.disable()
        self.gui.training_process.stop_button.disable()
        self.gui.training_process.tensorboard_button.disable()

        set_directory(remote_dir)
        self.gui.training_process.artifacts_thumbnail.set(file_info)
        self.gui.training_process.artifacts_thumbnail.show()
        self.gui.training_process.success_message.show()

    def _process_optional_widgets(self, app_options: Dict[str, Any]) -> None:
        """
        Process optional widget settings specified in the app options parameter.
        """
        if app_options.get("enable_device_selector", False):
            self.gui.training_process.select_device.disable()
            self.gui.training_process.select_device.hide()

    # Model Benchmark
    def _get_eval_results_dir_name(self) -> str:
        """
        Returns the evaluation results path.
        """
        task_info = self._api.task.get_info_by_id(self.task_id)
        task_dir = f"{self.task_id}_{task_info['meta']['app']['name']}"
        eval_res_dir = f"/model-benchmark/evaluation/{self.project_info.id}_{self.project_info.name}/{task_dir}/"
        eval_res_dir = self._api.storage.get_free_dir_name(self._team_id(), eval_res_dir)
        return eval_res_dir

    # Hot to pass inference_settings?
    def _run_model_benchmark(
        self,
        local_artifacts_dir: str,
        remote_artifacts_dir: str,
        experiment_info: dict,
        splits_data: dict,
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
        :return: Evaluation report and report ID.
        :rtype: tuple
        """
        report, report_id = None, None
        if self._inference_class is None:
            logger.warn(
                "Inference class is not registered, model benchmark disabled. "
                "Use 'register_inference_class' method to register inference class."
            )
            return report, report_id

        # Can't get task type from session. requires before session init
        supported_task_types = [
            TaskType.OBJECT_DETECTION,
            TaskType.INSTANCE_SEGMENTATION,
        ]
        task_type = experiment_info["task_type"]
        if task_type not in supported_task_types:
            logger.warn(
                f"Task type: '{task_type}' is not supported for Model Benchmark. "
                f"Supported tasks: {', '.join(task_type)}"
            )
            return report, report_id

        logger.info("Running Model Benchmark evaluation")
        try:
            remote_weights_dir = join(remote_artifacts_dir, "weights")
            best_checkpoint = experiment_info.get("best_checkpoint", None)
            best_filename = sly_fs.get_file_name_with_ext(best_checkpoint)
            remote_best_checkpoint = join(remote_weights_dir, best_filename)

            config_path = experiment_info["model_files"].get("config")
            if config_path is not None:
                remote_config_path = join(
                    remote_artifacts_dir, sly_fs.get_file_name_with_ext(config_path)
                )
            else:
                remote_config_path = None

            logger.info(f"Creating the report for the best model: {best_filename!r}")
            self.gui.training_process.model_benchmark_report_text.show()
            self.progress_bar_main(message="Starting Model Benchmark evaluation", total=1)
            self.progress_bar_main.show()

            # 0. Serve trained model
            m = self._inference_class(
                model_dir=self.model_dir,
                use_gui=False,
                custom_inference_settings=self._inference_settings,
            )

            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {device}")

            # Always download checkpoint from tf instead of using local one
            deploy_params = dict(
                device=device,
                runtime=RuntimeType.PYTORCH,
                model_source=ModelSource.CUSTOM,
                task_type=task_type,
                checkpoint_name=best_filename,
                checkpoint_url=remote_best_checkpoint,
                config_url=remote_config_path,  # @TODO: Not always needed
            )
            m._load_model(deploy_params)
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

            if task_type == TaskType.OBJECT_DETECTION:
                bm = ObjectDetectionBenchmark(
                    self._api,
                    self.project_info.id,
                    output_dir=benchmark_dir,
                    gt_dataset_ids=benchmark_dataset_ids,
                    gt_images_ids=benchmark_images_ids,
                    progress=self.progress_bar_main,
                    progress_secondary=self.progress_bar_secondary,
                    classes_whitelist=self.classes,
                )
            elif task_type == TaskType.INSTANCE_SEGMENTATION:
                bm = InstanceSegmentationBenchmark(
                    self._api,
                    self.project_info.id,
                    output_dir=benchmark_dir,
                    gt_dataset_ids=benchmark_dataset_ids,
                    gt_images_ids=benchmark_images_ids,
                    progress=self.progress_bar_main,
                    progress_secondary=self.progress_bar_secondary,
                    classes_whitelist=self.classes,
                )

            train_info = {
                "app_session_id": self.task_id,
                "train_dataset_ids": train_dataset_ids,
                "train_images_ids": train_images_ids,
                "images_count": len(train_images_ids),
            }
            bm.train_info = train_info

            # 2. Run inference
            bm.run_inference(session)

            # 3. Pull results from the server
            gt_project_path, dt_project_path = bm.download_projects(save_images=False)

            # 4. Evaluate
            bm._evaluate(gt_project_path, dt_project_path)

            # 5. Upload evaluation results
            eval_res_dir = self._get_eval_results_dir_name()
            bm.upload_eval_results(eval_res_dir + "/evaluation/")

            # 6. Speed test
            if self.gui.hyperparameters_selector.get_speedtest_checkbox_value() is True:
                bm.run_speedtest(session, self.project_info.id)
                self.progress_bar_secondary.hide()  # @TODO: add progress bar
                bm.upload_speedtest_results(eval_res_dir + "/speedtest/")

            # 7. Prepare visualizations, report and upload
            bm.visualize()
            remote_dir = bm.upload_visualizations(eval_res_dir + "/visualizations/")
            report = bm.upload_report_link(remote_dir)
            report_id = report.id

            # 8. UI updates
            benchmark_report_template = self._api.file.get_info_by_path(
                self._team_id(), remote_dir + "template.vue"
            )

            self.gui.training_process.model_benchmark_report_text.hide()
            self.gui.training_process.model_benchmark_report_thumbnail.set(
                benchmark_report_template
            )
            self.gui.training_process.model_benchmark_report_thumbnail.show()
            self.progress_bar_main.hide()
            self.progress_bar_secondary.hide()
            logger.info("Model benchmark evaluation completed successfully")
            logger.info(
                f"Predictions project name: {bm.dt_project_info.name}. Workspace_id: {bm.dt_project_info.workspace_id}"
            )
            logger.info(
                f"Differences project name: {bm.diff_project_info.name}. Workspace_id: {bm.diff_project_info.workspace_id}"
            )
        except Exception as e:
            logger.error(f"Model benchmark failed. {repr(e)}", exc_info=True)
            self.gui.training_process.model_benchmark_report_text.hide()
            self.progress_bar_main.hide()
            self.progress_bar_secondary.hide()
            try:
                if bm.dt_project_info:
                    self._api.project.remove(bm.dt_project_info.id)
                if bm.diff_project_info:
                    self._api.project.remove(bm.diff_project_info.id)
            except Exception as e2:
                return report, report_id
        return report, report_id

    # ----------------------------------------- #

    # Workflow
    def _workflow_input(self):
        """
        Adds the input data to the workflow.
        """
        try:
            project_version_id = self._api.project.version.create(
                self.project_info,
                self.app_name,
                f"This backup was created automatically by Supervisely before the {self.app_name} task with ID: {self._api.task_id}",
            )
        except Exception as e:
            logger.warning(f"Failed to create a project version: {repr(e)}")
            project_version_id = None

        try:
            if project_version_id is None:
                project_version_id = (
                    self.project_info.version.get("id", None) if self.project_info.version else None
                )
            self._api.app.workflow.add_input_project(
                self.project_info.id, version_id=project_version_id
            )

            if self.model_source == ModelSource.CUSTOM:
                file_info = self._api.file.get_info_by_path(
                    self._team_id,
                    self.gui.model_selector.custom_models_table.get_selected_checkpoint_path(),
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
    ):
        """
        Adds the output data to the workflow.
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
                title=self.app_name,
                url=(
                    f"/apps/{module_id}/sessions/{self._api.task_id}"
                    if module_id
                    else f"apps/sessions/{self._api.task_id}"
                ),
                url_title="Show Results",
            )

            if file_info:
                relation_settings = WorkflowSettings(
                    title="Train Artifacts",
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
                self._api.app.workflow.add_output_file(file_info, model_weight=True, meta=meta)
            else:
                logger.debug(
                    f"File with checkpoints not found in Team Files. Cannot set workflow output."
                )

            if model_benchmark_report:
                mb_relation_settings = WorkflowSettings(
                    title="Model Benchmark",
                    icon="assignment",
                    icon_color="#674EA7",
                    icon_bg_color="#CCCCFF",
                    url=f"/model-benchmark?id={model_benchmark_report.id}",
                    url_title="Open Report",
                )

                meta = WorkflowMeta(
                    relation_settings=mb_relation_settings, node_settings=node_settings
                )
                self._api.app.workflow.add_output_file(model_benchmark_report, meta=meta)
            else:
                logger.debug(
                    f"File with model benchmark report not found in Team Files. Cannot set workflow output."
                )
        except Exception as e:
            logger.debug(f"Failed to add output to the workflow: {repr(e)}")
        # ----------------------------------------- #

    # Logger
    def _init_logger(self):
        """
        Initialize training logger. Set up Tensorboard and callbacks.
        """
        tb_logger.set_log_dir(self.log_dir)
        tb_logger.start_tensorboard()
        self._setup_logger_callbacks()
        time.sleep(1)
        self.gui.training_process.tensorboard_button.enable()

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

            # @TODO: access tensorboard after training
            tb_logger.writer.close()
            tb_logger.stop_tensorboard()

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

        tb_logger.add_on_train_started_callback(start_training_callback)
        tb_logger.add_on_train_finish_callback(finish_training_callback)

        tb_logger.add_on_epoch_started_callback(start_epoch_callback)
        tb_logger.add_on_epoch_finish_callback(finish_epoch_callback)

        tb_logger.add_on_step_callback(step_callback)

    # ----------------------------------------- #
    def _wrapped_start_training(self):
        """
        Wrapper function to wrap the training process.
        """
        self.gui.training_process.start_button.loading = True

        if self._train_func is None:
            raise ValueError("Train function is not defined")

        self._prepare_working_dir()
        self._init_logger()
        experiment_info = None
        self._prepare()
        experiment_info = self._train_func()
        self._finalize(experiment_info)
        self.gui.training_process.start_button.loading = False

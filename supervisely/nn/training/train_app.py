import shutil
import time
from datetime import datetime
from os import listdir
from os.path import basename, dirname, isdir, isfile, join
from typing import Any, Dict, List, Optional, Union
from urllib.request import urlopen

import yaml
from fastapi import Request, Response

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
    WorkflowMeta,
    WorkflowSettings,
    download_project,
    is_development,
    is_production,
    logger,
)
from supervisely.api.file_api import FileInfo
from supervisely.app.widgets import Button, FolderThumbnail, Progress
from supervisely.nn.benchmark import (
    InstanceSegmentationBenchmark,
    ObjectDetectionBenchmark,
)
from supervisely.nn.inference import RuntimeType, SessionJSON
from supervisely.nn.task_type import TaskType
from supervisely.nn.training.gui.gui import TrainGUI
from supervisely.nn.training.train_logger import train_logger
from supervisely.nn.training.utils import load_file, validate_list_of_dicts
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

    def __init__(
        self,
        framework_name: str,
        models: Union[str, List[Dict[str, Any]]],
        hyperparameters: Union[str, List[Dict[str, Any]]],
        app_options: Union[str, List[Dict[str, Any]]] = None,
        work_dir: str = None,
    ):

        # Init
        self._api = Api.from_env()

        if is_production():
            self._task_id = sly_env.task_id()
        else:
            self._task_id = "debug-session"
            logger.info("TrainApp is running in debug mode")

        supported_frameworks = [
            "yolov5",
            "yolov5 2.0",
            "yolov8",
            "unet",
            "hrda",
            "ritm",
            "rt-detr",
            "mmdetection",
            "mmdetection 3.0",
            "mmsegmentation",
            "mmclassification",
            "detectron2",
        ]

        if framework_name not in supported_frameworks:
            logger.info(
                f"Framework: '{framework_name}' is not supported. Supported frameworks: {', '.join(supported_frameworks)}"
            )

        self._framework_name = framework_name
        self._team_id = sly_env.team_id()
        self._workspace_id = sly_env.workspace_id()
        self._app_name = sly_env.app_name()

        self._models = self._load_models(models)
        self._hyperparameters = self._load_hyperparameters(hyperparameters)
        self._app_options = self._load_app_options(app_options)

        self._inference_class = None
        # ----------------------------------------- #

        # Input
        self._work_dir = work_dir
        self._output_dir = join(self.work_dir, "result")
        self._project_dir = join(self._work_dir, "sly_project")
        self._project_meta_path = join(self._project_dir, "meta.json")  # No need?
        self._sly_project = None
        self._train_split, self._val_split = None, None
        # ----------------------------------------- #

        # Classes
        # ----------------------------------------- #

        # Model
        self._model_dir = join(self._work_dir, "model")
        self._model_name = None
        self._model_files = {}
        self._log_dir = join(self._work_dir, "logs")
        # ----------------------------------------- #

        # Hyperparameters
        # ----------------------------------------- #

        # Layout
        self._gui: TrainGUI = TrainGUI(
            self._framework_name, self._models, self._hyperparameters, self._app_options
        )
        self._app = Application(layout=self._gui.layout)
        self._server = self._app.get_server()
        self._train_func = None
        # -------------------------- #

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

    # General
    @property
    def app(self) -> Application:
        return self._app

    @property
    def gui(self) -> TrainGUI:
        return self._gui

    @property
    def app_name(self) -> str:
        return self._app_name

    @property
    def task_id(self) -> str:
        return self._task_id

    @property
    def work_dir(self) -> str:
        return self._work_dir

    @property
    def framework_name(self) -> str:
        return self._framework_name

    # ----------------------------------------- #

    # Input Data
    @property
    def project_id(self) -> int:
        return self._gui.project_id

    @property
    def project_name(self) -> str:
        return self._gui.project_info.name

    @property
    def project_info(self) -> ProjectInfo:
        return self._gui.project_info

    @property
    def sly_project(self):
        return self._sly_project

    @property
    def train_split(self):
        return self._train_split

    @property
    def val_split(self):
        return self._val_split

    # ----------------------------------------- #

    # Model
    @property
    def model_source(self) -> str:
        return self._gui.model_selector.get_model_source()

    @property
    def model_name(self) -> str:
        return self._gui.model_selector.get_model_name()

    @property
    def model_info(self) -> str:
        return self._gui.model_selector.get_model_info()

    @property
    def model_files(self) -> str:
        return self._model_files

    @property
    def log_dir(self) -> str:
        return self._log_dir

    # Classes
    @property
    def classes(self) -> List[str]:
        return self._gui.classes_selector.get_selected_classes()

    @property
    def num_classes(self) -> List[str]:
        return len(self._gui.classes_selector.get_selected_classes())

    # Hyperparameters
    @property
    def hyperparameters_json(self) -> Dict[str, Any]:
        return yaml.safe_load(self._gui.hyperparameters_selector.get_hyperparameters())

    @property
    def hyperparameters(self) -> str:
        return self._gui.hyperparameters_selector.get_hyperparameters()

    @property
    def use_model_benchmark(self) -> bool:
        return self._gui.hyperparameters_selector.get_model_benchmark_checkbox_value()

    @property
    def use_model_benchmark_speedtest(self) -> bool:
        return self._gui.hyperparameters_selector.get_speedtest_checkbox_value()

    # Train Process
    @property
    def progress_bar_download_project_main(self) -> Progress:
        return self._gui.training_process.project_download_progress_main

    @property
    def progress_bar_download_project_secondary(self) -> Progress:
        return self._gui.training_process.project_download_progress_secondary

    @property
    def progress_bar_download_model_main(self) -> Progress:
        return self._gui.training_process.model_download_progress_main

    @property
    def progress_bar_download_model_secondary(self) -> Progress:
        return self._gui.training_process.model_download_progress_secondary

    @property
    def progress_bar_epochs(self) -> Progress:
        return self._gui.training_process.epoch_progress

    @property
    def progress_bar_iters(self) -> Progress:
        return self._gui.training_process.iter_progress

    @property
    def progress_bar_upload_artifacts(self) -> Progress:
        return self._gui.training_process.artifacts_upload_progress

    # Output
    @property
    def artifacts_thumbnail(self) -> FolderThumbnail:
        return self._gui.training_process.artifacts_thumbnail

    @property
    def tensorboard_link(self) -> str:
        return self._gui.training_process.tensorboard_link

    # region TRAIN START
    @property
    def start(self):
        sly_fs.mkdir(self.work_dir, True)
        sly_fs.mkdir(self._output_dir, True)

        def decorator(func):
            self._train_func = func
            self._gui.training_process.start_button.click(self._wrapped_start_training)
            return func

        return decorator

    def preprocess(self):
        self.gui.disable_select_buttons()
        logger.info("Preprocessing")
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

    def postprocess(self, experiment_info: dict):
        logger.info("Postprocessing")

        # Step 1. Validate experiment_info
        success = self._validate_experiment_info(experiment_info)
        if not success:
            raise ValueError("Experiment info is not valid. Failed to upload artifacts")

        # Step 2. Preprocess artifacts
        output_dir = self._preprocess_artifacts(experiment_info)

        # Step3. Postprocess splits
        splits_data = self._postprocess_splits()

        # Step 3. Upload artifacts
        remote_dir, file_info = self._upload_artifacts(output_dir, experiment_info)

        # Step 4. Run Model Benchmark
        mb_eval_report, mb_eval_report_id = None, None
        if self.use_model_benchmark is True:
            if is_production() and self.use_model_benchmark is True:
                try:
                    mb_eval_report, mb_eval_report_id = self._run_model_benchmark(
                        output_dir, remote_dir, experiment_info, splits_data
                    )
                except Exception as e:
                    logger.error(f"Model benchmark failed: {e}")

        # Step 4. Generate and upload additional files
        self._generate_experiment_info(
            output_dir, remote_dir, experiment_info, splits_data, mb_eval_report_id
        )
        self._generate_app_state(output_dir, remote_dir, experiment_info)
        self._generate_train_val_splits(output_dir, remote_dir, splits_data)

        # Step 5. Disable widgets?
        self.gui.training_process.start_button.disable()
        self.gui.training_process.stop_button.disable()
        self.gui.training_process.tensorboard_button.disable()

        # Step 6. Workflow output
        if is_production():
            self._workflow_output(remote_dir, file_info, mb_eval_report)
        # Step 7. Shutdown app
        self._app.shutdown()

        # region TRAIN END

    def register_inference_class(self, inference_class: Any, inference_settings: dict = {}) -> None:
        self._inference_class = inference_class
        self._inference_settings = inference_settings

    # Loaders
    def _load_models(self, models: Union[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        if isinstance(models, str):
            models = load_file(models)
        return validate_list_of_dicts(models, "models")

    def _load_hyperparameters(self, hyperparameters: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        if isinstance(hyperparameters, str):
            hyperparameters = load_file(hyperparameters)
        if not isinstance(hyperparameters, (dict, str)):
            raise ValueError(
                "hyperparameters must be a dict, or a path to a '.json' or '.yaml' file."
            )
        return hyperparameters

    def _load_app_options(self, app_options: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        if isinstance(app_options, str):
            app_options = load_file(app_options)
        if not isinstance(app_options, dict):
            raise ValueError("app_options must be a dict, or a path to a '.json' or '.yaml' file.")
        return app_options

    # ----------------------------------------- #

    # Preprocess
    # Download Project
    def _download_project(self) -> None:
        sly_fs.mkdir(self._project_dir, True)

        dataset_infos = [dataset for _, dataset in self._api.dataset.tree(self.project_id)]

        if self.gui.train_val_splits_selector.get_split_method() == "Based on datasets":
            selected_ds_ids = (
                self.gui.train_val_splits_selector.get_train_dataset_ids()
                + self.gui.train_val_splits_selector.get_val_dataset_ids()
            )
            dataset_infos = [ds_info for ds_info in dataset_infos if ds_info.id in selected_ds_ids]

        total_images = sum(ds_info.images_count for ds_info in dataset_infos)
        if not self._gui.input_selector.get_cache_value() or is_development():
            self._download_no_cache(dataset_infos, total_images)
            self._sly_project = Project(self._project_dir, OpenMode.READ)
            return

        try:
            self._download_with_cache(dataset_infos, total_images)
        except Exception:
            logger.warning(
                "Failed to retrieve project from cache. Downloading it",
                exc_info=True,
            )
            if sly_fs.dir_exists(self._project_dir):
                sly_fs.clean_dir(self._project_dir)
            self._download_no_cache(dataset_infos, total_images)
        finally:
            self._sly_project = Project(self._project_dir, OpenMode.READ)
            logger.info(f"Project downloaded successfully to: '{self._project_dir}'")

    def _download_no_cache(self, dataset_infos: List[DatasetInfo], total_images: int) -> None:
        self.progress_bar_download_project_main.show()
        with self.progress_bar_download_project_main(
            message="Downloading input data", total=total_images
        ) as pbar:
            download_project(
                api=self._api,
                project_id=self.project_id,
                dest_dir=self._project_dir,
                dataset_ids=[ds_info.id for ds_info in dataset_infos],
                log_progress=True,
                progress_cb=pbar.update,
            )
        self.progress_bar_download_project_main.hide()

    def _download_with_cache(
        self,
        dataset_infos: List[DatasetInfo],
        total_images: int,
    ) -> None:
        to_download = [
            info for info in dataset_infos if not is_cached(self.project_info.id, info.name)
        ]
        cached = [info for info in dataset_infos if is_cached(self.project_info.id, info.name)]

        logger.info(self._get_cache_log_message(cached, to_download))
        with self.progress_bar_download_project_main(
            message="Downloading input data", total=total_images
        ) as pbar:
            self.progress_bar_download_project_main.show()
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
        with self.progress_bar_download_project_main(
            message="Retrieving data from cache",
            total=total_cache_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            copy_from_cache(
                project_id=self.project_info.id,
                dest_dir=self._project_dir,
                dataset_names=[ds_info.name for ds_info in dataset_infos],
                progress_cb=pbar.update,
            )
        self.progress_bar_download_project_main.hide()

    def _get_cache_log_message(self, cached: bool, to_download: List[DatasetInfo]) -> str:
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
        # Load splits
        self.gui.train_val_splits_selector.set_sly_project(self.sly_project)
        self._train_split, self._val_split = (
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
        items_count = max(len(self._train_split), len(self._val_split))
        num_digits = len(str(items_count))
        image_name_formats = {
            "train": f"train_img_{{:0{num_digits}d}}",
            "val": f"val_img_{{:0{num_digits}d}}",
        }

        # Utility function to move files
        def move_files(split, paths, img_name_format, pbar):
            for idx, item in enumerate(split, start=1):
                item_name = img_name_format.format(idx) + sly_fs.get_file_ext(item.name)
                ann_name = f"{item_name}.json"
                shutil.copy(item.img_path, join(paths["img_dir"], item_name))
                shutil.copy(item.ann_path, join(paths["ann_dir"], ann_name))
                pbar.update(1)

        # Main split processing
        with self.progress_bar_download_project_main(
            message="Applying train/val splits to project", total=2
        ) as main_pbar:
            self.progress_bar_download_project_main.show()
            for dataset in ["train", "val"]:
                split = self._train_split if dataset == "train" else self._val_split
                with self.progress_bar_download_project_secondary(
                    message=f"Preparing '{dataset}'", total=len(split)
                ) as second_pbar:
                    self.progress_bar_download_project_secondary.show()
                    move_files(split, paths[dataset], image_name_formats[dataset], second_pbar)
                    main_pbar.update(1)
                self.progress_bar_download_project_secondary.hide()
            self.progress_bar_download_project_main.hide()

        # Clean up project directory
        project_datasets = [
            join(self._project_dir, item)
            for item in listdir(self._project_dir)
            if isdir(join(self._project_dir, item))
        ]
        for dataset in project_datasets:
            sly_fs.remove_dir(dataset)

        # Move processed splits to final destination
        train_ds_path = join(self._project_dir, "train")
        val_ds_path = join(self._project_dir, "val")
        with self.progress_bar_download_project_main(message="Processing splits", total=2) as pbar:
            self.progress_bar_download_project_main.show()
            for dataset in ["train", "val"]:
                shutil.move(
                    paths[dataset]["split_path"],
                    train_ds_path if dataset == "train" else val_ds_path,
                )
                pbar.update(1)
            self.progress_bar_download_project_main.hide()

        # Clean up temporary directory
        sly_fs.remove_dir(project_split_path)
        self._sly_project = Project(self._project_dir, OpenMode.READ)

    # ----------------------------------------- #

    # ----------------------------------------- #
    # Download Model
    def _download_model(self) -> None:
        sly_fs.mkdir(self._model_dir, True)
        if self.model_source == ModelSource.PRETRAINED:
            self._download_pretrained_model()

        else:
            self._download_custom_model()
        logger.info(f"Model files have been downloaded successfully to: '{self._model_dir}'")

    def _download_pretrained_model(self):
        # General
        self._model_files = {}
        model_meta = self.model_info["meta"]
        model_files = model_meta["model_files"]

        with self.progress_bar_download_model_main(
            message="Downloading model files",
            total=len(model_files),
        ) as model_download_main_pbar:
            self.progress_bar_download_model_main.show()
            for file in model_files:
                file_url = model_files[file]

                with urlopen(file_url) as f:
                    weights_size = f.length

                file_path = join(self._model_dir, file)

                with self.progress_bar_download_model_secondary(
                    message=f"Downloading '{file}' ",
                    total=weights_size,
                    unit="bytes",
                    unit_scale=True,
                ) as model_download_secondary_pbar:
                    self.progress_bar_download_model_secondary.show()
                    sly_fs.download(
                        url=file_url,
                        save_path=file_path,
                        progress=model_download_secondary_pbar.update,
                    )

                model_download_main_pbar.update(1)
                self._model_files[file] = file_path

        self.progress_bar_download_model_main.hide()
        self.progress_bar_download_model_secondary.hide()

    def _download_custom_model(self):
        # General
        self._model_files = {}

        # Need to merge file_url with arts dir
        model_files = self.model_info["model_files"]
        for file in model_files:
            model_files[file] = join(self.model_info["artifacts_dir"], model_files[file])

        # Add selected checkpoint to model_files
        checkpoint = self._gui.model_selector.custom_models_table.get_selected_checkpoint_path()
        model_files["checkpoint"] = checkpoint

        with self.progress_bar_download_model_main(
            message="Downloading model files",
            total=len(model_files),
        ) as model_download_main_pbar:
            self.progress_bar_download_model_main.show()
            for file in model_files:
                file_url = model_files[file]

                file_info = self._api.file.get_info_by_path(self._team_id, file_url)
                file_path = join(self._model_dir, file)
                file_size = file_info.sizeb

                with self.progress_bar_download_model_secondary(
                    message=f"Downloading {file}",
                    total=file_size,
                    unit="bytes",
                    unit_scale=True,
                ) as model_download_secondary_pbar:
                    self.progress_bar_download_model_secondary.show()
                    self._api.file.download(
                        self._team_id,
                        file_url,
                        file_path,
                        progress_cb=model_download_secondary_pbar.update,
                    )
                model_download_main_pbar.update(1)
                self._model_files[file] = file_path

        self.progress_bar_download_model_main.hide()
        self.progress_bar_download_model_secondary.hide()

    # ----------------------------------------- #

    # Postprocess

    def _validate_experiment_info(self, experiment_info: dict) -> bool:
        if not isinstance(experiment_info, dict):
            logger.error(
                f"Validation failed: 'experiment_info' must be a dictionary not '{type(experiment_info)}'"
            )
            return False

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
                logger.error(f"Validation failed: Missing required key '{key}'")
                return False

            if not isinstance(experiment_info[key], expected_type):
                logger.error(
                    f"Validation failed: Key '{key}' should be of type {expected_type.__name__}"
                )
                return False

        if isinstance(experiment_info["checkpoints"], list):
            for checkpoint in experiment_info["checkpoints"]:
                if not isinstance(checkpoint, str):
                    logger.error(
                        "Validation failed: All items in 'checkpoints' list must be strings"
                    )
                    return False
                if not sly_fs.file_exists(checkpoint):
                    logger.error(
                        f"Validation failed: Checkpoint file: '{checkpoint}' does not exist"
                    )
                    return False

        if not sly_fs.file_exists(experiment_info["best_checkpoint"]):
            logger.error(
                f"Validation failed: Best checkpoint file: '{experiment_info['best_checkpoint']}' does not exist"
            )
            return False

        logger.info("Validation successful")
        return True

    def _postprocess_splits(self) -> dict:
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

    def _preprocess_artifacts(self, experiment_info: dict) -> str:
        logger.info("Preprocessing artifacts")
        output_dir = self._output_dir
        output_weights_dir = join(output_dir, "weights")

        if "model_files" not in experiment_info:
            experiment_info["model_files"] = {}
        else:
            # Move model files to output directory except config, config will be processed later
            files = {k: v for k, v in experiment_info["model_files"].items() if k != "config"}
            for file in files:
                if isfile:
                    shutil.move(file, join(output_dir, sly_fs.get_file_name_with_ext(file)))
                elif isdir:
                    shutil.move(file, join(output_dir, basename(file)))

        # Prepare or create config
        config = experiment_info["model_files"].get("config")
        if config is None:
            config = "config.yaml"
            experiment_info["model_files"]["config"] = config

        output_config_path = join(
            output_dir,
            sly_fs.get_file_name_with_ext(experiment_info["model_files"]["config"]),
        )
        sly_fs.mkdir(output_dir, True)
        sly_fs.mkdir(output_weights_dir, True)

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
                output_weights_dir, sly_fs.get_file_name_with_ext(checkpoint_path)
            )
            shutil.move(checkpoint_path, new_checkpoint_path)

        # Prepare logs
        if sly_fs.dir_exists(self.log_dir):
            logs_dir = join(output_dir, "logs")
            shutil.move(self.log_dir, logs_dir)
        return output_dir

    # Generate experiment_info.json and app_state.json
    def _upload_json_file(self, local_path: str, remote_path: str, message: str) -> None:
        """Helper function to upload a JSON file with progress."""
        logger.info(f"Uploading '{local_path}' to Supervisely")
        total_size = sly_fs.get_file_size(local_path)
        with self.progress_bar_upload_artifacts(
            message=message, total=total_size, unit="bytes", unit_scale=True
        ) as upload_artifacts_pbar:
            self.progress_bar_upload_artifacts.show()
            self._api.file.upload(
                self._team_id,
                local_path,
                remote_path,
                progress_cb=upload_artifacts_pbar,
            )
            self.progress_bar_upload_artifacts.hide()

    def _generate_train_val_splits(
        self, local_dir: str, remote_dir: str, splits_data: dict
    ) -> None:
        local_train_val_split_ids_path = join(local_dir, "train_val_ids_split.json")
        local_train_split_path = join(local_dir, "train_split.json")
        local_val_split_path = join(local_dir, "val_split.json")
        remote_train_val_split_ids_path = join(remote_dir, "train_val_ids_split.json")
        remote_train_split_path = join(remote_dir, "train_split.json")
        remote_val_split_path = join(remote_dir, "val_split.json")

        sly_json.dump_json_file(splits_data, local_train_val_split_ids_path)
        self._upload_json_file(
            local_train_val_split_ids_path,
            remote_train_val_split_ids_path,
            "Uploading 'train_val_ids_split.json' to Team Files",
        )

        sly_json.dump_json_file(self.train_split, local_train_split_path)
        self._upload_json_file(
            local_train_split_path,
            remote_train_split_path,
            "Uploading 'train_split.json' to Team Files",
        )

        sly_json.dump_json_file(self.val_split, local_val_split_path)
        self._upload_json_file(
            local_val_split_path,
            remote_val_split_path,
            "Uploading 'val_split.json' to Team Files",
        )

    def _generate_experiment_info(
        self,
        local_dir: str,
        remote_dir: str,
        experiment_info: Dict,
        splits_data: Dict,
        evaluation_report_id: Optional[int] = None,
    ) -> None:
        logger.info("Updating experiment info")
        experiment_info.update(
            {
                "framework_name": self.framework_name,
                "app_state": "app_state.json",
                "train_val_splits": {
                    "train": {
                        "split": "train_split.json",
                        "images_ids": splits_data["train"]["images_ids"],
                    },
                    "val": {
                        "split": "val_split.json",
                        "images_ids": splits_data["val"]["images_ids"],
                    },
                },
                "hyperparameters": self.hyperparameters,
                "artifacts_dir": remote_dir,
                "task_id": self.task_id,
                "project_id": self.project_info.id,
                "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "evaluation_report_id": evaluation_report_id,
                "eval_metrics": {"mAP": None, "mIoU": None, "f1_conf_threshold": None},
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

        local_path = join(local_dir, "experiment_info.json")
        remote_path = join(remote_dir, "experiment_info.json")
        sly_json.dump_json_file(experiment_info, local_path)
        self._upload_json_file(
            local_path, remote_path, "Uploading 'experiment_info.json' to Team Files"
        )

    def _generate_app_state(self, local_dir: str, remote_dir: str, experiment_info: Dict) -> None:
        input_data = {"project_id": self.project_id}
        train_val_splits = self._get_train_val_splits()
        model = self._get_model_config(experiment_info)

        options = {
            "model_benchmark": {
                "enable": self.use_model_benchmark,
                "speed_test": self.use_model_benchmark_speedtest,
            },
            "cache_project": self.gui.input_selector.get_cache_value(),
        }

        app_state = {
            "input": input_data,
            "train_val_splits": train_val_splits,
            "classes": self.classes,
            "model": model,
            "hyperparameters": self.hyperparameters,
            "options": options,
        }

        local_path = join(local_dir, "app_state.json")
        remote_path = join(remote_dir, "app_state.json")
        sly_json.dump_json_file(app_state, local_path)
        self._upload_json_file(local_path, remote_path, "Uploading 'app_state.json' to Team Files")

    def _get_train_val_splits(self) -> Dict:
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

    def _get_model_config(self, experiment_info: Dict) -> Dict:
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
    def _upload_artifacts(self, output_dir: str, experiment_info: dict) -> None:
        logger.info(f"Uploading directory: '{output_dir}' to Supervisely")

        experiments_dir = "experiments"
        task_id = self.task_id

        remote_artifacts_dir = f"/{experiments_dir}/{self.project_id}_{self.project_name}/{task_id}_{self.framework_name}/"

        # Clean debug directory if exists
        if task_id == "debug-session":
            if self._api.file.dir_exists(self._team_id, f"{remote_artifacts_dir}/", True):
                with self.progress_bar_upload_artifacts(
                    message=f"[Debug] Cleaning train artifacts: '{remote_artifacts_dir}/'",
                    total=1,
                ) as upload_artifacts_pbar:
                    self.progress_bar_upload_artifacts.show()
                    self._api.file.remove_dir(self._team_id, f"{remote_artifacts_dir}/", True)
                    upload_artifacts_pbar.update(1)
                    self.progress_bar_upload_artifacts.hide()

        # Generate link file
        if is_production():
            app_url = f"/apps/sessions/{task_id}"
        else:
            app_url = "This is a debug session. No link available."
        app_link_path = join(output_dir, "open_app.lnk")
        with open(app_link_path, "w") as text_file:
            print(app_url, file=text_file)

        local_files = sly_fs.list_files_recursively(output_dir)
        total_size = sum([sly_fs.get_file_size(file_path) for file_path in local_files])
        with self.progress_bar_upload_artifacts(
            message="Uploading train artifacts to Team Files",
            total=total_size,
            unit="bytes",
            unit_scale=True,
        ) as upload_artifacts_pbar:
            self.progress_bar_upload_artifacts.show()
            remote_dir = self._api.file.upload_directory(
                self._team_id,
                output_dir,
                remote_artifacts_dir,
                progress_size_cb=upload_artifacts_pbar,
            )
            self.progress_bar_upload_artifacts.hide()

        # Set output directory
        file_info = self._api.file.get_info_by_path(self._team_id, join(remote_dir, "open_app.lnk"))
        logger.info("Training artifacts uploaded successfully")
        set_directory(remote_artifacts_dir)

        self.artifacts_thumbnail.set(file_info)
        self.artifacts_thumbnail.show()
        self._gui.training_process.success_message.show()

        return remote_dir, file_info

    # Model Benchmark
    def _get_eval_results_dir_name(self) -> str:
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
    ) -> bool:

        report, report_id = None, None
        if self._inference_class is None:
            logger.warn(
                "Inference class is not registered, model benchmark disabled. "
                "Use 'register_inference_class' method to register inference class."
            )
            return report, report_id

        # can't get task type from session. requires before session init
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
            self._gui.training_process.model_benchmark_report_text.show()
            self._gui.training_process.model_benchmark_progress_main.show()
            self._gui.training_process.model_benchmark_progress_main(
                message="Starting Model Benchmark evaluation", total=1
            )

            # 0. Serve trained model
            m = self._inference_class(
                model_dir=self._model_dir,
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
                    progress=self._gui.training_process.model_benchmark_progress_main,
                    progress_secondary=self._gui.training_process.model_benchmark_progress_secondary,
                    classes_whitelist=self.classes,
                )
            elif task_type == TaskType.INSTANCE_SEGMENTATION:
                bm = InstanceSegmentationBenchmark(
                    self._api,
                    self.project_info.id,
                    output_dir=benchmark_dir,
                    gt_dataset_ids=benchmark_dataset_ids,
                    gt_images_ids=benchmark_images_ids,
                    progress=self._gui.training_process.model_benchmark_progress_main,
                    progress_secondary=self._gui.training_process.model_benchmark_progress_secondary,
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
            if self.use_model_benchmark_speedtest is True:
                bm.run_speedtest(session, self.project_info.id)
                self.model_benchmark_pbar_secondary.hide()  # @TODO: add progress bar
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

            self._gui.training_process.model_benchmark_report_text.hide()
            self._gui.training_process.model_benchmark_report_thumbnail.set(
                benchmark_report_template
            )
            self._gui.training_process.model_benchmark_report_thumbnail.show()
            self._gui.training_process.model_benchmark_progress_main.hide()
            self._gui.training_process.model_benchmark_progress_secondary.hide()
            logger.info("Model benchmark evaluation completed successfully")
            logger.info(
                f"Predictions project name: {bm.dt_project_info.name}. Workspace_id: {bm.dt_project_info.workspace_id}"
            )
            logger.info(
                f"Differences project name: {bm.diff_project_info.name}. Workspace_id: {bm.diff_project_info.workspace_id}"
            )
        except Exception as e:
            logger.error(f"Model benchmark failed. {repr(e)}", exc_info=True)
            self._gui.training_process.model_benchmark_report_text.hide()
            self._gui.training_process.model_benchmark_progress_main.hide()
            self._gui.training_process.model_benchmark_progress_secondary.hide()
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
                    self._gui.model_selector.custom_models_table.get_selected_checkpoint_path(),
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
        self._log_dir = join(self.work_dir, "logs")
        train_logger.set_log_dir(self._log_dir)
        train_logger.start_tensorboard()
        self._setup_logger_callbacks()
        time.sleep(1)
        self._gui.training_process.tensorboard_button.enable()

    def _setup_logger_callbacks(self):
        epoch_pbar = None
        step_pbar = None

        def start_training_callback(total_epochs: int):
            nonlocal epoch_pbar
            logger.info(f"Training started for {total_epochs} epochs")
            pbar_widget = self.progress_bar_epochs
            pbar_widget.show()
            epoch_pbar = pbar_widget(message=f"Epochs", total=total_epochs)

        def finish_training_callback():
            self.progress_bar_epochs.hide()
            self.progress_bar_iters.hide()

            # @TODO: access tensorboard after training
            train_logger.writer.close()
            train_logger.stop_tensorboard()

        def start_epoch_callback(total_steps: int):
            nonlocal step_pbar
            logger.info(f"Epoch started. Total steps: {total_steps}")
            pbar_widget = self.progress_bar_iters
            pbar_widget.show()
            step_pbar = pbar_widget(message=f"Steps", total=total_steps)

        def finish_epoch_callback():
            epoch_pbar.update(1)

        def step_callback():
            step_pbar.update(1)

        train_logger.add_on_train_started_callback(start_training_callback)
        train_logger.add_on_train_finish_callback(finish_training_callback)

        train_logger.add_on_epoch_started_callback(start_epoch_callback)
        train_logger.add_on_epoch_finish_callback(finish_epoch_callback)

        train_logger.add_on_step_callback(step_callback)

    # ----------------------------------------- #
    def _wrapped_start_training(self):
        self.gui.training_process.start_button.loading = True

        if self._train_func is None:
            raise ValueError("Train function is not defined")

        # Init logger and Tensorboard
        self._init_logger()
        experiment_info = None
        self.preprocess()
        try:
            experiment_info = self._train_func()
        except StopTrainingException as e:
            logger.error(f"Training stopped: {e}")
            raise e  # @TODO: add stop button
        self.postprocess(experiment_info)
        self.gui.training_process.start_button.loading = False

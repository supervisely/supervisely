import shutil
from datetime import datetime
from os.path import basename, dirname, join
from typing import Any, Dict, List, Optional, Union
from urllib.request import urlopen

import yaml

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
from supervisely.app.widgets import FolderThumbnail, Progress, SlyTqdm, Widget
from supervisely.nn.benchmark import InstanceSegmentationBenchmark, ObjectDetectionBenchmark
from supervisely.nn.inference import RuntimeType, SessionJSON
from supervisely.nn.task_type import TaskType
from supervisely.nn.training.gui.gui import TrainGUI
from supervisely.nn.training.utils import load_file, validate_list_of_dicts
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
        # if work_dir is None:
        # work_dir = sly.app.dir
        self._work_dir = work_dir
        self._project_dir = join(self._work_dir, "sly_project")
        self._project_meta_path = join(self._project_dir, "meta.json")  # No need?

        self._train_dataset_dir = join(self._project_dir, "train")
        self._val_dataset_dir = join(self._project_dir, "val")

        self._train_dataset_info = None
        self._val_dataset_info = None
        self._train_dataset_fs = None
        self._val_dataset_fs = None
        self._sly_project = None
        # ----------------------------------------- #

        # Classes
        # ----------------------------------------- #

        # Model
        self._model_dir = join(self._work_dir, "model")
        self._model_name = None
        self._model_files = {}
        # ----------------------------------------- #

        # Hyperparameters
        # ----------------------------------------- #

        # Layout
        self._layout: TrainGUI = TrainGUI(
            self._framework_name, self._models, self._hyperparameters, self._app_options
        )
        self._app = Application(layout=self._layout.layout)
        self._server = self._app.get_server()
        self._train_func = None
        # ----------------------------------------- #

    # General
    @property
    def app(self) -> Application:
        return self._app

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
        return self._layout.project_id

    @property
    def project_name(self) -> str:
        return self._layout.project_info.name

    @property
    def project_info(self) -> ProjectInfo:
        return self._layout.project_info

    @property
    def sly_project(self):
        return self._sly_project

    @property
    def use_cache(self) -> bool:
        return self._layout.input_selector.get_cache_value()

    @property
    def train_dataset_id(self) -> int:
        return self._layout.input_selector.get_train_dataset_id()

    @property
    def val_dataset_id(self) -> int:
        return self._layout.input_selector.get_val_dataset_id()

    @property
    def train_dataset_info(self) -> int:
        return self._train_dataset_info

    @property
    def val_dataset_info(self) -> int:
        return self._val_dataset_info

    @property
    def train_dataset_fs(self) -> int:
        return self._train_dataset_fs

    @property
    def val_dataset_fs(self) -> int:
        return self._val_dataset_fs

    # ----------------------------------------- #

    # Model
    @property
    def model_source(self) -> str:
        return self._layout.model_selector.get_model_source()

    @property
    def model_name(self) -> str:
        return self._layout.model_selector.get_model_name()

    @property
    def model_info(self) -> str:
        return self._layout.model_selector.get_model_info()

    @property
    def model_files(self) -> str:
        return self._model_files

    # Classes
    @property
    def classes(self) -> List[str]:
        return self._layout.classes_selector.get_selected_classes()

    @property
    def num_classes(self) -> List[str]:
        return len(self._layout.classes_selector.get_selected_classes())

    # Hyperparameters
    @property
    def hyperparameters(self) -> Dict[str, Any]:
        return yaml.safe_load(self._layout.hyperparameters_selector.get_hyperparameters())

    @property
    def hyperparameters_raw(self) -> str:
        return self._layout.hyperparameters_selector.get_hyperparameters()

    @property
    def use_model_benchmark(self) -> bool:
        return self._layout.hyperparameters_selector.get_model_benchmark_checkbox_value()

    @property
    def use_model_benchmark_speedtest(self) -> bool:
        return self._layout.hyperparameters_selector.get_speedtest_checkbox_value()

    # Train Process
    @property
    def progress_bar_download_project(self) -> Progress:
        return self._layout.training_process.project_download_progress

    @property
    def progress_bar_download_model_main(self) -> Progress:
        return self._layout.training_process.model_download_progress_main

    @property
    def progress_bar_download_model_secondary(self) -> Progress:
        return self._layout.training_process.model_download_progress_secondary

    @property
    def progress_bar_epochs(self) -> Progress:
        return self._layout.training_process.epoch_progress

    @property
    def progress_bar_iters(self) -> Progress:
        return self._layout.training_process.iter_progress

    @property
    def progress_bar_upload_artifacts(self) -> Progress:
        return self._layout.training_process.artifacts_upload_progress

    # Output
    @property
    def artifacts_thumbnail(self) -> FolderThumbnail:
        return self._layout.training_process.artifacts_thumbnail

    @property
    def start(self):
        sly_fs.mkdir(self.work_dir, True)

        def decorator(func):
            self._train_func = func

            def wrapped_start_training():
                if self._train_func is None:
                    raise ValueError("Train function is not defined")

                experiment_info = None
                self.preprocess()
                try:
                    experiment_info = self._train_func()
                except StopTrainingException as e:
                    print(f"Training stopped: {e}")
                    raise e  # @TODO: add stop button
                self.postprocess(experiment_info)

            self._layout.training_process.start_button.click(wrapped_start_training)
            return func

        return decorator

    # region PROCESS
    def preprocess(self):
        print("Preprocessing...")
        # Step 1. Workflow Input
        if is_production():
            self._workflow_input()
        # Step 2. Download Project
        self._download_project()
        # Step 3. Convert Supervisely to X format
        # Step 4. Download Model files
        self._download_model()

    def postprocess(self, experiment_info: dict):
        print("Postprocessing...")

        # Step 1. Validate experiment_info
        success = self._validate_experiment_info(experiment_info)
        if not success:
            raise ValueError("Experiment info is not valid. Failed to upload artifacts")

        # Step 2. Preprocess artifacts
        output_dir = self._preprocess_artifacts(experiment_info)

        # Step 3. Upload artifacts
        remote_dir, file_info = self._upload_artifacts(output_dir, experiment_info)

        # Step 4. Run Model Benchmark
        mb_eval_report, mb_eval_report_id = None, None
        if self.use_model_benchmark is True:
            if is_production() and self.use_model_benchmark is True:
                try:
                    mb_eval_report, mb_eval_report_id = self._run_model_benchmark(
                        output_dir, remote_dir, experiment_info
                    )
                except Exception as e:
                    logger.error(f"Model benchmark failed: {e}")

        # Step 4. Generate experiment_info.json
        self._generate_experiment_info(output_dir, remote_dir, experiment_info, mb_eval_report_id)

        # Step 5. Workflow output
        if is_production():
            self._workflow_output(remote_dir, file_info, mb_eval_report)
        # Step 6. Shutdown app
        self._app.shutdown()

    # region PROCESS END

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
        dataset_infos = [
            self._api.dataset.get_info_by_id(self.train_dataset_id),
            self._api.dataset.get_info_by_id(self.val_dataset_id),
        ]
        total_images = sum(ds_info.images_count for ds_info in dataset_infos)
        self._train_dataset_info, self._val_dataset_info = dataset_infos

        if not self.use_cache or is_development():
            self._download_no_cache(dataset_infos, total_images)
            self._sly_project = self._prepare_project()
            return

        try:
            self._download_with_cache(dataset_infos, total_images)
        except Exception:
            logger.warning(
                "Failed to retrieve project from cache. Downloading it...",
                exc_info=True,
            )
            if sly_fs.dir_exists(self._project_dir):
                sly_fs.clean_dir(self._project_dir)
            self._download_no_cache(dataset_infos, total_images)
        finally:
            self._sly_project = self._prepare_project()
            logger.info(f"Project downloaded successfully to: '{self._project_dir}'")

    def _prepare_project(self) -> None:
        # Preprocess project
        # Rename datasets to train and val
        project_fs = Project(self._project_dir, OpenMode.READ)
        for dataset in project_fs.datasets:
            dataset: Dataset
            if dataset.name == self._train_dataset_info.name:
                dataset_path = join(self._project_dir, dataset.name)
                shutil.move(dataset_path, self._train_dataset_dir)
                self._train_dataset_fs = Dataset(self._train_dataset_dir, OpenMode.READ)
            elif dataset.name == self._val_dataset_info.name:
                dataset_path = join(self._project_dir, dataset.name)
                shutil.move(dataset_path, self._val_dataset_dir)
                self._val_dataset_fs = Dataset(self._val_dataset_dir, OpenMode.READ)
            else:
                raise ValueError("Unknown dataset name")  # TODO: won't happen?
        return Project(self._project_dir, OpenMode.READ)

    def _download_no_cache(self, dataset_infos: List[DatasetInfo], total_images: int) -> None:
        self.progress_bar_download_project.show()
        with self.progress_bar_download_project(
            message="Downloading input data...", total=total_images
        ) as pbar:
            download_project(
                api=self._api,
                project_id=self.project_id,
                dest_dir=self._project_dir,
                dataset_ids=[ds_info.id for ds_info in dataset_infos],
                log_progress=True,
                progress_cb=pbar.update,
            )
        self.progress_bar_download_project.hide()

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
        self.progress_bar_download_project.show()
        with self.progress_bar_download_project(
            message="Downloading input data...", total=total_images
        ) as pbar:
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
        with self.progress_bar_download_project(
            message="Retrieving data from cache...",
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
        self.progress_bar_download_project.hide()

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

    # ----------------------------------------- #
    # Download Model
    def _download_model(self) -> None:
        sly_fs.mkdir(self._model_dir, True)
        if self.model_source == "Pretrained models":
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
            message="Downloading model files...",
            total=len(model_files),
        ) as model_download_main_pbar:
            self.progress_bar_download_model_main.show()
            for file in model_files:
                file_url = model_files[file]

                with urlopen(file_url) as f:
                    weights_size = f.length

                file_path = join(self._model_dir, file)

                with self.progress_bar_download_model_secondary(
                    message=f"Downloading '{file}' ...",
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
        checkpoint = self._layout.model_selector.custom_models_table.get_selected_checkpoint_path()
        model_files["checkpoint"] = checkpoint

        with self.progress_bar_download_model_main(
            message="Downloading model files...",
            total=len(model_files),
        ) as model_download_main_pbar:
            self.progress_bar_download_model_main.show()
            for file in model_files:
                file_url = model_files[file]

                file_info = self._api.file.get_info_by_path(self._team_id, file_url)
                file_path = join(self._model_dir, file)
                file_size = file_info.sizeb

                with self.progress_bar_download_model_secondary(
                    message=f"Downloading '{file}' ...",
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

        if "config" not in experiment_info["model_files"]:
            logger.error("Validation failed: 'model_files' must contain a 'config' key")
            return False
        sly_fs.file_exists(experiment_info["model_files"]["config"])

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

    def _preprocess_artifacts(self, experiment_info: dict) -> str:
        logger.info("Preprocessing artifacts...")
        output_dir = join(self.work_dir, "result")
        output_weights_dir = join(output_dir, "weights")
        output_config_path = join(
            output_dir, sly_fs.get_file_name_with_ext(experiment_info["model_files"]["config"])
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
        return output_dir

    # Generate train info
    def _generate_experiment_info(
        self,
        local_dir: str,
        remote_dir: str,
        experiment_info: dict,
        evaluation_report_id: int = None,
    ) -> None:
        logger.info("Updating experiment info")

        experiment_info["framework_name"] = self.framework_name

        experiment_info["hyperparameters"] = self.hyperparameters
        experiment_info["artifacts_dir"] = remote_dir
        experiment_info["task_id"] = self.task_id
        experiment_info["project_id"] = self.project_info.id
        experiment_info["train_dataset_id"] = self.train_dataset_id
        experiment_info["val_dataset_id"] = self.val_dataset_id

        experiment_info["datetime"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        experiment_info["evaluation_report_id"] = evaluation_report_id
        experiment_info["eval_metrics"] = {
            "mAP": None,
            "mIoU": None,
            "f1_conf_threshold": None,
        }

        remote_weights_dir = join(remote_dir, "weights")
        checkpoint_files = self._api.file.list(
            self._team_id, remote_weights_dir, return_type="fileinfo"
        )
        checkpoint_paths = [f"weights/{checkpoint.name}" for checkpoint in checkpoint_files]
        experiment_info["checkpoints"] = checkpoint_paths

        best_file_name = sly_fs.get_file_name_with_ext(experiment_info["best_checkpoint"])
        experiment_info["best_checkpoint"] = best_file_name

        config_name = sly_fs.get_file_name_with_ext(experiment_info["model_files"]["config"])
        # experiment_info["model_files"]["config"] = join(remote_dir, config_name)
        experiment_info["model_files"]["config"] = config_name

        logger.info("Uploading 'experiment_info.json' to Supervisely")
        # Dump experiment_info.json
        local_experiment_info_path = join(local_dir, "experiment_info.json")
        remote_experiment_info_path = join(remote_dir, "experiment_info.json")
        sly_json.dump_json_file(experiment_info, local_experiment_info_path)
        total_size = sly_fs.get_file_size(local_experiment_info_path)

        with self.progress_bar_upload_artifacts(
            message="Uploading 'experiment_info.json' to Team Files...",
            total=total_size,
            unit="bytes",
            unit_scale=True,
        ) as upload_artifacts_pbar:
            self.progress_bar_upload_artifacts.show()
            remote_dir = self._api.file.upload(
                self._team_id,
                local_experiment_info_path,
                remote_experiment_info_path,
                progress_cb=upload_artifacts_pbar,
            )
            self.progress_bar_upload_artifacts.hide()

    # Upload artifacts
    def _upload_artifacts(self, output_dir: str, experiment_info: dict) -> None:
        logger.info(f"Uploading directory: '{output_dir}' to Supervisely")

        experiments_dir = "/experiments"
        task_type = experiment_info["task_type"]

        task_id = self.task_id
        project_name = self.project_info.name

        remote_artifacts_dir = join(
            experiments_dir, self.framework_name, task_type, project_name, task_id
        )

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
            message="Uploading train artifacts to Team Files...",
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
        self._layout.training_process.success_message.show()

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
        self, local_artifacts_dir: str, remote_artifacts_dir: str, experiment_info: dict
    ) -> bool:
        if self._inference_class is None:
            logger.warn(
                "Inference class is not registered, model benchmark disabled. "
                "Use 'register_inference_class' method to register inference class."
            )
            return None, None

        # can't get task type from session. requires before session init
        supported_task_types = [TaskType.OBJECT_DETECTION, TaskType.INSTANCE_SEGMENTATION]
        task_type = experiment_info["task_type"]
        if task_type not in supported_task_types:
            logger.warn(
                f"Task type: '{task_type}' is not supported for Model Benchmark. "
                f"Supported tasks: {', '.join(task_type)}"
            )
            return None, None

        logger.info("Running Model Benchmark evaluation...")
        try:
            remote_weights_dir = join(remote_artifacts_dir, "weights")
            best_checkpoint = experiment_info.get("best_checkpoint", None)
            best_filename = sly_fs.get_file_name_with_ext(best_checkpoint)
            remote_best_checkpoint = join(remote_weights_dir, best_filename)

            config_path = experiment_info["model_files"]["config"]
            remote_config_path = join(
                remote_artifacts_dir, sly_fs.get_file_name_with_ext(config_path)
            )

            logger.info(f"Creating the report for the best model: {best_filename!r}")
            self._layout.training_process.model_benchmark_report_text.show()
            self._layout.training_process.model_benchmark_progress_main.show()
            self._layout.training_process.model_benchmark_progress_main(
                message="Starting Model Benchmark evaluation...", total=1
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
                model_source="Custom models",
                task_type=task_type,
                checkpoint_name=best_filename,
                checkpoint_url=remote_best_checkpoint,
                config_url=remote_config_path,
            )
            m._load_model(deploy_params)
            m.serve()

            # m.model.overrides["verbose"] = False
            session = SessionJSON(self._api, session_url="http://localhost:8000")
            benchmark_dir = join(local_artifacts_dir, "benchmark")
            sly_fs.mkdir(benchmark_dir, True)

            # 1. Init benchmark (todo: auto-detect task type)
            train_dataset_ids = [self.train_dataset_id]
            train_images_ids = [
                img_info.id for img_info in self._api.image.get_list(self.train_dataset_id)
            ]

            benchmark_dataset_ids = [self.val_dataset_id]
            benchmark_images_ids = [
                img_info.id for img_info in self._api.image.get_list(self.val_dataset_id)
            ]

            if task_type == TaskType.OBJECT_DETECTION:
                bm = ObjectDetectionBenchmark(
                    self._api,
                    self.project_info.id,
                    output_dir=benchmark_dir,
                    gt_dataset_ids=benchmark_dataset_ids,
                    gt_images_ids=benchmark_images_ids,
                    progress=self._layout.training_process.model_benchmark_progress_main,
                    progress_secondary=self._layout.training_process.model_benchmark_progress_secondary,
                    classes_whitelist=self.classes,
                )
            elif task_type == TaskType.INSTANCE_SEGMENTATION:
                bm = InstanceSegmentationBenchmark(
                    self._api,
                    self.project_info.id,
                    output_dir=benchmark_dir,
                    gt_dataset_ids=benchmark_dataset_ids,
                    gt_images_ids=benchmark_images_ids,
                    progress=self._layout.training_process.model_benchmark_progress_main,
                    progress_secondary=self._layout.training_process.model_benchmark_progress_secondary,
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
                self.model_benchmark_pbar_secondary.hide()
                bm.upload_speedtest_results(eval_res_dir + "/speedtest/")

            # 7. Prepare visualizations, report and upload
            bm.visualize()
            remote_dir = bm.upload_visualizations(eval_res_dir + "/visualizations/")
            report = bm.upload_report_link(remote_dir)

            # 8. UI updates
            benchmark_report_template = self._api.file.get_info_by_path(
                self._team_id(), remote_dir + "template.vue"
            )

            self._layout.training_process.model_benchmark_report_text.hide()
            self._layout.training_process.model_benchmark_report_thumbnail.set(
                benchmark_report_template
            )
            self._layout.training_process.model_benchmark_report_thumbnail.show()
            self._layout.training_process.model_benchmark_progress_main.hide()
            self._layout.training_process.model_benchmark_progress_secondary.hide()
            logger.info("Model benchmark evaluation completed successfully")
            logger.info(
                f"Predictions project name: {bm.dt_project_info.name}. Workspace_id: {bm.dt_project_info.workspace_id}"
            )
            logger.info(
                f"Differences project name: {bm.diff_project_info.name}. Workspace_id: {bm.diff_project_info.workspace_id}"
            )
        except Exception as e:
            logger.error(f"Model benchmark failed. {repr(e)}", exc_info=True)
            self._layout.training_process.model_benchmark_report_text.hide()
            self._layout.training_process.model_benchmark_progress_main.hide()
            self._layout.training_process.model_benchmark_progress_secondary.hide()
            try:
                if bm.dt_project_info:
                    self._api.project.remove(bm.dt_project_info.id)
                if bm.diff_project_info:
                    self._api.project.remove(bm.diff_project_info.id)
            except Exception as e2:
                return None, None
        return report, report.id

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

            if self.model_source == "Custom models":
                file_info = self._api.file.get_info_by_path(
                    self._team_id, self.model_parameters["checkpoint_url"]
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
        except Exception as e:
            logger.debug(f"Failed to add output to the workflow: {repr(e)}")
        # ----------------------------------------- #
        except Exception as e:
            logger.debug(f"Failed to add output to the workflow: {repr(e)}")
        # ----------------------------------------- #
        except Exception as e:
            logger.debug(f"Failed to add output to the workflow: {repr(e)}")
        # ----------------------------------------- #
        except Exception as e:
            logger.debug(f"Failed to add output to the workflow: {repr(e)}")
        # ----------------------------------------- #
        except Exception as e:
            logger.debug(f"Failed to add output to the workflow: {repr(e)}")
        # ----------------------------------------- #
        except Exception as e:
            logger.debug(f"Failed to add output to the workflow: {repr(e)}")
        # ----------------------------------------- #
        except Exception as e:
            logger.debug(f"Failed to add output to the workflow: {repr(e)}")
        # ----------------------------------------- #

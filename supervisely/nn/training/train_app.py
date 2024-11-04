from os import makedirs
from os.path import basename, exists, join
from typing import Any, Dict, List, Union
from urllib.request import urlopen

import yaml

import supervisely.io.env as sly_env
import supervisely.io.fs as sly_fs
from supervisely import (
    Api,
    Application,
    DatasetInfo,
    Project,
    ProjectInfo,
    download_project,
    logger,
)
from supervisely.app.widgets import Progress, SlyTqdm, Widget
from supervisely.io.fs import clean_dir
from supervisely.nn.training.gui.gui import TrainGUI
from supervisely.nn.training.utils import load_file, validate_list_of_dicts
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
        output_dir: str,
        models: Union[str, List[Dict[str, Any]]],
        hyperparameters: Union[str, List[Dict[str, Any]]],
        app_options: Union[str, List[Dict[str, Any]]] = None,
    ):

        # Init
        self._api = Api.from_env()

        self._team_id = sly_env.team_id()
        self._workspace_id = sly_env.workspace_id()

        self._models = self.__load_models(models)
        self._hyperparameters = self.__load_hyperparameters(hyperparameters)
        self._app_options = self.__load_app_options(app_options)

        # ----------------------------------------- #

        # Input
        self._output_dir = output_dir
        self._project_dir = join(self._output_dir, "sly_project")
        self._train_dataset_dir = join(self._output_dir, "train")
        self._val_dataset_dir = join(self._output_dir, "val")
        self._project_meta_path = join(self._project_dir, "meta.json")  # No need?

        self._project_fs = None
        # ----------------------------------------- #

        # Classes
        # ----------------------------------------- #

        # Model
        self._model_dir = join(self._output_dir, "model")
        self._model_path = None
        self._config_path = None
        # ----------------------------------------- #

        # Hyperparameters
        # ----------------------------------------- #

        # Layout
        self._layout: TrainGUI = TrainGUI(self._models, self._hyperparameters, self._app_options)
        self._app = Application(layout=self._layout.layout)
        self._server = self._app.get_server()
        self._train_func = None
        # ----------------------------------------- #

    @property
    def app(self) -> Application:
        return self._app

    @property
    def project_id(self) -> int:
        return self._layout.project_id

    @property
    def project_fs(self):
        return self._project_fs

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
    def model_source(self) -> str:
        return self._layout.model_selector.get_model_source()

    @property
    def model_parameters(self) -> str:
        return self._layout.model_selector.get_model_parameters()

    @property
    def hyperparameters(self) -> Dict[str, Any]:
        return yaml.safe_load(self.hyperparameters)(
            self._layout.hyperparameters_selector.get_hyperparameters()
        )

    @property
    def classes(self) -> List[str]:
        return self._layout.classes_selector.get_selected_classes()

    # Train Process
    @property
    def progress_bar_download_project(self) -> Progress:
        return self._layout.training_process.project_download_progress

    @property
    def progress_bar_download_model(self) -> Progress:
        return self._layout.training_process.model_download_progress

    @property
    def progress_bar_epochs(self) -> Progress:
        return self._layout.training_process.epoch_progress

    @property
    def progress_bar_iters(self) -> Progress:
        return self._layout.training_process.iter_progress

    @property
    def start(self):
        def decorator(func):
            self._train_func = func

            def wrapped_start_training():
                self.preprocess()
                try:
                    if self._train_func:
                        self._train_func()
                except StopTrainingException as e:
                    print(f"Training stopped: {e}")
                finally:
                    self.postprocess()

            self._layout.training_process.start_button.click(wrapped_start_training)
            return func

        return decorator

    def preprocess(self):
        print("Preprocessing...")
        # workflow input
        self.__download_project()
        # convert project to format?
        self.__download_model()

    def postprocess(self):
        print("Postprocessing...")
        # upload artifacts
        # model benchmark evaluation
        # workflow output

    # Utility methods

    # Loaders
    def __load_models(self, models: Union[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        if isinstance(models, str):
            models = load_file(models)
        return validate_list_of_dicts(models, "models")

    def __load_hyperparameters(self, hyperparameters: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        if isinstance(hyperparameters, str):
            hyperparameters = load_file(hyperparameters)
        if not isinstance(hyperparameters, (dict, str)):
            raise ValueError(
                "hyperparameters must be a dict, or a path to a '.json' or '.yaml' file."
            )
        return hyperparameters

    def __load_app_options(self, app_options: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        if isinstance(app_options, str):
            app_options = load_file(app_options)
        if not isinstance(app_options, dict):
            raise ValueError("app_options must be a dict, or a path to a '.json' or '.yaml' file.")
        return app_options

    # ----------------------------------------- #

    # Download Project
    def __download_project(self) -> None:
        makedirs(self._project_dir, exist_ok=True)
        project_info = self._api.project.get_info_by_id(self.project_id)
        dataset_infos = [
            self._api.dataset.get_info_by_id(self.train_dataset_id),
            self._api.dataset.get_info_by_id(self.val_dataset_id),
        ]
        total_images = sum(ds_info.images_count for ds_info in dataset_infos)

        if exists(self._project_dir):
            clean_dir(self._project_dir)

        if not self.use_cache:
            self.__download_no_cache(dataset_infos, total_images)
            return

        try:
            self.__download_with_cache(project_info, dataset_infos, total_images)
        except Exception:
            logger.warning(
                "Failed to retrieve project from cache. Downloading it...", exc_info=True
            )
            if exists(self._project_dir):
                clean_dir(self._project_dir)
            self.__download_no_cache(dataset_infos, total_images)
        finally:
            self._project_fs = Project(self._project_dir)
            logger.info(f"Project downloaded successfully to: '{self._project_dir}'")

    def __download_no_cache(self, dataset_infos: List[DatasetInfo], total_images: int) -> None:
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

    def __download_with_cache(
        self, project_info: ProjectInfo, dataset_infos: List[DatasetInfo], total_images: int
    ) -> None:
        to_download = [info for info in dataset_infos if not is_cached(project_info.id, info.name)]
        cached = [info for info in dataset_infos if is_cached(project_info.id, info.name)]

        logger.info(self.__get_cache_log_message(cached, to_download))
        self.progress_bar_download_project.show()
        with self.progress_bar_download_project(
            message="Downloading input data...", total=total_images
        ) as pbar:
            download_to_cache(
                api=self._api,
                project_id=project_info.id,
                dataset_infos=dataset_infos,
                log_progress=True,
                progress_cb=pbar.update,
            )

        total_cache_size = sum(get_cache_size(project_info.id, ds.name) for ds in dataset_infos)
        with self.progress_bar_download_project(
            message="Retrieving data from cache...",
            total=total_cache_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            copy_from_cache(
                project_id=project_info.id,
                dest_dir=self._project_dir,
                dataset_names=[ds_info.name for ds_info in dataset_infos],
                progress_cb=pbar.update,
            )
        self.progress_bar_download_project.hide()

    def __get_cache_log_message(self, cached: bool, to_download: List[DatasetInfo]) -> str:
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
    def __download_model(self) -> None:
        makedirs(self._model_dir, exist_ok=True)
        if self.model_source == "Pretrained models":
            self.__download_pretrained_model()

        else:
            self.__download_custom_model()
        logger.info(f"Model downloaded successfully to: '{self._model_path}'")

    # ----------------------------------------- #

    def __download_pretrained_model(self):
        # General
        model_meta = self.model_parameters["meta"]
        model_url = model_meta["weights_url"]
        model_name = basename(model_url)

        # Specific
        config_url = model_meta.get("config_url", None)
        config_name = basename(config_url) if config_url is not None else None
        arch_type = model_meta.get("arch_type", None)

        with urlopen(model_url) as file:
            weights_size = file.length

        self.progress_bar_download_model.show()
        self._model_path = join(self._model_dir, model_name)  # TODO handle ext?
        with self.progress_bar_download_model(
            message="Downloading model weights...",
            total=weights_size,
            unit="bytes",
            unit_scale=True,
        ) as model_download_pbar:
            sly_fs.download(
                url=model_url,
                save_path=self._model_path,
                progress=model_download_pbar.update,
            )

        if config_url is not None:
            with urlopen(config_url) as file:
                config_size = file.length

            self._config_path = join(self._model_dir, config_name)  # TODO handle ext?
            with self.progress_bar_download_model(
                message="Downloading model config...",
                total=config_size,
                unit="bytes",
                unit_scale=True,
            ) as config_pbar:
                sly_fs.download(
                    url=config_url,
                    save_path=self._config_path,
                    progress=config_pbar.update,
                )
        self.progress_bar_download_model.hide()

    def __download_custom_model(self):
        # General
        model_url = self.model_parameters["checkpoint_url"]
        model_name = basename(model_url)

        # Specific
        config_url = self.model_parameters("config_url", None)
        config_name = basename(config_url) if config_url is not None else None

        self._model_path = join(self._model_dir, model_name)  # TODO handle ext?

        checkpoint_info = self._api.file.get_info_by_path(self._team_id, model_url)
        weights_size = checkpoint_info.sizeb
        self.progress_bar_download_model.show()
        with self.progress_bar_download_model(
            message="Downloading model weights...",
            total=weights_size,
            unit="bytes",
            unit_scale=True,
        ) as model_download_pbar:
            if not exists(self._model_path):
                makedirs("models", exist_ok=True)
                self._api.file.download(
                    self._team_id,
                    model_url,
                    self._model_path,
                    progress_cb=model_download_pbar.update,
                )

        if config_url is not None:
            self._config_path = join(self._model_dir, config_name)
            config_info = self._api.file.get_info_by_path(self._team_id, config_url)
            config_size = config_info.sizeb
            with self.progress_bar_download_model(
                message="Downloading model config...",
                total=config_size,
                unit="bytes",
                unit_scale=True,
            ) as config_pbar:
                if not exists(self._config_path):
                    makedirs("models", exist_ok=True)
                    self._api.file.download(
                        self._team_id,
                        config_url,
                        self._config_path,
                        progress_cb=config_pbar.update,
                    )
        self.progress_bar_download_model.hide()

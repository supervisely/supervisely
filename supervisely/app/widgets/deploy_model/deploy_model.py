import datetime
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Literal, Tuple, Union

import pandas as pd
import yaml

from supervisely._utils import logger
from supervisely.api.api import Api
from supervisely.api.app_api import ModuleInfo
from supervisely.app.widgets.button.button import Button
from supervisely.app.widgets.container.container import Container
from supervisely.app.widgets.editor.editor import Editor
from supervisely.app.widgets.experiment_selector.experiment_selectorv2 import (
    ExperimentSelector,
)
from supervisely.app.widgets.fast_table.fast_table import FastTable
from supervisely.app.widgets.field.field import Field
from supervisely.app.widgets.tabs.tabs import Tabs
from supervisely.app.widgets.text.text import Text
from supervisely.app.widgets.widget import Widget
from supervisely.io import env
from supervisely.nn.experiments import ExperimentInfo, get_experiment_infos
from supervisely.nn.model.model_api import ModelAPI


class DeployModel(Widget):

    class Connect:

        class COLUMN:
            SESSION_ID = "Session ID"
            APP_NAME = "App Name"
            FRAMEWORK = "Framework"
            MODEL = "Model"

        COLUMNS = [
            str(COLUMN.SESSION_ID),
            str(COLUMN.APP_NAME),
            str(COLUMN.FRAMEWORK),
            str(COLUMN.MODEL),
        ]

        def __init__(self, deploy_model: "DeployModel"):
            self.api = deploy_model.api
            self.team_id = deploy_model.team_id
            self._cache = deploy_model._cache
            self.parent = deploy_model
            self._model_api = None
            self.layout = self._create_layout()
            self._update_sessions()

        def _create_layout(self) -> Container:
            self.inference_settings_editor = Editor(language_mode="yaml")
            self.inference_settings_editor_field = Field(
                content=self.inference_settings_editor, title="Inference Settings"
            )
            self.refresh_button = Button("", icon="zmdi zmdi-refresh")
            self.sessions_table = FastTable(
                columns=self.COLUMNS,
                page_size=10,
                is_radio=True,
                header_left_content=self.refresh_button,
            )
            self.deploy_button = Button("Connect")

            @self.refresh_button.click
            def _refresh_button_clicked():
                self._update_sessions()

            @self.deploy_button.click
            def _connect_button_clicked():
                self._connect()

            @self.sessions_table.selection_changed
            def _selection_changed(row):
                session_id = row.row[0]
                self._update_inference_settings(session_id)

            return Container(
                widgets=[
                    self.sessions_table,
                    self.inference_settings_editor_field,
                    self.deploy_button,
                ]
            )

        def _framework_from_task_info(self, task_info: Dict) -> str:
            module_id = task_info["meta"]["app"]["moduleId"]
            module = None
            for m in self.parent.get_modules():
                if m.id == module_id:
                    module = m
            if module is None:
                module = self.api.app.get_ecosystem_module_info(module_id=module_id)
                self._cache.setdefault("modules", []).append(module)
            for cat in module.config["categories"]:
                if cat.startswith("framework:"):
                    return cat[len("framework:") :]
            return "unknown"

        def _data_from_session(self, session: Dict) -> Dict[str, Any]:
            task_info = session["task_info"]
            deploy_info = session["model_info"]
            return {
                self.COLUMN.SESSION_ID: task_info["id"],
                self.COLUMN.APP_NAME: task_info["meta"]["app"]["name"],
                self.COLUMN.FRAMEWORK: self._framework_from_task_info(task_info),
                self.COLUMN.MODEL: deploy_info["model_name"],
            }

        def _update_sessions(self) -> None:
            self.sessions_table.loading = True
            try:
                self.sessions_table.clear()
                sessions = self.api.nn.list_deployed_models(team_id=self.team_id)
                data = [self._data_from_session(session) for session in sessions]
                df = pd.DataFrame.from_records(data=data, columns=self.COLUMNS)
                self.sessions_table.read_pandas(df)
            except Exception as e:
                logger.error(
                    f"Failed to load deployed models: {e}",
                    exc_info=True,
                )
            finally:
                self.sessions_table.loading = False

        def _update_inference_settings(self, session_id: int) -> None:
            self.inference_settings_editor.loading = True
            try:
                model_api = self.api.nn.connect(session_id)
                settings = model_api.get_settings()
                settings = yaml.safe_dump(settings)
                self.inference_settings_editor.set_text(settings)
            except Exception as e:
                logger.error(
                    f"Failed to load inference settings for session {session_id}: {e}",
                    exc_info=True,
                )
                self.inference_settings_editor.set_text("")
            finally:
                self.inference_settings_editor.loading = False

        def connect(self, deploy_parameters: Dict[str, Any]) -> ModelAPI:
            session_id = deploy_parameters["session_id"]
            model_api = self.api.nn.connect(task_id=session_id)
            task_info = self.api.task.get_info_by_id(model_api.task_id)
            model_info = model_api.get_info()
            model_name = model_info["model_name"]
            framework = self._framework_from_task_info(task_info)
            logger.info(
                f"Model {model_name} from framework {framework} deployed with session ID {model_api.task_id}."
            )
            self.parent.update_model_task_texts(task_info)
            return model_api

        def _connect(self) -> None:
            self.parent.status.text = "Connecting to model..."
            self.parent.status.status = "info"
            self.parent.status.show()
            try:
                deploy_parameters = self.get_deploy_parameters()
                logger.info(f"Connecting to model with parameters:", extra=deploy_parameters)
                model_api = self.connect(deploy_parameters)
                self._model_api = model_api
                self.parent.status.text = "Model connected successfully!"
                self.parent.status.status = "success"
                self.parent.sesson_link.show()
                return model_api
            except Exception as e:
                logger.error(f"Failed to connect to model: {e}", exc_info=True)
                self.parent.status.text = f"Error: {e}"
                self.parent.status.status = "error"
                self.parent.sesson_link.hide()

        def _deploy(self):
            return self._connect()

        def get_deploy_parameters(self) -> Dict[str, Any]:
            selected_row = self.sessions_table.get_selected_row()
            return {
                "session_id": selected_row.row[self.COLUMNS.index(str(self.COLUMN.SESSION_ID))],
            }

    class Pretrained:
        class COLUMN:
            FRAMEWORK = "Framework"
            MODEL_NAME = "Model"

        COLUMNS = [
            str(COLUMN.FRAMEWORK),
            str(COLUMN.MODEL_NAME),
        ]

        def __init__(self, deploy_model: "DeployModel"):
            self.api = deploy_model.api
            self.team_id = deploy_model.team_id
            self._cache = deploy_model._cache
            self.parent = deploy_model
            self._model_api = None
            self._last_selected_framework = None
            self._load_models()
            self.layout = self._create_layout()
            self._update_pretrained_models()

        def _create_layout(self) -> Container:
            self.inference_settings_editor = Editor(language_mode="yaml")
            self.inference_settings_editor_field = Field(
                content=self.inference_settings_editor, title="Inference Settings"
            )
            self.pretrained_table = FastTable(
                columns=self.COLUMNS,
                page_size=10,
                is_radio=True,
            )
            self.deploy_button = Button("Deploy")

            @self.deploy_button.click
            def _deploy_button_clicked():
                self._deploy()

            @self.pretrained_table.selection_changed
            def _selection_changed(row):
                self.inference_settings_editor.loading = True
                try:
                    framework = row[0] if row else None
                    if self._last_selected_framework == framework:
                        return
                    inference_settings = self.parent._get_inference_settings_for_framework(
                        framework
                    )
                    self.inference_settings_editor.set_text(inference_settings)
                    self._last_selected_framework = framework
                except Exception as e:
                    self.inference_settings_editor.set_text("")
                    raise
                finally:
                    self.inference_settings_editor.loading = False

            return Container(
                widgets=[
                    self.pretrained_table,
                    self.inference_settings_editor_field,
                    self.deploy_button,
                    self.parent.status,
                    self.parent.sesson_link,
                ]
            )

        def _load_models(self):
            models = self.api.nn._model_api.list_models(local=True)
            for model in models:
                self._cache.setdefault("pretrained_models", {}).setdefault(
                    model["framework"], []
                ).append(model)

        def _list_pretrained_models(self, framework: str):
            models = self._cache["pretrained_models"].get(framework, [])
            return models

        def _update_pretrained_models(self) -> None:
            self.pretrained_table.loading = True
            try:
                self.pretrained_table.clear()
                models_data = []
                for framework in self.parent.get_frameworks():
                    try:
                        framework_models = self._list_pretrained_models(framework)
                        models_data.extend(
                            [{"framework": framework, "model": model} for model in framework_models]
                        )
                    except Exception as e:
                        logger.warning(
                            f"Failed to load pretrained models for framework '{framework}': {e}",
                            exc_info=True,
                        )
                data = [
                    {
                        self.COLUMN.FRAMEWORK: model_data["framework"],
                        self.COLUMN.MODEL_NAME: model_data["model"]["name"],
                    }
                    for model_data in models_data
                ]
                df = pd.DataFrame.from_records(data=data, columns=self.COLUMNS)
                self.pretrained_table.read_pandas(df)
            except Exception as e:
                logger.error(
                    f"Failed to load pretrained models: {e}",
                    exc_info=True,
                )
            finally:
                self.pretrained_table.loading = False

        def get_deploy_parameters(self) -> Dict[str, Any]:
            selected_row = self.pretrained_table.get_selected_row()
            return {
                "framework": selected_row.row[self.COLUMNS.index(str(self.COLUMN.FRAMEWORK))],
                "model_name": selected_row.row[self.COLUMNS.index(str(self.COLUMN.MODEL_NAME))],
            }

        def deploy(self, deploy_parameters: Dict[str, Any]) -> ModelAPI:
            framework = deploy_parameters["framework"]
            model_name = deploy_parameters["model_name"]
            model_api = self.api.nn.deploy(model=f"{framework}/{model_name}")
            logger.info(
                f"Model {model_name} from framework {framework} deployed with session ID {model_api.task_id}."
            )
            return model_api

        def _deploy(self) -> None:
            self.parent.status.text = "Deploying pretrained model..."
            self.parent.status.status = "info"
            self.parent.status.show()
            try:
                deploy_parameters = self.get_deploy_parameters()
                logger.info(f"Deploying model with parameters:", extra=deploy_parameters)
                model_api = self.deploy(deploy_parameters)
                self._model_api = model_api
                self.parent.status.text = "Model deployed successfully!"
                self.parent.status.status = "success"
                task_info = self.api.task.get_info_by_id(model_api.task_id)
                self.parent.update_model_task_texts(task_info)
                self.parent.sesson_link.show()
                return model_api
            except Exception as e:
                logger.error(f"Failed to deploy model: {e}", exc_info=True)
                self.parent.status.text = f"Error: {e}"
                self.parent.status.status = "error"
                self.parent.sesson_link.hide()

    class Custom:
        def __init__(self, deploy_model: "DeployModel"):
            self.api = deploy_model.api
            self.team_id = deploy_model.team_id
            self._cache = deploy_model._cache
            self.parent = deploy_model
            self._model_api = None
            self.layout = self._create_layout()

        def _create_layout(self) -> Container:
            self.inference_settings_editor = Editor(language_mode="yaml")
            self.inference_settings_editor_field = Field(
                content=self.inference_settings_editor, title="Inference Settings"
            )
            frameworks = self.parent.get_frameworks()
            experiment_infos = []
            for framework_name in frameworks:
                experiment_infos.extend(
                    get_experiment_infos(self.api, self.team_id, framework_name=framework_name)
                )
            self.experiment_table = ExperimentSelector(
                experiment_infos=experiment_infos,
                team_id=self.team_id,
                api=self.api,
            )
            self.deploy_button = Button("Deploy")

            @self.deploy_button.click
            def _deploy_button_clicked():
                self._deploy()

            @self.experiment_table.selection_changed
            def _selection_changed(experiment_info: ExperimentInfo):
                self.inference_settings_editor.loading = True
                try:
                    task_id = experiment_info.task_id
                    train_task_info = self.api.task.get_info_by_id(task_id)
                    train_module_id = train_task_info["meta"]["app"]["moduleId"]
                    module = self.api.nn._deploy_api.get_serving_app_by_train_app(
                        module_id=train_module_id
                    )
                    inference_settings = self.parent._get_inference_settings_by_module(module)
                    self.inference_settings_editor.set_text(inference_settings)
                except Exception as e:
                    self.inference_settings_editor.set_text("")
                    raise
                finally:
                    self.inference_settings_editor.loading = False

            return Container(widgets=[self.experiment_table, self.inference_settings_editor_field])

        def get_deploy_parameters(self) -> Dict[str, Any]:
            experiment_info = self.experiment_table.get_selected_experiment_info()
            return {
                "experiment_info": experiment_info,
            }

        def _framework_from_task_info(self, task_info: Dict) -> str:
            module_id = task_info["meta"]["app"]["moduleId"]
            module = None
            for m in self.parent.get_modules():
                if m.id == module_id:
                    module = m
            if module is None:
                module = self.api.app.get_ecosystem_module_info(module_id=module_id)
                self._cache.setdefault("modules", []).append(module)
            for cat in module.config["categories"]:
                if cat.startswith("framework:"):
                    return cat[len("framework:") :]
            return "unknown"

        def deploy(self, deploy_parameters: Dict[str, Any]) -> ModelAPI:
            experiment_info = deploy_parameters["experiment_info"]
            task_info = self.api.nn._deploy_api.deploy_custom_model_from_experiment_info(
                experiment_info
            )
            model_api = ModelAPI(api=self.api, task_id=task_info["id"])
            model_info = model_api.get_info()
            model_name = model_info["model_name"]
            framework = self._framework_from_task_info(task_info)
            logger.info(
                f"Model {model_name} from framework {framework} deployed with session ID {model_api.task_id}."
            )
            return model_api

        def _deploy(self) -> None:
            self.parent.status.text = "Deploying custom model..."
            self.parent.status.status = "info"
            self.parent.status.show()
            try:
                deploy_parameters = self.get_deploy_parameters()
                logger.info(f"Deploying model with parameters:", extra=deploy_parameters)
                model_api = self.deploy(deploy_parameters)
                self._model_api = model_api
                self.parent.status.text = "Model deployed successfully!"
                self.parent.status.status = "success"
                task_info = self.api.task.get_info_by_id(model_api.task_id)
                self.parent.update_model_task_texts(task_info)
                self.parent.sesson_link.show()
                return model_api
            except Exception as e:
                logger.error(f"Failed to deploy model: {e}", exc_info=True)
                self.parent.status.text = f"Error: {e}"
                self.parent.status.status = "error"
                self.parent.sesson_link.hide()

    class MODE:
        CONNECT = "connect"
        PRETRAINED = "pretrained"
        CUSTOM = "custom"

    MODES = [str(MODE.CONNECT), str(MODE.PRETRAINED), str(MODE.CUSTOM)]
    MODE_TO_CLASS = {
        str(MODE.CONNECT): Connect,
        str(MODE.PRETRAINED): Pretrained,
        str(MODE.CUSTOM): Custom,
    }

    def __init__(
        self,
        modes: List[Literal["connect", "pretrained", "custom"]] = None,
        api: Api = None,
        team_id: int = None,
        widget_id: str = None,
    ):
        if modes is None:
            modes = self.MODES.copy()
        self.modes = modes
        self._validate_modes()
        if api is None:
            api = Api()
        self.api = api
        if team_id is None:
            team_id = env.team_id()
        self.team_id = team_id

        self._cache = {}

        # GUI
        self._modes_layouts = {}
        self.tabs: Tabs = None
        self.layout: Widget = None
        self._init_gui()

        super().__init__(widget_id=widget_id)

    def _validate_modes(self) -> None:
        if len(self.modes) < 0 or len(self.modes) > 3:
            raise ValueError(
                "Modes must be a list containing 1 to 3 of the following: 'connect', 'pretrained', 'custom'."
            )
        for mode in self.modes:
            if mode not in self.MODES:
                raise ValueError(
                    f"Invalid mode '{mode}'. Valid modes are 'connect', 'pretrained', 'custom'."
                )

    def get_modules(self) -> List[ModuleInfo]:
        modules = self._cache.setdefault("modules", [])
        if len(modules) > 0:
            return modules
        modules = self.api.app.get_list_ecosystem_modules(
            categories=["serve", "images"], categories_operation="and"
        )
        modules = [
            module
            for module in modules
            if any([cat for cat in module["config"]["categories"] if cat.startswith("framework:")])
        ]
        modules = [ModuleInfo.from_json(module) for module in modules]
        self._cache["modules"] = modules
        return modules

    def get_frameworks(self) -> List[str]:
        if len(self._cache.get("frameworks", [])) > 0:
            return self._cache["frameworks"]

        modules = self._cache.setdefault("modules", [])
        if len(modules) == 0:
            modules = self.get_modules()
        frameworks = [cat for module in modules for cat in module.config.get("categories", [])]
        frameworks = [
            cat[len("framework:") :] for cat in frameworks if cat.startswith("framework:")
        ]
        self._cache["frameworks"] = frameworks
        return frameworks

    @property
    def tabs_labels_dict(self) -> Dict:
        return {
            self.MODE.CONNECT: "Connect",
            self.MODE.PRETRAINED: "Pretrained",
            self.MODE.CUSTOM: "Custom",
        }

    def _init_modes(self) -> None:
        for mode in self.modes:
            self._modes_layouts[mode] = self.MODE_TO_CLASS[mode](self)

    def _create_task_link(self, task_id: int) -> str:
        return f"{self.api.server_address}/apps/sessions/{task_id}"

    def _get_inference_settings_by_module(self, module: Dict) -> str:
        config = module["config"]
        inference_settings_path = config.get("files", {}).get("inference_settings", None)
        if inference_settings_path is None:
            raise ValueError(
                f"No inference settings file found for framework app {module['meta']['app']['name']}."
            )
        save_path = tempfile.mktemp(suffix=".yaml")
        self.api.app.download_git_file(
            module_id=module["id"],
            file_path=inference_settings_path,
            save_path=save_path,
            version="model-files-in-configasd",
        )
        inference_settings = Path(save_path).read_text()
        return inference_settings

    def _get_inference_settings_for_framework(self, framework: str) -> str:
        inference_settings_cache = self._cache.setdefault("inference_settings", {})
        if framework not in inference_settings_cache:
            module = self.api.nn._deploy_api.find_serving_app_by_framework(
                framework, version="model-files-in-config"  # TODO: remove version
            )
            if module is None:
                raise ValueError(f"No serving app found for framework {framework}.")
            config = module["config"]
            inference_settings_path = config.get("files", {}).get("inference_settings", None)
            if inference_settings_path is None:
                raise ValueError(f"No inference settings file found for framework {framework}.")
            save_path = tempfile.mktemp(suffix=".yaml")
            self.api.app.download_git_file(
                module_id=module["id"],
                file_path=inference_settings_path,
                save_path=save_path,
                version="model-files-in-configasd",
            )
            inference_settings = Path(save_path).read_text()
            inference_settings_cache[framework] = inference_settings
        return inference_settings_cache[framework]

    def update_model_task_texts(self, task_info: Dict):
        task_id = task_info["id"]
        task_link = self._create_task_link(task_id)
        task_date = task_info["startedAt"]
        task_date = datetime.datetime.fromisoformat(task_date.replace("Z", "+00:00"))
        task_date = task_date.strftime("%Y-%m-%d %H:%M:%S")
        task_name = task_info["meta"]["app"]["name"]
        self.session_text_1.text = f"<i class='zmdi zmdi-link' style='color: #7f858e'></i> <a href='{task_link}' target='_blank'>{task_name}: {task_id}</a>"
        self.session_text_2.text = (
            f"<span class='field-description text-muted' style='color: #7f858e'>{task_date}</span>"
        )

    def _init_gui(self) -> None:
        self.status = Text("Deploying model...", status="info")
        self.session_text_1 = Text(
            "",
            "text",
        )
        self.session_text_2 = Text(
            "",
            "text",
            font_size=13,
        )
        self.sesson_link = Container(
            [
                self.session_text_1,
                self.session_text_2,
            ],
            gap=0,
            style="padding-left: 10px",
        )
        self.status.hide()
        self.sesson_link.hide()

        self._init_modes()
        _labels = [self.tabs_labels_dict[mode] for mode in self.modes]
        _contents = [self._modes_layouts[mode].layout for mode in self.modes]
        self.tabs = Tabs(labels=_labels, contents=_contents)
        widgets = []
        if len(self.modes) == 1:
            widgets.append(_contents[0])
        else:
            widgets.append(self.tabs)

        self.layout = Container(
            widgets=widgets,
        )

    def deploy(self) -> ModelAPI:
        mode_label = self.tabs.get_active_tab()
        mode = None
        for mode, label in self.tabs_labels_dict.items():
            if label == mode_label:
                break
        self._modes_layouts[mode]._deploy()
        return self._modes_layouts[mode]._model_api

    def load_from_json(self, data: Dict[str, Any]) -> None:
        """
        Load widget state from JSON data.
        :param data: Dictionary with widget data.
        """
        pass

    def get_deploy_parameters(self) -> Dict[str, Any]:
        mode_label = self.tabs.get_active_tab()
        mode = None
        for mode, label in self.tabs_labels_dict.items():
            if label == mode_label:
                break
        parameters = {
            "mode": mode,
        }
        parameters.update(self._modes_layouts[mode].get_deploy_parameters())
        return parameters

    def get_json_data(self) -> Dict[str, Any]:
        return {}

    def get_json_state(self) -> Dict[str, Any]:
        return {}

    def to_html(self):
        return self.layout.to_html()

import datetime
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Tuple, Union

import pandas as pd
import yaml

from supervisely._utils import logger
from supervisely.api.api import Api
from supervisely.api.app_api import ModuleInfo
from supervisely.app.widgets.agent_selector.agent_selector import AgentSelector
from supervisely.app.widgets.button.button import Button
from supervisely.app.widgets.container.container import Container
from supervisely.app.widgets.editor.editor import Editor
from supervisely.app.widgets.experiment_selector.experiment_selector import (
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

    class DeployMode:

        def deploy(self, agent_id: int = None) -> ModelAPI:
            raise NotImplementedError("This method should be implemented in subclasses.")

        def get_deploy_parameters(self) -> Dict[str, Any]:
            raise NotImplementedError("This method should be implemented in subclasses.")

        @property
        def layout(self) -> Widget:
            raise NotImplementedError("This property should be implemented in subclasses.")

    class Connect(DeployMode):

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
            self.deploy_model = deploy_model
            self._layout = self._create_layout()
            self._update_sessions()

        def _create_layout(self) -> Container:
            self.refresh_button = Button("", icon="zmdi zmdi-refresh")
            self.sessions_table = FastTable(
                columns=self.COLUMNS,
                page_size=10,
                is_radio=True,
                header_left_content=self.refresh_button,
            )

            @self.refresh_button.click
            def _refresh_button_clicked():
                self._update_sessions()

            return self.sessions_table

        @property
        def layout(self) -> FastTable:
            return self._layout

        def _data_from_session(self, session: Dict) -> Dict[str, Any]:
            task_info = session["task_info"]
            deploy_info = session["model_info"]
            return {
                self.COLUMN.SESSION_ID: task_info["id"],
                self.COLUMN.APP_NAME: task_info["meta"]["app"]["name"],
                self.COLUMN.FRAMEWORK: self.deploy_model._framework_from_task_info(task_info),
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

        def deploy(self, agent_id: int = None) -> ModelAPI:
            deploy_parameters = self.get_deploy_parameters()
            logger.info(f"Connecting to model with parameters:", extra=deploy_parameters)
            session_id = deploy_parameters["session_id"]
            model_api = self.api.nn.connect(task_id=session_id)
            return model_api

        def get_deploy_parameters(self) -> Dict[str, Any]:
            selected_row = self.sessions_table.get_selected_row()
            return {
                "session_id": selected_row.row[self.COLUMNS.index(str(self.COLUMN.SESSION_ID))],
            }

    class Pretrained(DeployMode):
        class COLUMN:
            FRAMEWORK = "Framework"
            MODEL_NAME = "Model"
            PARAMETERS = "Parameters (M)"
            MAP = "mAP"

        COLUMNS = [
            str(COLUMN.FRAMEWORK),
            str(COLUMN.MODEL_NAME),
            str(COLUMN.PARAMETERS),
            str(COLUMN.MAP),
        ]

        def __init__(self, deploy_model: "DeployModel"):
            self.api = deploy_model.api
            self.team_id = deploy_model.team_id
            self._cache = deploy_model._cache
            self.deploy_model = deploy_model
            self._model_api = None
            self._last_selected_framework = None
            self._load_models()
            self._layout = self._create_layout()
            self._update_pretrained_models()

        @property
        def layout(self) -> FastTable:
            return self._layout

        def _create_layout(self) -> Container:
            self.inference_settings_editor = Editor(language_mode="yaml", height_lines=10)
            self.inference_settings_editor_field = Field(
                content=self.inference_settings_editor, title="Inference Settings"
            )
            self.pretrained_table = FastTable(
                columns=self.COLUMNS,
                page_size=10,
                is_radio=True,
            )

            return self.pretrained_table

        def _load_models(self):
            models = self.api.nn.models_api.list_models()
            for model in models:
                self._cache.setdefault("pretrained_models", {}).setdefault(
                    model["framework"], []
                ).append(model)

        def _list_pretrained_models(self, framework: str):
            models = self._cache["pretrained_models"].get(framework, [])
            return models

        def _map_from_model(self, model: Dict):
            try:
                return model.get("evaluation", {}).get("metrics", {}).get("mAP", "N/A")
            except:
                return "N/A"

        def _update_pretrained_models(self) -> None:
            self.pretrained_table.loading = True
            try:
                self.pretrained_table.clear()
                models_data = []
                for framework in self.deploy_model.get_frameworks():
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
                        self.COLUMN.PARAMETERS: model_data["model"].get("paramsM", "N/A"),
                        self.COLUMN.MAP: self._map_from_model(model_data["model"]),
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

        def deploy(self, agent_id: int = None) -> ModelAPI:
            deploy_parameters = self.get_deploy_parameters()
            logger.info(f"Deploying pretrained model with parameters:", extra=deploy_parameters)
            framework = deploy_parameters["framework"]
            model_name = deploy_parameters["model_name"]
            model_api = self.api.nn.deploy(model=f"{framework}/{model_name}", agent_id=agent_id)
            return model_api

    class Custom(DeployMode):
        def __init__(self, deploy_model: "DeployModel"):
            self.api = deploy_model.api
            self.team_id = deploy_model.team_id
            self._cache = deploy_model._cache
            self.deploy_model = deploy_model
            self._model_api = None
            self._layout = self._create_layout()

        @property
        def layout(self) -> ExperimentSelector:
            return self._layout

        def _create_layout(self) -> Container:
            self.inference_settings_editor = Editor(language_mode="yaml", height_lines=10)
            self.inference_settings_editor_field = Field(
                content=self.inference_settings_editor, title="Inference Settings"
            )
            frameworks = self.deploy_model.get_frameworks()
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

            return self.experiment_table

        def get_deploy_parameters(self) -> Dict[str, Any]:
            experiment_info = self.experiment_table.get_selected_experiment_info()
            return {
                "experiment_info": experiment_info,
            }

        def deploy(self, agent_id: int) -> ModelAPI:
            deploy_parameters = self.get_deploy_parameters()
            logger.info(f"Deploying custom model with parameters:", extra=deploy_parameters)
            experiment_info = deploy_parameters["experiment_info"]
            task_info = self.api.nn._deploy_api.deploy_custom_model_from_experiment_info(
                agent_id=agent_id, experiment_info=experiment_info
            )
            model_api = ModelAPI(api=self.api, task_id=task_info["id"])
            return model_api

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
        api: Api = None,
        team_id: int = None,
        modes: List[Literal["connect", "pretrained", "custom"]] = None,
        widget_id: str = None,
    ):
        self.modes: Dict[str, DeployModel.DeployMode] = {}
        if modes is None:
            modes = self.MODES.copy()
        self._validate_modes(modes)
        if api is None:
            api = Api()
        self.api = api
        if team_id is None:
            team_id = env.team_id()
        self.team_id = team_id
        self._cache = {}
        self._init_modes(modes)

        self.modes_labels = {
            self.MODE.CONNECT: "Connect",
            self.MODE.PRETRAINED: "Pretrained",
            self.MODE.CUSTOM: "Custom",
        }

        # GUI
        self.layout: Widget = None
        self._init_gui()

        self.model_api: ModelAPI = None

        super().__init__(widget_id=widget_id)

    def _validate_modes(self, modes) -> None:
        if len(modes) < 1 or len(modes) > len(self.MODES):
            raise ValueError(
                f"Modes must be a list containing 1 to {len(self.MODES)} of the following: {', '.join(self.MODES)}."
            )
        for mode in modes:
            if mode not in self.MODES:
                raise ValueError(f"Invalid mode '{mode}'. Valid modes are {', '.join(self.MODES)}.")

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

    def _init_modes(self, modes: str) -> None:
        for mode in modes:
            self.modes[mode] = self.MODE_TO_CLASS[mode](self)

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
        )
        inference_settings = Path(save_path).read_text()
        return inference_settings

    def _get_inference_settings_for_framework(self, framework: str) -> str:
        inference_settings_cache = self._cache.setdefault("inference_settings", {})
        if framework not in inference_settings_cache:
            module = self.api.nn._deploy_api.find_serving_app_by_framework(framework)
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
            )
            inference_settings = Path(save_path).read_text()
            inference_settings_cache[framework] = inference_settings
        return inference_settings_cache[framework]

    def _framework_from_task_info(self, task_info: Dict) -> str:
        module_id = task_info["meta"]["app"]["moduleId"]
        module = None
        for m in self.get_modules():
            if m.id == module_id:
                module = m
        if module is None:
            module = self.api.app.get_ecosystem_module_info(module_id=module_id)
            self._cache.setdefault("modules", []).append(module)
        for cat in module.config["categories"]:
            if cat.startswith("framework:"):
                return cat[len("framework:") :]
        return "unknown"

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

        self.select_agent = AgentSelector(self.team_id)
        self.select_agent_field = Field(content=self.select_agent, title="Select Agent")

        self.deploy_button = Button("Deploy", icon="zmdi zmdi-play")
        self.stop_button = Button("Stop", icon="zmdi zmdi-stop", button_type="danger")
        self.stop_button.hide()

        _labels = [self.modes_labels[mode] for mode in self.modes]
        _contents = [self.modes[mode].layout for mode in self.modes]
        self.tabs = Tabs(labels=_labels, contents=_contents)
        widgets = []
        if len(self.modes) == 1:
            widgets.append(_contents[0])
        else:
            widgets.append(self.tabs)

        @self.deploy_button.click
        def _deploy_button_clicked():
            self._deploy()

        @self.stop_button.click
        def _stop_button_clicked():
            self.stop()
            self.tabs.enable()

        deploy_stop_buttons = Container(widgets=[self.deploy_button, self.stop_button], gap=0)

        widgets = [
            *widgets,
            self.select_agent_field,
            deploy_stop_buttons,
            self.status,
            self.sesson_link,
        ]
        self.layout = Container(
            widgets=widgets,
        )

    def set_model_status(
        self,
        status: Literal["deployed", "stopped", "deploying", "connecting", "error", "hide"],
        extra_text: str = None,
    ) -> None:
        if status == "hide":
            self.status.hide()
            return
        status_args = {
            "deployed": {"text": "Model deployed successfully!", "status": "success"},
            "stopped": {"text": "Model stopped", "status": "info"},
            "deploying": {"text": "Deploying model...", "status": "info"},
            "connecting": {"text": "Connecting to model...", "status": "info"},
            "error": {"text": "Error occurred during model deployment.", "status": "error"},
        }
        args = status_args[status]
        if extra_text:
            args["text"] += f" {extra_text}"
        self.status.set(**args)
        self.status.show()

    def set_session_info(self, task_info: Dict):
        if task_info is None:
            self.sesson_link.hide()
            return
        task_id = task_info["id"]
        task_link = self._create_task_link(task_id)
        task_date = task_info["startedAt"]
        task_date = datetime.datetime.fromisoformat(task_date.replace("Z", "+00:00"))
        task_date = task_date.strftime("%Y-%m-%d %H:%M:%S")
        task_name = task_info["meta"]["app"]["name"]
        self.session_text_1.text = f"<i class='zmdi zmdi-link' style='color: #7f858e'></i> <a href='{task_link}' target='_blank'>{task_name}: {task_id}</a>"
        self.session_text_2.text = f"<span class='field-description text-muted' style='color: #7f858e'>{task_date} (UTC)</span>"
        self.sesson_link.show()

    def _deploy(self) -> None:
        self.set_model_status("connecting")
        self.set_session_info(None)
        try:
            self.tabs.disable()
            model_api = self.deploy()
            task_info = self.api.task.get_info_by_id(model_api.task_id)
            model_info = model_api.get_info()
            model_name = model_info["model_name"]
            framework = self._framework_from_task_info(task_info)
            logger.info(
                f"Model {framework}: {model_name} deployed with session ID {model_api.task_id}."
            )
            self.model_api = model_api
            self.set_model_status("deployed")
            self.set_session_info(task_info)
        except Exception as e:
            logger.error(f"Failed to connect to model: {e}", exc_info=True)
            self.set_model_status("error", str(e))
            self.set_session_info(None)
            self.tabs.enable()

    def deploy(self) -> ModelAPI:
        mode_label = self.tabs.get_active_tab()
        mode = None
        for mode, label in self.modes_labels.items():
            if label == mode_label:
                break
        agent_id = self.select_agent.get_value()
        self.model_api = self.modes[mode].deploy(agent_id=agent_id)
        return self.model_api

    def stop(self) -> None:
        if self.model_api is not None:
            self.model_api.shutdown()
            self.model_api = None
            self.set_model_status("stopped")
            self.stop_button.hide()
            self.tabs.enable()

    def load_from_json(self, data: Dict[str, Any]) -> None:
        """
        Load widget state from JSON data.
        :param data: Dictionary with widget data.
        """
        pass

    def get_deploy_parameters(self) -> Dict[str, Any]:
        mode_label = self.tabs.get_active_tab()
        mode = None
        for mode, label in self.modes_labels.items():
            if label == mode_label:
                break
        parameters = {
            "mode": mode,
        }
        parameters.update(self.modes[mode].get_deploy_parameters())
        return parameters

    def get_json_data(self) -> Dict[str, Any]:
        return {}

    def get_json_state(self) -> Dict[str, Any]:
        return {}

    def to_html(self):
        return self.layout.to_html()

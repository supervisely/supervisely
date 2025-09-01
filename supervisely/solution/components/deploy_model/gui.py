from typing import Dict, Optional, Tuple

import supervisely.io.env as sly_env
from supervisely.api.api import Api
from supervisely.app.content import DataJson
from supervisely.app.exceptions import show_dialog
from supervisely.app.widgets import (
    AgentSelector,
    Button,
    Container,
    Dialog,
    Field,
    Input,
    Switch,
    Widget,
)
from supervisely.sly_logger import logger


class DeployModelGUI(Widget):
    def __init__(
        self,
        team_id: Optional[int] = None,
        widget_id: Optional[str] = None,
    ):
        self._api = Api.from_env()
        self.team_id = team_id or sly_env.team_id()
        self.model = None
        super().__init__(widget_id=widget_id)
        self.content = self._init_gui()

    @property
    def modal(self) -> Dialog:
        """
        Returns the settings modal dialog for the Compare widget.
        """
        if not hasattr(self, "_settings_modal"):
            self._settings_modal = Dialog(
                title="Settings",
                content=self.content,
                size="tiny",
            )
        return self._settings_modal

    @property
    def agent_selector(self) -> AgentSelector:
        if not hasattr(self, "_agent_selector"):
            self._agent_selector = AgentSelector(self.team_id)
        return self._agent_selector

    @property
    def model_name_input(self):
        if not hasattr(self, "_model_name_input"):
            self._model_name_input = Input(
                placeholder="Enter model name. E.g. 'RT-DETRv2/rt-detrv2-s"
            )
        return self._model_name_input

    @property
    def model_input_field(self) -> Field:
        if not hasattr(self, "_model_input_field"):
            self._model_input_field = Field(
                self.model_name_input,
                title="Model",
                description="Enter the name of the Pre-trained model or the path to a Custom checkpoint in Team Files.",
                icon=Field.Icon(
                    zmdi_class="zmdi zmdi-memory",
                    color_rgb=(21, 101, 192),
                    bg_color_rgb=(227, 242, 253),
                ),
            )
        return self._model_input_field

    def _disable_model_input_field(self):
        self.model_name_input.disable()

    def _enable_model_input_field(self):
        self.model_name_input.enable()

    def _get_model_input_value(self) -> str:
        return self.model_name_input.get_value().strip()

    def _set_model_input_value(self, model: str):
        self.model_name_input.set_value(model)

    def _init_gui(self):
        agent_selector_field = Field(
            Container([self.agent_selector, self.change_agent_button], gap=15),
            title="Select Agent",
            description="Select the agent to deploy the model on.",
            icon=Field.Icon(
                zmdi_class="zmdi zmdi-storage",
                color_rgb=(21, 101, 192),
                bg_color_rgb=(227, 242, 253),
            ),
        )
        optimize_gpu_field = Field(
            self.optimize_gpu_switch,
            title="Optimize Memory Usage",
            description="The model will be automatically unloaded from memory after a period of inactivity or to free up GPU memory for other tasks (e.g. training).",
            icon=Field.Icon(
                zmdi_class="zmdi zmdi-settings",
                color_rgb=(21, 101, 192),
                bg_color_rgb=(227, 242, 253),
            ),
        )

        deploy_button_container = Container(
            widgets=[self.freeze_model_button, self.stop_button, self.deploy_button],
            direction="horizontal",
            overflow="wrap",
            style="display: flex; justify-content: flex-end;",
            widgets_style="display: flex; flex: none;",
        )

        return Container(
            [
                self.model_input_field,
                agent_selector_field,
                optimize_gpu_field,
                deploy_button_container,
            ],
            gap=20,
        )

    @property
    def optimize_gpu_switch(self) -> Switch:
        if not hasattr(self, "_optimize_gpu_switch"):
            self._optimize_gpu_switch = Switch(switched=True)
            self._optimize_gpu_switch.disable()
        return self._optimize_gpu_switch

    @property
    def agent_selector(self) -> AgentSelector:
        if not hasattr(self, "_agent_selector"):
            self._agent_selector = AgentSelector(self.team_id, show_only_gpu=True)
        return self._agent_selector

    @property
    def freeze_model_button(self):
        if not hasattr(self, "_freeze_model_button"):
            self._freeze_model_button = Button(
                text="Freeze Model",
                icon="zmdi zmdi-eject",
                plain=True,
            )

            @self._freeze_model_button.click
            def _on_freeze_model_click():
                if self.model is None:
                    show_dialog(
                        title="Warning",
                        description="No model is currently deployed. Nothing to freeze.",
                        status="warning",
                    )
                    return
                logger.info("Freezing model...")
                res = self.model.freeze_model()
                if isinstance(res, dict) and "message" in res:
                    logger.info(res["message"])

            self._freeze_model_button.hide()

        return self._freeze_model_button

    @property
    def change_agent_button(self):
        if not hasattr(self, "_change_agent_button"):
            self._change_agent_button = Button(
                text="Change Agent",
                button_type="text",
                button_size="mini",
                plain=True,
            )

            @self._change_agent_button.click
            def _on_change_agent_click():
                self.agent_selector.enable()
                if self.model is not None:
                    self.enable_gui()
                    self.model.shutdown()
                    self.model = None

            self._change_agent_button.hide()

        return self._change_agent_button

    @property
    def deploy_button(self):
        if not hasattr(self, "_deploy_button"):
            self._deploy_button = Button(text="Deploy")
        return self._deploy_button

    @property
    def stop_button(self):
        if not hasattr(self, "_stop_button"):
            self._stop_button = Button(text="Stop", button_type="danger")
            self._stop_button.hide()
        return self._stop_button

    def deploy(
        self,
        model: Optional[str] = None,
        agent_id: Optional[int] = None,
        stop_current: bool = True,
    ) -> None:
        try:
            if self.model is not None and stop_current:
                self.model.shutdown()
                self.enable_gui()
                self.model = None

            if model is None:
                model = self._get_model_input_value()
            else:
                self._set_model_input_value(model)
            if agent_id is None:
                agent_id = self.agent_selector.get_value()
            else:
                self.agent_selector.set_value(agent_id)
            if not model:
                show_dialog(
                    title="Error", description="Model name cannot be empty.", status="error"
                )
                return
            self.model = self._api.nn.deploy(model=model, agent_id=agent_id)
            self.disable_gui()
        except Exception as e:
            show_dialog(
                title="Deployment Error",
                description=f"An error occurred while deploying the model: {str(e)}",
                status="error",
            )
            self.model = None
            self.enable_gui()

    def get_json_data(self) -> dict:
        return {}

    def get_json_state(self) -> dict:
        return {}

    def enable_gui(self):
        self.agent_selector.enable()
        self._enable_model_input_field()
        self.deploy_button.show()
        self.freeze_model_button.hide()
        self.stop_button.hide()
        self.change_agent_button.hide()

    def disable_gui(self):
        self.agent_selector.disable()
        self._disable_model_input_field()
        self.deploy_button.hide()
        self.freeze_model_button.show()
        self.stop_button.show()
        self.change_agent_button.show()

    def save_settings(self, agent_id: Optional[int] = None):
        agent_id = agent_id if agent_id is not None else self.agent_selector.get_value()
        DataJson()[self.widget_id]["settings"] = {"agent_id": agent_id}
        DataJson().send_changes()

    def load_settings(self):
        data = DataJson().get(self.widget_id, {}).get("settings", {})
        agent_id = data.get("agent_id")
        self.update_widgets(agent_id=agent_id)

    def update_widgets(self, agent_id: Optional[int] = None):
        """Set the state of widgets based on the provided parameters."""
        if agent_id is not None:
            self.agent_selector.set_value(agent_id)
            self.agent_selector.set_value(agent_id)

    def _get_deployed_model_info(self) -> Tuple[Dict, Dict]:
        """Returns the deployment information."""
        if self.model is None:
            return {}, {}
        task_info = self._api.task.get_info_by_id(self.model.task_id)
        deploy_info = self.model.get_info()
        return task_info, deploy_info

    def _get_agent_info(self) -> Optional[Dict]:
        """
        Returns GPU information with available and total memory.
        If the model is not deployed or the agent does not have GPU info, returns None.
        """
        try:
            agent_id = self.agent_selector.get_value()
            if agent_id is None:
                return None
            agent_info = self._api.agent.get_info_by_id(agent_id)
            if agent_info is None or not hasattr(agent_info, "gpu_info"):
                return None
            if not isinstance(agent_info.gpu_info, dict):
                return None
            if "device_memory" not in agent_info.gpu_info:
                return None
            return {
                "available": agent_info.gpu_info["device_memory"][0]["available"],
                "total": agent_info.gpu_info["device_memory"][0]["total"],
                "agent_name": agent_info.name,
            }
        except Exception as e:
            logger.warning(f"Failed to get GPU info: {e}")
            return None

from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import supervisely.io.env as sly_env
from supervisely._utils import abs_url, is_development
from supervisely.api.api import Api
from supervisely.sly_logger import logger
from supervisely.solution.base_node import BaseCardNode
from supervisely.solution.components.deploy_model.automation import (
    DeployTasksAutomation,
)
from supervisely.solution.components.deploy_model.gui import DeployModelGUI
from supervisely.solution.components.deploy_model.history import DeployTasksHistory
from supervisely.solution.engine.models import ModelDeployMessage


class DeployModelNode(BaseCardNode):
    TITLE = "Deploy Model"
    DESCRIPTION = "Deploy the selected model for inference."
    ICON = "mdi mdi-memory"
    ICON_COLOR = "#4CAF50"
    ICON_BG_COLOR = "#E8F5E9"

    HISTORY_CLASS = DeployTasksHistory
    GUI_CLASS = DeployModelGUI

    def __init__(
        self,
        tooltip_position: Literal["left", "right"] = "right",
        *args,
        **kwargs,
    ):
        """A node for comparing evaluation reports of different models in Supervisely."""
        self._api = Api.from_env()
        self.team_id = sly_env.team_id()
        self.workspace_id = sly_env.workspace_id()

        self.history = self.HISTORY_CLASS(api=self._api)
        self.gui = self.GUI_CLASS(team_id=self.team_id)
        self._automation = DeployTasksAutomation()

        # --- init node ------------------------------------------------------
        title = kwargs.pop("title", self.TITLE)
        description = kwargs.pop("description", self.DESCRIPTION)
        icon = kwargs.pop("icon", self.ICON)
        icon_color = kwargs.pop("icon_color", self.ICON_COLOR)
        icon_bg_color = kwargs.pop("icon_bg_color", self.ICON_BG_COLOR)
        self.modal_content = self.gui.content
        super().__init__(
            title=title,
            description=description,
            icon=icon,
            icon_color=icon_color,
            icon_bg_color=icon_bg_color,
            *args,
            **kwargs,
        )

        @self.gui.content.on_deploy
        def _on_deploy(model_api):
            self.gui.model = model_api
            self._refresh_node()

        @self.gui.content.on_stop
        def _on_stop():
            self.gui.model = None
            self.gui.content.disconnect()
            self._refresh_node()

        @self.gui.content.select_agent.value_changed
        def _on_agent_selector_change(value: int):
            self.save(agent_id=value)

        @self.click
        def _on_node_click():
            self.gui.modal.show()

        self.modals = [self.history.modal, self.history.logs_modal, self.gui.modal]

    # ------------------------------------------------------------------
    # Node methods -----------------------------------------------------
    # ------------------------------------------------------------------
    def _get_tooltip_buttons(self):
        if not hasattr(self, "tooltip_buttons"):
            self.tooltip_buttons = [self.history.open_modal_button]
        return self.tooltip_buttons

    # ------------------------------------------------------------------
    # Handles ----------------------------------------------------------
    # ------------------------------------------------------------------
    def _get_handles(self):
        return [
            {
                "id": "model_deployed",
                "type": "source",
                "position": "left",
                "connectable": True,
            },
            {
                "id": "best_model",
                "type": "target",
                "position": "right",
                "connectable": True,
            },
        ]

    # ------------------------------------------------------------------
    # Events -----------------------------------------------------------
    # ------------------------------------------------------------------
    def _available_subscribe_methods(self) -> Dict[str, Callable]:
        return {
            "best_model": self._process_incoming_message,
        }

    def _available_publish_methods(self) -> Dict[str, Callable]:
        return {
            "model_deployed": self._send_model_deployed_message,
        }

    def _process_incoming_message(self, message: ModelDeployMessage):
        pass

    def _send_model_deployed_message(self, session_id: int = None) -> ModelDeployMessage:
        return ModelDeployMessage(session_id=session_id)

    # --------------------------------------------------------------------
    # Private methods ----------------------------------------------------
    # --------------------------------------------------------------------
    def save(self, agent_id: Optional[int] = None):
        """Save re-deploy settings."""
        if agent_id is None:
            agent_id = self.gui.agent_selector.get_value()

        self.gui.save_settings(agent_id)

    def load_settings(self):
        """Load re-deploy settings from DataJson."""
        self.gui.load_settings()

    def _refresh_node(self):
        self._refresh_model_info()
        self._refresh_memory_usage_info()

    def _refresh_memory_usage_info(self) -> None:
        """
        Refreshes the GPU memory usage information.
        """
        agent_info = self.gui._get_agent_info()
        if agent_info is not None:
            total = agent_info["total"]
            used = total - agent_info["available"]
            value = f"{used / (1024 ** 3):.2f} GB / {total / (1024 ** 3):.2f} GB"
            self.update_property("Agent", agent_info["agent_name"])
            self.update_property("GPU Memory", value, highlight=True)
        else:
            self.remove_property_by_key("GPU Memory")
            self.remove_property_by_key("Agent")

    def _refresh_model_info(self) -> None:
        """
        Refreshes the deployed model information.
        """
        deploy_info = None
        if self.gui.model is not None:
            task_info, deploy_info = self.gui._get_deployed_model_info()
            self._add_task_to_history(task_info.get("id"), deploy_info)
        self._update_deploy_props(deploy_info)

    def _add_task_to_history(self, task_id: int, deploy_info) -> None:
        existing_task_ids = {task["id"] for task in self.history.get_tasks()}
        if task_id in existing_task_ids:
            return
        task_info = self._api.task.get_info_by_id(task_id)
        if task_info is None:
            return
        task_data = {
            "id": task_info.get("id"),
            "app_name": task_info.get("meta", {}).get("app", {}).get("name"),
            "model_name": task_info.get("meta", {}).get("model_name"),
            "started_at": task_info.get("startedAt"),
            "status": task_info.get("status"),
        }
        self.history.add_task(task_data)
        self._send_model_deployed_message(session_id=task_info.get("id"))
        logger.info(
            f"Model '{deploy_info.get('model_name')}' deployed successfully. Task ID: {task_info.get('id')}"
        )

    def _update_deploy_props(self, deploy_info: Dict[str, Any]) -> None:
        mandatory_keys = ["model_name", "model_source", "hardware"]
        if deploy_info is None or not all(key in deploy_info for key in mandatory_keys):
            self.remove_property_by_key("Model")
            self.remove_property_by_key("Source")
            self.remove_property_by_key("Hardware")
            self.remove_badge_by_key("Deployed")
            return
        self.update_property("Model", deploy_info.get("model_name"), highlight=True)
        self.update_property("Source", deploy_info.get("model_source"))
        self.update_property("Hardware", deploy_info.get("hardware"))
        self.update_badge_by_key(key="Deployed", label="⚡", plain=True)

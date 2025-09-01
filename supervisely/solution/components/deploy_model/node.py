import time
from faulthandler import enable
from pathlib import Path
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

        self.history = self.HISTORY_CLASS()
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

        @self.gui.agent_selector.value_changed
        def _on_agent_selector_change(value: int):
            self.save(agent_id=value)

        @self.gui.deploy_button.click
        def _on_deploy_button_click():
            self.gui.modal.hide()
            self.deploy(model=self.gui._get_model_input_value())

        @self.gui.stop_button.click
        def _on_stop_button_click():
            if self.gui.model is None:
                return
            try:
                self.gui.model.shutdown()
                self.gui.enable_gui()
                self.gui.model = None
                self._update_properties()
            except Exception as e:
                logger.error(
                    f"An error occurred while stopping the model: {repr(e)}", exc_info=True
                )

        @self.click
        def _on_node_click():
            self.gui.modal.show()

        self.modals = [self.history.modal, self.gui.modal]
        self._automation.apply(self._refresh_memory_usage_info, self._automation.REFRESH_GPU_USAGE)

    # ------------------------------------------------------------------
    # Node methods -----------------------------------------------------
    # ------------------------------------------------------------------
    def _get_tooltip_buttons(self):
        if not hasattr(self, "tooltip_buttons"):
            self.tooltip_buttons = [self.history.history_btn]
        return self.tooltip_buttons

    def _update_properties(self, deploy_info: Dict[str, Any] = None):
        """Update node properties with current settings."""

        if self.gui.model is not None:
            self._refresh_memory_usage_info()
            deploy_info = deploy_info or self.gui._get_deployed_model_info()[1]
            self.update_property("Source", deploy_info.get("model_source"))
            self.update_property("Hardware", deploy_info.get("hardware"))
            self.update_property("Model", deploy_info.get("model_name"), highlight=True)
            self.update_property("Status", "Model deployed", highlight=True)
            self.show_automation_badge("Model Deployed")
        else:
            self.hide_automation_badge("Model Deployed")
            self.remove_property_by_key("Status")
            self.remove_property_by_key("Source")
            self.remove_property_by_key("Hardware")
            self.remove_property_by_key("Model")

    # --------------------------------------------------------------------
    # Main methods -------------------------------------------------------
    # --------------------------------------------------------------------
    def deploy(
        self,
        model: Optional[str] = None,
        agent_id: Optional[int] = None,
    ):
        """
        Sends a request to the backend to start the model deployment process.
        """
        try:
            self.gui.disable_gui()
            self.show_in_progress_badge("Deploying Model...")

            self.gui.deploy(model=model, agent_id=agent_id)
            time.sleep(10)  # Wait for the model to be fully deployed

            task_info, deploy_info = self.gui._get_deployed_model_info()
            self._update_properties(deploy_info=deploy_info)

            task_data = {
                "id": task_info.get("id"),
                "app_name": task_info.get("meta", {}).get("app", {}).get("name"),
                "model_name": deploy_info.get("model_name"),
                "started_at": task_info.get("startedAt"),
                "runtime": deploy_info.get("runtime"),
                "hardware": deploy_info.get("hardware"),
                "device": deploy_info.get("device"),
            }
            self.history.add_task(task_data)

            task_id = task_info.get("id")
            self._send_model_deployed_message(session_id=task_id)

            logger.info(f"Model deployed. Task ID: {task_id}")

        except Exception as e:
            logger.error(f"Model deployment failed. {repr(e)}", exc_info=True)
        finally:
            self.hide_in_progress_badge("Model Deployment...")
            self.gui.enable_gui()

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
            self.remove_property_by_key("Agent")
            self.remove_property_by_key("GPU Memory")
            self.remove_property_by_key("Agent")
            self.remove_property_by_key("GPU Memory")

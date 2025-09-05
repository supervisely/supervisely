from typing import Dict, Optional, Tuple

import supervisely.io.env as sly_env
from supervisely.api.api import Api
from supervisely.app.content import DataJson
from supervisely.app.widgets import DeployModel, Dialog, Widget
from supervisely.sly_logger import logger


class DeployModelGUI(Widget):
    MODES = ["pretrained", "custom", "connect"]

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
                # size="tiny",
            )
        return self._settings_modal

    def _init_gui(self):
        return DeployModel(api=self._api, team_id=self.team_id, modes=self.MODES)

    def get_json_data(self) -> dict:
        return {}

    def get_json_state(self) -> dict:
        return {}

    def save_settings(self, agent_id: Optional[int] = None):
        if agent_id is None:
            agent_id = self.content.select_agent.get_value()
        DataJson()[self.widget_id]["settings"] = {"agent_id": agent_id}
        DataJson().send_changes()

    def load_settings(self):
        data = DataJson().get(self.widget_id, {}).get("settings", {})
        agent_id = data.get("agent_id")
        self.update_widgets(agent_id=agent_id)

    def update_widgets(self, agent_id: Optional[int] = None):
        """Set the state of widgets based on the provided parameters."""
        if agent_id is not None:
            self.content.select_agent.set_value(agent_id)

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
            task_id = self.content.model_api.task_id
            task_info = self._api.task.get_info_by_id(task_id)
            if not task_info:
                return None
            agent_id = task_info.get("agentId")
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

from typing import Callable, Dict, Optional

from supervisely.io.env import team_id as env_team_id
from supervisely.sly_logger import logger
from supervisely.solution.components.deploy_model.node import DeployModelNode
from supervisely.solution.engine.models import (
    ComparisonFinishedMessage,
    ModelDeployMessage,
)
from supervisely.solution.nodes.deploy_custom_model.gui import DeployCustomModelGUI
from supervisely.solution.nodes.deploy_custom_model.history import (
    DeployCustomModelHistory,
)
from supervisely.solution.utils import find_agent


class DeployCustomModelNode(DeployModelNode):
    TITLE = "Custom Model"
    DESCRIPTION = "Deploy a custom model for pre-labeling to speed up the labeling process."

    GUI_CLASS = DeployCustomModelGUI
    HISTORY_CLASS = DeployCustomModelHistory

    def _process_incoming_message(self, message: ComparisonFinishedMessage):
        self._deploy(message.train_task_id)

    def _send_model_deployed_message(self, session_id: int = None) -> ModelDeployMessage:
        return ModelDeployMessage(session_id=session_id)

    def _deploy(self, train_task_id: int):
        agent_id = self.gui.content.select_agent.get_value()
        if not agent_id:
            agent_id = find_agent(api=self._api, team_id=env_team_id())
        if not agent_id:
            raise RuntimeError("No available agents found for model deployment.")
        data = {"mode": "custom", "train_task_id": train_task_id}
        key = str(self.gui.content.MODE.CUSTOM)
        self.gui.content.modes[key].update_table()
        self.gui.content.load_from_json(data)
        self.gui.content._deploy()
        self._refresh_node()
        # self.columns_keys = [
        #     ["id"],
        #     ["model_name"],
        #     ["experiment_name"],
        #     ["started_at"],
        #     ["hardware"],
        #     ["device"],
        # ]
        task = {
            "id": train_task_id,
            "model_name": self.gui.content.model_name.get_value(),
            "experiment_name": self.gui.content.experiment_name.get_value(),
            "started_at": self.gui.content.started_at.get_value(),
            "hardware": self.gui.content.hardware.get_value(),
            "device": self.gui.content.device.get_value(),
        }
        self.history.add_task()

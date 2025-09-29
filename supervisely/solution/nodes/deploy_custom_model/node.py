from supervisely.app.widgets import DeployModel
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
        self.gui.content.add_new_experiment_to_table(train_task_id)
        self.gui.content.load_from_json(data)
        self.gui.content._deploy()
        self._refresh_node()

    def _refresh_model_info(self) -> None:
        """
        Refreshes the deployed model information.
        """
        if self.gui.model is not None:
            task_info, deploy_info = self.gui._get_deployed_model_info()
            gui: DeployModel.Custom = self.gui.content.modes[str(self.gui.content.MODE.CUSTOM)]
            experiment_info = gui.experiment_table.get_selected_experiment_info()

            # !TODO: validate whether selected experiment matches deployed model

            tasks = self.history.get_tasks()
            task_ids = {task["id"] for task in tasks}
            if task_info.get("id") not in task_ids:
                task_data = {
                    "id": task_info.get("id"),
                    "model_name": experiment_info.model_name,
                    "experiment_name": experiment_info.experiment_name,
                    "started_at": task_info.get("startedAt"),
                    "runtime": deploy_info.get("runtime"),
                    "hardware": deploy_info.get("hardware"),
                    "device": deploy_info.get("device"),
                }
                self.history.add_task(task_data)
                self.update_property("Model", deploy_info.get("model_name"), highlight=True)
                # self.update_property("Status", "Model deployed", highlight=True)
                self.update_property("Source", deploy_info.get("model_source"))
                self.update_property("Hardware", deploy_info.get("hardware"))
                self.update_badge_by_key(key="Deployed", label="⚡", plain=True)
                self._send_model_deployed_message(session_id=task_info.get("id"))
                logger.info(
                    f"Model '{deploy_info.get('model_name')}' deployed successfully. Task ID: {task_info.get('id')}"
                )
        else:
            self.remove_property_by_key("Model")
            # self.remove_property_by_key("Status")
            self.remove_property_by_key("Source")
            self.remove_property_by_key("Hardware")
            self.remove_badge_by_key("Deployed")

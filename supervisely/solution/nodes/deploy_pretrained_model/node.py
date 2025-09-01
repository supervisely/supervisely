from typing import Callable, Dict

from supervisely.solution.components.deploy_model.node import DeployModelNode
from supervisely.solution.engine.models import ModelDeployMessage
from supervisely.solution.nodes.deploy_pretrained_model.gui import (
    DeployPretrainedModelGUI,
)


class DeployPretrainedModelNode(DeployModelNode):
    TITLE = "Pretrained Model"
    DESCRIPTION = "Deploy a pretrained model for inference."

    GUI_CLASS = DeployPretrainedModelGUI

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
                "id": "deploy_model",
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
            "deploy_model": self._process_incoming_message,
        }

    def _available_publish_methods(self) -> Dict[str, Callable]:
        return {
            "model_deployed": self._send_model_deployed_message,
        }

    def _process_incoming_message(self, message: ModelDeployMessage):
        pass

    def _send_model_deployed_message(self, session_id: int = None) -> ModelDeployMessage:
        return ModelDeployMessage(session_id=session_id)

from typing import Callable, Dict

from supervisely.solution.components.deploy_model.node import DeployModelNode
from supervisely.solution.nodes.deploy_pretrained_model.gui import (
    DeployPretrainedModelGUI,
)


class DeployPretrainedModelNode(DeployModelNode):
    TITLE = "Pretrained Model"
    DESCRIPTION = "Deploy a pretrained model for pre-labeling to speed up the labeling process."

    GUI_CLASS = DeployPretrainedModelGUI

    def _get_handles(self):
        return [
            {
                "id": "model_deployed",
                "type": "source",
                "position": "left",
                "connectable": True,
            }
        ]

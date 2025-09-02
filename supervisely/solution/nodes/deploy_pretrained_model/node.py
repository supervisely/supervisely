from typing import Callable, Dict

from supervisely.solution.components.deploy_model.node import DeployModelNode
from supervisely.solution.nodes.deploy_pretrained_model.gui import (
    DeployPretrainedModelGUI,
)


class DeployPretrainedModelNode(DeployModelNode):
    TITLE = "Pretrained Model"
    DESCRIPTION = "Deploy a pretrained model available in Supervisely."

    GUI_CLASS = DeployPretrainedModelGUI

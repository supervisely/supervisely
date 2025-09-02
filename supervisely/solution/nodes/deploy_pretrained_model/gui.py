from supervisely.solution.components.deploy_model.gui import DeployModelGUI


class DeployPretrainedModelGUI(DeployModelGUI):
    MODES = ["pretrained", "custom", "connect"]

from supervisely.solution.components.deploy_model.gui import DeployModelGUI


class DeployCustomModelGUI(DeployModelGUI):
    MODES = ["custom", "connect"]

# @TODO: not needed mb

from supervisely.app.widgets import Text


class BaseTrainGUIStep:
    def __init__(self):
        self.validator_text = Text("")
        self.validator_text.hide()

    @property
    def widgets_to_disable(self):
        return []

    def validate_step(self):
        return True

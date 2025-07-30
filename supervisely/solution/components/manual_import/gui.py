from __future__ import annotations

from supervisely.app.widgets import Text, Widget
from supervisely.solution.components.base import BaseGUI


# @TODO: will be implemented in the future
class ManualImportGUI(Widget, BaseGUI):
    def __init__(self):
        super().__init__()
        self.content = Text("Use the card link to open Import Wizard", status="text", color="gray")

    def run(self):
        pass

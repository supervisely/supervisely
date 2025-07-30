from __future__ import annotations

from supervisely.app.widgets import Text, Container
from supervisely.solution.components.base import BaseAutomation


# @TODO: will be implemented in the future
class ManualImportAuto(BaseAutomation):
    def __init__(self):
        super().__init__()
        self.apply_btn = None  # not needed
        self.widget = Container([Text("No automation available for Manual Import", status="text")])

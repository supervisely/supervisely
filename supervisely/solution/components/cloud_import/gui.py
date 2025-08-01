from supervisely.app.widgets import CloudImport as CloudImportWidget, Dialog
from supervisely.solution.components.base import BaseGUI


class CloudImportGUI(BaseGUI):
    def __init__(self, project_id: int):
        self._modal = None
        self.widget = CloudImportWidget(project_id)

    @property
    def modal(self) -> Dialog:
        if self._modal is None:
            self._modal = Dialog(
                title="Import from Cloud Storage", content=self.widget, size="tiny"
            )
        return self._modal

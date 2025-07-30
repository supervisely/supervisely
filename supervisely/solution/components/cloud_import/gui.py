from supervisely.app.widgets import CloudImport as CloudImportWidget
from supervisely.solution.components.base import BaseGUI


class CloudImportGUI(CloudImportWidget, BaseGUI):
    def __init__(self, project_id: int):
        CloudImportWidget.__init__(self, project_id)
        BaseGUI.__init__(self)

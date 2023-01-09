from supervisely.app import DataJson
from supervisely.app.widgets import Widget
from supervisely.api.project_api import ProjectInfo
from supervisely.project.project import Project


class ProjectThumbnail(Widget):
    def __init__(self, info: ProjectInfo = None, widget_id: str = None):
        self._info: ProjectInfo = None
        self._description: str = None
        self._url: str = None
        self._set_info(info)

        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self):
        if self._info is None:
            return {
                "id": None,
                "name": None,
                "description": None,
                "url": None,
                "image_preview_url": None,
            }

        return {
            "id": self._info.id,
            "name": self._info.name,
            "description": self._description,
            "url": self._url,
            "image_preview_url": self._info.image_preview_url,
        }

    def get_json_state(self):
        return None

    def _set_info(self, info: ProjectInfo = None):
        if info is None:
            return
        self._info = info
        self._description = f"{self._info.items_count} {self._info.type} in project"
        self._url = Project.get_url(self._info.id)

    def set(self, info: ProjectInfo):
        self._set_info(info)
        self.update_data()
        DataJson().send_changes()

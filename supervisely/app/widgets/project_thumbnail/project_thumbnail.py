from supervisely.app.widgets import Widget
from supervisely.api.project_api import ProjectInfo


class ProjectThumbnail(Widget):
    def __init__(self, info: ProjectInfo):
        self._info = info
        self._description = f"{self._info.items_count} {self._info.type}"
        super().__init__(file_path=__file__)

    def get_json_data(self):
        return {
            "id": self._info.id,
            "name": self._info.name,
            "description": self._description,
            "image_preview_url": self._info.image_preview_url,
        }

    def get_json_state(self):
        return None

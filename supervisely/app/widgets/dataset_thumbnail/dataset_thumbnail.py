from supervisely.api.project_api import ProjectInfo
from supervisely.app.widgets import Widget
from supervisely.api.project_api import ProjectInfo
from supervisely.api.dataset_api import DatasetInfo
from supervisely.project.project import Dataset


class DatasetThumbnail(Widget):
    def __init__(
        self,
        project_info: ProjectInfo,
        dataset_info: DatasetInfo,
        widget_id: str = None,
    ):
        self._project_info = project_info
        self._dataset_info = dataset_info

        self._description = (
            f"{self._dataset_info.items_count} {self._project_info.type} in dataset"
        )
        self._url = Dataset.get_url(
            project_id=self._project_info.id, dataset_id=self._dataset_info.id
        )
        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self):
        return {
            "id": self._dataset_info.id,
            "name": self._dataset_info.name,
            "description": self._description,
            "url": self._url,
            "image_preview_url": self._dataset_info.image_preview_url,
        }

    def get_json_state(self):
        return None

from typing import Dict, Optional, Union

from supervisely.api.dataset_api import DatasetInfo
from supervisely.api.project_api import ProjectInfo
from supervisely.app import DataJson
from supervisely.app.widgets import Widget
from supervisely.project.project import Dataset, Project


class DatasetThumbnail(Widget):
    """DatasetThumbnail widget in Supervisely is a widget that allows to display a thumbnail image that represents supervisely dataset.

    Read about it in `Developer Portal <https://developer.supervisely.com/app-development/widgets/thumbnails/datasetthumbnail>`_
        (including screenshots and examples).

    :param project_info: project info
    :type project_info: Optional[ProjectInfo]
    :param dataset_info: dataset info
    :type dataset_info: Optional[DatasetInfo]
    :param show_project_name: if True, project name will be shown
    :type show_project_name: Optional[bool]
    :param remove_margins: if True, removes margins around the widget
    :type remove_margins: bool, optional
    :param custom_name: custom name for the dataset (if None, actual dataset name will be used)
    :type custom_name: str, optional
    :param widget_id: An identifier of the widget.
    :type widget_id: str, optional

    :Usage example:
    .. code-block:: python

            from supervisely.app.widgets import DatasetThumbnail

            dataset_thumbnail = DatasetThumbnail(
                project_info=project_info,
                dataset_info=dataset_info,
            )
    """

    def __init__(
        self,
        project_info: Optional[ProjectInfo] = None,
        dataset_info: Optional[DatasetInfo] = None,
        show_project_name: Optional[bool] = True,
        remove_margins: bool = False,
        custom_name: str = None,
        widget_id: Optional[str] = None,
    ):
        self._project_info: ProjectInfo = None
        self._dataset_info: DatasetInfo = None
        self._id: int = None
        self._name: str = custom_name
        self._description: str = None
        self._url: str = None
        self._image_preview_url: str = None
        self._show_project_name: bool = show_project_name
        self._project_name: str = None
        self._project_url: str = None
        self._remove_margins: bool = remove_margins
        self._set_info(project_info, dataset_info, show_project_name)

        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self) -> Dict[str, Union[int, str, bool]]:
        """Returns dictionary with widget data, which defines the appearance and behavior of the widget.

        Dictionary contains the following fields:
            - id: dataset id
            - name: dataset name
            - description: dataset description
            - url: dataset url
            - image_preview_url: dataset image preview url
            - show_project_name: if True, project name will be shown
            - project_name: project name
            - project_url: project url

        :return: dictionary with widget data
        :rtype: Dict[str, Union[int, str, bool]]
        """
        return {
            "id": self._id,
            "name": self._name,
            "description": self._description,
            "url": self._url,
            "image_preview_url": self._image_preview_url,
            "show_project_name": self._show_project_name,
            "project_name": self._project_name,
            "project_url": self._project_url,
            "removeMargins": self._remove_margins,
        }

    def get_json_state(self) -> None:
        """DatasetThumbnail widget doesn't have state and returns None."""
        return None

    def _set_info(
        self, project_info: ProjectInfo, dataset_info: DatasetInfo, show_project_name: bool
    ):
        if project_info is None:
            return
        if dataset_info is None:
            return

        self._project_info = project_info
        self._dataset_info = dataset_info
        self._id = dataset_info.id
        if self._name is None:
            self._name = dataset_info.name
        self._description = f"{self._dataset_info.items_count} {self._project_info.type} in dataset"
        self._url = Dataset.get_url(project_id=project_info.id, dataset_id=dataset_info.id)
        self._image_preview_url = dataset_info.image_preview_url
        self._show_project_name = show_project_name
        self._project_name = project_info.name
        self._project_url = Project.get_url(project_info.id)

    def set(
        self,
        project_info: ProjectInfo,
        dataset_info: DatasetInfo,
        show_project_name: Optional[bool] = True,
    ):
        """Sets the data for the widget.

        :param project_info: project info
        :type project_info: ProjectInfo
        :param dataset_info: dataset info
        :type dataset_info: DatasetInfo
        :param show_project_name: if True, project name will be shown
        :type show_project_name: Optional[bool]
        """
        self._set_info(project_info, dataset_info, show_project_name)
        self.update_data()
        DataJson().send_changes()

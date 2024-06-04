from typing import Dict, Literal, Optional, Union

from supervisely.api.api import Api
from supervisely.app import StateJson
from supervisely.app.widgets import Widget
from supervisely.project.project_type import ProjectType


class DestinationProject(Widget):
    """DestinationProject widget in Supervisely provides several options for selecting
    the destination project and dataset when transferring data.

    Read about it in `Developer Portal <https://developer.supervisely.com/app-development/widgets/input/destinationproject>`_
        (including screenshots and examples).

    :param workspace_id: workspace id
    :type workspace_id: int
    :param project_type: project type, one of: images, videos, volumes, point_clouds, point_cloud_episodes, default: images
    :type project_type: Optional[Literal["images", "videos", "volumes", "point_clouds", "point_cloud_episodes"]]
    :param widget_id: An identifier of the widget.
    :type widget_id: str, optional

    :Usage example:
    .. code-block:: python

            from supervisely.app.widgets import DestinationProject

            destination_project = DestinationProject(
                workspace_id=1,
                project_type="images",
            )

    """

    def __init__(
        self,
        workspace_id: int,
        project_type: Optional[
            Literal[
                ProjectType.IMAGES,
                ProjectType.VIDEOS,
                ProjectType.VOLUMES,
                ProjectType.POINT_CLOUDS,
                ProjectType.POINT_CLOUD_EPISODES,
            ]
        ] = ProjectType.IMAGES,
        widget_id: Optional[str] = None,
    ):
        self._api = Api()

        self._project_mode = "new_project"
        self._dataset_mode = "new_dataset"

        self._project_id = None
        self._dataset_id = None

        self._project_name = ""
        self._dataset_name = ""
        self._conflict_resolution = None

        self._use_project_datasets_structure = False

        self._workspace_id = workspace_id
        self._project_type = str(project_type)
        self._changes_handled = False

        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self) -> None:
        """DestinationProject widget in Supervisely has no JSON data,
        the method returns None."""
        return None

    def get_json_state(self) -> Dict[str, Union[str, int, bool]]:
        """Returns dictionary with widget state.

        Dictionary contains the following fields:
            - workspace_id: workspace id
            - project_mode: project mode, one of: new_project, existing_project
            - project_id: project id
            - project_name: project name
            - project_type: project type, one of: images, videos, volumes, point_clouds, point_cloud_episodes
            - dataset_mode: dataset mode, one of new_dataset, existing_dataset
            - dataset_id: dataset id
            - dataset_name: dataset name
            - use_project_datasets_structure: if True, project structure will be used

        :return: dictionary with widget state
        :rtype: Dict[str, Union[str, int, bool]]
        """

        return {
            "workspace_id": str(self._workspace_id),
            "project_mode": self._project_mode,
            "project_id": self._project_id,
            "project_name": self._project_name,
            "project_type": self._project_type,
            "dataset_mode": self._dataset_mode,
            "dataset_id": self._dataset_id,
            "dataset_name": self._dataset_name,
            "use_project_datasets_structure": self._use_project_datasets_structure,
            "conflict_resolution": self._conflict_resolution,
        }

    def get_selected_project_id(self) -> int:
        """Returns selected project id.

        :return: selected project id
        :rtype: int
        """
        return StateJson()[self.widget_id]["project_id"]

    def get_selected_dataset_id(self) -> int:
        """Returns selected dataset id.

        :return: selected dataset id
        :rtype: int
        """
        project_id = StateJson()[self.widget_id]["project_id"]
        dataset_mode = StateJson()[self.widget_id]["dataset_mode"]
        ds_name = StateJson()[self.widget_id]["dataset_id"]
        if project_id is not None and dataset_mode == "existing_dataset" and ds_name is not None:
            ds = self._api.dataset.get_info_by_name(project_id=project_id, name=ds_name)
            return ds.id
        return None

    def get_project_name(self) -> str:
        """Returns selected project name.

        :return: selected project name
        :rtype: str
        """
        return StateJson()[self.widget_id]["project_name"]

    def get_dataset_name(self) -> str:
        """Returns selected dataset name.

        :return: selected dataset name
        :rtype: str
        """
        return StateJson()[self.widget_id]["dataset_name"]

    def use_project_datasets_structure(self) -> bool:
        """Returns True if project structure will be used.

        :return: True if project structure will be used
        :rtype: bool
        """
        return StateJson()[self.widget_id]["use_project_datasets_structure"]

    def get_conflict_resolution(self):
        """Returns selected conflict resolution method.

        :return: selected conflict resolution method.
        :rtype: str
        """
        return StateJson()[self.widget_id]["conflict_resolution"]

from typing import Dict, List, Union

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from supervisely.app import StateJson, DataJson
from supervisely.app.widgets import Widget
from supervisely.api.api import Api


class TeamFilesSelector(Widget):
    """File/folder picker for Team Files; supports single/multiple selection and optional file type filter."""

    def __init__(
        self,
        team_id: int,
        multiple_selection: bool = False,
        max_height: Union[int, str] = 500,
        selection_file_type: Literal["folder", "file"] = None,
        hide_header: bool = True,
        hide_empty_table: bool = True,
        additional_fields: List[
            Literal["id", "createdAt", "updatedAt", "type", "size", "mimeType"]
        ] = [],
        widget_id: str = None,
        initial_folder: str = None,
        show_cloud_storage: bool = False,
        show_agent_storage: bool = False,
        disable_bucket_selection: bool = True,
    ):
        """
        :param team_id: Team ID for Team Files.
        :type team_id: int
        :param multiple_selection: If True, allow multiple selection.
        :type multiple_selection: bool
        :param max_height: Max height in pixels or CSS units.
        :type max_height: int or str
        :param selection_file_type: Filter: "folder" or "file".
        :type selection_file_type: Literal["folder", "file"], optional
        :param hide_header: If True, hide table header.
        :type hide_header: bool
        :param hide_empty_table: If True, hide when empty.
        :type hide_empty_table: bool
        :param additional_fields: Extra columns to show.
        :type additional_fields: List[Literal["id", "createdAt", "updatedAt", "type", "size", "mimeType"]]
        :param widget_id: Unique widget identifier.
        :type widget_id: str, optional
        :param initial_folder: Initial folder path.
        :type initial_folder: str, optional
        :param show_cloud_storage: If True, show "Cloud Storages" option in the root.
        :type show_cloud_storage: bool
        :param show_agent_storage: If True, show agent storages option in the root.
        :type show_agent_storage: bool
        :param disable_bucket_selection: If True, disables bucket selection in the cloud storage.
        :type disable_bucket_selection: bool

        :raises ValueError: If additional_fields contains invalid field names, or if
            max_height is not an int or str.
        """
        self._api = Api()
        self._team_id = team_id

        self._multiple_selection = multiple_selection
        self._max_height = self._normalize_max_height(max_height)
        self._selection_file_type = selection_file_type
        self._hide_header = hide_header
        self._hide_empty_table = hide_empty_table
        self._show_cloud_storage = show_cloud_storage
        self._show_agent_storage = show_agent_storage
        self._disable_bucket_selection = disable_bucket_selection

        available_fields = ["id", "createdAt", "updatedAt", "type", "size", "mimeType"]
        for field in additional_fields:
            if field not in available_fields:
                raise ValueError(
                    f'"{field}" is not a valid field. Available fields: {available_fields}.'
                )

        self._additional_fields = additional_fields
        self._selected = []
        self._initial_folder = initial_folder or "/"

        super().__init__(widget_id=widget_id, file_path=__file__)

    @staticmethod
    def _normalize_max_height(max_height: Union[int, str]) -> str:
        if type(max_height) == int:
            return f"{max_height}px"
        if type(max_height) == str:
            return max_height
        raise ValueError(f"max_height must be int or str, got {type(max_height)}")

    def get_json_data(self) -> Dict:
        return {
            "teamId": self._team_id,
            "options": {
                "multipleSelection": self._multiple_selection,
                "maxHeight": self._max_height,
                "selectionFileType": self._selection_file_type,
                "hideHeader": self._hide_header,
                "hideEmptyTable": self._hide_empty_table,
                "additionalFields": self._additional_fields,
                "showCloudStorage": self._show_cloud_storage,
                "showAgentStorage": self._show_agent_storage,
                "disableBucketSelection": self._disable_bucket_selection,
                "initialFolder": {"path": self._initial_folder},
            },
        }

    def get_json_state(self) -> Dict:
        return {"selected": self._selected}

    def get_selected_items(self) -> List[dict]:
        selected = StateJson()[self.widget_id]["selected"]
        return [item for item in selected]

    def get_selected_paths(self) -> List[str]:
        selected = StateJson()[self.widget_id]["selected"]
        return [item["path"] for item in selected]

    def set_team_id(self, team_id: int):
        if type(team_id) != int:
            raise ValueError(f"team_id must be int, got {type(team_id)}")
        if team_id == self._team_id:
            return
        DataJson()[self.widget_id]["teamId"] = None
        DataJson().send_changes()
        self._team_id = team_id
        DataJson()[self.widget_id]["teamId"] = self._team_id
        DataJson().send_changes()

from typing import Dict, List, Optional, Union

from supervisely.api.api import Api
from supervisely.app import DataJson, StateJson
from supervisely.app.widgets import Widget


class FileStorageUpload(Widget):
    """FileStorageUpload is a widget in Supervisely's web interface that allows users to
    upload files directly to Team files by given path.

    Read about it in `Developer Portal <https://developer.supervisely.com/app-development/widgets/controls/filestorageupload>`_
        (including screenshots and examples).

    :param team_id: id of the team to upload files to
    :type team_id: int
    :param path: path to upload files to
    :type path: str
    :param change_name_if_conflict: if True, will change the name of the file if it already exists
    :type change_name_if_conflict: Optional[bool]
    :param widget_id: id of the widget
    :type widget_id: Optional[str]

    """

    def __init__(
        self,
        team_id: int,
        path: str,
        change_name_if_conflict: Optional[bool] = False,
        widget_id: Optional[str] = None,
    ):
        self._api = Api()
        self._team_id = team_id
        self._change_name_if_conflict = change_name_if_conflict
        self._path = self._get_path(path)
        self._files = []

        super().__init__(widget_id=widget_id, file_path=__file__)

    def _set_path(self, path: str):
        self._path = self._get_path(path)
        DataJson()[self.widget_id]["path"] = self._path
        DataJson().send_changes()

    def _get_path(self, path: str):
        if self._change_name_if_conflict is True:
            path = f"/{path}" if not path.startswith("/") else path
            return self._api.file.get_free_dir_name(self._team_id, path)
        return path

    def get_json_data(self) -> Dict[str, Union[int, str]]:
        """Returns dictionary with widget data, which defines the appearance and behavior of the widget.
        Dictionary contains the following fields:
            - team_id: id of the team to upload files to
            - path: path to upload files to

        :return: dictionary with widget data
        :rtype: Dict[str, Union[int, str]]
        """
        return {"team_id": self._team_id, "path": self._path}

    def get_json_state(self) -> Dict[str, List[str]]:
        """Returns dictionary with widget state.

        The dictionary contains the following fields:
            - files: list of uploaded files

        :return: dictionary with widget state
        :rtype: Dict[str, List[str]]
        """
        return {"files": self._files}

    @property
    def path(self) -> str:
        """Returns path to upload files to.

        :return: path to upload files to
        :rtype: str
        """
        return DataJson()[self.widget_id]["path"]

    @path.setter
    def path(self, path: str) -> None:
        """Sets path to upload files to.

        Note: same as ``set_path`` method.

        :param path: path to upload files to
        :type path: str
        """
        self._set_path(path)

    def set_path(self, path: str) -> None:
        """Sets path to upload files to.

        Note: same as ``path`` property.

        :param path: path to upload files to
        :type path: str
        """
        self._set_path(path)

    def get_team_id(self) -> int:
        """Returns id of the team to upload files to.

        :return: id of the team to upload files to
        :rtype: int
        """
        return self._team_id

    def get_uploaded_paths(self) -> Union[List[str], None]:
        """Returns list of uploaded files paths.

        :return: list of uploaded files paths
        :rtype: Union[List[str], None]
        """
        response = StateJson()[self.widget_id]["files"]
        if len(response) == 0 or response is None:
            return []
        uploaded_files = response["uploadedFiles"]
        if len(uploaded_files) == 0:
            return None
        paths = [file["path"] for file in uploaded_files]
        return paths

    # def get_uploaded_info(self) -> Union[List[sly.api.file_api.FileInfo], None]:
    #     response = StateJson()[self.widget_id]["files"]
    #     uploaded_files = response["uploadedFiles"]
    #     if len(uploaded_files) == 0:
    #         return None

    #     files_infos = []
    #     for item in uploaded_files:
    #         TODO: convert from json instead of api queries
    #         files_infos.append(self._api.file.get_info_by_id(item["id"]))
    #     return files_infos

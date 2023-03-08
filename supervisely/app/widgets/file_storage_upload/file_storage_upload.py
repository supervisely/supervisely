from typing import List, Optional, Union

import supervisely as sly
from supervisely.api.api import Api
from supervisely.app import DataJson, StateJson
from supervisely.app.widgets import Widget


class FileStorageUpload(Widget):
    def __init__(
        self,
        folder_name: str = None,
        change_name_if_conflict: Optional[bool] = False,
        widget_id: str = None,
    ):
        self._api = Api()
        self._team_id = sly.env.team_id()
        self._folder_name = None
        self._files = []
        self._change_name_if_conflict = change_name_if_conflict
        if folder_name is not None:
            path = folder_name
        else:
            task_id = sly.env.task_id()
            path = f"/import-from-widget/{task_id}/"

        self._folder_name = self._get_folder_name(path)

        super().__init__(widget_id=widget_id, file_path=__file__)

    def _set_folder_name(self, path: str):
        self._folder_name = self._get_folder_name(path)
        DataJson()[self.widget_id]["folderName"] = self._folder_name
        DataJson().send_changes()

    def _get_folder_name(self, path):
        if self._change_name_if_conflict is True:
            return self._api.file.get_free_dir_name(self._team_id, path)
        return path

    def get_json_data(self):
        return {"folderName": self._folder_name}

    def get_json_state(self):
        return {"files": self._files}

    @property
    def folder_path(self):
        return DataJson()[self.widget_id]["folderName"]

    @folder_path.setter
    def folder_path(self, path: str):
        self._set_folder_name(path)

    def set_folder_name(self, path: str):
        self._set_folder_name(path)

    def get_uploaded_info(self) -> Union[List[sly.api.file_api.FileInfo], None]:
        response = StateJson()[self.widget_id]["files"]
        uploaded_files = response["uploadedFiles"]
        if len(uploaded_files) == 0:
            return None

        files_infos = []
        for item in uploaded_files:
            files_infos.append(self._api.file.get_info_by_id(item["id"]))
        return files_infos

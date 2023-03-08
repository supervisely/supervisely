from supervisely.api.api import Api
from supervisely.app import DataJson
from supervisely.app.widgets import Widget


class FileStorageUpload(Widget):
    def __init__(
        self,
        team_id: int,
        folder_name: str = None,
        widget_id: str = None,
    ):
        self._api = Api()
        self._team_id = team_id
        self._folder_name = None
        if folder_name is not None:
            self._folder_name = self._api.file.get_free_dir_name(self._team_id, dir_path=folder_name)
        else:
            path = "/import-from-widget/"
            self._folder_name = self._api.file.get_free_dir_name(self._team_id, dir_path=path)

        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self):
        return {
            "teamId": self._team_id,
            "folderName": self._folder_name,
        }

    def get_json_state(self):
        return None


    def set_folder_name(self, value):
        self._folder_name = self._api.file.get_free_dir_name(self._team_id, dir_path=value)
        DataJson()[self.widget_id]["folderName"] = self._folder_name
        DataJson().send_changes()

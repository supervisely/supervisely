from typing import Any, Dict, Optional

from supervisely._utils import abs_url
from supervisely.api.file_api import FileInfo
from supervisely.app import DataJson
from supervisely.app.widgets import Widget


class FileThumbnail(Widget):
    """FileThumbnail widget in Supervisely displays an icon, link, and path to the file in Team Files.

    Read about it in `Developer Portal <https://developer.supervisely.com/app-development/widgets/thumbnails/filethumbnail`_
        (including screenshots and examples).

    """

    def __init__(self, info: Optional[FileInfo] = None, widget_id: Optional[str] = None):
        self._id: int = None
        self._info: FileInfo = None
        self._description: str = None
        self._url: str = None
        self._set_info(info)

        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self) -> Dict[str, Any]:
        """Returns dictionary with widget data, which defines the appearance and behavior of the widget.
        Dictionary contains the following fields:
            - id: id of the file
            - name: name of the file
            - description: path to the file
            - url: url to the file
            - icon: dictionary with following fields:
                - className: class name of the icon
                - color: color of the icon
                - bgColor: background color of the icon

        :return: dictionary with widget data
        :rtype: Dict[str, Any]
        """
        return {
            "id": self._id,
            "name": "File in Team files",
            "description": self._description,
            "url": self._url,
            "icon": {
                "className": "zmdi zmdi-file-text",
                "color": "#4977ff",
                "bgColor": "#ddf2ff",
            },
        }

    def get_json_state(self) -> None:
        """FileThumbnail widget does not have state, the method returns None."""
        return None

    def _set_info(self, info: FileInfo = None):
        if info is None:
            self._id: int = None
            self._info: FileInfo = None
            self._description: str = None
            self._url: str = None
            return
        self._id = info.id
        self._info = info
        self._description = info.path
        self._url = abs_url(f"/files/{info.id}")

    def set(self, info: FileInfo) -> None:
        """Sets the file to display.

        :param info: file info
        :type info: FileInfo
        """
        self._set_info(info)
        self.update_data()
        DataJson().send_changes()

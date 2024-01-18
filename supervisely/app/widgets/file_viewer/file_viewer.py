from typing import List, Literal, Union

from supervisely.api.api import Api
from supervisely.app import DataJson, StateJson
from supervisely.app.widgets import Widget


class FileViewer(Widget):
    class Routes:
        VALUE_CHANGED = "value_changed"
        PATH_CHANGED = "path_changed"

    def __init__(
        self,
        files_list: List[dict],
        selection_type: Literal[None, "file", "folder"] = None,
        extended_selection: bool = False,
        widget_id: str = None,
    ):
        """
        Widget for selecting files and folders in convenient GUI

        :param files_list: list of dicts with files and folders info
        :type files_list: List[dict]
        :param selection_type: type of selection, defaults to None
        :type selection_type: Literal[None, "file", "folder"], optional
        :param extended_selection: If True method 'get_selected_items()' returns an array of objects { path: '', type: 'file or folder' } instead of array with paths
        :type extended_selection: bool, optional
        :param widget_id: widget id, defaults to None
        :type widget_id: str, optional
        :raises ValueError: if files_list is not a list
        :raises ValueError: if files_list contains non-dict element
        :raises KeyError: if files_list contains dict without 'path' key

        """
        self._api = Api()

        if type(files_list) is not list:
            raise ValueError(
                f"Argument 'files_list' must be a list, got '{type(files_list)}' instead"
            )

        for idx, f in enumerate(files_list):
            if type(f) is not dict:
                raise ValueError(
                    f"All elements in 'files_list' must be dicts, index: {idx}, element: {f}"
                )
            if "path" not in f:
                raise KeyError(
                    f"One of the files dicts missing required key 'path', index: {idx}, element: {f}"
                )

        self._files_list = files_list
        self._selection_type = selection_type
        self._extended_selection = extended_selection
        self._selected = []
        self._viewer_path = ""
        self._changes_handled = False
        self._viewer_path_changed = False
        self._loading = False

        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self):
        return {
            "list": self._files_list,
            "loading": self._loading,
            "options": {
                "flatMode": True,
                "extendedSelection": self._extended_selection,
                "selectionFileType": self._selection_type,
            },
        }

    def get_json_state(self):
        return {
            "viewer_path": self._viewer_path,
            "selected": self._selected,
        }

    @property
    def loading(self):
        return self._loading

    @loading.setter
    def loading(self, value):
        self._loading = value
        DataJson()[self.widget_id]["loading"] = self._loading
        DataJson().send_changes()

    def get_selected_items(self):
        return StateJson()[self.widget_id]["selected"]

    def get_current_path(self):
        return StateJson()[self.widget_id]["viewer_path"]

    def update_file_tree(self, files_list):
        self._files_list = files_list
        DataJson()[self.widget_id]["list"] = self._files_list
        DataJson().send_changes()

    def path_changed(self, func):
        route_path = self.get_route_path(FileViewer.Routes.PATH_CHANGED)
        server = self._sly_app.get_server()
        self._viewer_path_changed = True

        @server.post(route_path)
        def _path_changed():
            self.loading = True
            res = self.get_current_path()
            func(res)
            self.loading = False

        return _path_changed

    def value_changed(self, func):
        # TODO: throttle
        route_path = self.get_route_path(FileViewer.Routes.VALUE_CHANGED)
        server = self._sly_app.get_server()
        self._changes_handled = True

        @server.post(route_path)
        def _value_changed():
            res = self.get_selected_items()
            func(res)

        return _value_changed

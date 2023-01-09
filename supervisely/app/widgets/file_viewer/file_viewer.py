try:
    from typing import Literal, List
except ImportError:
    from typing_extensions import Literal

from supervisely.app import StateJson, DataJson
from supervisely.app.widgets import Widget
from supervisely.api.api import Api
from supervisely.project.project_type import ProjectType


class FileViewer(Widget):
    class Routes:
        VALUE_CHANGED = "value_changed"

    def __init__(
        self,
        files_list: List[dict],
        widget_id: str = None,
    ):
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
        self._selected = []
        self._changes_handled = False

        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self):
        return {"list": self._files_list}

    def get_json_state(self):
        return {"selected": self._selected}

    def get_selected_items(self):
        return StateJson()[self.widget_id]["selected"]

    def update_file_tree(self, files_list):
        self._files_list = files_list
        DataJson()[self.widget_id]["list"] = self._files_list
        DataJson().send_changes()

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

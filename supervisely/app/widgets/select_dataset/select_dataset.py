from typing import Dict

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from supervisely.app import StateJson
from supervisely.app.widgets import Widget, SelectProject, generate_id, Checkbox
from supervisely.api.api import Api
from supervisely.sly_logger import logger
from supervisely.app.widgets.select_sly_utils import _get_int_or_env


class SelectDataset(Widget):
    class Routes:
        VALUE_CHANGED = "value_changed"

    def __init__(
        self,
        default_id: int = None,
        project_id: int = None,
        multiselect: bool = False,
        compact: bool = False,
        show_label: bool = True,
        size: Literal["large", "small", "mini"] = None,
        widget_id: str = None,
    ):
        self._api = Api()
        self._default_id = default_id
        self._project_id = project_id
        self._multiselect = multiselect
        self._compact = compact
        self._show_label = show_label
        self._size = size
        self._team_selector = None
        self._all_datasets_checkbox = None
        self._changes_handled = False

        self._default_id = _get_int_or_env(self._default_id, "modal.state.slyDatasetId")
        if self._default_id is not None:
            info = self._api.dataset.get_info_by_id(self._default_id, raise_error=True)
            self._project_id = info.project_id
        self._project_id = _get_int_or_env(self._project_id, "modal.state.slyProjectId")

        if compact is True:
            if self._project_id is None:
                raise ValueError(
                    '"project_id" have to be passed as argument or "compact" has to be False'
                )
        else:
            # if self._show_label is False:
            #     logger.warn(
            #         "show_label can not be false if compact is True and default_id / project_id are not defined"
            #     )
            self._show_label = True
            self._project_selector = SelectProject(
                default_id=self._project_id,
                show_label=True,
                size=self._size,
                widget_id=generate_id(),
            )

        if self._multiselect is True:
            self._all_datasets_checkbox = Checkbox(
                "Select all datasets", checked=False, widget_id=generate_id()
            )

        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self) -> Dict:
        res = {}
        res["projectId"] = self._project_id
        res["options"] = {
            "showLabel": self._show_label,
            "compact": self._compact,
            "filterable": True,
            "valueProperty": "id",
            "multiple": self._multiselect,
        }
        if self._size is not None:
            res["options"]["size"] = self._size
        return res

    def get_json_state(self) -> Dict:
        return {
            "datasets": [self._default_id],
        }

    def get_selected_id(self):
        if self._multiselect is True:
            raise ValueError(
                "Multiselect is enabled. Use another method 'get_selected_ids' insted of 'get_selected_id'"
            )
        return StateJson()[self.widget_id]["datasets"]

    def get_selected_ids(self):
        if self._multiselect is False:
            raise ValueError(
                "Multiselect is disabled. Use another method 'get_selected_id' insted of 'get_selected_ids'"
            )
        if self._all_datasets_checkbox.is_checked():
            if self._compact is True:
                project_id = self._project_id
            else:
                project_id = self._project_selector.get_selected_id()
            datasets = self._api.dataset.get_list(project_id)
            ids = [ds.id for ds in datasets]
            return ids
        else:
            return StateJson()[self.widget_id]["datasets"]

    def value_changed(self, func):
        route_path = self.get_route_path(SelectDataset.Routes.VALUE_CHANGED)
        server = self._sly_app.get_server()
        self._changes_handled = True

        @server.post(route_path)
        def _click():
            if self._multiselect is True:
                value = self.get_selected_ids()
                if value == "":
                    value = None
                func(value)
            else:
                value = self.get_selected_id()
                if value == "":
                    value = None
                func(value)

        return _click

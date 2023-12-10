from typing import Dict, Optional, List

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from supervisely.project.project_type import ProjectType
from supervisely.app import DataJson, StateJson
from supervisely.app.widgets import Widget, SelectProject, generate_id, Checkbox, Empty
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
        disabled: Optional[bool] = False,
        widget_id: str = None,
        select_all_datasets: bool = False,
        allowed_project_types: List[ProjectType] = [],
    ):
        self._api = Api()
        self._default_id = default_id
        self._project_id = project_id
        self._multiselect = multiselect
        self._compact = compact
        self._show_label = show_label
        self._size = size
        self._team_selector = None
        self._all_datasets_checkbox = Empty()
        self._project_selector = Empty()
        self._project_types = allowed_project_types
        self._changes_handled = False
        self._disabled = disabled

        self._default_id = _get_int_or_env(self._default_id, "modal.state.slyDatasetId")
        if self._default_id is not None:
            info = self._api.dataset.get_info_by_id(self._default_id, raise_error=True)
            self._project_id = info.project_id
        self._project_id = _get_int_or_env(self._project_id, "modal.state.slyProjectId")

        # NOW PROJECT CAN BE SET LATER WITH SET_PROJECT_ID METHOD
        # if compact is True:
        #     if self._project_id is None:
        #         raise ValueError(
        #             '"project_id" have to be passed as argument or "compact" has to be False'
        #         )
        # else:
        self._show_label = True
        self._project_selector = SelectProject(
            default_id=self._project_id,
            show_label=True,
            size=self._size,
            allowed_types=allowed_project_types,
            widget_id=generate_id(),
        )
        if self._disabled is True:
            self._project_selector.disable()

        if self._multiselect is True:
            self._all_datasets_checkbox = Checkbox(
                "Select all datasets", checked=select_all_datasets, widget_id=generate_id()
            )

        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self) -> Dict:
        res = {}
        res["disabled"] = self._disabled
        if self._compact is True:
            res["projectId"] = self._project_id
        res["options"] = {
            "showLabel": self._show_label,
            "compact": self._compact,
            "filterable": True,
            "valueProperty": "id",
            "multiple": self._multiselect,
            "flat": True,
            "availableProjectTypes": [ptype.value for ptype in self._project_types],
        }
        if self._size is not None:
            res["options"]["size"] = self._size
        return res

    def get_json_state(self) -> Dict:
        res = {"datasets": [self._default_id]}
        if self._compact is False:
            res["projectId"] = self._project_id
        return res

    def get_selected_project_id(self):
        if self._compact is True:
            raise ValueError(
                '"project_id" have to be passed as argument or "compact" has to be False'
            )
        return self._project_selector.get_selected_id()

    def get_selected_id(self):
        if self._multiselect is True:
            raise ValueError(
                "Multiselect is enabled. Use another method 'get_selected_ids' instead of 'get_selected_id'"
            )
        return StateJson()[self.widget_id]["datasets"]

    def get_selected_ids(self):
        if self._multiselect is False:
            raise ValueError(
                "Multiselect is disabled. Use another method 'get_selected_id' instead of 'get_selected_ids'"
            )
        if self._all_datasets_checkbox.is_checked():
            if self._compact is True:
                project_id = self._project_id
            else:
                project_id = self._project_selector.get_selected_id()
            if project_id is None:
                return [None]
            datasets = self._api.dataset.get_list(project_id)
            ids = [ds.id for ds in datasets]
            return ids
        else:
            return StateJson()[self.widget_id]["datasets"]

    def set_project_id(self, id: int):
        self._project_id = id
        if self._compact is True:
            DataJson()[self.widget_id]["projectId"] = self._project_id
            DataJson().send_changes()
        else:
            StateJson()[self.widget_id]["projectId"] = self._project_id
            StateJson().send_changes()

    def set_dataset_id(self, id: int):
        if self._multiselect is True:
            raise ValueError(
                "Multiselect is enabled. Use another method 'set_dataset_ids' instead of 'set_dataset_id'"
            )

        self._default_id = [id]
        StateJson()[self.widget_id]["datasets"] = self._default_id
        StateJson().send_changes()

    def set_dataset_ids(self, ids: List[int]):
        if self._multiselect is False:
            raise ValueError(
                "Multiselect is disabled. Use another method 'set_dataset_id' instead of 'set_dataset_ids'"
            )
        self._default_id = ids
        StateJson()[self.widget_id]["datasets"] = self._default_id
        StateJson().send_changes()

    def value_changed(self, func):
        route_path = self.get_route_path(SelectDataset.Routes.VALUE_CHANGED)
        server = self._sly_app.get_server()
        self._changes_handled = True

        def _process():
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

        if self._multiselect is True:

            @self._all_datasets_checkbox.value_changed
            def _select_all_datasets(is_checked):
                _process()

        @server.post(route_path)
        def _click():
            _process()

        return _click

    def disable(self):
        self._project_selector.disable()
        self._all_datasets_checkbox.disable()
        self._disabled = True
        DataJson()[self.widget_id]["disabled"] = self._disabled
        DataJson().send_changes()

    def enable(self):
        self._all_datasets_checkbox.enable()
        self._project_selector.enable()
        self._disabled = False
        DataJson()[self.widget_id]["disabled"] = self._disabled
        DataJson().send_changes()

    @property
    def is_disabled(self) -> bool:
        return self._disabled

    @is_disabled.setter
    def is_disabled(self, value: int):
        self._disabled = value
        DataJson()[self.widget_id]["disabled"] = self._disabled
        DataJson().send_changes()

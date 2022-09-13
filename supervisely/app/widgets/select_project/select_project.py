import os
from typing import Dict, List

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from supervisely.app.widgets import (
    Widget,
    SelectWorkspace,
    generate_id,
)
from supervisely.api.api import Api
from supervisely.project.project_type import ProjectType

import os


def _get_int_env(env_key: str) -> int:
    res = os.environ.get(env_key)
    if res is not None:
        res = int(res)
    return res


def _get_int_value_or_env(value: int, env_key: str) -> int:
    if value is not None:
        return int(value)
    return _get_int_env(env_key)


class SelectProject(Widget):
    def __init__(
        self,
        workspace_id: int = None,
        compact: bool = False,
        project_types: List[ProjectType] = [],
        show_label: bool = True,
        size: Literal["large", "small", "mini"] = None,
        widget_id: str = None,
    ):
        self._api = Api()
        self._ws_id = workspace_id
        self._compact = compact
        self._project_types = project_types
        self._show_label = show_label
        self._size = size
        self._tw_selector = None

        self._default_id = _get_int_env("modal.state.slyProjectId")
        self._ws_id = _get_int_value_or_env(self._ws_id, "context.workspaceId")
        if compact is True:
            if self._ws_id is None:
                raise ValueError(
                    '"workspace_id" have to be passed as argument or "compact" has to be False'
                )
        else:
            self._show_label = True
            self._tw_selector = SelectWorkspace(
                compact=False, show_label=True, widget_id=generate_id()
            )

            # if self._workspace_id is None:
            #     self._team_id = _get_int_env("context.workspaceId")
            # else:
            #     ws_info = self._api.workspace.get_info_by_id(self._workspace_id)
            #     if ws_info is None:
            #         raise KeyError(
            #             f"Workspace with id={self._workspace_id} not found in your account"
            #         )
            #     self._team_id = ws_info.team_id

        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self) -> Dict:
        res = {}
        res["workspaceId"] = self._ws_id
        res["options"] = {
            "availableTypes": [ptype.value for ptype in self._project_types],
            "showLabel": self._show_label,
            "compact": self._compact,
            "filterable": True,
        }
        if self._size is not None:
            res["options"]["size"] = self._size
        return res

    def get_json_state(self) -> Dict:
        return {
            "projectId": self._default_id,
        }

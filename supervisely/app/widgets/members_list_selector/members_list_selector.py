from typing import List, Optional

from supervisely.api.user_api import UserInfo
from supervisely.app.content import StateJson
from supervisely.app.widgets import Widget


class MembersListSelector(Widget):
    class Routes:
        CHECKBOX_CHANGED = "checkbox_cb"

    def __init__(
        self,
        users: Optional[List[UserInfo]] = [],
        multiple: Optional[bool] = False,
        widget_id: Optional[str] = None,
    ):
        self._users = users
        self._multiple = multiple
        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self):
        return {
            "users": [
                {
                    "login": f"<b>{user.login}</b>",
                    "role": user.role,
                }
                for user in self._users
            ]
        }

    def get_json_state(self):
        return {"selected": [False for _ in self._users]}

    def set(self, users: List[UserInfo]):
        selected_members = [name for name in self.get_selected_members()]
        self._users = users
        StateJson()[self.widget_id]["selected"] = [
            user.login in selected_members for user in self._users
        ]
        self.update_data()
        StateJson().send_changes()

    def get_selected_members(self):
        selected = StateJson()[self.widget_id]["selected"]
        return [user for user, is_selected in zip(self._users, selected) if is_selected]

    def select_all(self):
        StateJson()[self.widget_id]["selected"] = [True for _ in self._users]
        StateJson().send_changes()

    def deselect_all(self):
        StateJson()[self.widget_id]["selected"] = [False for _ in self._users]
        StateJson().send_changes()

    def select(self, names: List[str]):
        selected = [user.login in names for user in self._users]
        StateJson()[self.widget_id]["selected"] = selected
        StateJson().send_changes()

    def deselect(self, names: List[str]):
        selected = StateJson()[self.widget_id]["selected"]
        for idx, user in enumerate(self._users):
            if user.name in names:
                selected[idx] = False
        StateJson()[self.widget_id]["selected"] = selected
        StateJson().send_changes()

    def set_multiple(self, value: bool):
        self._multiple = value

    def get_all_members(self):
        return self._users

    def selection_changed(self, func):
        route_path = self.get_route_path(MembersListSelector.Routes.CHECKBOX_CHANGED)
        server = self._sly_app.get_server()
        self._checkboxes_handled = True

        @server.post(route_path)
        def _click():
            selected = self.get_selected_members()
            func(selected)

        return _click

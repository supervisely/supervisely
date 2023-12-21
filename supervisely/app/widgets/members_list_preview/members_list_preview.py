from typing import List, Dict, Optional

from supervisely.app.widgets import Widget
from supervisely.app.content import StateJson
from supervisely.api.user_api import UserInfo


class MembersListPreview(Widget):
    def __init__(
        self,
        users: Optional[List[UserInfo]] = [],
        max_width: int = 300,
        empty_text: str = None,
        widget_id: int = None,
    ):
        self._users = users
        self._max_width = self._get_max_width(max_width)
        self._empty_text = empty_text
        super().__init__(widget_id=widget_id, file_path=__file__)

    def _get_max_width(self, value):
        if value < 150:
            value = 150
        return f"{value}px"

    def get_json_data(self):
        return {
            "maxWidth": self._max_width,
        }

    def get_json_state(self) -> Dict:
        return {
            "users": [
                {
                    "login": f"<b>{user.login}</b>",
                    "role": user.role,
                }
                for user in self._users
            ]
        }

    def set(self, users: List[UserInfo]):
        self._users = users
        StateJson()[self.widget_id] = self.get_json_state()
        StateJson().send_changes()

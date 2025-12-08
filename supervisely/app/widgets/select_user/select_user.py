from typing import Callable, List, Optional, Union

from supervisely.api.user_api import UserInfo
from supervisely.app import DataJson, StateJson
from supervisely.app.widgets import Widget

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


class SelectUser(Widget):
    """
    SelectUser is a dropdown widget for selecting users from a team.
    Extends the Select widget with user-specific functionality.

    :param users: Initial list of UserInfo instances
    :type users: Optional[List[UserInfo]]
    :param team_id: Team ID to fetch users from
    :type team_id: Optional[int]
    :param roles: List of allowed user roles to filter by (e.g., ['admin', 'developer'])
    :type roles: Optional[List[str]]
    :param filterable: Enable search/filter functionality in dropdown
    :type filterable: Optional[bool]
    :param placeholder: Placeholder text when no user is selected
    :type placeholder: Optional[str]
    :param size: Size of the select dropdown
    :type size: Optional[Literal["large", "small", "mini"]]
    :param multiple: Enable multiple selection
    :type multiple: bool
    :param widget_id: Unique widget identifier
    :type widget_id: Optional[str]

    :Usage example:

     .. code-block:: python

        import supervisely as sly
        from supervisely.app.widgets import SelectUser

        # Initialize with team_id and filter by roles
        select_user = SelectUser(
            team_id=123,
            roles=['admin', 'developer'],
            multiple=True
        )

        # Or initialize empty and set users later
        select_user = SelectUser(roles=['annotator', 'reviewer'])
        select_user.set_users(user_list)

        # Handle selection changes
        @select_user.value_changed
        def on_user_selected(users):
            if isinstance(users, list):
                print(f"Selected users: {[u.login for u in users]}")
            else:
                print(f"Selected user: {users.login}")
    """

    class Routes:
        VALUE_CHANGED = "value_changed"

    def __init__(
        self,
        users: Optional[List[UserInfo]] = None,
        team_id: Optional[int] = None,
        roles: Optional[List[str]] = None,
        filterable: Optional[bool] = True,
        placeholder: Optional[str] = "Select user",
        size: Optional[Literal["large", "small", "mini"]] = None,
        multiple: bool = False,
        widget_id: Optional[str] = None,
    ):
        self._users = []
        self._team_id = team_id
        self._allowed_roles = roles
        self._filterable = filterable
        self._placeholder = placeholder
        self._size = size
        self._multiple = multiple
        self._changes_handled = False

        # Load users from team_id if provided
        if team_id is not None:
            self._load_users_from_team(team_id)
        elif users is not None:
            self._users = self._filter_users_by_role(list(users))

        # Initialize parent Widget
        super().__init__(widget_id=widget_id, file_path=__file__)

    def _filter_users_by_role(self, users: List[UserInfo]) -> List[UserInfo]:
        """Filter users by allowed roles."""
        if self._allowed_roles is None:
            return users

        return [user for user in users if user.role in self._allowed_roles]

    def _load_users_from_team(self, team_id: int):
        """Load users from a team using the API."""
        from supervisely import Api

        api = Api.from_env()
        all_users = api.user.get_team_members(team_id)
        self._users = self._filter_users_by_role(all_users)

    def get_json_data(self):
        """Build JSON data for the widget."""
        items = []
        for user in self._users:
            user_name = None
            if user.name:
                name = user.name.strip()
                if name:
                    user_name = name[:15] + "…" if len(name) > 15 else name

            right_text = ""
            user_role = user.role.upper() if user.role else "NONE"
            if len(user_role) > 10:
                user_role = user_role[:10] + "…"
            if user_name and user_name != user.login:
                right_text = f"{user_name} • {user_role}"
            else:
                right_text = user_role

            items.append(
                {
                    "value": user.login,
                    "label": user.login,
                    "rightText": right_text,
                }
            )

        return {
            "items": items,
            "placeholder": self._placeholder,
            "filterable": self._filterable,
            "multiple": self._multiple,
            "size": self._size,
        }

    def get_json_state(self):
        """Build JSON state for the widget."""
        value = None
        if self._multiple:
            value = []
        return {"value": value}

    def get_value(self) -> Union[str, List[str], None]:
        """Get the currently selected user login(s)."""
        return StateJson()[self.widget_id]["value"]

    def get_selected_user(self) -> Union[UserInfo, List[UserInfo], None]:
        """Get the currently selected UserInfo object(s)."""
        value = self.get_value()
        if value is None:
            return None

        if self._multiple:
            if not isinstance(value, list):
                return []
            result = []
            for login in value:
                for user in self._users:
                    if user.login == login:
                        result.append(user)
                        break
            return result
        else:
            for user in self._users:
                if user.login == value:
                    return user
            return None

    def set_value(self, login: Union[str, List[str]]):
        """Set the selected user by login."""
        StateJson()[self.widget_id]["value"] = login
        StateJson().send_changes()

    def get_all_users(self) -> List[UserInfo]:
        """Get all available users."""
        return self._users.copy()

    def set_users(self, users: List[UserInfo]):
        """Update the list of available users."""
        self._users = self._filter_users_by_role(list(users))

        # Update data
        DataJson()[self.widget_id] = self.get_json_data()
        DataJson().send_changes()

        # Reset value if current selection is not in new users
        current_value = StateJson()[self.widget_id]["value"]
        if current_value:
            if self._multiple:
                if isinstance(current_value, list):
                    # Keep only valid selections
                    valid = [v for v in current_value if any(u.login == v for u in self._users)]
                    if valid != current_value:
                        StateJson()[self.widget_id]["value"] = valid
                        StateJson().send_changes()
            else:
                if not any(u.login == current_value for u in self._users):
                    StateJson()[self.widget_id]["value"] = (
                        self._users[0].login if self._users else None
                    )
                    StateJson().send_changes()

    def set_team_id(self, team_id: int):
        """Load users from a team by team_id."""
        self._team_id = team_id
        self._load_users_from_team(team_id)

        # Update data
        DataJson()[self.widget_id] = self.get_json_data()
        DataJson().send_changes()

        # Reset selection
        if self._multiple:
            StateJson()[self.widget_id]["value"] = []
        else:
            StateJson()[self.widget_id]["value"] = self._users[0].login if self._users else None
        StateJson().send_changes()

    def set_selected_users_by_ids(self, user_ids: Union[int, List[int]]):
        """Set the selected user(s) by user ID(s).

        :param user_ids: Single user ID or list of user IDs to select
        :type user_ids: Union[int, List[int]]
        """
        if isinstance(user_ids, int):
            user_ids = [user_ids]

        # Find logins for the given user IDs
        selected_logins = []
        for user_id in user_ids:
            for user in self._users:
                if user.id == user_id:
                    selected_logins.append(user.login)
                    break

        # Set value based on multiple mode
        if self._multiple:
            StateJson()[self.widget_id]["value"] = selected_logins
        else:
            StateJson()[self.widget_id]["value"] = selected_logins[0] if selected_logins else None

        StateJson().send_changes()

    def value_changed(self, func: Callable[[Union[UserInfo, List[UserInfo]]], None]):
        """
        Decorator to handle value change event.
        The decorated function receives the selected UserInfo (or list of UserInfo if multiple=True).

        :param func: Function to be called when selection changes
        :type func: Callable[[Union[UserInfo, List[UserInfo]]], None]
        """
        route_path = self.get_route_path(SelectUser.Routes.VALUE_CHANGED)
        server = self._sly_app.get_server()
        self._changes_handled = True

        @server.post(route_path)
        def _value_changed():
            selected = self.get_selected_user()
            if selected is not None:
                func(selected)

        return _value_changed

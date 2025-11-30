from typing import Callable, List, Optional, Union

from supervisely.api.user_api import UserInfo
from supervisely.app import StateJson
from supervisely.app.widgets import Select

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


class SelectUser(Select):
    """
    SelectUser is a dropdown widget for selecting users from a team.
    Extends the Select widget with user-specific functionality.

    :param users: Initial list of UserInfo instances
    :type users: Optional[List[UserInfo]]
    :param team_id: Team ID to fetch users from
    :type team_id: Optional[int]
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

        # Initialize with team_id
        select_user = SelectUser(team_id=123, multiple=True)

        # Or initialize empty and set users later
        select_user = SelectUser()
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
        filterable: Optional[bool] = True,
        placeholder: Optional[str] = "Select user",
        size: Optional[Literal["large", "small", "mini"]] = None,
        multiple: bool = False,
        widget_id: Optional[str] = None,
    ):
        self._users = []
        self._team_id = team_id
        self._value_changed_callback = None

        # Load users from team_id if provided
        if team_id is not None:
            self._load_users_from_team(team_id)
        elif users is not None:
            self._users = list(users)

        # Build Select.Item list from users
        items = self._build_items()

        # Initialize parent Select widget
        super().__init__(
            items=items,
            filterable=filterable,
            placeholder=placeholder,
            size=size,
            multiple=multiple,
            widget_id=widget_id,
        )

    def _load_users_from_team(self, team_id: int):
        """Load users from a team using the API."""
        from supervisely import Api

        api = Api.from_env()
        self._users = api.user.get_team_members(team_id)

    def _build_items(self) -> List[Select.Item]:
        """Build Select.Item list from UserInfo list."""
        items = []
        for user in self._users:
            items.append(
                Select.Item(
                    value=user.login,
                    label=user.login,
                    right_text=user.name if user.name != user.login else "",
                )
            )
        return items

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
        self._users = list(users)

        # Rebuild items
        items = self._build_items()

        # Use parent's set method
        super().set(items=items)

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

        # Rebuild items and update
        items = self._build_items()
        super().set(items=items)

        # Reset selection
        StateJson()[self.widget_id]["value"] = (
            self._users[0].login if self._users else None
        )
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
        self._value_changed_callback = func

        @server.post(route_path)
        def _value_changed():
            selected = self.get_selected_user()
            if selected is not None:
                func(selected)

        return _value_changed

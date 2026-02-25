from typing import List

from supervisely.app.content import DataJson, StateJson
from supervisely.app.widgets.widget import Widget


class DropdownCheckboxSelector(Widget):
    """Dropdown widget that lets a user select one or multiple items via checkboxes."""

    class Routes:
        """Callback route names used by the widget frontend to notify Python."""

        VALUE_CHANGED = "value_changed"

    class Item:
        """Selectable item descriptor (id + optional display name/description)."""

        def __init__(self, id: str, name: str = None, description: str = None):
            """
            :param id: Unique item identifier.
            :type id: str
            :param name: Display name. Defaults to id.
            :type name: str, optional
            :param description: Optional description text.
            :type description: str, optional
            """
            self.id = id
            if name is None:
                name = id
            self.name = name
            self.description = description

        def to_json(self):
            return {
                "id": self.id,
                "name": self.name,
                "description": self.description,
            }

        @classmethod
        def from_json(cls, data):
            return cls(
                id=data["id"],
                name=data["name"],
                description=data.get("description", None),
            )

    def __init__(
        self, items: List[Item], label: str = None, widget_id: str = None, multiple: bool = True
    ):
        """

        :param items: List of DropdownCheckboxSelector.Item.
        :type items: List[:class:`~supervisely.app.widgets.dropdown_checkbox_selector.DropdownCheckboxSelector.Item`]
        :param label: Optional label for the dropdown.
        :type label: str, optional
        :param widget_id: Unique widget identifier.
        :type widget_id: str, optional
        :param multiple: If True, allow multiple selection.
        :type multiple: bool, optional
        """
        self._items = items
        self._label = label
        self._multiple = multiple
        self._changes_handled = False

        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self):
        return {
            "items": [item.to_json() for item in self._items],
            "label": self._label,
        }

    def get_json_state(self):
        return {
            "selected": [],
        }

    def set(self, items: List[Item]):
        self._items = items
        DataJson()[self.widget_id]["items"] = [
            item.to_json() for item in self._items
        ]
        DataJson().send_changes()

    def get_selected(self) -> List[Item]:
        selected =  StateJson()[self.widget_id].get("selected", [])
        return [item for item in self._items if item.id in selected]

    def select(self, ids: List[str]):
        selected = [item for item in self._items if item.id in ids]
        StateJson()[self.widget_id]["selected"] = [item.to_json() for item in selected]
        StateJson().send_changes()

    def value_changed(self, func):
        """
        Decorator to handle value changes.
        :param func: function to call when value changes
        """

        route_path = self.get_route_path(self.Routes.VALUE_CHANGED)
        server = self._sly_app.get_server()

        @server.post(route_path)
        def on_value_changed():
            selected_items = self.get_selected()
            return func(selected_items)

        self._changes_handled = True

        return on_value_changed

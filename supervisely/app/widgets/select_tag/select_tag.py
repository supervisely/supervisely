from typing import Callable, List, Optional, Union

from supervisely.annotation.tag_meta import TagMeta, TagValueType
from supervisely.annotation.tag_meta_collection import TagMetaCollection
from supervisely.app import DataJson, StateJson
from supervisely.app.widgets import Text, Widget
from supervisely.imaging.color import generate_rgb

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


# Available value types for tag creation
available_value_types = [
    {"value": TagValueType.NONE, "label": "None"},
    {"value": TagValueType.ANY_STRING, "label": "Any string"},
    {"value": TagValueType.ANY_NUMBER, "label": "Any number"},
    {"value": TagValueType.ONEOF_STRING, "label": "One of"},
]


class SelectTag(Widget):
    """
    SelectTag is a compact dropdown widget for selecting tag metadata with an option to create
    new tags on the fly.

    :param tags: Initial list of TagMeta instances
    :type tags: Optional[Union[List[TagMeta], TagMetaCollection]]
    :param filterable: Enable search/filter functionality in dropdown
    :type filterable: Optional[bool]
    :param placeholder: Placeholder text when no tag is selected
    :type placeholder: Optional[str]
    :param show_add_new_tag: Show "Add new tag" option at the end of the list
    :type show_add_new_tag: Optional[bool]
    :param size: Size of the select dropdown
    :type size: Optional[Literal["large", "small", "mini"]]
    :param multiple: Enable multiple selection
    :type multiple: bool
    :param widget_id: Unique widget identifier
    :type widget_id: Optional[str]

    :Usage example:

     .. code-block:: python

        import supervisely as sly
        from supervisely.app.widgets import SelectTag

        # Create some initial tags
        tag_weather = sly.TagMeta('weather', sly.TagValueType.ANY_STRING)
        tag_count = sly.TagMeta('count', sly.TagValueType.ANY_NUMBER)

        colors = ["red", "green", "blue"]
        tag_color = sly.TagMeta('color', sly.TagValueType.ONEOF_STRING, possible_values=colors)

        # Create SelectTag widget
        select_tag = SelectTag(
            tags=[tag_weather, tag_count, tag_color],
            filterable=True,
            show_add_new_tag=True
        )

        # Handle selection changes
        @select_tag.value_changed
        def on_tag_selected(tag_name):
            print(f"Selected tag: {tag_name}")
            selected_tag = select_tag.get_selected_tag()
            print(f"Tag object: {selected_tag}")

        # Handle new tag creation
        @select_tag.tag_created
        def on_tag_created(new_tag: sly.TagMeta):
            print(f"New tag created: {new_tag.name}")
    """

    class Routes:
        VALUE_CHANGED = "value_changed"
        TAG_CREATED = "tag_created_cb"

    def __init__(
        self,
        tags: Optional[Union[List[TagMeta], TagMetaCollection]] = [],
        filterable: Optional[bool] = True,
        placeholder: Optional[str] = "Select tag",
        show_add_new_tag: Optional[bool] = True,
        size: Optional[Literal["large", "small", "mini"]] = None,
        multiple: bool = False,
        widget_id: Optional[str] = None,
    ):
        # Convert to list for internal use to allow mutations when adding new tags
        if isinstance(tags, TagMetaCollection):
            self._tags = list(tags)
        else:
            self._tags = list(tags) if tags else []

        self._filterable = filterable
        self._placeholder = placeholder
        self._show_add_new_tag = show_add_new_tag
        self._size = size
        self._multiple = multiple

        self._changes_handled = False
        self._tag_created_callback = None

        # Store error message widget
        self._error_message = Text("", status="error", font_size=13)

        # Initialize parent Widget
        super().__init__(widget_id=widget_id, file_path=__file__)

        # Register tag_created route if show_add_new_tag is enabled
        if self._show_add_new_tag:
            self._register_tag_created_route()

    def get_json_data(self):
        """Build JSON data for the widget."""
        # Build items list with tag info
        items = []
        for tag in self._tags:
            value_type_text = tag.value_type.replace("_", " ").upper()

            items.append(
                {
                    "value": tag.name,
                    "label": tag.name,
                    "color": tag.color,
                    "valueType": value_type_text if value_type_text else "",
                }
            )

        return {
            "items": items,
            "placeholder": self._placeholder,
            "filterable": self._filterable,
            "multiple": self._multiple,
            "size": self._size,
            "showAddNewTag": self._show_add_new_tag,
            "availableValueTypes": available_value_types,
        }

    def get_json_state(self):
        """Build JSON state for the widget."""
        # Set initial value based on multiple mode
        if self._multiple:
            initial_value = []
        else:
            initial_value = self._tags[0].name if self._tags else None

        return {
            "value": initial_value,
            "createTagDialog": {
                "visible": False,
                "tagName": "",
                "valueType": TagValueType.NONE,
                "possibleValues": "",
                "showError": False,
                "showPossibleValues": False,
            },
        }

    def get_value(self) -> Union[str, List[str], None]:
        """Get the currently selected tag name(s)."""
        return StateJson()[self.widget_id]["value"]

    def get_selected_tag(self) -> Union[TagMeta, List[TagMeta], None]:
        """Get the currently selected TagMeta object(s)."""
        value = self.get_value()
        if value is None:
            return None

        if self._multiple:
            if not isinstance(value, list):
                return []
            result = []
            for tag_name in value:
                for tag in self._tags:
                    if tag.name == tag_name:
                        result.append(tag)
                        break
            return result
        else:
            for tag in self._tags:
                if tag.name == value:
                    return tag
            return None

    def set_value(self, tag_name: Union[str, List[str]]):
        """Set the selected tag by name."""
        StateJson()[self.widget_id]["value"] = tag_name
        StateJson().send_changes()

    def get_all_tags(self) -> List[TagMeta]:
        """Get all available tags."""
        return self._tags.copy()

    def set(self, tags: Union[List[TagMeta], TagMetaCollection]):
        """Update the list of available tags."""
        # Convert to list for internal use
        if isinstance(tags, TagMetaCollection):
            self._tags = list(tags)
        else:
            self._tags = list(tags) if tags else []

        # Update data
        DataJson()[self.widget_id] = self.get_json_data()
        DataJson().send_changes()

        # Reset value if current selection is not in new tags
        current_value = StateJson()[self.widget_id]["value"]
        if current_value:
            if self._multiple:
                if isinstance(current_value, list):
                    # Keep only valid selections
                    valid = [v for v in current_value if any(tag.name == v for tag in self._tags)]
                    if valid != current_value:
                        StateJson()[self.widget_id]["value"] = valid
                        StateJson().send_changes()
            else:
                if not any(tag.name == current_value for tag in self._tags):
                    StateJson()[self.widget_id]["value"] = (
                        self._tags[0].name if self._tags else None
                    )
                    StateJson().send_changes()

    def _show_error(self, message: str):
        """Show error message in the create tag dialog."""
        self._error_message.text = message
        StateJson()[self.widget_id]["createTagDialog"]["showError"] = True
        StateJson().send_changes()

    def _hide_dialog(self):
        """Hide the create tag dialog and reset its state."""
        state_obj = StateJson()[self.widget_id]["createTagDialog"]
        state_obj["visible"] = False
        state_obj["tagName"] = ""
        state_obj["possibleValues"] = ""
        state_obj["showError"] = False
        state_obj["showPossibleValues"] = False
        StateJson().send_changes()

    def _add_new_tag(self, new_tag: TagMeta):
        """Add a new tag to the widget and update the UI."""
        # Add to tags list
        self._tags.append(new_tag)

        # Update data
        DataJson()[self.widget_id] = self.get_json_data()
        DataJson().send_changes()

        # Set the new tag as selected
        if self._multiple:
            current = StateJson()[self.widget_id]["value"]
            if isinstance(current, list):
                current.append(new_tag.name)
            else:
                StateJson()[self.widget_id]["value"] = [new_tag.name]
        else:
            StateJson()[self.widget_id]["value"] = new_tag.name
        StateJson().send_changes()

    def value_changed(self, func: Callable[[Union[TagMeta, List[TagMeta]]], None]):
        """
        Decorator to handle value change event.
        The decorated function receives the selected TagMeta (or list of TagMeta if multiple=True).

        :param func: Function to be called when selection changes
        :type func: Callable[[Union[TagMeta, List[TagMeta]]], None]
        """
        route_path = self.get_route_path(SelectTag.Routes.VALUE_CHANGED)
        server = self._sly_app.get_server()
        self._changes_handled = True

        @server.post(route_path)
        def _value_changed():
            selected = self.get_selected_tag()
            if selected is not None:
                func(selected)

        return _value_changed

    def _register_tag_created_route(self):
        """Register the tag_created route."""
        route_path = self.get_route_path(SelectTag.Routes.TAG_CREATED)
        server = self._sly_app.get_server()

        @server.post(route_path)
        def _tag_created():
            state = StateJson()[self.widget_id]["createTagDialog"]
            tag_name = state["tagName"].strip()
            value_type = state["valueType"]
            possible_values_str = state["possibleValues"].strip()

            # Validate tag name
            if not tag_name:
                self._show_error("Tag name cannot be empty")
                return

            # Check if tag with this name already exists
            if any(tag.name == tag_name for tag in self._tags):
                self._show_error(f"Tag '{tag_name}' already exists")
                return

            # Parse possible values for ONEOF_STRING
            possible_values = None
            if value_type == TagValueType.ONEOF_STRING:
                if not possible_values_str:
                    self._show_error("Possible values are required for 'One of' type")
                    return
                # Split by comma and strip whitespace
                possible_values = [v.strip() for v in possible_values_str.split(",") if v.strip()]
                if len(possible_values) == 0:
                    self._show_error("At least one possible value is required")
                    return

            # Generate color for the new tag
            existing_colors = [tag.color for tag in self._tags]
            new_color = generate_rgb(existing_colors)

            # Create new tag
            try:
                new_tag = TagMeta(
                    name=tag_name,
                    value_type=value_type,
                    possible_values=possible_values,
                    color=new_color,
                )
            except Exception as e:
                self._show_error(f"Error creating tag: {str(e)}")
                return

            # Add to widget
            self._add_new_tag(new_tag)

            # Hide dialog
            self._hide_dialog()

            # Call user's callback if set
            if self._tag_created_callback:
                self._tag_created_callback(new_tag)

    def tag_created(self, func: Callable[[TagMeta], None]):
        """
        Decorator to handle new tag creation event.
        The decorated function receives the newly created TagMeta.

        :param func: Function to be called when a new tag is created
        :type func: Callable[[TagMeta], None]
        """
        self._tag_created_callback = func
        return func

from typing import List, Union, Dict, Callable

from supervisely.app.widgets import Widget
from supervisely import TagMeta, TagMetaCollection, Tag
from supervisely.annotation.tag_meta import TagValueType
from supervisely.app.content import DataJson, StateJson
from supervisely.imaging.color import rgb2hex


class InputTagList(Widget):
    """Store and manage a list of input tags. Class accepts a list of TagMeta objects and provides methods to interact with them.

    :param tag_metas: List of TagMeta objects or a TagMetaCollection, defaults to an empty list
    :type tag_metas: Union[List[TagMeta], TagMetaCollection], optional
    :param max_width: Maximum width of the widget in pixels, defaults to 300
    :type max_width: int, optional
    :param max_height: Maximum height of the widget in pixels, defaults to 50
    :type max_height: int, optional
    :param multiple: Whether to allow multiple tags to be selected, defaults to False
    :type multiple: bool, optional
    :param widget_id: Unique identifier for the widget, defaults to None
    :type widget_id: int, optional
    """

    class VALUE_TYPES:
        """Value types for input tags. Classifies the different types of values that tags can have."""

        none = str(TagValueType.NONE)
        any_string = str(TagValueType.ANY_STRING)
        one_of = str(TagValueType.ONEOF_STRING)
        number = str(TagValueType.ANY_NUMBER)

    VALUE_TYPE_NAME = {
        str(TagValueType.NONE): "NONE",
        str(TagValueType.ANY_STRING): "TEXT",
        str(TagValueType.ONEOF_STRING): "ONE OF",
        str(TagValueType.ANY_NUMBER): "NUMBER",
    }

    def get_default_value(self, tag_meta: TagMeta) -> Union[str, int, None]:
        """Get default value for the tag based on its meta information.
        If the tag has a predefined set of possible values (ONEOF_STRING), return the first one.
        For other types, return a standard default value:
        1. NONE: None
        2. ANY_STRING: ""
        3. ANY_NUMBER: 0

        :param tag_meta: Tag metadata
        :type tag_meta: TagMeta
        :return: Default value for the tag
        :rtype: Union[str, int, None]
        """
        DEFAULT_VALUES = {
            str(TagValueType.NONE): None,
            str(TagValueType.ANY_STRING): "",
            str(TagValueType.ANY_NUMBER): 0,
        }
        if tag_meta.value_type == str(TagValueType.ONEOF_STRING):
            return tag_meta.possible_values[0]
        else:
            return DEFAULT_VALUES[tag_meta.value_type]

    class Routes:
        """Routes for the widget events. Classifies the different types of events that can occur within the widget."""

        CHECKBOX_CHANGED = "checkbox_cb"

    def __init__(
        self,
        tag_metas: Union[List[TagMeta], TagMetaCollection] = [],
        max_width: int = 300,
        max_height: int = 50,
        multiple: bool = False,
        widget_id: int = None,
    ):
        self._tag_metas = tag_metas
        self._max_width = self._get_max_width(max_width)
        self._max_height = self._get_max_height(max_height)
        self._multiple = multiple

        super().__init__(widget_id=widget_id, file_path=__file__)

    def _get_max_width(self, value) -> str:
        """Get the maximum width for the widget.
        Ensures the width is at least 150 pixels.

        :param value: Desired maximum width in pixels.
        :type value: int
        :return: Maximum width for the widget
        :rtype: str
        """
        if value < 150:
            value = 150
        return f"{value}px"

    def _get_max_height(self, value) -> str:
        """Get the maximum height for the widget.
        Ensures the height is at least 100 pixels.

        :param value: Desired maximum height in pixels.
        :type value: int
        :return: Maximum height for the widget
        :rtype: str
        """
        if value < 100:
            value = 100
        return f"{value}px"

    def get_json_data(self) -> Dict:
        """Get JSON data for the widget.

        :return: JSON data for the widget
        :rtype: Dict
        """
        return {
            "maxWidth": self._max_width,
            "maxHeight": self._max_height,
            "tags": [
                {
                    "name": tag_meta.name,
                    "valueType": tag_meta.value_type,
                    "valueTypeText": self.VALUE_TYPE_NAME[tag_meta.value_type],
                    "color": rgb2hex(tag_meta.color),
                    "possible_values": tag_meta.possible_values,
                }
                for tag_meta in self._tag_metas
            ],
        }

    def get_json_state(self) -> Dict:
        """Get JSON state for the widget.

        :return: JSON state for the widget
        :rtype: Dict
        """
        return {
            "selected": [False for _ in self._tag_metas],
            "values": [self.get_default_value(tm) for tm in self._tag_metas],
        }

    def get_selected_tag_metas(self) -> List[TagMeta]:
        """Get selected tag metas for the widget.

        :return: List of selected tag metas
        :rtype: List[TagMeta]
        """
        return [
            tm
            for selected, tm in zip(StateJson()[self.widget_id]["selected"], self._tag_metas)
            if selected
        ]

    def get_selected_tags(self) -> List[Tag]:
        """Get selected tags for the widget.

        :return: List of selected tags
        :rtype: List[Tag]
        """
        return [
            Tag(meta=tm, value=value)
            for selected, value, tm in zip(
                StateJson()[self.widget_id]["selected"],
                StateJson()[self.widget_id]["values"],
                self._tag_metas,
            )
            if selected
        ]

    def get_all_tags(self) -> Union[List[TagMeta], TagMetaCollection]:
        """Get all tags for the widget.

        :return: List of all tag metas
        :rtype: Union[List[TagMeta], TagMetaCollection]
        """
        return [
            Tag(meta=tm, value=value)
            for value, tm in zip(
                StateJson()[self.widget_id]["values"],
                self._tag_metas,
            )
        ]

    def set(self, tag_metas: Union[List[TagMeta], TagMetaCollection]) -> None:
        """Set tag metas for the widget.

        :param tag_metas: Tag metas to set
        :type tag_metas: Union[List[TagMeta], TagMetaCollection]
        :return: None
        """
        self._tag_metas = tag_metas
        DataJson()[self.widget_id] = self.get_json_data()
        DataJson().send_changes()
        StateJson()[self.widget_id] = self.get_json_state()
        StateJson().send_changes()

    def set_values(self, values_dict: Dict) -> None:
        """Set values for the widget.

        :param values_dict: Dictionary of values to set
        :type values_dict: Dict
        :return: None
        """
        current_values = StateJson()[self.widget_id]["values"]
        values = [
            values_dict.get(tm.name, current_values[idx]) for idx, tm in enumerate(self._tag_metas)
        ]
        StateJson()[self.widget_id]["values"] = values
        StateJson().send_changes()

    def select_all(self) -> None:
        """Select all tags for the widget.

        :return: None
        """
        StateJson()[self.widget_id]["selected"] = [True for _ in self._tag_metas]
        StateJson().send_changes()

    def deselect_all(self) -> None:
        """Deselect all tags for the widget.

        :return: None
        """
        StateJson()[self.widget_id]["selected"] = [False for _ in self._tag_metas]
        StateJson().send_changes()

    def select(self, names: List[str]) -> None:
        """Select tags for the widget.

        :param names: List of tag names to select
        :type names: List[str]
        :return: None
        """
        selected = [tm.name in names for tm in self._tag_metas]
        StateJson()[self.widget_id]["selected"] = selected
        StateJson().send_changes()

    def deselect(self, names: List[str]) -> None:
        """Deselect tags for the widget.

        :param names: List of tag names to deselect
        :type names: List[str]
        :return: None
        """
        selected = StateJson()[self.widget_id]["selected"]
        for idx, tm in enumerate(self._tag_metas):
            if tm.name in names:
                selected[idx] = False
        StateJson()[self.widget_id]["selected"] = selected
        StateJson().send_changes()

    def get_all_tag_metas(self) -> List[TagMeta]:
        """Get all tag metas for the widget.

        :return: List of all tag metas
        :rtype: List[TagMeta]
        """
        return self._tag_metas

    def selection_changed(self, func: Callable) -> Callable:
        """Decorator to register a callback function for selection changes.

        :param func: Callback function
        :type func: Callable
        """
        route_path = self.get_route_path(InputTagList.Routes.CHECKBOX_CHANGED)
        server = self._sly_app.get_server()
        self._checkboxes_handled = True

        @server.post(route_path)
        def _click():
            selected = self.get_selected_tag_metas()
            func(selected)

        return _click

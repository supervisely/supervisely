from typing import Dict, Union, Callable

from supervisely.annotation.tag import Tag
from supervisely.annotation.tag_meta import TagMeta, TagValueType
from supervisely.app import DataJson
from supervisely.app.widgets import (
    Empty,
    Input,
    InputNumber,
    OneOf,
    RadioGroup,
    Select,
    Switch,
    Widget,
)

VALUE_TYPE_NAME = {
    str(TagValueType.NONE): "NONE",
    str(TagValueType.ANY_STRING): "TEXT",
    str(TagValueType.ONEOF_STRING): "ONE OF",
    str(TagValueType.ANY_NUMBER): "NUMBER",
}

VALUE_TYPES = [
    str(TagValueType.NONE),
    str(TagValueType.ANY_NUMBER),
    str(TagValueType.ANY_STRING),
    str(TagValueType.ONEOF_STRING),
]


class InputTag(Widget):
    """Widget for inputting a single tag value based on its TagMeta information. Accepts various input types depending on the tag's value type. Returns the tag value when requested.

    :param tag_meta: Tag metadata
    :type tag_meta: TagMeta
    :param max_width: Maximum width of the widget in pixels, defaults to 300
    :type max_width: int
    :param hide_switch: Whether to hide the activation switch, defaults to False
    :type hide_switch: bool
    :param widget_id: Unique identifier for the widget, defaults to None
    :type widget_id: int
    """

    def __init__(
        self,
        tag_meta: TagMeta,
        max_width: int = 300,
        hide_switch: bool = False,
        widget_id: int = None,
    ):
        self._input_widgets = {}
        self._init_input_components()

        self._conditional_widget = Select(
            items=[
                Select.Item(value_type, content=self._input_widgets[value_type])
                for value_type in VALUE_TYPES
            ]
        )
        self._value_changed_callbacks = {}

        self._tag_meta = tag_meta
        # if TagMeta ValueType is ONEOF_STRING, then we need to set items (possible values options) for RadioGroup
        if self._tag_meta.value_type == str(TagValueType.ONEOF_STRING):
            items = [RadioGroup.Item(pv, pv) for pv in self._tag_meta.possible_values]
            self._input_widgets[str(TagValueType.ONEOF_STRING)].set(items)
        self._conditional_widget.set_value(str(self._tag_meta.value_type))

        self._value_type_name = VALUE_TYPE_NAME[self._tag_meta.value_type]
        self._name = f"<b>{self._tag_meta.name}</b>"
        self._max_width = self._get_max_width(max_width)
        self._hide_switch = hide_switch
        self._activation_widget = Switch()
        self._input_widget = OneOf(self._conditional_widget)

        if self._hide_switch:
            self._activation_widget.hide()
            self._activation_widget.on()

        super().__init__(widget_id=widget_id, file_path=__file__)

    def _init_input_components(self):
        self._input_widgets[str(TagValueType.NONE)] = Empty()
        self._input_widgets[str(TagValueType.ANY_NUMBER)] = InputNumber(debounce=500)
        self._input_widgets[str(TagValueType.ANY_STRING)] = Input(type="textarea")
        self._input_widgets[str(TagValueType.ONEOF_STRING)] = RadioGroup(items=[])

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

    def get_tag_meta(self) -> TagMeta:
        """Get the tag metadata.

        :return: Tag metadata
        :rtype: TagMeta
        """
        return self._tag_meta

    def activate(self) -> None:
        """Activate the widget."""
        self._activation_widget.on()

    def deactivate(self) -> None:
        """Deactivate the widget."""
        self._activation_widget.off()

    def is_active(self) -> bool:
        """Check if the widget is active.

        :return: True if the widget is active, False otherwise
        :rtype: bool
        """
        return self._activation_widget.is_switched()

    @property
    def value(self) -> Union[str, int, None]:
        """Get the current value of the tag.

        :return: Current value of the tag
        :rtype: Union[str, int, None]
        """
        return self._get_value()

    @value.setter
    def value(self, value: Union[str, int, None]) -> None:
        """Set the current value of the tag.

        :param value: Current value of the tag
        :type value: Union[str, int, None]
        :return: None
        """
        self._set_value(value)

    def is_valid_value(self, value: Union[str, int, None]) -> bool:
        """Check if the value is valid for the tag.

        :param value: Value to check
        :type value: Union[str, int, None]
        :return: True if the value is valid, False otherwise
        :rtype: bool
        """
        return self._tag_meta.is_valid_value(value)

    def set(self, tag: Union[Tag, None]) -> None:
        """Set the tag value.

        :param tag: Tag to set
        :type tag: Union[Tag, None]
        :return: None
        """
        if tag is None:
            self._set_default_value()
            self.deactivate()
        else:
            self._set_value(tag.value)
            self.activate()

    def get_tag(self) -> Union[Tag, None]:
        """Get the current tag.

        :return: Current tag
        :rtype: Union[Tag, None]
        """
        if not self._hide_switch and not self.is_active():
            return None
        tag_value = self._get_value()
        return Tag(self._tag_meta, tag_value)

    def _get_value(self) -> Union[str, int, None]:
        """Get the current value of the tag.

        :return: Current value of the tag
        :rtype: Union[str, int, None]
        """
        input_widget = self._input_widgets[self._tag_meta.value_type]
        if isinstance(input_widget, Empty):
            return None
        else:
            return input_widget.get_value()

    def _set_value(self, value):
        if not self.is_valid_value(value):
            raise ValueError(f'Tag value "{value}" is invalid')
        input_widget = self._input_widgets[self._tag_meta.value_type]
        if isinstance(input_widget, InputNumber):
            input_widget.value = value
        if isinstance(input_widget, Input):
            input_widget.set_value(value)
        if isinstance(input_widget, RadioGroup):
            input_widget.set_value(value)

    def _set_default_value(self):
        input_widget = self._input_widgets[self._tag_meta.value_type]
        if isinstance(input_widget, InputNumber):
            input_widget.value = 0
        if isinstance(input_widget, Input):
            input_widget.set_value("")
        if isinstance(input_widget, RadioGroup):
            input_widget.set_value(None)

    def get_json_data(self) -> Dict:
        """Get the JSON representation of the tag.

        :return: JSON representation of the tag
        :rtype: Dict
        """
        return {
            "name": self._name,
            "valueType": self._value_type_name,
            "maxWidth": self._max_width,
        }

    def get_json_state(self) -> Dict:
        """Get the JSON representation of the tag state.

        :return: JSON representation of the tag state
        :rtype: Dict
        """
        return None

    def value_changed(self, func) -> Callable:
        """Decorator to register a callback function for selection changes.

        :param func: Callback function
        :type func: Callable
        """
        for value_type, input_widget in self._input_widgets.items():
            if isinstance(input_widget, Empty):
                self._value_changed_callbacks[value_type] = func
            else:
                self._value_changed_callbacks[value_type] = input_widget.value_changed(func)

        def inner(*args, **kwargs):
            return self._value_changed_callbacks[self._tag_meta.value_type](*args, **kwargs)

        return inner

    def selection_changed(self, func):
        return self._activation_widget.value_changed(func)

    def set_tag_meta(self, tag_meta: TagMeta) -> None:
        """Set the tag metadata.

        :param tag_meta: Tag metadata to set
        :type tag_meta: TagMeta
        :return: None
        """
        self._tag_meta = tag_meta
        self._value_type_name = VALUE_TYPE_NAME[self._tag_meta.value_type]
        self._name = f"<b>{self._tag_meta.name}</b>"
        # if TagMeta ValueType is ONEOF_STRING, then we need to set items (possible values options) for RadioGroup
        if self._tag_meta.value_type == str(TagValueType.ONEOF_STRING):
            items = [RadioGroup.Item(pv, pv) for pv in self._tag_meta.possible_values]
            self._input_widgets[str(TagValueType.ONEOF_STRING)].set(items)

        self._conditional_widget.set_value(str(self._tag_meta.value_type))
        self._set_default_value()
        if self._hide_switch:
            self.activate()
        else:
            self.deactivate()
        self.update_data()
        DataJson().send_changes()

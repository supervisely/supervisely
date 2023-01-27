from typing import Dict, Union
from supervisely.annotation.tag import Tag
from supervisely.annotation.tag_meta import TagMeta, TagValueType
from supervisely.app.widgets import Widget
from supervisely.app.widgets import (
    Switch,
    Empty,
    Input,
    InputNumber,
    RadioGroup,
)


VALUE_TYPE_NAME = {
    "none": "NONE",
    "any_string": "TEXT",
    "oneof_string": "ONE OF",
    "any_number": "NUMBER"
}


class InputTag(Widget):
    def __init__(
        self,
        tag_meta: TagMeta,
        max_width: int = 300,
        widget_id: int = None,
    ):
        self._tag_meta = tag_meta
        self._value_type_name = VALUE_TYPE_NAME[self._tag_meta.value_type]
        self._name = f"<b>{self._tag_meta.name}</b>"
        self._max_width = self._get_max_width(max_width)
        self._activation_widget = Switch()
        self._input_widget = self._get_input_component()

        super().__init__(widget_id=widget_id, file_path=__file__)

    def _get_input_component(self):
        if self._tag_meta.value_type == str(TagValueType.NONE):
            return Empty()
        if self._tag_meta.value_type == str(TagValueType.ANY_NUMBER):
            return InputNumber(debounce=500)
        if self._tag_meta.value_type == str(TagValueType.ANY_STRING):
            return Input()
        if self._tag_meta.value_type == str(TagValueType.ONEOF_STRING):
            items = [
                RadioGroup.Item(pv, pv, Empty()) for pv in self._tag_meta.possible_values
            ]
            return RadioGroup(items=items)

    def _get_max_width(self, value):
        if value < 150:
            value = 150
        return f"{value}px"

    def get_tag_meta(self):
        return self._tag_meta

    def activate(self):
        self._activation_widget.on()

    def deactivate(self):
        self._activation_widget.off()

    def is_active(self):
        return self._activation_widget.is_switched()

    @property
    def value(self):
        return self._get_value()

    @value.setter
    def value(self, value):
        self._set_value(value)

    def is_valid_value(self, value):
        return self._tag_meta.is_valid_value(value)

    def set(self, tag: Union[Tag, None]):
        if tag is None:
            self._set_default_value()
            self.deactivate()
        else:
            self._set_value(tag.value)
            self.activate()

    def get_tag(self):
        if not self.is_active():
            return None
        tag_value = self._get_value()
        return Tag(self._tag_meta, tag_value)

    def _get_value(self):
        if type(self._input_widget) is Empty:
            return None
        else:
            return self._input_widget.get_value()

    def _set_value(self, value):
        if not self.is_valid_value(value):
            raise ValueError(f'Tag value "{value}" is invalid')
        if type(self._input_widget) is InputNumber:
            self._input_widget.value = value
        if type(self._input_widget) is Input:
            self._input_widget.set_value(value)
        if type(self._input_widget) is RadioGroup:
            self._input_widget.set_value(value)

    def _set_default_value(self):
        if type(self._input_widget) is InputNumber:
            self._input_widget.value = 0
        if type(self._input_widget) is Input:
            self._input_widget.set_value("")
        if type(self._input_widget) is RadioGroup:
            self._input_widget.set_value(None)

    def get_json_data(self):
        return {
            "name": self._name,
            "valueType": self._value_type_name,
            "maxWidth": self._max_width,
        }

    def get_json_state(self) -> Dict:
        return None

    def value_changed(self, func):
        if type(self._input_widget) is Empty:
            return func
        return self._input_widget.value_changed(func)

    def selection_changed(self, func):
        return self._activation_widget.value_changed(func)

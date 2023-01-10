from typing import Dict, Union
from supervisely.annotation.tag import Tag
from supervisely.annotation.tag_meta import TagMeta, TagValueType
from supervisely.app.widgets import Widget
from supervisely.app.widgets import (
    Checkbox,
    Field,
    Empty,
    Input,
    InputNumber,
    RadioGroup,
)


class InputTag(Widget):
    def __init__(self, tag_meta: TagMeta, widget_id: int = None):
        self._tag_meta = tag_meta
        self._component = self._get_tag_component(tag_meta)

        super().__init__(widget_id=widget_id, file_path=__file__)

    def _get_tag_component(self, tag_meta: TagMeta):
        if tag_meta.value_type == str(TagValueType.NONE):
            return Checkbox(content=Field(content=Empty(), title=tag_meta.name))
        if tag_meta.value_type == str(TagValueType.ANY_NUMBER):
            return Checkbox(
                content=Field(
                    content=InputNumber(controls=False, debounce=500),
                    title=tag_meta.name,
                )
            )
        if tag_meta.value_type == str(TagValueType.ANY_STRING):
            return Checkbox(content=Field(content=Input(), title=tag_meta.name))
        if tag_meta.value_type == str(TagValueType.ONEOF_STRING):
            items = [
                RadioGroup.Item(pv, pv, Empty()) for pv in tag_meta.possible_values
            ]
            return Checkbox(
                content=Field(content=RadioGroup(items=items), title=tag_meta.name)
            )

    def get_tag_meta(self):
        return self._tag_meta

    def activate(self):
        tag_component = self._component
        tag_component.check()

    def deactivate(self):
        tag_component = self._component
        tag_component.uncheck()

    def is_active(self):
        return self._component.is_checked()

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
        tag_component = self._component
        content = tag_component._content._content
        if type(content) is Empty:
            return None
        else:
            return content.get_value()

    def _set_value(self, value):
        if not self.is_valid_value(value):
            raise ValueError(f'Tag value "{value}" is invalid')
        tag_component = self._component
        content = tag_component._content._content
        if type(content) is InputNumber:
            content.value = value
        if type(content) is Input:
            content.set_value(value)
        if type(content) is RadioGroup:
            content.set_value(value)

    def _set_default_value(self):
        tag_component = self._component
        content = tag_component._content._content
        if type(content) is InputNumber:
            content.value = 0
        if type(content) is Input:
            content.set_value("")
        if type(content) is RadioGroup:
            content.set_value(None)

    def get_json_data(self):
        return None

    def get_json_state(self) -> Dict:
        return None

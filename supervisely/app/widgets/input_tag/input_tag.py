from typing import Dict, Union
from supervisely.annotation.tag import Tag
from supervisely.annotation.tag_meta import TagMeta, TagValueType
from supervisely.app.content import StateJson
from supervisely.app.widgets import Widget
from supervisely.app.widgets import Checkbox, Field, Empty, Input, InputNumber, RadioGroup


class InputTag(Widget):
    def __init__(self, tag_meta: TagMeta, widget_id: int = None):
        self._tag_meta = tag_meta
        self._component = self._get_tag_component(tag_meta)

        super().__init__(widget_id=widget_id, file_path=__file__)

    def _get_tag_component(self, tag_meta: TagMeta):
        if tag_meta.value_type == str(TagValueType.NONE):
            return Checkbox(content=Field(content=Empty(), title=tag_meta.name))
        if tag_meta.value_type == str(TagValueType.ANY_NUMBER):
            return Checkbox(content=Field(content=InputNumber(), title=tag_meta.name))
        if tag_meta.value_type == str(TagValueType.ANY_STRING):
            return Checkbox(content=Field(content=Input(), title=tag_meta.name))
        if tag_meta.value_type == str(TagValueType.ONEOF_STRING):
            items = [RadioGroup.Item(pv, pv, Empty()) for pv in tag_meta.possible_values]
            return Checkbox(content=Field(content=RadioGroup(items=items), title=tag_meta.name))

    def activate(self):
        tag_component = self._component
        tag_component.check()
        
    def deactivate(self):
        tag_component = self._component
        tag_component.uncheck()

    def set(self, value: Union[bool, int, str, None] = None, tag: Tag = None):
        """
        Sets value for the tag. If argument "tag" is present, then value is ignored.
        Pass None as value to deactivate tag
        """
        if not tag is None:
            value = tag.value
            if tag.meta.value_type == str(TagValueType.NONE):
                value = True
        if value is None:
            self.deactivate()
            return
        if self._tag_meta.value_type == str(TagValueType.NONE):
            if value == True:
                self.activate()
            elif value == False:
                self.deactivate()
            else:
                raise ValueError(f'Tag value "{value}" is invalid')
        else:
            self._set_tag_value(value)
        StateJson().send_changes()

    def get_value(self):
        tag_component = self._component
        if not tag_component.is_checked():
            return None
        content = tag_component._content._content
        if type(content) is Empty:
            return True
        if type(content) is InputNumber:
            return content.get_value()
        if type(content) is Input:
            return content.get_value()
        if type(content) is RadioGroup:
            return content.get_value()

    def get_tag(self):
        return Tag(self._tag_meta, self.get_value())

    def _set_tag_value(self, value):
        if not self._tag_meta.is_valid_value(value):
            raise ValueError(f'Tag value "{value}" is invalid')
        tag_component = self._component
        content = tag_component._content._content
        if type(content) is InputNumber:
            content.value = value
            content.update_state()
        if type(content) is Input:
            content.set_value(value)
        if type(content) is RadioGroup:
            content.set_value(value)
        
    def get_json_data(self):
        return None

    def get_json_state(self) -> Dict:
        return None

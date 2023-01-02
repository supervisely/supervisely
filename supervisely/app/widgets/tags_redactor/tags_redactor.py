from __future__ import annotations
from supervisely.app import DataJson, StateJson
from supervisely.app.widgets import Widget, Checkbox, Empty, Field, InputNumber, Input, RadioGroup, Container
from typing import List, Dict, Union

from supervisely.annotation.label import Label 
from supervisely.annotation.tag_meta import TagApplicableTo, TagMeta
from supervisely.annotation.tag import Tag, TagValueType


class TagsRedactor(Container):
    def __init__(self, project_meta, widget_id=None):
        self._project_meta = project_meta
        self._tag_inputs = [self._get_tag_component(tag) for tag in self._project_meta.tag_metas]
        return super().__init__(widgets=self._tag_inputs, widget_id=widget_id)

    def _get_tag_component(self, tag: TagMeta):
        if tag.value_type == str(TagValueType.NONE):
            return Checkbox(content=Field(content=Empty(), title=tag.name))
        if tag.value_type == str(TagValueType.ANY_NUMBER):
            return Checkbox(content=Field(content=InputNumber(), title=tag.name))
        if tag.value_type == str(TagValueType.ANY_STRING):
            return Checkbox(content=Field(content=Input(), title=tag.name))
        if tag.value_type == str(TagValueType.ONEOF_STRING):
            items = [RadioGroup.Item(pv, pv, Empty()) for pv in tag.possible_values]
            return Checkbox(content=Field(content=RadioGroup(items=items), title=tag.name))

    def set(self, label: Label):
        self._label = label
        label_class = label.obj_class
        for i, tm in enumerate(self._project_meta.tag_metas):
            if tm.applicable_to == TagApplicableTo.OBJECTS_ONLY:
                if label_class.name in tm.applicable_classes:
                    self._tag_inputs[i].show()
                else:
                    self._tag_inputs[i].hide()
            else:
                self._tag_inputs[i].show()
            tag = label.tags.get(tm.name)
            if tag is None:
                self._set_null_tag_value(i)
            else:
                self._set_tag_value(i, tag.value)

    def _set_tag_value(self, idx, value):
        tag_component = self._tag_inputs[idx]
        tag_component.check()
        content = tag_component._content._content
        if type(content) is InputNumber:
            content.value = value
            content.update_state()
        if type(content) is Input:
            content.set_value(value)
        if type(content) is RadioGroup:
            content.set_value(value)

    def _get_tag_value(self, idx):
        tag_component = self._tag_inputs[idx]
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

    def _set_null_tag_value(self, idx):
        tag_component = self._tag_inputs[idx]
        tag_component.uncheck()
        content = tag_component._content._content
        if type(content) is InputNumber:
            content.value = None
            content.update_state()
        if type(content) is Input:
            content.set_value(None)
        if type(content) is RadioGroup:
            content.set_value(None)

    def get_label(self):
        tags = []
        for i, tm in enumerate(self._project_meta.tag_metas):
            tag_value = self._get_tag_value(i)
            if tag_value is not None:
                if type(tag_value) is bool:
                    tag_value = None
                tag = Tag(tm, tag_value)
                tags.append(tag)
        return self._label.clone(tags=tags)

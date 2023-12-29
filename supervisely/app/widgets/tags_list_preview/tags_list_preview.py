from typing import List, Union, Dict

from supervisely.app.widgets import Widget
from supervisely import TagMeta, TagMetaCollection
from supervisely.annotation.tag_meta import TagValueType
from supervisely.app.content import StateJson
from supervisely.imaging.color import rgb2hex


VALUE_TYPE_NAME = {
    str(TagValueType.NONE): "NONE",
    str(TagValueType.ANY_STRING): "TEXT",
    str(TagValueType.ONEOF_STRING): "ONE OF",
    str(TagValueType.ANY_NUMBER): "NUMBER",
}


class TagsListPreview(Widget):
    def __init__(
        self,
        tag_metas: Union[List[TagMeta], TagMetaCollection] = [],
        max_width: int = 300,
        empty_text: str = None,
        widget_id: int = None,
    ):
        self._tag_metas = tag_metas
        self._max_width = self._get_max_width(max_width)
        self._empty_text = empty_text
        super().__init__(widget_id=widget_id, file_path=__file__)

    def _get_max_width(self, value):
        if value < 150:
            value = 150
        return f"{value}px"

    def get_json_data(self):
        return {
            "maxWidth": self._max_width,
        }

    def get_json_state(self) -> Dict:
        return {
            "tags": [
                {
                    "name": f"<b>{tag_meta.name}</b>",
                    "valueType": VALUE_TYPE_NAME[tag_meta.value_type],
                    "color": rgb2hex(tag_meta.color),
                }
                for tag_meta in self._tag_metas
            ]
        }

    def set(self, tag_metas: Union[List[TagMeta], TagMetaCollection]):
        self._tag_metas = tag_metas
        StateJson()[self.widget_id] = self.get_json_state()
        StateJson().send_changes()

from typing import List, Optional, Union

from supervisely import TagMeta, TagMetaCollection, TagValueType
from supervisely.app.content import StateJson
from supervisely.app.widgets import NotificationBox, Widget
from supervisely.imaging.color import rgb2hex

VALUE_TYPE_NAME = {
    str(TagValueType.NONE): "NONE",
    str(TagValueType.ANY_STRING): "TEXT",
    str(TagValueType.ONEOF_STRING): "ONE OF",
    str(TagValueType.ANY_NUMBER): "NUMBER",
}


class TagsListSelector(Widget):
    class Routes:
        CHECKBOX_CHANGED = "checkbox_cb"

    def __init__(
        self,
        tag_metas: Optional[Union[List[TagMeta], TagMetaCollection]] = [],
        multiple: Optional[bool] = False,
        empty_notification: Optional[NotificationBox] = None,
        widget_id: Optional[str] = None,
    ):
        self._tag_metas = tag_metas
        self._multiple = multiple
        if empty_notification is None:
            empty_notification = NotificationBox(
                title="No tags",
                description="No tags to select.",
            )
        self.empty_notification = empty_notification
        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self):
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

    def get_json_state(self):
        return {"selected": [False for _ in self._tag_metas]}

    def set(self, tags: Union[List[TagMeta], TagMetaCollection]):
        selected_tags = [cls.name for cls in self.get_selected_tags()]
        self._tag_metas = tags
        StateJson()[self.widget_id]["selected"] = [
            cls.name in selected_tags for cls in self._tag_metas
        ]
        self.update_data()
        StateJson().send_changes()

    def get_selected_tags(self):
        selected = StateJson()[self.widget_id]["selected"]
        return [cls for cls, is_selected in zip(self._tag_metas, selected) if is_selected]

    def select_all(self):
        StateJson()[self.widget_id]["selected"] = [True for _ in self._tag_metas]
        StateJson().send_changes()

    def deselect_all(self):
        # pylint: disable=no-member
        StateJson()[self.widget_id]["selected"] = [False for _ in self._tags]
        StateJson().send_changes()

    def select(self, names: List[str]):
        selected = [cls.name in names for cls in self._tag_metas]
        StateJson()[self.widget_id]["selected"] = selected
        StateJson().send_changes()

    def deselect(self, names: List[str]):
        selected = StateJson()[self.widget_id]["selected"]
        for idx, cls in enumerate(self._tag_metas):
            if cls.name in names:
                selected[idx] = False
        StateJson()[self.widget_id]["selected"] = selected
        StateJson().send_changes()

    def set_multiple(self, value: bool):
        self._multiple = value

    def get_all_tags(self):
        return self._tag_metas

    def selection_changed(self, func):
        route_path = self.get_route_path(TagsListSelector.Routes.CHECKBOX_CHANGED)
        server = self._sly_app.get_server()
        self._checkboxes_handled = True

        @server.post(route_path)
        def _click():
            selected = self.get_selected_tags()
            func(selected)

        return _click

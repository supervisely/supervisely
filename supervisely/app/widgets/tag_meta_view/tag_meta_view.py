from supervisely.app.widgets import Widget
from supervisely.annotation.tag_meta import TagMeta


class TagMetaView(Widget):
    def __init__(
        self,
        tag_meta: TagMeta,
        show_type_text: bool = True,
        limit_long_names: bool = False,
        widget_id: str = None,
    ):
        self._tag_meta = tag_meta
        self._show_type_text = show_type_text
        self._limit_long_names = limit_long_names
        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self):
        res = self._tag_meta.to_json()
        res["limit_long_names"] = self._limit_long_names
        res["type_text"] = None
        if self._show_type_text is True:
            res["type_text"] = self._tag_meta.value_type.upper()
        return res

    def get_json_state(self):
        return None

from supervisely.app import StateJson
from supervisely.app.widgets import Widget
from typing import List

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

SUPPORTED_TAG_WIDGET_TYPES = ["primary", "gray", "success", "warning", "danger"]

# @TODO: fix problem with close transition id


class ElementTagsList(Widget):
    class Routes:
        CLOSE = "tag_close_cb"

    class Tag:
        def __init__(
            self,
            text: str,
            type: Literal["primary", "gray", "success", "warning", "danger"] = "primary",
            hit: bool = False,
            color: str = "",
            closable: bool = False,
            close_transition: bool = False,
        ):
            self._text = text
            self._type = type
            self._hit = hit
            self._color = color
            self._closable = closable
            self._close_transition = close_transition

        @property
        def text(self):
            return self._text

        @property
        def type(self):
            return self._type

        @property
        def hit(self):
            return self._hit

        @property
        def color(self):
            return self._color

        def to_json(self):
            return {
                "text": self._text,
                "type": self._type,
                "hit": self._hit,
                "color": self._color,
                "closable": self._closable,
                "close_transition": self._close_transition,
            }

        @classmethod
        def from_json(cls, tag_json):
            return cls(
                tag_json["text"],
                tag_json["type"],
                tag_json["hit"],
                tag_json["color"],
                tag_json["closable"],
                tag_json["close_transition"],
            )

    def __init__(
        self,
        tags: List[Tag] = [],
        widget_id: str = None,
    ):
        self._clicked_tag = None

        self._validate_tags(tags)
        self._tags = tags

        self._close_handled = False

        super().__init__(widget_id=widget_id, file_path=__file__)

    def _validate_tags(self, tags):
        for tag in tags:
            if not isinstance(tag, ElementTagsList.Tag):
                raise ValueError(f"tag type should be class ElementTagsList.Tag")

    def get_json_data(self):
        return {}

    def get_json_state(self):
        return {"tags": [tag.to_json() for tag in self._tags]}

    def set_tags(self, tags: List[Tag]):
        self._tags = tags
        self.update_state()
        StateJson().send_changes()

    def get_tags(self):
        return [ElementTagsList.Tag.from_json(tag) for tag in StateJson()[self.widget_id]["tags"]]

    def add_tags(self, tags: List[Tag]):
        self._tags = self.get_tags()
        self._tags.extend(tags)
        self.update_state()
        StateJson().send_changes()

    def close(self, func):
        route_path = self.get_route_path(ElementTagsList.Routes.CLOSE)
        server = self._sly_app.get_server()
        self._changes_handled = True

        @server.post(route_path)
        async def _click():
            res = self.get_tags()
            func(res)

        return _click

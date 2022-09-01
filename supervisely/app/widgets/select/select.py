from supervisely.app import DataJson
from supervisely.app.widgets import Widget
from typing import List
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


class SelectItem:
    def __init__(self, value, label: str = None, group: str = None) -> None:
        self.value = value
        self.label = label
        if label is None:
            self.label = str(self.value)
        self.group = group


class Select(Widget):
    def __init__(
        self,
        item: List[SelectItem],
        filterable: bool = False,
        placeholder: str = None,
        widget_id: str = None,
    ):
        self._text = None
        self._status = None
        self._icon = None
        self._icon_color = None
        self._text_color = None
        super().__init__(widget_id=widget_id, file_path=__file__)
        self.set(text, status)

    def get_json_data(self):
        return {
            "status": self._status,
            "text": self._text,
            "text_color": self._text_color,
            "icon": self._icon,
            "icon_color": self._icon_color,
        }

    def get_json_state(self):
        return None

    @property
    def text(self):
        return self._text

    @text.setter
    def text(self, value):
        self._text = value
        self.update_data()
        DataJson().send_changes()

    @property
    def status(self):
        return self._status

    @status.setter
    def status(self, value: Literal["text", "info", "success", "warning", "error"]):
        if value not in type_to_icon:
            raise ValueError(f'Unknown status "{value}"')
        self._status = value
        self._icon = type_to_icon[self._status]
        self._icon_color = type_to_icon_color[self._status]
        self._text_color = type_to_text_color[self._status]
        self.update_data()
        DataJson().send_changes()

    def set(
        self, text: str, status: Literal["text", "info", "success", "warning", "error"]
    ):
        self.text = text
        self.status = status

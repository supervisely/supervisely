from supervisely.app import DataJson, StateJson
from supervisely.app.widgets import Widget


class MenuItem(Widget):
    def __init__(
        self,
        index: str,
        name: str,
        content: Widget,
        icon: str = None,
        widget_id: str = None,
    ):
        self.index = index
        self.name = name
        self.icon = icon
        self._content = content
        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self):
        return {"index": self.index, "name": self.name, "icon": self.icon}

    def get_json_state(self):
        return {}

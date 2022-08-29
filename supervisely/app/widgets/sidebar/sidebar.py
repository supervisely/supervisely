from supervisely.app import DataJson, StateJson
from supervisely.app.widgets import Widget


class Sidebar(Widget):
    def __init__(self, widget_id: str = None):
        super().__init__(widget_id=widget_id, file_path=__file__)
        StateJson()["app_body_padding"] = "0px"

    def get_json_data(self):
        return {}

    def get_json_state(self):
        return {}

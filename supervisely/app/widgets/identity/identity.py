from supervisely.app.widgets import Widget


class Identity(Widget):
    """Pass-through wrapper: renders child widget without adding layout or behavior. Useful for conditional display."""

    def __init__(self, content: Widget, widget_id: str = None):
        self._content = content
        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self):
        return {}

    def get_json_state(self):
        return {}

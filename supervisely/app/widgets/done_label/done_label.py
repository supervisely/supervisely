from supervisely.app.widgets import Widget

INFO = "info"
WARNING = "warning"
ERROR = "error"


class DoneLabel(Widget):
    def __init__(
        self,
        text: str = None,
        widget_id: str = None,
    ):
        self.text = text
        super().__init__(widget_id=widget_id, file_path=__file__)

    def init_data(self):
        return {"text": self.text}

    def init_state(self):
        return None

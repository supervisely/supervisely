from supervisely.app.widgets import Widget


class FlowNode(Widget):
    def __init__(
        self,
        x: int = 0,
        y: int = 0,
        content: Widget = None,
        widget_id: str = None,
    ):
        self.position = {"x": x, "y": y}
        self.x = x
        self.y = y
        self.content = content
        super().__init__(widget_id=widget_id, file_path=__file__)

    def to_json(self):
        return {
            "id": self.widget_id,
            "position": self.position,
            "content_html": self.content.to_html() if self.content else "",
        }

    @property
    def key(self) -> str:
        return f"node-{self.widget_id}"

    def get_json_state(self):
        return self.content.get_json_state() if self.content else {}

    def get_json_data(self):
        return self.content.get_json_data() if self.content else {}

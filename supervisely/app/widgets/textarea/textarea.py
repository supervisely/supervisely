from supervisely.app.widgets import Widget


class Textarea(Widget):
    def __init__(
        self,
        text: str = None,
        placeholder: str = "Please input",
        rows: int = 2,
        autosize: bool = True,
        widget_id=None,
    ):
        self._text = text
        self._placeholder = placeholder
        self._rows = rows
        self._autosize = autosize
        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self):
        return {
            "text": self._text,
            "placeholder": self._placeholder,
            "rows": self._rows,
            "autosize": self._autosize,
        }

    def get_json_state(self):
        return None


    @property
    def text(self):
        return self._text
from supervisely.app import DataJson, StateJson
from supervisely.app.widgets import Widget
from typing import Dict, Optional
from supervisely.app.widgets import Editor, Text, TextArea, Input


class CopyToClipboard(Widget):
    def __init__(
        self,
        content: Optional[Editor or Text or TextArea or Input or str] = "",
        widget_id: str = None,
    ):
        self._content = content
        self._content_text = None

        self._init_content(content)

        super().__init__(widget_id=widget_id, file_path=__file__)

    def _init_content(self, content):
        self._res_data = {}
        self._res_state = {}
        self._editor_or_input = False
        self._text = False
        self._only_string = False
        if type(content) is Editor:
            self._editor_or_input = True
            self._content_text = content.get_text()
            self._res_state = {"content": self._content_text, "curr_widget_text": "text"}
        elif type(content) is Text:
            self._text = True
            self._content_text = content.text
            self._res_data = {"content": self._content_text, "curr_widget_text": "text"}
        elif type(content) is TextArea:
            self._text = True
            self._content_text = content.get_value()
            self._res_data = {"content": self._content_text, "curr_widget_text": "value"}
        elif type(content) is Input:
            self._editor_or_input = True
            self._content_text = content.get_value()
            self._res_state = {
                "content": self._content_text,
                "value": self._content_text,
                "curr_widget_text": "value",
            }
        else:
            self._only_string = True
            self._content_text = content
            self._res_state = {"content": self._content_text}

    def get_json_data(self) -> Dict:
        return self._res_data

    def get_json_state(self) -> Dict:
        return self._res_state

    def get_content_text(self):
        if self._editor_or_input is True or self._only_string is True:
            return StateJson()[self.widget_id]["content"]
        else:
            return DataJson()[self.widget_id]["content"]

    def get_content(self):
        return self._content

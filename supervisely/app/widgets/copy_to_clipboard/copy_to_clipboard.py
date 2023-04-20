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

    # def _init_content(self, content):
    #     self._editor = False
    #     self._text = False
    #     self._textarea = False
    #     self._input = False
    #     if type(content) is Editor:
    #         self._content_text = content.get_text()
    #         self._editor = True
    #         self._res_data = {
    #             "options": {
    #                 "height": f"{content._height_px}px",
    #                 "mode": f"ace/mode/{content._language_mode}",
    #                 "readOnly": content._readonly,
    #                 "showGutter": content._show_line_numbers,
    #                 "maxLines": content._height_lines,
    #                 "highlightActiveLine": content._highlight_active_line,
    #             }
    #         }
    #         self._res_state = {"content": self._content_text}

    #     elif type(content) is Text:
    #         self._content_text = content.text
    #         self._text = True
    #         self._res_data = {
    #             "status": content._status,
    #             "text": content._text,
    #             "text_color": content._text_color,
    #             "icon": content._icon,
    #             "icon_color": content._icon_color,
    #         }
    #         self._res_state = {"content": self._content_text}

    #     elif type(content) is TextArea:
    #         self._content_text = content.get_value()
    #         self._textarea = True
    #         self._res_data = {
    #             "value": content._value,
    #             "placeholder": content._placeholder,
    #             "rows": content._rows,
    #             "autosize": content._autosize,
    #             "readonly": content._readonly,
    #         }
    #         self._res_state = {"content": self._content_text}

    #     elif type(content) is Input:
    #         self._content_text = content.get_value()
    #         self._input = True
    #         self._res_data = {
    #             "minlength": content._minlength,
    #             "maxlength": content._maxlength,
    #             "placeholder": content._placeholder,
    #             "size": content._size,
    #             "readonly": content._readonly,
    #             "type": content._type,
    #         }

    #         self._input_state = content._value
    #         self._res_state = {"content": self._content_text, "value": self._input_state}
    #     else:
    #         self._content_text = content
    #         self._res_state = {"content": self._content_text}

    def get_json_data(self) -> Dict:
        return self._res_data

    def get_json_state(self) -> Dict:
        return self._res_state

    # def set_content(self, content: Widget):
    #     self._init_content(content)
    #     StateJson()[self.widget_id] = self._res_state
    #     StateJson().send_changes()
    #     DataJson()[self.widget_id] = self._res_data
    #     DataJson().send_changes()

    def get_content_text(self):
        if self._editor_or_input is True or self._only_string is True:
            return StateJson()[self.widget_id]["content"]
        else:
            return DataJson()[self.widget_id]["content"]

    def get_content(self):
        return self._content

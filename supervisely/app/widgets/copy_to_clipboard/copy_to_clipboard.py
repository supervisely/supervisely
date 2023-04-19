from supervisely.app import DataJson, StateJson
from supervisely.app.widgets import Widget
from typing import Dict, Optional
from supervisely.app.widgets import Editor, Text, TextArea, Input


class CopyToClipboard(Widget):
    def __init__(
        self,
        content: Optional[Editor or Text or TextArea or str] = "",
        widget_id: str = None,
    ):
        self._content = content
        self._editor = False
        self._text = False
        self._textarea = False
        self._input = False
        self._res_data = {}

        if type(content) is Editor:
            self._content_text = content.get_text()
            self._editor = True
            self._res_data = {
                "options": {
                    "height": f"{content._height_px}px",
                    "mode": f"ace/mode/{content._language_mode}",
                    "readOnly": content._readonly,
                    "showGutter": content._show_line_numbers,
                    "maxLines": content._height_lines,
                    "highlightActiveLine": content._highlight_active_line,
                }
            }

        elif type(content) is Text:
            self._content_text = content.text
            self._text = True
            self._res_data = {
                "status": content._status,
                "text": content._text,
                "text_color": content._text_color,
                "icon": content._icon,
                "icon_color": content._icon_color,
            }
        elif type(content) is TextArea:
            self._content_text = content.get_value()
            self._textarea = True
            self._res_data = {
                "value": content._value,
                "placeholder": content._placeholder,
                "rows": content._rows,
                "autosize": content._autosize,
                "readonly": content._readonly,
            }

        elif type(content) is Input:
            self._content_text = content.get_value()
            self._input = True
            self._res_data = {
                "minlength": content._minlength,
                "maxlength": content._maxlength,
                "placeholder": content._placeholder,
                "size": content._size,
                "readonly": content._readonly,
                "type": content._type,
            }

            self._input_state = content._value
        else:
            self._content_text = content

        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self) -> Dict:
        return self._res_data

    def get_json_state(self) -> Dict:
        if self._input is True:
            return {"content": self._content_text, "value": self._input_state}
        else:
            return {"content": self._content_text}

    def set_content(self, content: Widget):
        self._content_text = content
        StateJson()[self.widget_id]["content"] = self._content_text
        StateJson().send_changes()

    def get_content_text(self):
        return StateJson()[self.widget_id]["content"]

    def get_content(self):
        return self._content

from supervisely.app import DataJson
from supervisely.app.widgets import Widget
from typing import Optional, Literal

class Editor(Widget):
    def __init__(
        self,
        initial_code: Optional[str] = "",
        height_px: Optional[int] = 100,
        height_lines: Optional[int] = None, # overwrites height_px if specified. If >= 1000, all lines will be displayed.
        language_mode: Optional[Literal['json', 'html', 'plain_text', 'yaml', 'python']] = 'json',
        readonly: Optional[bool] = False,
        show_line_numbers: Optional[bool] = True,
        highlight_active_line: Optional[bool] = True,
        restore_default_button: Optional[bool] = True,
        widget_id: Optional[int] = None,
    ):
        self._initial_code = initial_code
        self._current_code = initial_code
        self._height_px = height_px
        self._height_lines = height_lines
        self._language_mode = language_mode
        self._readonly = readonly
        self._show_line_numbers = show_line_numbers
        self._highlight_active_line = highlight_active_line
        self._restore_button = restore_default_button

        super(Editor, self).__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self):
        return {
            "editor_options": {
                "height": f"{self._height_px}px",
                "mode": f"ace/mode/{self._language_mode}",
                "readOnly": self._readonly,
                "showGutter": self._show_line_numbers,
                "maxLines": self._height_lines,
                "highlightActiveLine": self._highlight_active_line,
            },
        }

    def get_json_state(self):
        return {"code": self._current_code}

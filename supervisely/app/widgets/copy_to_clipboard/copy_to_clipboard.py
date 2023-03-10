from supervisely.app import DataJson, StateJson
from supervisely.app.widgets import Widget
from typing import Dict, Optional

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


class CopyToClipboard(Widget):
    def __init__(
        self,
        content: str = "",
        max_lines: Optional[int] = 200,
        language_mode: Optional[Literal["json", "html", "plain_text", "yaml", "python"]] = "json",
        readonly: Optional[bool] = True,
        show_line_numbers: Optional[bool] = False,
        highlight_active_line: Optional[bool] = False,
        widget_id: str = None,
    ):

        self._content = content
        self._max_lines = max_lines
        self._language_mode = language_mode
        self._readonly = readonly
        self._show_line_numbers = show_line_numbers
        self._highlight_active_line = highlight_active_line

        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self) -> Dict:
        return {
            "options": {
                "mode": self._language_mode,
                "showGutter": self._show_line_numbers,
                "readOnly": self._readonly,
                "maxLines": self._max_lines,
                "highlightActiveLine": self._highlight_active_line,
            },
        }

    def get_json_state(self) -> Dict:
        return {"content": self._content}

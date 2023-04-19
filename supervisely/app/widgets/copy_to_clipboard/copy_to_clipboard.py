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
        content: Optional[str] = "",
        # content: Widget = None,
        max_lines: Optional[int] = 200,
        language_mode: Optional[
            Literal["json", "html", "plain_text", "yaml", "python"]
        ] = "ace/mode/json",
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
        return {}

    def get_json_state(self) -> Dict:
        return {"content": self._content}

    # def get_json_data(self) -> Dict:
    #     return {
    #         "options": {
    #             "mode": self._language_mode,
    #             "showGutter": self._show_line_numbers,
    #             "readOnly": self._readonly,
    #             "maxLines": self._max_lines,
    #             "highlightActiveLine": self._highlight_active_line,
    #         },
    #     }

    # def get_json_state(self) -> Dict:
    #     return {"content": self._content}

    # def get_content(self) -> str:
    #     return StateJson()[self.widget_id]["content"]

    # def set_content(
    #     self,
    #     content: Optional[str] = "",
    #     language_mode: Optional[Literal["json", "html", "plain_text", "yaml", "python"]] = None,
    # ) -> None:
    #     self._language_mode = language_mode
    #     StateJson()[self.widget_id]["content"] = content
    #     StateJson().send_changes()
    #     if language_mode is not None:
    #         self._language_mode = f"ace/mode/{language_mode}"
    #         DataJson()[self.widget_id]["options"]["mode"] = self._language_mode
    #         DataJson().send_changes()

    # @property
    # def readonly(self) -> bool:
    #     return self._readonly

    # @readonly.setter
    # def readonly(self, value: bool):
    #     self._readonly = value
    #     DataJson()[self.widget_id]["options"]["readOnly"] = self._readonly
    #     DataJson().send_changes()

    # def show_line_numbers(self):
    #     self._show_line_numbers = True
    #     DataJson()[self.widget_id]["options"]["showGutter"] = self._show_line_numbers
    #     DataJson().send_changes()

    # def hide_line_numbers(self):
    #     self._show_line_numbers = False
    #     DataJson()[self.widget_id]["options"]["showGutter"] = self._show_line_numbers
    #     DataJson().send_changes()

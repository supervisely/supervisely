from typing import Dict, Literal, Optional, Union

from supervisely.app import DataJson, StateJson
from supervisely.app.widgets import Button, Widget


class Editor(Widget):
    """Code editor with syntax highlighting."""

    def __init__(
        self,
        initial_text: Optional[str] = "",
        height_px: Optional[int] = 100,
        height_lines: Optional[int] = None,
        language_mode: Optional[Literal["json", "html", "plain_text", "yaml", "python"]] = "json",
        readonly: Optional[bool] = False,
        show_line_numbers: Optional[bool] = True,
        highlight_active_line: Optional[bool] = True,
        restore_default_button: Optional[bool] = True,
        widget_id: Optional[int] = None,
        auto_format: bool = False,
    ):
        """
        :param initial_text: Initial content.
        :param height_px: Height in pixels.
        :param height_lines: Height in lines (overrides height_px; >=1000 shows all).
        :param language_mode: json, html, plain_text, yaml, python.
        :param readonly: Read-only mode.
        :param show_line_numbers: Show line numbers.
        :param highlight_active_line: Highlight current line.
        :param restore_default_button: Show restore button.
        :param auto_format: Auto-format JSON on init.
        :param widget_id: Widget identifier.

        :Usage Example:

            .. code-block:: python

                from supervisely.app.widgets import Editor
                editor = Editor(initial_text="print('hi')", language_mode="python")
        """
        self._initial_code = initial_text
        self._current_code = initial_text
        self._height_px = height_px
        self._height_lines = height_lines
        self._language_mode = language_mode
        self._readonly = readonly
        self._show_line_numbers = show_line_numbers
        self._highlight_active_line = highlight_active_line
        self._restore_button = None
        self._auto_format = auto_format

        if restore_default_button:
            self._restore_button = Button("Restore Default", button_type="text", plain=True)

            @self._restore_button.click
            def restore_default():
                self._current_code = self._initial_code
                StateJson()[self.widget_id]["text"] = self._current_code
                StateJson().send_changes()

        super(Editor, self).__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self) -> Dict[str, Union[bool, str, int]]:
        """Returns dictionary with widget data, which defines the appearance and behavior of the widget.

        Dictionary contains the following fields:
            - editor_options: Dictionary with editor options
                - height: Height of the editor in pixels
                - mode: Language mode of the editor, one of: json, html, plain_text, yaml, python
                - readOnly: If True, editor will be readonly
                - showGutter: If True, line numbers will be shown
                - maxLines: Overwrites height if specified. If >= 1000, all lines will be displayed
                - highlightActiveLine: If True, active line will be highlighted

        :returns: Dictionary with widget data
        :rtype: Dict[str, Union[bool, str, int]]
        """
        return {
            "editor_options": {
                "height": f"{self._height_px}px",
                "mode": f"ace/mode/{self._language_mode}",
                "readOnly": self._readonly,
                "showGutter": self._show_line_numbers,
                "maxLines": self._height_lines,
                "highlightActiveLine": self._highlight_active_line,
                "formatJsonOnInit": self._auto_format,
            },
        }

    def get_json_state(self) -> Dict[str, str]:
        """Returns dictionary with widget state.

        Dictionary contains the following fields:
            - text: current text in the editor

        :returns: Dictionary with widget state
        :rtype: Dict[str, str]
        """
        return {"text": self._current_code}

    def get_text(self) -> str:
        """Returns current text in the editor.
        Same as get_value().

        :returns: current text in the editor
        :rtype: str
        """
        return StateJson()[self.widget_id]["text"]

    def get_value(self) -> str:
        """Returns current text in the editor.
        Same as get_text().

        :returns: current text in the editor
        :rtype: str
        """
        return StateJson()[self.widget_id]["text"]

    def set_text(
        self,
        text: Optional[str] = "",
        language_mode: Optional[Literal["json", "html", "plain_text", "yaml", "python"]] = None,
    ) -> None:
        """Sets current text in the editor.
        This method overwrites the current text in the editor.

        :param text: current text in the editor
        :type text: Optional[str]
        :param language_mode: Language mode of the editor, one of: json, html, plain_text, yaml, python
        :type language_mode: Optional[Literal["json", "html", "plain_text", "yaml", "python"]]
        """

        self._initial_code = text
        self._current_code = text
        self._language_mode = language_mode
        StateJson()[self.widget_id]["text"] = text
        StateJson().send_changes()
        if language_mode is not None:
            self._language_mode = f"ace/mode/{language_mode}"
            DataJson()[self.widget_id]["editor_options"]["mode"] = self._language_mode
            DataJson().send_changes()

    @property
    def readonly(self) -> bool:
        """Returns True if editor is readonly, False otherwise.

        :returns: True if editor is readonly, False otherwise
        :rtype: bool
        """
        return self._readonly

    @readonly.setter
    def readonly(self, value: bool) -> None:
        """Sets editor readonly mode.

        :param value: True if editor is readonly, False otherwise
        :type value: bool
        """
        self._readonly = value
        DataJson()[self.widget_id]["editor_options"]["readOnly"] = self._readonly
        DataJson().send_changes()

    def show_line_numbers(self) -> None:
        """Enables line numbers in the editor."""
        self._show_line_numbers = True
        DataJson()[self.widget_id]["editor_options"]["showGutter"] = self._show_line_numbers
        DataJson().send_changes()

    def hide_line_numbers(self):
        """Disables line numbers in the editor."""
        self._show_line_numbers = False
        DataJson()[self.widget_id]["editor_options"]["showGutter"] = self._show_line_numbers
        DataJson().send_changes()

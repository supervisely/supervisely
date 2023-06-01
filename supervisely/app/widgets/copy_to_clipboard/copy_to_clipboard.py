from supervisely.app import DataJson, StateJson
from supervisely.app.widgets import Widget
from typing import Dict, Optional, Union
from supervisely.app.widgets import Editor, Text, TextArea, Input


class CopyToClipboard(Widget):
    def __init__(
        self,
        content: Union[Editor, Text, TextArea, Input, str] = "",
        widget_id: Optional[str] = None,
    ):
        self._content = content

        if isinstance(content, Editor):
            self._editor_or_input = True
            self._content_text = content.get_text()
            self._curr_widget_text = "text"
        elif isinstance(content, Input):
            self._editor_or_input = True
            self._content_text = content.get_value()
            self._curr_widget_text = "value"
        elif isinstance(content, Text):
            self._text_or_textarea = True
            self._content_text = content.text
            self._curr_widget_text = "text"
        elif isinstance(content, TextArea):
            self._text_or_textarea = True
            self._content_text = content.get_value()
            self._curr_widget_text = "value"
        elif isinstance(content, str):
            self._only_string = True
            self._content_text = content
            self._curr_widget_text = None
        else:
            raise TypeError(
                f"Supported types: str, Editor, Text, TextArea, Input. Your type: {type(content).__name__}"
            )

        super().__init__(widget_id=widget_id, file_path=__file__)


    def get_json_data(self) -> Dict:
        return {
            "content": self._content_text,
            "curr_widget_text": self._curr_widget_text
        }

    def get_json_state(self) -> Dict:
        return {
            "content": self._content_text,
            "curr_widget_text": self._curr_widget_text
        }

    def get_content(self) -> Union[Editor, Input, Text, TextArea , str]:
        return self._content

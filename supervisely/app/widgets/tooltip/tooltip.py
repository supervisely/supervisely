from typing import List, Optional, Dict, Union, Literal
from supervisely.app import StateJson, DataJson
from supervisely.app.widgets import Widget


class Tooltip(Widget):
    def __init__(
        self,
        content: Union[str, List[str]],
        element: Widget,
        effect: Optional[Literal["dark", "light"]] = "dark",
        placement: Optional[
            Literal[
                "top",
                "top-start",
                "top-end",
                "bottom",
                "bottom-start",
                "bottom-end",
                "left",
                "left-start",
                "left-end",
                "right",
                "right-start",
                "right-end",
            ]
        ] = "bottom",
        disabled: Optional[bool] = False,
        offset: Optional[int] = 0,
        transition: Optional[
            Literal[
                "el-fade-in-linear",
                "el-fade-in",
            ]
        ] = "el-fade-in-linear",
        open_delay: Optional[int] = 0,
        widget_id: Optional[str] = None,
    ):
        self._placement = placement
        self._element = element
        self._content = content
        self._effect = effect
        self._disabled = disabled
        self._offset = offset
        self._transition = transition
        self._open_delay = open_delay
        self._multiline = True if isinstance(self._content, List) else False

        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self) -> Dict:
        return {
            "content": self._content,
            "effect": self._effect,
            "placement": self._placement,
            "disabled": self._disabled,
            "offset": self._offset,
            "transition": self._transition,
            "open_delay": self._open_delay,
            "multiline": self._multiline,
        }

    def get_json_state(self):
        return None

    def set_content(self, content: Union[str, List[str]]):
        """
        To make tooltip multiline - pass content as list of lines
        """
        self._content = content
        self._multiline = True if isinstance(self._content, List) else False
        DataJson()[self.widget_id]["content"] = self._content
        DataJson()[self.widget_id]["multiline"] = self._multiline
        DataJson().send_changes()

    def set_placement(
        self,
        placement: Literal[
            "top",
            "top-start",
            "top-end",
            "bottom",
            "bottom-start",
            "bottom-end",
            "left",
            "left-start",
            "left-end",
            "right",
            "right-start",
            "right-end",
        ],
    ):
        self._placement = placement
        DataJson()[self.widget_id]["placement"] = self._placement
        DataJson().send_changes()

    def set_disabled(self, disabled: bool):
        self._disabled = disabled
        DataJson()[self.widget_id]["disabled"] = self._disabled
        DataJson().send_changes()

    def set_offset(self, offset: int):
        self._offset = offset
        DataJson()[self.widget_id]["offset"] = self._offset
        DataJson().send_changes()

    def set_transition(
        self,
        transition: Literal["el-fade-in-linear", "el-fade-in"],
    ):
        self._transition = transition
        DataJson()[self.widget_id]["transition"] = self._transition
        DataJson().send_changes()

    def set_open_delay(self, open_delay: int):
        """
        Milliseconds
        """
        self._open_delay = open_delay
        DataJson()[self.widget_id]["open_delay"] = self._open_delay
        DataJson().send_changes()

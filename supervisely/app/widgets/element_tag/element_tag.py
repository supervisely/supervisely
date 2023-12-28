from typing import Dict, Literal, Optional, Union

from supervisely.app import StateJson
from supervisely.app.widgets import Widget

SUPPORTED_TAG_WIDGET_TYPES = ["primary", "gray", "success", "warning", "danger"]


class ElementTag(Widget):
    """ElementTag widget in Supervisely is a widget that allows users to display elements tag in the UI.

    Read about it in `Developer Portal <https://developer.supervisely.com/app-development/widgets/text-elements/elementtag>`_
        (including screenshots and examples).

    :param text: Tag text
    :type text: Optional[str]
    :param type: Tag type, one of: primary, gray, success, warning, danger
    :type type: Optional[Literal["primary", "gray", "success", "warning", "danger"]]
    :param hit: If True, tag will be highlighted
    :type hit: Optional[bool]
    :param color: Tag color
    :type color: Optional[str]
    :param widget_id: An identifier of the widget.
    :type widget_id: str, optional

    :Usage example:
    .. code-block:: python

            from supervisely.app.widgets import ElementTag

            element_tag = ElementTag(
                text="Tag",
                type="primary",
                hit=True,
                color="#20a0ff",
            )
    """

    class Routes:
        CLOSE = "tag_close_cb"

    def __init__(
        self,
        text: Optional[str] = "",
        type: Optional[Literal["primary", "gray", "success", "warning", "danger", None]] = None,
        hit: Optional[bool] = False,
        color: Optional[str] = "",
        widget_id: Optional[str] = None,
    ):
        self._text = text
        self._validate_type(type)
        self._type = type
        self._color = color
        self._hit = hit
        self._tag_close = False
        self._clicked_tag = None

        super().__init__(widget_id=widget_id, file_path=__file__)

    def _validate_type(self, value):
        if value is None:
            return
        if value not in SUPPORTED_TAG_WIDGET_TYPES:
            raise ValueError(
                "type = {!r} is unknown, should be one of {}".format(
                    value, SUPPORTED_TAG_WIDGET_TYPES
                )
            )

    def get_json_data(self) -> Dict:
        """ElementTag widget has no data, the method returns an empty dict."""
        return {}

    def get_json_state(self) -> Dict[str, Union[str, bool]]:
        """Returns dictionary with widget state.

        The dictionary contains the following fields:
            - text: Tag text
            - type: Tag type, one of: primary, gray, success, warning, danger
            - hit: If True, tag will be highlighted
            - color: Tag color

        :return: dictionary with widget state
        :rtype: Dict[str, Union[str, bool]]
        """
        return {
            "text": self._text,
            "type": self._type,
            "hit": self._hit,
            "color": self._color,
        }

    @property
    def text(self) -> str:
        """Returns current tag text

        :return: current tag text
        :rtype: str
        """
        return self._text

    @property
    def type(self) -> Literal["primary", "gray", "success", "warning", "danger"]:
        """Returns current tag type

        :return: current tag type
        :rtype: Literal["primary", "gray", "success", "warning", "danger"]
        """
        return self._type

    @property
    def hit(self) -> bool:
        """Returns True if tag is highlighted, False otherwise

        :return: True if tag is highlighted, False otherwise
        :rtype: bool
        """
        return self._hit

    @property
    def color(self) -> str:
        """Returns current tag color.

        :return: current tag color
        :rtype: str
        """
        return self._color

    def set_text(self, value: str) -> None:
        """Sets current tag text.

        :param value: current tag text
        :type value: str
        """
        StateJson()[self.widget_id]["text"] = value
        StateJson().send_changes()

    def get_text(self) -> str:
        """Returns current tag text.

        :return: current tag text
        :rtype: str
        """
        return StateJson()[self.widget_id]["text"]

    def set_type(self, value: Literal["primary", "gray", "success", "warning", "danger"]) -> None:
        """Sets current tag type.

        :param value: current tag type
        :type value: Literal["primary", "gray", "success", "warning", "danger"]
        """
        self._validate_type(value)
        StateJson()[self.widget_id]["type"] = value
        StateJson().send_changes()

    def get_type(self) -> Literal["primary", "gray", "success", "warning", "danger"]:
        """Returns current tag type.

        :return: current tag type
        :rtype: Literal["primary", "gray", "success", "warning", "danger"]
        """
        return StateJson()[self.widget_id]["type"]

    def set_hit(self, value: bool) -> None:
        """Sets current tag highlight.

        :param value: current tag highlight
        :type value: bool
        """
        StateJson()[self.widget_id]["hit"] = value
        StateJson().send_changes()

    def get_hit(self) -> bool:
        """Returns True if tag is highlighted, False otherwise.

        :return: True if tag is highlighted, False otherwise
        :rtype: bool
        """
        return StateJson()[self.widget_id]["hit"]

    def set_color(self, value: str) -> None:
        """Sets current tag color.

        :param value: current tag color
        :type value: str
        """
        StateJson()[self.widget_id]["color"] = value
        StateJson().send_changes()

    def get_color(self) -> str:
        """Returns current tag color.

        :return: current tag color
        :rtype: str
        """
        return StateJson()[self.widget_id]["color"]

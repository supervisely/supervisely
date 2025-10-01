from typing import Any, Dict, Literal, Optional

from supervisely.app import StateJson
from supervisely.app.widgets import Widget
from supervisely.sly_logger import logger


class Card(Widget):
    """Card widget in Supervisely is a simple widget that can be used to display information or content in a compact format.

    Read about it in `Developer Portal <https://developer.supervisely.com/app-development/widgets/layouts-and-containers/card>`_
        (including screenshots and examples).


    :param title: Title of the card, will be displayed in the UI.
    :type title: Optional[str]
    :param description: Description of the card, will be displayed in the UI.
    :type description: Optional[str]
    :param collapsable: If True, the card will be collapsable in the UI.
    :type collapsable: Optional[bool]
    :param content: Widget to be displayed in the card.
    :type content: Optional[Widget]
    :param content_top_right: Widget to be displayed in the top right corner of the card.
    :type content_top_right: Optional[Widget]
    :param lock_message: Message to be displayed when the card is locked.
    :type lock_message: Optional[str]
    :param remove_padding: If True, padding will be removed from the card.
    :type remove_padding: Optional[bool]
    :param overflow: Overflow property of the card. Can be "auto", "unset" or "scroll".
    :type overflow: Optional[Literal["auto", "unset", "scroll"]]
    :param style: CSS styles string of the card.
    :type style: Optional[str]
    :param widget_id: Unique widget identifier.
    :type widget_id: str

    :Usage example:
    .. code-block:: python

        from supervisely.app.widgets import Card, Text

        text = Text("Hello, world!")
        card = Card(title="Card title", description="Card description", content=text)
    """

    def __init__(
        self,
        title: Optional[str] = None,
        description: Optional[str] = None,
        collapsable: Optional[bool] = False,
        content: Optional[Widget] = None,
        content_top_right: Optional[Widget] = None,
        lock_message: Optional[str] = "Card content is locked",
        widget_id: Optional[str] = None,
        remove_padding: Optional[bool] = False,
        overflow: Optional[Literal["auto", "unset", "scroll"]] = "auto",
        style: Optional[str] = "",
    ):
        self._title = title
        self._description = description
        self._collapsable = collapsable
        self._collapsed = False
        self._content = content
        self._show_slot = False
        self._slot_content = content_top_right
        self._remove_padding = remove_padding
        if self._slot_content is not None:
            self._show_slot = True
        self._overflow = overflow
        self._style = style
        self._options = {
            "collapsable": self._collapsable,
            "marginBottom": "0px",
            "contentOverflow": self._overflow,
        }
        self._lock_message = lock_message
        super().__init__(widget_id=widget_id, file_path=__file__)
        self._disabled = {"disabled": False, "message": self._lock_message}

    def get_json_data(self) -> Dict[str, Any]:
        """Returns dictionary with widget data, which defines the appearance and behavior of the widget.

        Dictionary contains the following fields:
            - title: Title of the card, will be displayed in the UI.
            - description: Description of the card, will be displayed in the UI.
            - collapsable: If True, the card will be collapsable in the UI.
            - options: Dictionary with the following fields:
                - collapsable: If True, the card will be collapsable in the UI.
                - marginBottom: Margin bottom of the card.
            - showSlot: If True, the slot content will be displayed.

        :return: Dictionary with widget data.
        :rtype: Dict[str, Any]
        """
        return {
            "title": self._title,
            "description": self._description,
            "collapsable": self._collapsable,
            "options": self._options,
            "show_slot": self._show_slot,
        }

    def get_json_state(self) -> Dict[str, bool]:
        """Returns dictionary with widget state.

        Dictionary contains the following fields:
            - disabled: True if card is disabled.
            - collapsed: True if card is collapsed.

        :return: Dictionary with widget state.
        :rtype: Dict[str, bool]
        """
        return {"disabled": self._disabled, "collapsed": self._collapsed}

    def collapse(self) -> None:
        """Collapses the card."""
        if self._collapsable is False:
            logger.warn(f"Card {self.widget_id} can not be collapsed")
            return
        self._collapsed = True
        StateJson()[self.widget_id]["collapsed"] = self._collapsed
        StateJson().send_changes()

    def uncollapse(self) -> None:
        """Uncollapses the card."""
        if self._collapsable is False:
            logger.warn(f"Card {self.widget_id} can not be uncollapsed")
            return
        self._collapsed = False
        StateJson()[self.widget_id]["collapsed"] = self._collapsed
        StateJson().send_changes()

    def is_collapsed(self) -> bool:
        return StateJson()[self.widget_id]["collapsed"]

    def lock(self, message: Optional[str] = None) -> None:
        """Locks the card, changes the lock message if specified.

        :param message: Message to be displayed when the card is locked.
        :type message: Optional[str]
        """
        if message is not None:
            self._lock_message = message
        self._disabled = {"disabled": True, "message": self._lock_message}
        StateJson()[self.widget_id]["disabled"] = self._disabled
        StateJson().send_changes()

    def unlock(self) -> None:
        """Unlocks the card."""
        self._disabled["disabled"] = False
        StateJson()[self.widget_id]["disabled"] = self._disabled
        StateJson().send_changes()

    def is_locked(self) -> bool:
        """Returns True if the card is locked, False otherwise.

        :return: True if the card is locked, False otherwise.
        :rtype: bool
        """
        return self._disabled["disabled"]

    @property
    def description(self) -> Optional[str]:
        """Description of the card.

        :return: Description of the card.
        :rtype: Optional[str]
        """
        return self._description

    @description.setter
    def description(self, value: str) -> None:
        """Sets the description of the card.

        :param value: Description of the card.
        :type value: str
        """
        self._description = value
        StateJson()[self.widget_id]["description"] = self._description
        StateJson().send_changes()

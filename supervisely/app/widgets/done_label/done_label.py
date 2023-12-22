from typing import Dict, Optional

from supervisely.app import DataJson
from supervisely.app.widgets import Widget

INFO = "info"
WARNING = "warning"
ERROR = "error"


class DoneLabel(Widget):
    """DoneLabel is a widget in Supervisely that is used to display messages about completed tasks or operations.

    Read about it in `Developer Portal <https://developer.supervisely.com/app-development/widgets/status-elements/donelabel>`_
        (including screenshots and examples).

    :param text: DoneLabel text
    :type text: str
    :param widget_id: An identifier of the widget.
    :type widget_id: str, optional

    :Usage example:
    .. code-block:: python

        from supervisely.app.widgets import DoneLabel

        done_label = DoneLabel(text="Done!")
    """

    def __init__(
        self,
        text: Optional[str] = None,
        widget_id: Optional[str] = None,
    ):
        self._text = text
        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self) -> Dict[str, str]:
        """Returns dictionary with widget data, which defines the appearance and behavior of the widget.

        Dictionary contains the following fields:
            - text: DoneLabel text

        :return: Dictionary with widget data
        :rtype: Dict[str, str]
        """
        return {"text": self._text}

    def get_json_state(self) -> None:
        """DoneLabel widget does not have state,
        the method returns None."""
        return None

    @property
    def text(self) -> str:
        """Returns DoneLabel text.

        :return: DoneLabel text
        :rtype: str
        """
        return self._text

    @text.setter
    def text(self, value: str) -> None:
        """Sets DoneLabel text.

        :param value: DoneLabel text
        :type value: str
        """
        self._text = value
        DataJson()[self.widget_id]["text"] = self._text
        DataJson().send_changes()

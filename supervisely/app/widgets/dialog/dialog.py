from typing import Dict, Literal, Optional

from supervisely.app.content import DataJson, StateJson
from supervisely.app.widgets import Widget


class Dialog(Widget):
    """Dialog is a widget that allows to show a dialog window that contain any other widgets.
    It can be used to show a message to the user or to ask for confirmation.

    Read about it in `Developer Portal <https://developer.supervisely.com/app-development/widgets/layouts-and-containers/dialog>`_
        (including screenshots and examples).

    :param title: Dialog title
    :type title: str
    :param content: Dialog content
    :type content: Widget
    :param size: Dialog size, one of: tiny, small, large, full
    :type size: Literal["tiny", "small", "large", "full"]
    :param widget_id: An identifier of the widget.
    :type widget_id: str, optional

    :Usage example:
    .. code-block:: python

        from supervisely.app.widgets import Dialog, Input, Button

        dialog = Dialog(title="Dialog title", content=Input("Input"), size="large")
        dialog.show()
    """
    class Routes:
        ON_CLOSE = "close_cb"

    def __init__(
        self,
        title: Optional[str] = "",
        content: Optional[Widget] = None,
        size: Optional[Literal["tiny", "small", "large", "full"]] = "small",
        widget_id: Optional[str] = None,
    ):
        self._title = title
        self._content = content
        self._size = size
        super().__init__(widget_id=widget_id, file_path=__file__)

        server = self._sly_app.get_server()
        route = self.get_route_path(Dialog.Routes.ON_CLOSE)
        @server.post(route)
        def _on_close():
            # * Change visibility state to False when dialog is closed on client side
            visible = StateJson()[self.widget_id]["visible"]
            if visible is True:
                StateJson()[self.widget_id]["visible"] = False
                # * no need to call send_changes(), as it is already changed on client side

    def get_json_data(self) -> Dict[str, str]:
        """Returns dictionary with widget data, which defines the appearance and behavior of the widget.

        Dictionary contains the following fields:
            - title: Dialog title
            - size: Dialog size, one of: tiny, small, large, full

        :return: Dictionary with widget data
        :rtype: Dict[str, str]
        """
        return {
            "title": self._title,
            "size": self._size,
        }

    def get_json_state(self) -> Dict[str, bool]:
        """Returns dictionary with widget state.

        Dictionary contains the following fields:
            - visible: Dialog visibility

        :return: Dictionary with widget state
        :rtype: Dict[str, bool]
        """
        return {
            "visible": False,
        }

    def show(self) -> None:
        """Shows the dialog."""
        StateJson()[self.widget_id]["visible"] = True
        StateJson().send_changes()

    def hide(self) -> None:
        """Hides the dialog."""
        StateJson()[self.widget_id]["visible"] = False
        StateJson().send_changes()

    @property
    def title(self) -> str:
        """Returns dialog title.

        :return: Dialog title
        :rtype: str
        """
        return self._title

    @title.setter
    def title(self, title: str) -> None:
        """Sets dialog title.

        :param title: Dialog title
        :type title: str
        """
        self._title = title
        DataJson()[self.widget_id]["title"] = title
        DataJson().send_changes()

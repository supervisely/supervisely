from typing import Dict, Literal, Optional

from supervisely.app.content import DataJson, StateJson
from supervisely.app.widgets import Widget


class Dialog(Widget):
    """Dialog window that can contain any other widgets."""
    class Routes:
        """Route name constants for this widget."""
        ON_CLOSE = "close_cb"

    def __init__(
        self,
        title: Optional[str] = "",
        content: Optional[Widget] = None,
        size: Optional[Literal["tiny", "small", "large", "full"]] = "small",
        widget_id: Optional[str] = None,
    ):
        """
        :param title: Dialog title.
        :type title: Optional[str]
        :param content: Content widget.
        :type content: Optional[Widget]
        :param size: Dialog size, one of: tiny, small, large, full.
        :type size: Optional[Literal["tiny", "small", "large", "full"]]
        :param widget_id: Widget identifier.
        :type widget_id: Optional[str]

        :Usage Example:

            .. code-block:: python

                from supervisely.app.widgets import Dialog, Input
                dialog = Dialog(title="Title", content=Input(""), size="large")
                dialog.show()
        """
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

        :returns: Dictionary with widget data
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

        :returns: Dictionary with widget state
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

        :returns: Dialog title
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

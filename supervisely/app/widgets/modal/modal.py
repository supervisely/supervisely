from typing import Dict, List, Literal, Optional

from supervisely.app.content import DataJson, StateJson
from supervisely.app.widgets import Widget


class Modal(Widget):
    """Modal overlay with close button; contains widgets, show/hide programmatically."""

    class Routes:
        """Route name constants for this widget."""
        ON_CLOSE = "close_cb"
        VALUE_CHANGED = "value_changed"

    def __init__(
        self,
        title: Optional[str] = "",
        widgets: Optional[List[Widget]] = None,
        size: Optional[Literal["tiny", "small", "large", "full"]] = "small",
        widget_id: Optional[str] = None,
    ):
        """
        :param title: Modal title.
        :param widgets: List of content widgets.
        :param size: tiny, small, large, or full.
        :param widget_id: Widget identifier.

        :Usage Example:

            .. code-block:: python

                from supervisely.app.widgets import Modal, Text, Input
                modal = Modal(title="Title", widgets=[Text("Content"), Input("")], size="small")
                modal.show()
        """
        self._title = title
        self._widgets = widgets if widgets is not None else []
        self._size = size
        self._value_changed_handled = False
        super().__init__(widget_id=widget_id, file_path=__file__)

        server = self._sly_app.get_server()
        route = self.get_route_path(Modal.Routes.ON_CLOSE)

        @server.post(route)
        def _on_close():
            # Change visibility state to False when modal is closed on client side
            visible = StateJson()[self.widget_id]["visible"]
            if visible is True:
                StateJson()[self.widget_id]["visible"] = False
                # no need to call send_changes(), as it is already changed on client side

    def get_json_data(self) -> Dict[str, str]:
        """Returns dictionary with widget data, which defines the appearance and behavior of the widget.

        Dictionary contains the following fields:
            - title: Modal title
            - size: Modal size, one of: tiny, small, large, full

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
            - visible: Modal visibility

        :returns: Dictionary with widget state
        :rtype: Dict[str, bool]
        """
        return {
            "visible": False,
        }

    def show(self) -> None:
        """Shows the modal window."""
        StateJson()[self.widget_id]["visible"] = True
        StateJson().send_changes()

    def hide(self) -> None:
        """Hides the modal window."""
        StateJson()[self.widget_id]["visible"] = False
        StateJson().send_changes()

    def show_modal(self) -> None:
        """Shows the modal window. Alias for show() method."""
        self.show()

    def close_modal(self) -> None:
        """Closes the modal window. Alias for hide() method."""
        self.hide()

    def is_opened(self) -> bool:
        """Returns whether the modal is currently open.

        :returns: True if modal is visible, False otherwise
        :rtype: bool
        """
        return StateJson()[self.widget_id]["visible"]

    def value_changed(self, func):
        """Decorator to handle modal visibility changes.
        The callback function will receive a boolean parameter: True when opened, False when closed.

        :param func: Callback function that takes a boolean parameter
        :type func: Callable[[bool], None]

        :Usage Example:

            .. code-block:: python

                @modal.value_changed
                def on_modal_state_changed(is_opened):
                    if is_opened:
                        print("Modal opened")
                    else:
                        print("Modal closed")
        """
        route_path = self.get_route_path(Modal.Routes.VALUE_CHANGED)
        server = self._sly_app.get_server()
        self._value_changed_handled = True

        @server.post(route_path)
        def _value_changed():
            is_opened = StateJson()[self.widget_id]["visible"]
            func(is_opened)

        return _value_changed

    @property
    def title(self) -> str:
        """Returns modal title.

        :returns: Modal title
        :rtype: str
        """
        return self._title

    @title.setter
    def title(self, title: str) -> None:
        """Sets modal title.

        :param title: Modal title
        :type title: str
        """
        self._title = title
        DataJson()[self.widget_id]["title"] = title
        DataJson().send_changes()

    @property
    def widgets(self) -> List[Widget]:
        """Returns list of widgets inside the modal.

        :returns: List of widgets
        :rtype: List[:class:`~supervisely.app.widgets.widget.Widget`]
        """
        return self._widgets

    @widgets.setter
    def widgets(self, widgets: List[Widget]) -> None:
        """Sets the list of widgets to display in the modal.
        Note: Changing widgets dynamically may require re-rendering.

        :param widgets: List of widgets
        :type widgets: List[:class:`~supervisely.app.widgets.widget.Widget`]
        """
        self._widgets = widgets

from typing import Dict, List, Literal, Optional

from supervisely.app.content import DataJson, StateJson
from supervisely.app.widgets import Widget


class Modal(Widget):
    """Modal is a widget that displays content in a modal overlay window with a close button.
    It can contain any other widgets, similar to Container, and provides programmatic control
    to show/hide the modal.

    Read about it in `Developer Portal <https://developer.supervisely.com/app-development/widgets/layouts-and-containers/modal>`_
        (including screenshots and examples).

    :param title: Modal window title
    :type title: str
    :param widgets: List of widgets to be displayed inside the modal
    :type widgets: Optional[List[Widget]]
    :param size: Modal size, one of: tiny, small, large, full
    :type size: Literal["tiny", "small", "large", "full"]
    :param widget_id: An identifier of the widget.
    :type widget_id: str, optional

    :Usage example:
    .. code-block:: python

        from supervisely.app.widgets import Modal, Text, Button, Input, Container

        # Create widgets to show in modal
        input_widget = Input("Enter value")
        text_widget = Text("This is modal content")

        # Create modal with multiple widgets
        modal = Modal(
            title="My Modal Window",
            widgets=[text_widget, input_widget],
            size="small"
        )

        # Show modal programmatically
        modal.show()

        # Hide modal programmatically
        modal.hide()

        # Alternative methods
        modal.show_modal()
        modal.close_modal()
    """

    class Routes:
        ON_CLOSE = "close_cb"
        VALUE_CHANGED = "value_changed"

    def __init__(
        self,
        title: Optional[str] = "",
        widgets: Optional[List[Widget]] = None,
        size: Optional[Literal["tiny", "small", "large", "full"]] = "small",
        widget_id: Optional[str] = None,
    ):
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
            - visible: Modal visibility

        :return: Dictionary with widget state
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

        :return: True if modal is visible, False otherwise
        :rtype: bool
        """
        return StateJson()[self.widget_id]["visible"]

    def value_changed(self, func):
        """Decorator to handle modal visibility changes.
        The callback function will receive a boolean parameter: True when opened, False when closed.

        :param func: Callback function that takes a boolean parameter
        :type func: Callable[[bool], None]

        :Usage example:
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

        :return: Modal title
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

        :return: List of widgets
        :rtype: List[Widget]
        """
        return self._widgets

    @widgets.setter
    def widgets(self, widgets: List[Widget]) -> None:
        """Sets the list of widgets to display in the modal.
        Note: Changing widgets dynamically may require re-rendering.

        :param widgets: List of widgets
        :type widgets: List[Widget]
        """
        self._widgets = widgets

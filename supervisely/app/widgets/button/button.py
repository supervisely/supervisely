try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from typing import Callable, Optional

from supervisely.app import DataJson
from supervisely.app.widgets import Widget


class Button(Widget):
    """Button widget in Supervisely is a user interface element that allows users to create clickable buttons in the applications.

    Read about it in `Developer Portal <https://developer.supervisely.com/app-development/widgets/controls/button>`_
        (including screenshots and examples).

    :param text: Text to be displayed on the button.
    :type text: Optional[str]
    :param button_type: Type of the button.
    :type button_type: Optional[Literal["primary", "info", "warning", "danger", "success", "text"]]
    :param button_size: Size of the button.
    :type button_size: Optional[Literal["mini", "small", "large"]]
    :param plain: If True, the button will be plain.
    :type plain: Optional[bool]
    :param show_loading: If True, the button will show loading animation.
    :type show_loading: Optional[bool]
    :param icon: Icon to be displayed on the button. E.g. "zmdi zmdi-play", "zmdi zmdi-stop", "zmdi zmdi-pause".
        List of available icons can be found `here <http://zavoloklom.github.io/material-design-iconic-font/icons.html>`_.
    :type icon: Optional[str]
    :param icon_gap: Gap between the icon and the text in pixels.
    :type icon_gap: Optional[int]
    :param widget_id: Unique widget identifier.
    :type widget_id: Optional[str]
    :param link: Link to be opened on button click.
    :type link: Optional[str]
    :param emit_on_click: Name of the event to be emitted on button click.
    :type emit_on_click: Optional[str]
    :param style: CSS style to be applied to the button.
    :type style: Optional[str]
    :param call_on_click: Function to be called on button click.
    :type call_on_click: Optional[str]
    :param icon_color: Color of the icon.
    :type icon_color: Optional[str]

    :Usage example:
    .. code-block:: python
        from supervisely.app.widgets import Button

        button = Button(text="Button", button_type="primary", button_size="large")
    """

    class Routes:
        CLICK = "button_clicked_cb"

    def __init__(
        self,
        text: Optional[str] = "Button",
        button_type: Optional[
            Literal["primary", "info", "warning", "danger", "success", "text"]
        ] = "primary",
        button_size: Optional[Literal["mini", "small", "large"]] = None,
        plain: Optional[bool] = False,
        show_loading: Optional[bool] = True,
        icon: Optional[str] = None,
        icon_gap: Optional[int] = 5,
        widget_id: Optional[str] = None,
        link: Optional[str] = None,
        emit_on_click: Optional[str] = None,
        style: Optional[str] = None,
        call_on_click: Optional[str] = None,
        visible_by_vue_field: Optional[str] = "",
        icon_color: Optional[str] = None,
    ):
        self._widget_routes = {}

        self._text = text
        self._button_type = button_type
        self._button_size = button_size
        self._plain = plain
        self._icon_gap = icon_gap
        self._link = link
        if icon is None:
            self._icon = ""
        else:
            icon_style = f"margin-right: {icon_gap}px"
            if icon_color is not None:
                icon_style += f"; color: {icon_color}"
            self._icon = f'<i class="{icon}" style="{icon_style}"></i>'

        self._loading = False
        self._disabled = False
        self._show_loading = show_loading
        self._click_handled = False
        self._emit_on_click = emit_on_click
        self._style = style
        self._call_on_click = call_on_click
        self._visible_by_vue_field = visible_by_vue_field

        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self):
        """Returns dictionary with widget data, which defines the appearance and behavior of the widget.

        Dictionary contains the following fields:
            - text: Text to be displayed on the button.
            - button_type: Type of the button.
            - plain: If True, the button will be plain.
            - button_size: Size of the button.
            - loading: If True, the button will show loading animation.
            - disabled: If True, the button will be disabled.
            - icon: Icon to be displayed on the button.
            - link: Link to be opened on button click.
        """
        return {
            "text": self._text,
            "button_type": self._button_type,
            "plain": self._plain,
            "button_size": self._button_size,
            "loading": self._loading,
            "disabled": self._disabled,
            "icon": self._icon,
            "link": self._link,
            "style": self._style,
        }

    def get_json_state(self) -> None:
        """Button widget doesn't have state, so this method returns None."""
        return {"visible_by_vue_field": self._visible_by_vue_field}

    @property
    def text(self) -> str:
        """Returns the text to be displayed on the button.

        :return: Text to be displayed on the button.
        :rtype: str
        """
        return self._text

    @text.setter
    def text(self, value: str) -> None:
        """Sets the text to be displayed on the button.

        :param value: Text to be displayed on the button.
        :type value: str
        """
        self._text = value
        DataJson()[self.widget_id]["text"] = self._text
        DataJson().send_changes()

    @property
    def icon(self) -> str:
        """Returns the icon to be displayed on the button.

        :return: Icon to be displayed on the button.
        :rtype: str
        """
        return self._icon

    @icon.setter
    def icon(self, value: str) -> None:
        """Sets the icon to be displayed on the button.

        :param value: Icon to be displayed on the button.
        :type value: str
        """
        if value is None:
            self._icon = ""
        else:
            self._icon = f'<i class="{value}" style="margin-right: {self._icon_gap}px"></i>'
        DataJson()[self.widget_id]["icon"] = self._icon
        DataJson().send_changes()

    @property
    def button_type(self) -> str:
        """Returns the type of the button.

        :return: Type of the button.
        :rtype: str
        """
        return self._button_type

    @button_type.setter
    def button_type(
        self, value: Literal["primary", "info", "warning", "danger", "success", "text"]
    ) -> None:
        """Sets the type of the button.

        :param value: Type of the button.
        :type value: Literal["primary", "info", "warning", "danger", "success", "text"]
        """
        self._button_type = value
        DataJson()[self.widget_id]["button_type"] = self._button_type
        DataJson().send_changes()

    @property
    def plain(self) -> bool:
        """Returns True if the button is plain, False otherwise.

        :return: True if the button is plain, False otherwise.
        :rtype: bool
        """
        return self._plain

    @plain.setter
    def plain(self, value: bool) -> None:
        """Sets the button to be plain or not.

        :param value: If True, the button will be plain.
        :type value: bool
        """
        self._plain = value
        DataJson()[self.widget_id]["plain"] = self._plain
        DataJson().send_changes()

    @property
    def link(self) -> str:
        """Returns the link to be opened on button click.

        :return: Link to be opened on button click.
        :rtype: str
        """
        return self._link

    @link.setter
    def link(self, value: str) -> None:
        """Sets the link to be opened on button click.

        :param value: Link to be opened on button click.
        :type value: str
        """
        self._link = value
        DataJson()[self.widget_id]["link"] = self._link
        DataJson().send_changes()

    @property
    def loading(self) -> bool:
        """Returns True if the button shows loading animation, False otherwise.

        :return: True if the button shows loading animation, False otherwise.
        :rtype: bool
        """
        return self._loading

    @loading.setter
    def loading(self, value: bool) -> None:
        """Sets the button loading animation.

        :param value: If True, the animation will be enabled, otherwise disabled.
        :type value: bool
        """
        self._loading = value
        DataJson()[self.widget_id]["loading"] = self._loading
        DataJson().send_changes()

    @property
    def show_loading(self) -> bool:
        """Returns True if the button shows loading animation, False otherwise.

        :return: True if the button shows loading animation, False otherwise.
        :rtype: bool
        """
        return self._show_loading

    @property
    def disabled(self) -> bool:
        """Returns True if the button is disabled, False otherwise.

        :return: True if the button is disabled, False otherwise.
        :rtype: bool
        """
        return self._disabled
    
    @property
    def style(self) -> Optional[str]:
        """Returns the CSS style applied to the button.

        :return: CSS style applied to the button.
        :rtype: Optional[str]
        """
        return self._style
    
    @style.setter
    def style(self, value: Optional[str]) -> None:
        """Sets the CSS style to be applied to the button.

        :param value: CSS style to be applied to the button.
        :type value: Optional[str]
        """
        self._style = value
        DataJson()[self.widget_id]["style"] = self._style
        DataJson().send_changes()

    @disabled.setter
    def disabled(self, value: bool) -> None:
        """Sets the button to be disabled or not.

        :param value: If True, the button will be disabled.
        :type value: bool
        """
        self._disabled = value
        DataJson()[self.widget_id]["disabled"] = self._disabled

    def click(self, func: Callable[[], None]) -> Callable[[], None]:
        """Decorator that allows to handle button click. Decorated function
        will be called on button click.

        :param func: Function to be called on button click.
        :type func: Callable
        :return: Decorated function.
        :rtype: Callable
        """
        route_path = self.get_route_path(Button.Routes.CLICK)
        server = self._sly_app.get_server()
        self._click_handled = True

        @server.post(route_path)
        def _click():
            if self.show_loading:
                self.loading = True
            try:
                func()
            except Exception as e:
                if self.show_loading and self.loading:
                    self.loading = False
                raise e
            if self.show_loading:
                self.loading = False

        return _click

    def _get_on_click(self):
        on_click_actions = []
        if self._emit_on_click:
            on_click_actions.append(f"$emit('{self._emit_on_click}');")

        if self._call_on_click:
            on_click_actions.append(f"{self._call_on_click}")

        if self._click_handled:
            on_click_actions.append(f"post('/{self.widget_id}/button_clicked_cb');")
        return " ".join(on_click_actions) if on_click_actions else None

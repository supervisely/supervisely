try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from typing import Optional

from supervisely.app import DataJson
from supervisely.app.widgets import Widget
from supervisely.app.widgets_context import JinjaWidgets


class RunAppButton(Widget):
    class Routes:
        CLICK = "button_clicked_cb"

    def __init__(
        self,
        workspace_id: int,
        module_id: int,
        payload: dict = None,
        text: Optional[str] = "Button",
        button_type: Optional[
            Literal["primary", "info", "warning", "danger", "success", "text"]
        ] = "primary",
        button_size: Optional[Literal["mini", "small", "large"]] = None,
        plain: Optional[bool] = False,
        icon: Optional[str] = None,
        icon_gap: Optional[int] = 5,
        available_in_offline: Optional[bool] = False,
        visible_by_vue_field: Optional[str] = "",
        widget_id: Optional[str] = None,
    ):
        self._widget_routes = {}

        self._text = text
        self._button_type = button_type
        self._button_size = button_size
        self._plain = plain
        self._icon_gap = icon_gap
        if icon is None:
            self._icon = ""
        else:
            self._icon = f'<i class="{icon}" style="margin-right: {icon_gap}px"></i>'

        self._available_in_offline = available_in_offline
        self._visible_by_vue_field = visible_by_vue_field

        self._loading = False
        self._disabled = False
        self._workspace_id = workspace_id
        self._module_id = module_id
        self._payload = payload

        super().__init__(widget_id=widget_id, file_path=__file__)

        script_path = "./sly/css/app/widgets/run_app_button/script.js"
        JinjaWidgets().context["__widget_scripts__"][self.__class__.__name__] = script_path

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
            "options": {
                "text": self._text,
                "button_type": self._button_type,
                "plain": self._plain,
                "button_size": self._button_size,
                "loading": self._loading,
                "disabled": self._disabled,
                "icon": self._icon,
                "available_in_offline": self._available_in_offline,
            }
        }

    def get_json_state(self) -> None:
        """Button widget doesn't have state, so this method returns None."""
        return {
            "workspace_id": self._workspace_id,
            "module_id": self._module_id,
            "payload": self._payload,
            "visible_by_vue_field": self._visible_by_vue_field,
        }

    @property
    def workspace_id(self) -> int:
        """Returns the workspace ID.

        :return: Workspace ID.
        :rtype: int
        """
        return self._workspace_id

    @workspace_id.setter
    def workspace_id(self, value: int) -> None:
        """Sets the workspace ID.

        :param value: Workspace ID.
        :type value: int
        """
        self._workspace_id = value
        DataJson()[self.widget_id]["workspace_id"] = self._workspace_id
        DataJson().send_changes()

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
    def disabled(self) -> bool:
        """Returns True if the button is disabled, False otherwise.

        :return: True if the button is disabled, False otherwise.
        :rtype: bool
        """
        return self._disabled

    @disabled.setter
    def disabled(self, value: bool) -> None:
        """Sets the button to be disabled or not.

        :param value: If True, the button will be disabled.
        :type value: bool
        """
        self._disabled = value
        DataJson()[self.widget_id]["disabled"] = self._disabled

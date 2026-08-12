from typing import Callable, Dict, List, Optional

from supervisely.app.content import DataJson, StateJson
from supervisely.app.widgets import Widget
from supervisely.app.widgets_context import JinjaWidgets


class Hotkeys(Widget):
    """Invisible widget that catches keyboard shortcuts anywhere on the page
    (not only while a specific input is focused) and notifies Python.

    Read about it in `Developer Portal <https://developer.supervisely.com/app-development/widgets/controls/hotkeys>`_.

    :Usage example:

        .. code-block:: python

            from supervisely.app.widgets import Hotkeys

            hotkeys = Hotkeys(hotkeys=["ctrl+s", "ctrl+z"])

            @hotkeys.key_pressed("ctrl+s")
            def on_save():
                print("Ctrl+S pressed")

            @hotkeys.key_pressed("ctrl+z")
            def on_undo():
                print("Ctrl+Z pressed")
    """

    class Routes:
        """Callback route names used by the widget frontend to notify Python."""

        KEY_PRESSED = "key_pressed_cb"

    def __init__(
        self,
        hotkeys: Optional[List[str]] = None,
        prevent_default: Optional[bool] = True,
        ignore_input_focus: Optional[bool] = True,
        widget_id: Optional[str] = None,
    ):
        """
        :param hotkeys: List of key combinations to catch, e.g. ``["ctrl+s", "shift+arrowright", "a"]``.
            Modifiers must go before the key and in the order ``ctrl``, ``alt``, ``shift``, joined with ``+``.
        :type hotkeys: List[str], optional
        :param prevent_default: If True, calls ``event.preventDefault()`` for matched combinations
            (useful to stop the browser's own shortcuts, e.g. Ctrl+S opening the "Save page" dialog).
        :type prevent_default: bool, optional
        :param ignore_input_focus: If True, combinations are not caught while the user is typing
            in an input, textarea, select or a contenteditable element.
        :type ignore_input_focus: bool, optional
        :param widget_id: Unique widget identifier.
        :type widget_id: str, optional
        """
        self._hotkeys = [combo.lower() for combo in (hotkeys or [])]
        self._prevent_default = prevent_default
        self._ignore_input_focus = ignore_input_focus
        self._handlers: Dict[Optional[str], List[Callable]] = {}
        self._route_registered = False

        super().__init__(widget_id=widget_id, file_path=__file__)

        script_path = "./sly/css/app/widgets/hotkeys/script.js"
        JinjaWidgets().context["__widget_scripts__"][self.__class__.__name__] = script_path

    def get_json_data(self) -> Dict:
        """Returns dictionary with widget data.

        :returns: Dictionary with ``hotkeys``, ``prevent_default`` and ``ignore_input_focus`` fields.
        :rtype: Dict
        """
        return {
            "hotkeys": self._hotkeys,
            "prevent_default": self._prevent_default,
            "ignore_input_focus": self._ignore_input_focus,
        }

    def get_json_state(self) -> Dict:
        """Returns dictionary with widget state.

        :returns: Dictionary with the last pressed combination (``pressed_key``).
        :rtype: Dict
        """
        return {"pressed_key": None}

    @property
    def hotkeys(self) -> List[str]:
        """Returns the list of key combinations this widget catches.

        :returns: List of key combinations.
        :rtype: List[str]
        """
        return self._hotkeys

    @property
    def pressed_key(self) -> Optional[str]:
        """Returns the last key combination that was caught.

        :returns: Last caught key combination, or None.
        :rtype: Optional[str]
        """
        return StateJson()[self.widget_id]["pressed_key"]

    def add_hotkey(self, combo: str) -> None:
        """Adds a new key combination to catch, without restarting the app.

        :param combo: Key combination, e.g. ``"ctrl+s"``.
        :type combo: str
        """
        combo = combo.lower()
        if combo not in self._hotkeys:
            self._hotkeys.append(combo)
            DataJson()[self.widget_id]["hotkeys"] = self._hotkeys
            DataJson().send_changes()

    def key_pressed(self, combo: Optional[str] = None) -> Callable:
        """Decorator that registers a Python callback for a key combination.

        :param combo: Key combination to react to, e.g. ``"ctrl+s"``. If the combination
            was not passed to the widget constructor, it is added automatically.
            If None, the decorated function is called for every combination caught by this widget.
        :type combo: str, optional
        :returns: Decorator.
        :rtype: Callable
        """
        if combo is not None:
            combo = combo.lower()
            self.add_hotkey(combo)

        def decorator(func: Callable) -> Callable:
            self._handlers.setdefault(combo, []).append(func)
            self._ensure_route()
            return func

        return decorator

    def _ensure_route(self) -> None:
        if self._route_registered:
            return
        self._route_registered = True

        route_path = self.get_route_path(Hotkeys.Routes.KEY_PRESSED)
        server = self._sly_app.get_server()

        @server.post(route_path)
        def _key_pressed():
            pressed = StateJson()[self.widget_id]["pressed_key"]
            callbacks = list(self._handlers.get(pressed, [])) + list(
                self._handlers.get(None, [])
            )
            for callback in callbacks:
                callback(pressed)

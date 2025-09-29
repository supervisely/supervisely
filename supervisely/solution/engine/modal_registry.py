from typing import Any, Dict, Iterable, List, Literal, Optional, Union

from supervisely.app.singleton import Singleton
from supervisely.app.widgets import Container, Dialog, Widget


# ------------------------------------------------------------------
# ModalRegistry ----------------------------------------------------
# ------------------------------------------------------------------
class ModalRegistry(metaclass=Singleton):
    """Keeps a consistent set of Dialogs that can be shared across nodes."""

    _AUTOMATION_KEY = "automation"
    _HISTORY_KEY = "history"
    _SETTINGS_TINY_KEY = "settings_tiny"
    _SETTINGS_SMALL_KEY = "settings_small"
    _SETTINGS_LARGE_KEY = "settings_large"
    _LOGS_KEY = "logs"
    _PREVIEW_KEY = "preview"

    def __init__(self) -> None:
        self._dialogs: Dict[str, Dialog] = {}
        self._automation_widgets: Dict[str, Widget] = {}
        self._history_widgets: Dict[str, Widget] = {}
        self._settings_widgets_tiny: Dict[str, Widget] = {}
        self._settings_widgets_small: Dict[str, Widget] = {}
        self._settings_widgets_large: Dict[str, Widget] = {}
        self._logs_widgets: Dict[str, Widget] = {}
        self._preview_widgets: Dict[str, Widget] = {}

        self._automation_dialog = self._create_automation_dialog()
        self._history_dialog = self._create_history_dialog()
        self._settings_dialog_tiny = self._create_tiny_settings_dialog()
        self._settings_dialog_small = self._create_small_settings_dialog()
        self._settings_dialog_large = self._create_large_settings_dialog()
        self._logs_dialog = self._create_logs_dialog()
        self._preview_dialog = self._create_preview_dialog()

    # ------------------------------------------------------------------ #
    # Generic modal API
    # ------------------------------------------------------------------ #
    def register_modal(self, name: str, dialog: Dialog, *, replace: bool = False) -> None:
        """Add a Dialog under `name` so GraphBuilder can expose it to VueFlow."""
        if not replace and name in self._dialogs:
            raise ValueError(f"Modal '{name}' is already registered")

        self._dialogs[name] = dialog

    def modals(self) -> Iterable[Dialog]:
        """Return all dialogs that should be sent to the frontend."""
        return self._dialogs.values()

    def get_modal(self, name: str) -> Dialog:
        return self._dialogs[name]

    # ------------------------------------------------------------------ #
    # Common Helpers
    # ------------------------------------------------------------------ #
    def _create_modal(
        self,
        key: str,
        title: str,
        size: Literal["tiny", "small", "large"] = "small",
    ) -> Dialog:
        """Create a generic dialog if it does not exist yet."""
        if key not in self._dialogs:
            container = Container(widgets=[], gap=0)
            dialog = Dialog(
                title=title,
                content=container,
                size=size,
            )
            self.register_modal(key, dialog)
        return self._dialogs[key]

    def _attach_widget_to_dialog(self, dialog: Dialog, widget: Widget) -> None:
        container: Container = dialog._content  # type: ignore[attr-defined]
        if widget not in container._widgets:  # pylint: disable=protected-access
            container._widgets.append(widget)  # reuse hidden container slots
        widget.hide()
        dialog.hide()

    def _open_modal(
        self, key: str, owner_id: str, widgets: Dict[str, Widget], dialog: Dialog
    ) -> None:
        target = widgets.get(owner_id)
        if target is None:
            raise KeyError(f"No widget registered for '{owner_id}' in modal '{key}'")
        for widget in widgets.values():
            widget.hide()
        target.show()
        dialog.show()

    # ------------------------------------------------------------------ #
    # Automation helpers
    # ------------------------------------------------------------------ #
    def _create_automation_dialog(self) -> None:
        """Create the automation dialog if it does not exist yet."""
        return self._create_modal(self._AUTOMATION_KEY, "Automation")

    def attach_automation_widget(self, owner_id: str, widget: Widget) -> Dialog:
        """
        Register a node's automation widget inside the shared automation dialog.
        Returns the dialog so the node can pass it further (e.g. to AutomationWidget).
        """
        self._automation_widgets[owner_id] = widget
        self._attach_widget_to_dialog(self._automation_dialog, widget)
        return self._automation_dialog

    def open_automation(self, owner_id: str) -> None:
        """
        Show the automation dialog for the node that owns `owner_id`.
        All other registered automation widgets are hidden.
        """
        self._open_modal(
            self._AUTOMATION_KEY, owner_id, self._automation_widgets, self._automation_dialog
        )

    @property
    def automation_dialog(self) -> Dialog:
        return self._automation_dialog

    # ------------------------------------------------------------------ #
    # History helpers
    # ------------------------------------------------------------------ #
    def _create_history_dialog(self) -> Dialog:
        """Create the history dialog if it does not exist yet."""
        return self._create_modal(self._HISTORY_KEY, "History", size="small")

    def attach_history_widget(self, owner_id: str, widget: Widget) -> Dialog:
        """
        Register a node's history widget inside the shared history dialog.
        Returns the dialog so the node can pass it further (e.g. to HistoryWidget).
        """
        self._history_widgets[owner_id] = widget
        self._attach_widget_to_dialog(self._history_dialog, widget)
        return self._history_dialog

    def open_history(self, owner_id: str) -> None:
        """
        Show the history dialog for the node that owns `owner_id`.
        All other registered history widgets are hidden.
        """
        self._open_modal(self._HISTORY_KEY, owner_id, self._history_widgets, self._history_dialog)

    @property
    def history_dialog(self) -> Dialog:
        return self._history_dialog

    # ------------------------------------------------------------------ #
    # Settings helpers
    # ------------------------------------------------------------------ #
    def _create_tiny_settings_dialog(self) -> Dialog:
        """Create the tiny settings dialog if it does not exist yet."""
        return self._create_modal(self._SETTINGS_TINY_KEY, "Settings", size="tiny")

    def _create_small_settings_dialog(self) -> Dialog:
        """Create the small settings dialog if it does not exist yet."""
        return self._create_modal(self._SETTINGS_SMALL_KEY, "Settings", size="small")

    def _create_large_settings_dialog(self) -> Dialog:
        """Create the large settings dialog if it does not exist yet."""
        return self._create_modal(self._SETTINGS_LARGE_KEY, "Settings", size="large")

    def attach_settings_widget(
        self, owner_id: str, widget: Widget, size: Literal["tiny", "small", "large"] = "small"
    ) -> Dialog:
        """
        Register a node's settings widget inside the shared settings dialog.
        Returns the dialog so the node can pass it further (e.g. to SettingsWidget).
        """
        if size == "tiny":
            self._settings_widgets_tiny[owner_id] = widget
            self._attach_widget_to_dialog(self._settings_dialog_tiny, widget)
            return self._settings_dialog_tiny
        elif size == "small":
            self._settings_widgets_small[owner_id] = widget
            self._attach_widget_to_dialog(self._settings_dialog_small, widget)
            return self._settings_dialog_small
        elif size == "large":
            self._settings_widgets_large[owner_id] = widget
            self._attach_widget_to_dialog(self._settings_dialog_large, widget)
            return self._settings_dialog_large
        else:
            raise ValueError(f"Unknown settings dialog size '{size}'")

    def open_settings(
        self, owner_id: str, size: Literal["tiny", "small", "large"] = "small"
    ) -> None:
        """
        Show the settings dialog for the node that owns `owner_id`.
        All other registered settings widgets are hidden.
        """
        if size == "tiny":
            self._open_modal(
                self._SETTINGS_TINY_KEY,
                owner_id,
                self._settings_widgets_tiny,
                self._settings_dialog_tiny,
            )
        elif size == "small":
            self._open_modal(
                self._SETTINGS_SMALL_KEY,
                owner_id,
                self._settings_widgets_small,
                self._settings_dialog_small,
            )
        elif size == "large":
            self._open_modal(
                self._SETTINGS_LARGE_KEY,
                owner_id,
                self._settings_widgets_large,
                self._settings_dialog_large,
            )
        else:
            raise ValueError(f"Unknown settings dialog size '{size}'")

    @property
    def settings_dialog_tiny(self) -> Dialog:
        return self._settings_dialog_tiny

    @property
    def settings_dialog_small(self) -> Dialog:
        return self._settings_dialog_small

    @property
    def settings_dialog_large(self) -> Dialog:
        return self._settings_dialog_large

    # ------------------------------------------------------------------ #
    # Logs helpers
    # ------------------------------------------------------------------ #
    def _create_logs_dialog(self) -> Dialog:
        """Create the logs dialog if it does not exist yet."""
        return self._create_modal(self._LOGS_KEY, "Logs", size="large")

    def attach_logs_widget(self, owner_id: str, widget: Widget) -> Dialog:
        """
        Register a node's logs widget inside the shared logs dialog.
        Returns the dialog so the node can pass it further (e.g. to LogsWidget).
        """
        self._logs_widgets[owner_id] = widget
        self._attach_widget_to_dialog(self._logs_dialog, widget)
        return self._logs_dialog

    def open_logs(self, owner_id: str) -> None:
        """
        Show the logs dialog for the node that owns `owner_id`.
        All other registered logs widgets are hidden.
        """
        self._open_modal(self._LOGS_KEY, owner_id, self._logs_widgets, self._logs_dialog)

    @property
    def logs_dialog(self) -> Dialog:
        return self._logs_dialog

    # ------------------------------------------------------------------ #
    # Preview helpers
    # ------------------------------------------------------------------ #
    def _create_preview_dialog(self) -> Dialog:
        """Create the preview dialog if it does not exist yet."""
        return self._create_modal(self._PREVIEW_KEY, "Preview", size="large")

    def attach_preview_widget(self, owner_id: str, widget: Widget) -> Dialog:
        """
        Register a node's preview widget inside the shared preview dialog.
        Returns the dialog so the node can pass it further (e.g. to PreviewWidget).
        """
        self._preview_widgets[owner_id] = widget
        self._attach_widget_to_dialog(self._preview_dialog, widget)
        return self._preview_dialog

    def open_preview(self, owner_id: str) -> None:
        """
        Show the preview dialog for the node that owns `owner_id`.
        All other registered preview widgets are hidden.
        """
        self._open_modal(self._PREVIEW_KEY, owner_id, self._preview_widgets, self._preview_dialog)

    @property
    def preview_dialog(self) -> Dialog:
        return self._preview_dialog

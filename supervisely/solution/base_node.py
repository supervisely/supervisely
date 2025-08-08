from __future__ import annotations

from typing import Any, Callable, Dict, List, Literal, Optional, Union

from supervisely.api.project_api import ProjectInfo
from supervisely.app import DataJson
from supervisely.app.widgets import (
    Button,
    Container,
    Dialog,
    SolutionCard,
    SolutionGraph,
    SolutionProject,
    Widget,
)
from supervisely.app.widgets_context import JinjaWidgets
from supervisely.solution.engine.events import PubSub, PubSubAsync, publish_event
from supervisely.solution.engine.scheduler import TasksScheduler


# ------------------------------------------------------------------
# EventMixin ------------------------------------------------------
# ------------------------------------------------------------------
class EventMixin:
    def __init__(self):
        self._register_event_handlers()

    @property
    def broker(self):
        """Get the event broker instance."""
        return self._event_broker

    def _register_event_handlers(self):
        """Register all methods decorated with @on_event."""
        for method_name in dir(self):
            try:
                method = getattr(self, method_name)
                if hasattr(method, "_event_topic"):
                    for topic in method._event_topic:
                        PubSub().subscribe(topic, method)
            except Exception as e:
                pass


# ------------------------------------------------------------------
# Node -------------------------------------------------------------
# ------------------------------------------------------------------
class SolutionElement(Widget, EventMixin):
    progress_badge_key = "Task"

    def __new__(cls, *args, **kwargs):
        JinjaWidgets().incremental_widget_id_mode = True
        return super().__new__(cls)

    def __init__(self, *args, **kwargs):
        """Base class for all solution elements.

        This class is used to create a common interface for all solution elements.
        It can be extended to create specific solution elements with their own properties and methods.
        """
        self.id = kwargs.pop("id", None)
        self._subscribe = kwargs.pop("subscribe", [])  # List of topics to subscribe to
        self._publish = kwargs.pop("publish", [])  # List of topics to publish to
        widget_id = kwargs.get("widget_id", None)
        if not hasattr(self, "widget_id"):
            self.widget_id = widget_id
        # Widget.__init__(self, widget_id=self.widget_id)
        super().__init__(widget_id=self.widget_id, *args, **kwargs)
        self.enable_subscribtions()
        self.enable_publishing()
        EventMixin.__init__(self)

    # ------------------------------------------------------------------
    # JSON Methods ----------------------------------------------------
    # ------------------------------------------------------------------
    @classmethod
    def from_json(cls, json_data):
        node_id = json_data.get("id")
        kwargs = json_data.get("parameters", {})
        x = kwargs.pop("x", 0)
        y = kwargs.pop("y", 0)
        subscribe = json_data.get("events", {}).get("subscribe", [])
        publish = json_data.get("events", {}).get("publish", [])
        return cls(id=node_id, x=x, y=y, subscribe=subscribe, publish=publish, **kwargs)

    # ------------------------------------------------------------------
    # Base Widget Methods ----------------------------------------------
    # ------------------------------------------------------------------
    def get_json_data(self) -> Dict:
        return {}

    def get_json_state(self) -> Dict:
        return {}

    def save_to_state(self, data: Dict) -> None:
        """Save data to the state JSON."""
        if not isinstance(data, dict):
            raise TypeError("data must be a dictionary")
        if self.widget_id not in DataJson():
            DataJson()[self.widget_id] = {}
        DataJson()[self.widget_id].update(data)
        DataJson().send_changes()

    # ------------------------------------------------------------------
    # Card  ------------------------------------------------------------
    # ------------------------------------------------------------------
    def _build_card(
        self,
        title: str,
        tooltip_description: str,
        width: int = 250,
        buttons: Optional[List] = None,
        link: Optional[str] = None,
    ) -> SolutionCard:
        if buttons is None:
            buttons = []
            if hasattr(self, "history") and self.history is not None:
                if hasattr(self.history, "open_modal_button"):
                    buttons.append(self.history.open_modal_button)
            if hasattr(self, "automation") and self.automation is not None:
                if hasattr(self.automation, "open_modal_button"):
                    buttons.append(self.automation.open_modal_button)

        return SolutionCard(
            title=title,
            tooltip=SolutionCard.Tooltip(
                description=tooltip_description,
                content=buttons,
            ),
            width=width,
            link=link,
        )

    # ------------------------------------------------------------------
    # Progress badge wrappers ------------------------------------------
    # ------------------------------------------------------------------
    @staticmethod
    def _wrap_in_progress(node: SolutionElement, func: Callable):
        def wrapped(*args, **kwargs):
            node.show_in_progress_badge(node.progress_badge_key)
            res = func(*args, **kwargs)
            node.hide_in_progress_badge(node.progress_badge_key)
            return res

        return wrapped

    @staticmethod
    def _wrap_start(node: SolutionElement, func: Callable):
        def wrapped():
            node.show_in_progress_badge(node.progress_badge_key)
            if callable(func):
                func()

        return wrapped

    @staticmethod
    def _wrap_finish(node: "SolutionElement", func: Callable[[int], None]):
        import inspect

        def wrapped(task_id: int):
            if callable(func):
                # Call with task_id only if the callback expects at least one parameter
                if len(inspect.signature(func).parameters) > 0:
                    func(task_id)
                else:
                    func()
            node.hide_in_progress_badge(node.progress_badge_key)

        return wrapped

    # ------------------------------------------------------------------
    # Badge proxies for SolutionCardNode -------------------------------
    # ------------------------------------------------------------------
    def show_in_progress_badge(self, key: Optional[str] = None):
        if hasattr(self, "node") and hasattr(self.node, "show_in_progress_badge"):
            self.node.show_in_progress_badge(key)

    def hide_in_progress_badge(self, key: Optional[str] = None):
        if hasattr(self, "node") and hasattr(self.node, "hide_in_progress_badge"):
            self.node.hide_in_progress_badge(key)

    def show_finished_badge(self):
        if hasattr(self, "node") and hasattr(self.node, "show_finished_badge"):
            self.node.show_finished_badge()

    def hide_finished_badge(self):
        if hasattr(self, "node") and hasattr(self.node, "hide_finished_badge"):
            self.node.hide_finished_badge()

    def show_failed_badge(self):
        if hasattr(self, "node") and hasattr(self.node, "show_failed_badge"):
            self.node.show_failed_badge()

    def hide_failed_badge(self):
        if hasattr(self, "node") and hasattr(self.node, "hide_failed_badge"):
            self.node.hide_failed_badge()

    # ------------------------------------------------------------------
    # Event registration ------------------------------------------------
    # ------------------------------------------------------------------
    def _available_publish_methods(self) -> Dict[str, Callable]:
        """Returns a dictionary of methods that can be used for publishing events."""
        return {}

    def _available_subscribe_methods(self) -> Dict[str, Union[Callable, List[Callable]]]:
        """Returns a dictionary of methods that can be used for subscribing to events."""
        return {}

    def enable_subscribtions(self):
        """Subscribe to events defined in the class."""
        for topic, method in self._available_subscribe_methods().items():
            if topic in self._subscribe:
                if not isinstance(method, list):
                    method = [method]
                for m in method:
                    if not callable(m):
                        raise TypeError(f"Method {m} for topic {topic} is not callable")
                    PubSubAsync().subscribe(topic, m)

    def enable_publishing(self):
        """Publish events defined in the class."""
        for topic, method in self._available_publish_methods().items():
            if topic in self._publish:
                # not publish, but wrap the method to publish the event when called
                wrapped = publish_event(topic)(self._wrap_in_progress(self, method))
                setattr(self, method.__name__, wrapped)  # replace the method with the wrapped one

    def on_start(self, func: Callable[[], None] = None):
        """Register a callback to run *before* the node's main task.
        The callback is wrapped so the in-progress badge appears automatically."""
        if not hasattr(self, "gui") or not hasattr(self.gui, "on_start"):
            raise AttributeError(f"{self.__class__.__name__} has no gui.on_start() method")
        wrapped = self._wrap_start(self, func)
        return self.gui.on_start(wrapped)

    def on_finish(self, func: Callable[[int], None] = None):
        """Register a callback to run *after* the node's main task.
        The callback is wrapped so the in-progress badge disappears automatically."""
        if not hasattr(self, "gui") or not hasattr(self.gui, "on_finish"):
            raise AttributeError(f"{self.__class__.__name__} has no gui.on_finish() method")
        wrapped = self._wrap_finish(self, func)
        return self.gui.on_finish(wrapped)

    # ------------------------------------------------------------------
    # Convenience ------------------------------------------------------
    # ------------------------------------------------------------------
    def run(self, *args, **kwargs):
        if hasattr(self, "gui"):
            if hasattr(self.gui, "widget"):
                if hasattr(self.gui.widget, "run"):
                    return self.gui.widget.run(*args, **kwargs)
        raise NotImplementedError("Subclasses must implement this method")


# ------------------------------------------------------------------
# Automation -------------------------------------------------------
# ------------------------------------------------------------------
class Automation:
    @property
    def scheduler(self):
        """Returns the scheduler for the automation."""
        return TasksScheduler()


class AutomationWidget(Automation):

    def __init__(self, func: Callable):
        """
        Initializes the automation widget.

        :param func: Function to be called when the automation is applied.
        :type func: Callable
        """
        super().__init__()
        self.func = func
        self.apply_button = Button("Apply", plain=True, button_size="small")
        self.widget = self._create_widget()
        self.job_id = self.widget.widget_id

        # --- modal ----------------------------------------------------
        self._modal: Optional[Dialog] = None
        self._open_modal_button: Optional[Button] = None

        # --- apply button ---------------------------------------------
        # should be implemented in subclasses (can not be overridden)
        # @self.apply_button.click
        # def on_apply_button_click():
        #     self.modal.hide()
        #     self.apply()

    # ------------------------------------------------------------------
    # Automation -------------------------------------------------------
    # ------------------------------------------------------------------
    def apply(self) -> None:
        """Applies the automation settings."""
        sec, _, _ = self.get_details()
        if sec is None:
            if self.scheduler.is_job_scheduled(self.job_id):
                self.scheduler.remove_job(self.job_id)
        else:
            self.scheduler.add_job(self.func, sec, self.job_id, True)
        if self._on_apply is not None:
            self._on_apply()

    def on_apply(self, func: Callable) -> None:
        """Sets the function to be called when the automation is applied."""
        self._on_apply = func

    # ------------------------------------------------------------------
    # Automation Settings ----------------------------------------------
    # Depends on the automation GUI implementation `_create_widget()` --
    # ------------------------------------------------------------------
    def get_details(self) -> Any:
        """Returns the details of the automation."""
        raise NotImplementedError("Subclasses must implement this method")

    def save_details(self) -> None:
        """
        Saves the automation settings.

        :param enabled: Whether the automation is enabled.
        :type enabled: bool
        :param interval: Interval for synchronization.
        :type interval: int
        :param period: Period unit for synchronization (e.g., "minutes", "hours", "days").
        :type period: str
        """
        raise NotImplementedError("Subclasses must implement this method")

    # ------------------------------------------------------------------
    # GUI --------------------------------------------------------------
    # ------------------------------------------------------------------
    def _create_widget(self) -> Container:
        """Create the widget for the automation."""
        raise NotImplementedError("Subclasses must implement this method")

    @property
    def is_enabled(self) -> bool:
        """Implement checkbox in `_create_widget()` to check if the automation is enabled."""
        raise NotImplementedError("Subclasses must implement this method")

    # ------------------------------------------------------------------
    # Modal ------------------------------------------------------------
    # ------------------------------------------------------------------
    @property
    def modal(self) -> Dialog:
        """Returns the modal for the automation."""
        if self._modal is None:
            self._modal = Dialog(title="Automation Settings", content=self.widget, size="tiny")
        return self._modal

    @property
    def open_modal_button(self) -> Button:
        """Returns the open modal button."""
        if self._open_modal_button is None:
            btn = Button(
                text="Automate",
                icon="zmdi zmdi-flash-auto",
                button_size="mini",
                plain=True,
                button_type="text",
            )

            @btn.click
            def _on_click():
                self.modal.show()

            self._open_modal_button = btn
        return self._open_modal_button


# ------------------------------------------------------------------
# Card -------------------------------------------------------------
# only SolutionCard/SolutionProject can be used as content ---------
# ------------------------------------------------------------------
class SolutionCardNode(SolutionGraph.Node):
    def __new__(
        cls, content: Widget, x: int = 0, y: int = 0, *args, **kwargs
    ) -> SolutionGraph.Node:
        JinjaWidgets().incremental_widget_id_mode = True
        if not isinstance(content, (SolutionCard, SolutionProject)):
            raise TypeError("Content must be one of SolutionCard or SolutionProject")
        return super().__new__(cls, *args, **kwargs)

    # ------------------------------------------------------------------
    # Base Widget Settings ---------------------------------------------
    # ------------------------------------------------------------------
    def disable(self):
        """Disables the card widget."""
        self.content.disable()
        super().disable()

    def enable(self):
        """Enables the card widget."""
        self.content.enable()
        super().enable()

    # ------------------------------------------------------------------
    # Tooltip Methods --------------------------------------------------
    # ------------------------------------------------------------------
    def update_property(self, key: str, value: str, link: str = None, highlight: bool = None):
        """
        Updates the property of the card.

        :param key: Key of the property.
        :type key: str
        :param value: Value of the property.
        :type value: str
        :param link: Link of the property.
        :type link: str
        :param highlight: Whether to highlight the property.
        :type highlight: bool
        """
        for prop in self.content.tooltip_properties:
            if prop["key"] == key:
                self.content.update_property(key, value, link, highlight)
                return
        self.content.add_property(key, value, link, highlight)

    def remove_property_by_key(self, key: str):
        """
        Removes the property by key of the card.

        :param key: Key of the property.
        :type key: str
        """
        self.content.remove_property_by_key(key)

    # ------------------------------------------------------------------
    # Badge Methods ----------------------------------------------------
    # ------------------------------------------------------------------
    def add_badge(self, badge):
        """
        Adds a badge to the card.

        :param badge: Badge to add.
        :type badge: Badge
        """
        self.content.add_badge(badge)

    def remove_badge(self, idx: int):
        """
        Removes the badge by index of the card.

        :param idx: Index of the badge.
        :type idx: int
        """
        self.content.remove_badge(idx)

    def update_badge(
        self,
        idx: int,
        label: str,
        on_hover: str = None,
        badge_type: Literal["info", "success", "warning", "error"] = "info",
    ):
        """
        Updates the badge by index of the card.

        :param idx: Index of the badge.
        :type idx: int
        :param label: Label of the badge.
        :type label: str
        :param on_hover: On hover text of the badge.
        :type on_hover: str
        :param badge_type: Type of the badge.
        :type badge_type: Literal["info", "success", "warning", "error"]
        """
        self.content.update_badge(idx, label, on_hover, badge_type)

    def update_badge_by_key(
        self,
        key: str,
        label: str,
        badge_type: Literal["info", "success", "warning", "error"] = None,
        new_key: str = None,
        plain: Optional[bool] = None,
    ):
        """
        Updates the badge by key of the card.

        :param key: Key of the badge.
        :type key: str
        :param label: Label of the badge.
        :type label: str
        """
        self.content.update_badge_by_key(
            key=key,
            label=label,
            new_key=new_key,
            badge_type=badge_type,
            plain=plain,
        )

    def remove_badge_by_key(self, key: str):
        """
        Removes the badge by key from the card.

        :param key: Key of the badge.
        :type key: str
        """
        self.content.remove_badge_by_key(key)

    # ------------------------------------------------------------------
    # Automation Badge Methods -----------------------------------------
    # ------------------------------------------------------------------
    def update_automation_badge(self, enable: bool) -> None:
        """
        Updates the automation badge of the card.

        :param enable: Whether to enable the automation.
        :type enable: bool
        """
        for idx, prop in enumerate(self.content.badges):
            if prop["on_hover"] == "Automation":
                if enable:
                    pass  # already enabled
                else:
                    self.content.remove_badge(idx)
                return

        if enable:  # if not found
            self.content.add_badge(
                SolutionCard.Badge(
                    label="⚡",
                    on_hover="Automation",
                    badge_type="warning",
                    plain=True,
                )
            )

    def show_automation_badge(self) -> None:
        """Updates the card to show that automation is enabled."""
        self.update_automation_badge(True)

    def hide_automation_badge(self) -> None:
        """Updates the card to show that automation is disabled."""
        self.update_automation_badge(False)

    # ------------------------------------------------------------------
    # In Progress Badge Methods ----------------------------------------
    # ------------------------------------------------------------------
    def show_in_progress_badge(self, key: Optional[str] = None):
        """
        Updates the card to show that the main task is in progress.

        :param key: Key of the badge.
        :type key: Optional[str]
        """
        key = key or "⏳"
        self.content.update_badge_by_key(key=key, label="In progress", badge_type="info")

    def hide_in_progress_badge(self, key: Optional[str] = None):
        """
        Hides the in-progress badge from the card.

        :param key: Key of the badge.
        :type key: Optional[str]
        """
        key = key or "⏳"
        self.content.remove_badge_by_key(key=key)

    # ------------------------------------------------------------------
    # Finished Badge Methods -------------------------------------------
    # ------------------------------------------------------------------
    def show_finished_badge(self):
        """Updates the card to show that main task is finished."""
        self.content.update_badge_by_key(
            key="Finished", label="✅", plain=True, badge_type="success"
        )

    def hide_finished_badge(self):
        """Hides the finished badge from the card."""
        self.content.remove_badge_by_key(key="Finished")

    # ------------------------------------------------------------------
    # Failed Badge Methods ---------------------------------------------
    # ------------------------------------------------------------------
    def show_failed_badge(self):
        """Updates the card to show that the main task has failed."""
        self.content.update_badge_by_key(key="Failed", label="❌", plain=True, badge_type="error")

    def hide_failed_badge(self):
        """Hides the failed badge from the card."""
        self.content.remove_badge_by_key(key="Failed")


# ------------------------------------------------------------------
# Project ----------------------------------------------------------
# only SolutionProject can be used as content ----------------------
# ------------------------------------------------------------------
class SolutionProjectNode(SolutionCardNode):
    def __new__(
        cls, content: Widget, x: int = 0, y: int = 0, *args, **kwargs
    ) -> SolutionGraph.Node:
        if not isinstance(content, SolutionProject):
            raise TypeError("Content must be an instance of SolutionProject")
        return super().__new__(cls, content, x, y, *args, **kwargs)

    # ------------------------------------------------------------------
    # Preview Methods --------------------------------------------------
    # ------------------------------------------------------------------
    def update_preview(self, imgs: List[str], counts: List[int]):
        """
        Updates the preview of the project.

        :param imgs: List of image urls.
        :type imgs: List[str]
        :param counts: List of counts.
        :type counts: List[int]
        """
        self.content.update_preview_url(imgs)
        self.content.update_items_count(counts)

    def update_preview_url(self, imgs: List[str]):
        """
        Updates the preview url of the project.

        :param imgs: List of image urls.
        :type imgs: List[str]
        """
        self.content.update_preview_url(imgs)

    def update_items_count(self, counts: List[int]):
        """
        Updates the items count of the project.

        :param counts: List of counts.
        :type counts: List[int]
        """
        self.content.update_items_count(counts)

    # ------------------------------------------------------------------
    # Update Methods ---------------------------------------------------
    # ------------------------------------------------------------------
    def update(
        self,
        project: ProjectInfo = None,
        new_items_count: int = None,
        urls: List[Union[int, str, None]] = None,
        counts: List[Union[int, None]] = None,
    ):
        """
        Updates the node with the new project.

        :param project: Project info.
        :type project: ProjectInfo
        :param new_items_count: New items count.
        :type new_items_count: int
        :param urls: List of image urls.
        :type urls: List[Union[int, str, None]]
        :param counts: List of counts.
        :type counts: List[Union[int, None]]
        """
        if project is not None:
            self.project = project
        if new_items_count is not None:
            self.update_property(key="Last update", value=f"+{new_items_count}")
            self.update_property(key="Total", value=f"{self.project.items_count} images")
            self.update_badge_by_key(key="Last update:", label=f"+{new_items_count}")

        if self.is_training and urls is not None and counts is not None:
            self.update_preview(urls, counts)
        else:
            self.update_preview(
                [self.project.image_preview_url],
                [self.project.items_count],
            )

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Literal, Optional, Union

from supervisely.api.project_api import ProjectInfo
from supervisely.app import DataJson, StateJson
from supervisely.app.widgets import (
    Button,
    Container,
    Dialog,
    Icons,
    SolutionCard,
    SolutionGraph,
    SolutionProject,
    VueFlow,
    Widget,
)
from supervisely.app.widgets.vue_flow.modal import VueFlowModal
from supervisely.app.widgets.vue_flow.models import (
    Handle,
    NodeBadge,
    NodeQueueInfo,
    NodeSettings,
    NodeTooltip,
)
from supervisely.app.widgets.widget import generate_id
from supervisely.app.widgets_context import JinjaWidgets
from supervisely.solution.engine.events import PubSub, PubSubAsync, publish_event
from supervisely.solution.engine.scheduler import TasksScheduler


# ------------------------------------------------------------------
# EventMixin ------------------------------------------------------
# ------------------------------------------------------------------
class EventMixin:
    def __init__(self, *args, **kwargs):
        self._subscribe = kwargs.pop("subscribe", [])  # List of topics to subscribe to
        self._publish = kwargs.pop("publish", [])  # List of topics to publish to
        # self.enable_publishing()  # order matters (publish methods must be wrapped before subscribing)
        # self.enable_subscriptions()
        super().__init__()

    # ------------------------------------------------------------------
    # Event registration ------------------------------------------------
    # ------------------------------------------------------------------
    def _available_publish_methods(self) -> Dict[str, Callable]:
        """Returns a dictionary of methods that can be used for publishing events."""
        return {}

    def _available_subscribe_methods(self) -> Dict[str, Union[Callable, List[Callable]]]:
        """Returns a dictionary of methods that can be used for subscribing to events."""
        return {}

    def enable_subscriptions(self, source: Optional[str] = None, topic: Optional[str] = None):
        """Subscribe to events defined in the class."""
        if topic is not None:
            if topic not in self._available_subscribe_methods():
                return
            method = self._available_subscribe_methods()[topic]
            PubSubAsync().subscribe(topic=topic, callback=method, source=source)
        else:
            for topic, method in self._available_subscribe_methods().items():
                if topic in self._subscribe:
                    if not isinstance(method, list):
                        method = [method]
                    for m in method:
                        PubSubAsync().subscribe(topic=topic, callback=m, source=source)

    def enable_publishing(self, source: Optional[str] = None, topic: Optional[str] = None):
        """Publish events defined in the class."""
        if topic is not None and source is not None:
            if topic not in self._available_publish_methods():
                return
            method = self._available_publish_methods()[topic]
        else:
            for topic, method in self._available_publish_methods().items():
                if topic in self._publish:
                    # not publish, but wrap the method to publish the event when called
                    break
            else:
                return
        cb = publish_event(topic, source=source)(method)  # pylint: disable=E1101
        setattr(self, method.__name__, cb)  # replace the method with the wrapped one

    def disable_subscriptions(self, source: Optional[str] = None):
        """Unsubscribe from all events."""
        for topic, method in self._available_subscribe_methods().items():
            PubSubAsync().unsubscribe(topic=topic, callback=method, source=source)

    def disable_publishing(self, source: Optional[str] = None):
        """Unpublish all events."""
        for key, subsctions in PubSubAsync().subscribers.items():
            topic, src = key
            if src == source:
                for callback in subsctions:
                    PubSubAsync().unsubscribe(topic=topic, callback=callback, source=source)


# ------------------------------------------------------------------
# Base Node (all nodes) --------------------------------------------
# ------------------------------------------------------------------
class BaseNode(Widget, VueFlow.Node, EventMixin):
    NODE_TYPE = "base"
    TITLE = None
    DESCRIPTION = None
    ICON = None
    ICON_COLOR = None
    ICON_BG_COLOR = None

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        self.x = kwargs.pop("x", 0)
        self.y = kwargs.pop("y", 0)
        self.parent_id = kwargs.pop("parent_id", None)
        self.id = kwargs.pop("id", generate_id(cls_name=self.__class__.__name__))
        self.title = kwargs.pop("title", "Node")
        # self._modal = kwargs.pop("modal", None)

        Widget.__init__(self, widget_id=self.id)  # Widget does not call super()
        EventMixin.__init__(self, *args, **kwargs)
        VueFlow.Node.__init__(
            self,
            id=self.id,
            x=self.x,
            y=self.y,
            label=self.title,
            data=self._create_settings(*args, **kwargs),
            parent_id=self.parent_id,
            link=kwargs.pop("link", None),
        )

    # ------------------------------------------------------------------
    # Base Widget Methods ----------------------------------------------
    # ------------------------------------------------------------------
    def get_json_data(self) -> Dict:
        return {}

    def get_json_state(self) -> Dict:
        return {}

    # ------------------------------------------------------------------
    # VueFlow Node deserialization -------------------------------------
    # These methods create the node from JSON data
    # ------------------------------------------------------------------
    @classmethod
    def from_json(
        cls,
        json_data,
        parent_id: Optional[str] = None,
        # modal: Optional[VueFlowModal] = None,
    ) -> "BaseNode":
        node_id = json_data.get("id")
        kwargs = json_data.get("parameters", {})
        x = kwargs.pop("x", 0)
        y = kwargs.pop("y", 0)
        kwargs["subscribe"] = json_data.get("events", {}).get("subscribe", [])
        kwargs["publish"] = json_data.get("events", {}).get("publish", [])
        return cls(
            id=node_id,
            x=x,
            y=y,
            parent_id=parent_id,
            # modal=modal,
            **kwargs,
        )

    def configure_automation(self, *args, **kwargs):
        """Method to call after all nodes are created and subscribed to events."""
        pass

    # ------------------------------------------------------------------
    # VueFlow Node Methods ----------------------------------------------
    # These methods create the settings for the node in the graph
    # ------------------------------------------------------------------
    def _create_settings(self, *args, **kwargs) -> NodeSettings:
        return NodeSettings(
            type=self.NODE_TYPE,
            tooltip=self._create_tooltip(*args, **kwargs),
            icon=self._create_icon(*args, **kwargs),
            queue_info=self._create_queue_info(*args, **kwargs),
            handles=self._create_handles(*args, **kwargs),
        )

    def _create_tooltip(self, *args, **kwargs) -> NodeTooltip:
        btns = self._get_tooltip_buttons()
        res_buttons = []
        for btn in btns:
            btn: Button
            btn_icon = re.sub(r'<i class="(.*?)".*?</i>', r"\1", btn.icon) if btn.icon else None
            if btn.link is not None and btn.link != "":
                btn = {"label": btn.text, "link": {"url": btn.link}, "icon": btn_icon}
            else:
                route_path = btn.get_route_path(Button.Routes.CLICK)
                btn = {"label": btn.text, "link": {"action": route_path}, "icon": btn_icon}
            res_buttons.append(btn)
        return NodeTooltip(description=kwargs.get("description", None), buttons=res_buttons)

    def _get_tooltip_buttons(self) -> List[Button]:
        if not hasattr(self, "tooltip_buttons"):
            buttons = []
            if hasattr(self, "history"):
                if self.history is not None:
                    if hasattr(self.history, "open_modal_button"):
                        buttons.append(self.history.open_modal_button)
            if hasattr(self, "automation"):
                if self.automation is not None:
                    if hasattr(self.automation, "open_modal_button"):
                        buttons.append(self.automation.open_modal_button)
            self.tooltip_buttons = buttons
        return self.tooltip_buttons

    def _create_icon(self, *args, **kwargs) -> Optional[Icons]:
        icon = kwargs.get("icon", None)
        icon_color = kwargs.get("icon_color", None)
        icon_bg_color = kwargs.get("icon_bg_color", None)
        if icon is not None:
            return {
                "name": icon,
                "color": icon_color,
                "backgroundColor": icon_bg_color,
            }
        return None

    def _create_queue_info(self, *args, **kwargs) -> Optional[NodeQueueInfo]:
        return None

    def _create_handles(self, *args, **kwargs) -> List[Handle]:
        handles = self._get_handles()
        return [Handle(**handle) for handle in handles]

    def _get_handles(self) -> List[Dict[str, Any]]:
        return []


# ------------------------------------------------------------------
# Card Node (action nodes) -----------------------------------------
# ------------------------------------------------------------------
# SolutionCardNode
class BaseCardNode(BaseNode):
    NODE_TYPE = "action"

    # Automation Badge Methods -----------------------------------------
    def show_automation_badge(self) -> None:
        """Updates the card to show that automation is enabled."""
        self.update_badge_by_key(key="Automation", label="⚡", plain=True)

    def hide_automation_badge(self) -> None:
        """Updates the card to show that automation is disabled."""
        self.remove_badge_by_key(key="Automation")

    # In Progress Badge Methods ----------------------------------------
    def show_in_progress_badge(self, key: Optional[str] = None):
        """Updates the card to show that the main task is in progress."""
        key = key or "In progress"
        self.update_badge_by_key(key=key, label="in progress", badge_type="info")

    def hide_in_progress_badge(self, key: Optional[str] = None):
        """Hides the in-progress badge from the card."""
        key = key or "In progress"
        self.remove_badge_by_key(key=key)

    # Finished Badge Methods -------------------------------------------
    def show_finished_badge(self):
        """Updates the card to show that main task is finished."""
        self.update_badge_by_key(key="Finished", label="done", badge_type="success")

    def hide_finished_badge(self):
        """Hides the finished badge from the card."""
        self.remove_badge_by_key(key="Finished")

    # Failed Badge Methods ---------------------------------------------
    def show_failed_badge(self):
        """Updates the card to show that the main task has failed."""
        self.update_badge_by_key(key="Failed", label="error", badge_type="error")

    def hide_failed_badge(self):
        """Hides the failed badge from the card."""
        self.remove_badge_by_key(key="Failed")


# ------------------------------------------------------------------
# Project Node -----------------------------------------------------
# ------------------------------------------------------------------
# SolutionProjectNode
class BaseProjectNode(BaseNode):
    NODE_TYPE = "project"

    # Preview Methods --------------------------------------------------
    def update_preview(self, imgs: List[str], counts: List[int]):
        """Updates the preview of the project."""
        self.update_preview_url(imgs)
        self.update_items_count(counts)

    def update_preview_url(self, imgs: List[str]):
        """Updates the preview url of the project."""
        if isinstance(imgs, str):
            imgs = [imgs]
        if not isinstance(imgs, list):
            raise TypeError("imgs must be a list of strings")
        for idx, img in enumerate(imgs):
            if idx < len(self.settings.previews):
                self.settings.previews[idx]["src"] = img
            else:
                self.settings.previews.append({"src": img, "label": "0 images"})
        StateJson().send_changes()
        VueFlow.update_node(self)

    def update_items_count(self, counts: List[int]):
        """Updates the items count of the project."""

        def _pretty_count(count: Union[List, int, None, str]) -> str:
            if count is None:
                return "0 images"
            if isinstance(count, int):
                return f"{count} images"
            if isinstance(count, str):
                return count
            raise TypeError("count must be an int, str, or None")

        if isinstance(counts, (int, str)):
            counts = [counts]
        if not isinstance(counts, list):
            raise TypeError("counts must be a list of integers or strings")
        for idx, count in enumerate(counts):
            if idx < len(self.settings.previews):
                self.settings.previews[idx]["label"] = _pretty_count(count)
            else:
                self.settings.previews.append({"src": "", "label": _pretty_count(count)})
        StateJson().send_changes()
        VueFlow.update_node(self)

    # Update Methods ---------------------------------------------------
    def update(
        self,
        project: ProjectInfo = None,
        new_items_count: int = None,
        urls: List[Union[int, str, None]] = None,
        counts: List[Union[int, None]] = None,
    ):
        """Updates the node with the new project."""
        if project is not None:
            self.update_property(key="Total", value=f"{project.items_count} images")
        if new_items_count is not None:
            self.update_property(key="Last update", value=f"+{new_items_count}")
            self.update_badge_by_key(key="Last update", label=f"+{new_items_count}")
        if urls is not None and counts is not None and self.is_training:
            self.update_preview(imgs=urls, counts=counts)
        else:
            self.update_preview(
                imgs=[project.image_preview_url] if project else [],
                counts=[project.items_count] if project else [],
            )
            self.update_preview(
                imgs=[project.image_preview_url] if project else [],
                counts=[project.items_count] if project else [],
            )


# ------------------------------------------------------------------
# Queue Node -------------------------------------------------------
# ------------------------------------------------------------------
# SolutionQueueNode
class BaseQueueNode(BaseNode):
    NODE_TYPE = "queue"

    def _create_queue_info(self, *args, **kwargs) -> Optional[NodeQueueInfo]:
        return NodeQueueInfo()

    def update_pending(self, pending: int):
        """Updates the pending count of the queue."""
        self.settings.queue_info.pending = pending
        VueFlow.update_node(self)

    def update_annotation(self, annotating: int):
        """Updates the annotating count of the queue."""
        self.settings.queue_info.annotating = annotating
        VueFlow.update_node(self)

    def update_review(self, reviewing: int):
        """Updates the reviewing count of the queue."""
        self.settings.queue_info.reviewing = reviewing
        VueFlow.update_node(self)

    def update_finished(self, finished: int):
        """Updates the finished count of the queue."""
        self.settings.queue_info.finished = finished
        VueFlow.update_node(self)

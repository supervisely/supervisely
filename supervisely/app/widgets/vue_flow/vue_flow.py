import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Union

from supervisely.app.content import DataJson, StateJson
from supervisely.app.fastapi.subapp import Application
from supervisely.app.fastapi.utils import run_sync
from supervisely.app.fastapi.websocket import WebsocketManager
from supervisely.app.widgets import Button, DynamicWidget, Widget
from supervisely.app.widgets_context import JinjaWidgets
from supervisely.io.fs import clean_dir, copy_dir_recursively


@dataclass
class NodeBadge:
    label: str = ""
    value: str = None

    def to_json(self):
        return asdict(self)

    # def __init__(self, label: str, value: str = None):
    #     self._label = label
    #     self._value = value

    # def to_json(self):
    #     return {"label": self._label, "value": self._value}


@dataclass
class TooltipButton:
    text: str
    link: Optional[str] = None
    icon: Optional[str] = None

    def to_json(self):
        return {"text": self.text, "link": self.link, "icon": self.icon}


@dataclass
class TooltipProperty:
    label: str
    value: str
    link: Optional[Dict[str, str]] = None
    highlight: bool = False

    def to_json(self):
        return asdict(self)


@dataclass
class NodeTooltip:
    description: Optional[str] = ""
    properties: List[TooltipProperty] = field(default_factory=list)
    buttons: List[Union[Button, TooltipButton]] = field(default_factory=list)

    def __post_init__(self):
        """Post-initialization to convert Button instances to dicts."""
        for idx, btn in enumerate(self.buttons):
            if isinstance(btn, Button):
                btn: Button
                btn_icon = re.sub(r'<i class="(.*?)"', r"\1", btn.icon) if btn.icon else None
                if btn.link is not None:
                    btn = TooltipButton(
                        text=btn.text,
                        link={"url": btn.link},
                        icon=btn_icon,
                    )
                else:
                    btn = TooltipButton(
                        text=btn.text,
                        link={"action": btn.get_route_path(Button.Routes.CLICK)},
                        icon=btn_icon,
                    )
                self.buttons[idx] = btn

    def to_json(self):
        return asdict(self)


@dataclass
class NodeQueueInfo:
    """Information about the queue."""

    pending: int = 0
    annotating: int = 0
    reviewing: int = 0
    finished: int = 0

    def to_json(self):
        return asdict(self)
    
@dataclass
class Handle:
    """Represents a handle in the VueFlow node."""

    id: str
    type: Literal["source", "target"] = "source"
    position: Literal["left", "right", "top", "bottom"] = "left"
    label: Optional[str] = None
    connectable: bool = True

    def to_json(self):
        return asdict(self)


@dataclass
class NodeSettings:
    """Settings for the VueFlow node."""

    type: Literal["project", "action", "queue"] = "action"
    icon: Optional[Dict[str, str]] = None
    previews: List[Dict[str, str]] = field(default_factory=list)
    badges: List[NodeBadge] = field(default_factory=list)
    tooltip: NodeTooltip = NodeTooltip()
    queue_info: Optional[NodeQueueInfo] = field(
        default_factory=NodeQueueInfo, metadata={"alias": "queueInfo"}
    )
    handles: List[Handle] = field(default_factory=list)

    def to_json(self):
        return asdict(self)


class VueFlow(Widget):
    """
    VueFlow widget to wrap VueFlow component.
    This widget is used to create interactive flow diagrams in Supervisely for visualizing workflows.
    """

    class Node:
        """Represents a node in the VueFlow diagram."""

        def __init__(
            self,
            id: str,  # * widget ID
            label: str,
            x: int,
            y: int,
            parent_id: Optional[str] = None,
            settings: Optional[NodeSettings] = None,
        ):
            self.id = id
            self.label = label
            self.position = {"x": x, "y": y}
            self.settings = settings if settings is not None else NodeSettings()
            self.parent_id = parent_id

        def to_json(self):
            """
            Converts the node to a JSON serializable format.
            """
            return {
                "id": self.id,
                "type": "sly-flow",
                "label": self.label,
                "position": self.position,
                "data": self.settings.to_json(),
            }

        def _wrap_actions(self, func: Callable) -> Callable:
            """Decorator to call the update_node method when a node is changed."""

            def decorator(*args, **kwargs):
                res = func(*args, **kwargs)
                for idx, node_json in enumerate(StateJson()[self.parent_id]["nodes"]):
                    if node_json["id"] == self.id:
                        StateJson()[self.parent_id]["nodes"][idx] = self.to_json()
                        StateJson().send_changes()
                        VueFlow.update_node(self.parent_id, self)
                        break
                return res

            return decorator

    class ActionNode(Node):

        def __init__(
            self,
            id: str,
            label: str,
            x: int,
            y: int,
            parent_id: Optional[str] = None,
            settings: Optional[NodeSettings] = None,
        ):
            super().__init__(id, label, x, y, parent_id, settings)
            # wrap all methods that change the node state
            self.update_property = self._wrap_actions(self._update_property)
            self.remove_property_by_key = self._wrap_actions(self._remove_property_by_key)
            self.add_badge = self._wrap_actions(self._add_badge)
            self.remove_badge = self._wrap_actions(self._remove_badge)
            self.update_badge = self._wrap_actions(self._update_badge)
            self.update_badge_by_key = self._wrap_actions(self._update_badge_by_key)
            self.remove_badge_by_key = self._wrap_actions(self._remove_badge_by_key)

        # ------------------------------------------------------------------
        # Tooltip Methods --------------------------------------------------
        # ------------------------------------------------------------------
        def _update_property(self, key: str, value: str, link: str = None, highlight: bool = None):
            """Updates the property of the card."""
            for prop in self.settings.tooltip.properties:
                if prop.get("label") == key:
                    prop["value"] = value
                    prop["link"] = {"url": link} if link else None
                    prop["highlight"] = highlight if highlight is not None else False
                    return
            # If property not found, add it
            new_prop = {
                "label": key,
                "value": value,
                "link": {"url": link} if link else None,
                "highlight": highlight if highlight is not None else False,
            }
            self.settings.tooltip.properties.append(new_prop)

        def _remove_property_by_key(self, key: str, silent: bool = True):
            """Removes the property by key of the card."""
            for idx, prop in enumerate(self.settings.tooltip.properties):
                if prop.get("label") == key:
                    self.settings.tooltip.properties.pop(idx, None)
                    return
            if not silent:
                raise KeyError(f"Property with key '{key}' not found in tooltip properties.")

        # ------------------------------------------------------------------
        # Badge Methods ----------------------------------------------------
        # ------------------------------------------------------------------
        def _add_badge(self, badge: Union[dict, NodeBadge]):
            """Adds a badge to the card."""
            # if self.settings.badges is None:
            #     self.settings.badges = []
            # if isinstance(badge, NodeBadge):
            #     badge = badge.to_json()
            if not isinstance(badge, (dict, NodeBadge)):
                raise TypeError("Badge must be an instance of NodeBadge or a dict")
            self.settings.badges.append(badge)

        def _remove_badge(self, idx: int, silent: bool = True):
            """Removes the badge by index of the card."""
            if not self.settings.badges or idx >= len(self.settings.badges):
                if not silent:
                    raise IndexError("Badge index out of range")
            self.settings.badges.pop(idx, None)

        def _update_badge(
            self,
            idx: int,
            label: str,
            on_hover: str = None,
            badge_type: Literal["info", "success", "warning", "error"] = "info",  # TODO: remove
        ):
            """Updates the badge by index of the card."""
            if not self.settings.badges or idx >= len(self.settings.badges):
                raise IndexError("Badge index out of range")
            badge = self.settings.badges[idx]
            badge._label = label
            badge._value = on_hover if on_hover else ""

        def _update_badge_by_key(
            self,
            key: str,
            label: str,
            badge_type: Literal["info", "success", "warning", "error"] = None,  # TODO: remove
            new_key: str = None,
            plain: Optional[bool] = None,  # TODO: remove
        ):
            """Updates the badge by key of the card."""
            for badge in self.settings.badges:
                if badge.label == key:
                    badge.value = label
                    if new_key is not None:
                        badge.label = new_key
                    return
            # If badge not found, add it
            new_badge = NodeBadge(label=key, value=label)
            self.add_badge(new_badge)

        def _remove_badge_by_key(self, key: str):
            """Removes the badge by key from the card."""
            for idx, badge in enumerate(self.settings.badges):
                if badge.label == key:
                    self.settings.badges.pop(idx, None)
                    return
            raise KeyError(f"Badge with key '{key}' not found in card badges.")

        # ------------------------------------------------------------------
        # Automation Badge Methods -----------------------------------------
        # ------------------------------------------------------------------
        def _show_automation_badge(self) -> None:
            """Updates the card to show that automation is enabled."""
            self.update_badge_by_key(key="Automation", label="⚡")

        def _hide_automation_badge(self) -> None:
            """Updates the card to show that automation is disabled."""
            self.remove_badge_by_key(key="Automation")

        # ------------------------------------------------------------------
        # In Progress Badge Methods ----------------------------------------
        # ------------------------------------------------------------------
        def _show_in_progress_badge(self, key: Optional[str] = None):
            """Updates the card to show that the main task is in progress."""
            key = key or "In progress"
            self.update_badge_by_key(key=key, label="⏳")

        def _hide_in_progress_badge(self, key: Optional[str] = None):
            """Hides the in-progress badge from the card."""
            key = key or "In progress"
            self.remove_badge_by_key(key=key)

        # ------------------------------------------------------------------
        # Finished Badge Methods -------------------------------------------
        # ------------------------------------------------------------------
        def _show_finished_badge(self):
            """Updates the card to show that main task is finished."""
            self.update_badge_by_key(key="Finished", label="✅")

        def _hide_finished_badge(self):
            """Hides the finished badge from the card."""
            self.remove_badge_by_key(key="Finished")

        # ------------------------------------------------------------------
        # Failed Badge Methods ---------------------------------------------
        # ------------------------------------------------------------------
        def _show_failed_badge(self):
            """Updates the card to show that the main task has failed."""
            self.update_badge_by_key(key="Failed", label="❌")

        def _hide_failed_badge(self):
            """Hides the failed badge from the card."""
            self.remove_badge_by_key(key="Failed")

    def __init__(self, nodes: Optional[List[Node]] = None, widget_id: str = None):
        self.nodes = nodes if nodes is not None else []
        self._url = None
        super().__init__(widget_id=widget_id, file_path=__file__)
        script_path = "./sly/css/app/widgets/vue_flow/script.js"
        JinjaWidgets().context["__widget_scripts__"][self.__class__.__name__] = script_path

        # ! TODO: move vue_flow_ui folder to static folder
        self._prepare_ui_static()

    def get_json_data(self):
        return {}

    def get_json_state(self):
        return {
            "nodes": [node.to_json() for node in self.nodes],
            "edges": [],  # Edges can be added later if needed
            "url": self._url if self._url is not None else "",
        }

    def add_nodes(self, nodes: List[Node]) -> None:
        """
        Adds nodes to the VueFlow widget.

        :param nodes: List of Node objects to be added.
        """
        self.nodes.extend(nodes)
        if "nodes" not in StateJson()[self.widget_id]:
            StateJson()[self.widget_id]["nodes"] = []
        StateJson()[self.widget_id]["nodes"].extend([node.to_json() for node in nodes])
        StateJson().send_changes()

    def add_node(self, node: Node) -> None:
        """
        Adds a single node to the VueFlow widget.

        :param node: Node object to be added.
        """
        self.add_nodes([node])

    def remove_node(self, node: Union[Node, str]) -> None:
        """
        Removes a node from the VueFlow widget by its ID.

        :param node_id: Widget or node ID to be removed.
        """
        node_id = node.id if isinstance(node, self.Node) else node
        if not isinstance(node_id, str):
            raise ValueError("Node ID must be a string.")
        self.nodes = [node for node in self.nodes if node.id != node_id]
        StateJson()[self.widget_id]["nodes"] = [node.to_json() for node in self.nodes]
        StateJson().send_changes()

    def _prepare_ui_static(self) -> None:
        """
        Prepares the static files for the VueFlow widget.
        This method is called to ensure that the necessary static files are available for the widget.
        """
        # app = Application()
        vue_flow_ui_dir = Path(__file__).parent / "vue_flow_ui"
        # static_dir = Path(app.get_static_dir()) # will not work when initializing app (None)
        static_dir = Path("static")
        new_vue_flow_ui_dir = static_dir / "vue_flow_ui"
        new_vue_flow_ui_dir.mkdir(parents=True, exist_ok=True)
        clean_dir(str(new_vue_flow_ui_dir))
        copy_dir_recursively(str(vue_flow_ui_dir), str(new_vue_flow_ui_dir))
        self._url = f"http://0.0.0.0:8000/{str(new_vue_flow_ui_dir)}/index.html"  # ! TODO: ???
        StateJson()[self.widget_id]["url"] = self._url
        StateJson().send_changes()

    def reload(self) -> None:
        """
        Reloads the VueFlow widget to reflect any changes made to the nodes.
        """
        run_sync(
            WebsocketManager().broadcast(
                {
                    "runAction": {
                        "action": f"sly-flow-{self.widget_id}",
                        "payload": {"action": "refresh-nodes"},
                    }
                }
            )
        )

    @staticmethod
    def update_node(parent_id: str, node: Union[str, Node]) -> None:
        """
        Updates a node in the VueFlow widget.

        :param node: Node object or ID to be updated.
        """
        pass
        node_id = node.id if isinstance(node, VueFlow.Node) else node
        if not isinstance(node_id, str):
            raise ValueError("Node ID must be a string.")
        run_sync(
            WebsocketManager().broadcast(
                {
                    "runAction": {
                        "action": f"sly-flow-{parent_id}",
                        "payload": {"action": "update-node", "data": {"nodeId": node_id}},
                    }
                }
            )
        )

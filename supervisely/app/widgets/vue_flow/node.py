from abc import abstractmethod
from typing import Any, Callable, Dict, Literal, Optional, Union

from supervisely.app.fastapi import _MainServer
from supervisely.app.widgets.vue_flow.models import (
    NodeBadge,
    NodeBadgeStyle,
    NodeLink,
    NodeSettings,
    TooltipProperty,
    get_badge_style,
    node_badge_style_map,
)


class Node:
    """Represents a node in the VueFlow diagram."""

    def __init__(
        self,
        id: str,  # * widget ID
        label: str,
        x: Optional[int] = None,
        y: Optional[int] = None,
        position: Optional[Dict[str, int]] = None,
        parent_id: Optional[str] = None,
        data: Optional[NodeSettings] = None,
    ):
        self.id = id
        self.label = label
        if position is not None:
            if "x" not in position or "y" not in position:
                raise ValueError("Position must contain both 'x' and 'y' keys.")
            self.position = position
        elif x is not None or y is not None:
            self.position = {"x": x, "y": y}
        else:
            raise ValueError("Either position or both x and y must be provided.")
        if data is not None:
            if isinstance(data, dict):
                self.settings = NodeSettings(**data)
            elif isinstance(data, NodeSettings):
                self.settings = data
            else:
                self.settings = NodeSettings()
        self.parent_id = parent_id  # ID of the VueFlow widget
        self._server = _MainServer().get_server()

        # ----------------------------------------------------------------
        # --- Connected Nodes --------------------------------------------
        # --- Every node have info about source (parent) nodes -----------
        self._sources = []
        # ----------------------------------------------------------------
        super().__init__()

    @abstractmethod
    def update_node(node: "Node") -> None:
        """Updates the node in the VueFlow diagram."""
        raise NotImplementedError("Subclasses must implement this method in VueFlow.")

    def to_json(self):
        """
        Converts the node to a JSON serializable format.
        """
        return {
            "id": self.id,
            "type": "sly-flow",
            "label": self.label,
            "position": self.position,
            "data": self.settings.model_dump(by_alias=True),
        }

    # ------------------------------------------------------------------
    # Card Methods -----------------------------------------------------
    # ------------------------------------------------------------------
    def set_link(self, link: str):
        """Sets the link of the card."""
        self.settings.link = NodeLink(url=link)
        self.update_node(self)

    def remove_link(self):
        """Removes the link of the card."""
        self.settings.link = NodeLink(url=None)
        self.update_node(self)

    # ------------------------------------------------------------------
    # Tooltip Methods --------------------------------------------------
    # ------------------------------------------------------------------
    def _find_free_property_idx(self) -> int:
        free_idx = 0
        while True:
            if not self.settings.tooltip.properties:
                break
            if self._format_idx(free_idx) not in self.settings.tooltip.properties.keys():
                break
            free_idx += 1
        return free_idx

    def update_property(self, key: str, value: str, link: str = None, highlight: bool = None):
        """Updates the property of the card."""
        for prop in self.settings.tooltip.properties.values():
            if prop.label == key:
                prop.value = value
                if link is not None:
                    prop.link = {"url": link}
                if highlight is not None:
                    prop.highlight = highlight
                self.update_node(self)
                return
        new_prop = TooltipProperty(
            label=key,
            value=value,
            link={"url": link} if link else None,
            highlight=highlight if highlight is not None else False,
        )
        idx = self._find_free_property_idx()
        self.settings.tooltip.properties[self._format_idx(idx)] = new_prop
        self.update_node(self)

    def remove_property_by_key(self, key: str, silent: bool = True):
        """Removes the property by key of the card."""
        for idx, prop in self.settings.tooltip.properties.items():
            if prop.label == key:
                self.settings.tooltip.properties.pop(idx)
                self.update_node(self)
                return
        if not silent:
            raise KeyError(f"Property with key '{key}' not found in tooltip properties.")

    # ------------------------------------------------------------------
    # Badge Methods ----------------------------------------------------
    # ------------------------------------------------------------------
    def _format_idx(self, idx: int) -> str:
        return f"{idx:3d}"

    def _find_free_badge_idx(self) -> int:
        free_idx = 0
        while True:
            if not self.settings.badges:
                break
            if self._format_idx(free_idx) not in self.settings.badges.keys():
                break
            free_idx += 1
        return free_idx

    def add_badge(self, badge: Union[dict, NodeBadge]):
        """Adds a badge to the card."""
        if not isinstance(badge, (dict, NodeBadge)):
            raise TypeError("Badge must be an instance of NodeBadge or a dict")
        idx = self._find_free_badge_idx()
        self.settings.badges[self._format_idx(idx)] = badge
        self.update_node(self)

    def remove_badge(self, idx: int, silent: bool = True):
        """Removes the badge by index of the card."""
        if not self.settings.badges or self._format_idx(idx) not in self.settings.badges:
            if not silent:
                raise IndexError("Badge index out of range")
            return
        self.settings.badges.pop(self._format_idx(idx))
        self.update_node(self)

    def update_badge(
        self,
        idx: int,
        key: Optional[str] = None,
        label: Optional[str] = None,
        badge_type: Literal["info", "success", "warning", "error"] = None,
        plain: Optional[bool] = None,
    ):
        """Updates the badge by index of the card."""
        if not self.settings.badges or self._format_idx(idx) not in self.settings.badges:
            raise IndexError("Badge index out of range")
        badge = self.settings.badges[self._format_idx(idx)]
        if key is not None:
            badge.label = key
        if label is not None:
            badge.value = label
        if plain is not None and plain:
            badge.style = NodeBadgeStyle()
        elif badge_type is not None:
            if badge_type in node_badge_style_map.keys():
                badge.style = get_badge_style(badge_type)
        self.update_node(self)

    def update_badge_by_key(
        self,
        key: str,
        label: str,
        badge_type: Optional[Literal["info", "success", "warning", "error"]] = None,
        new_key: Optional[str] = None,
        plain: Optional[bool] = None,
    ):
        """Updates the badge by key of the card."""
        for idx, badge in self.settings.badges.items():
            if badge.label == key:
                self.update_badge(
                    idx=int(idx),
                    key=new_key,
                    label=label,
                    badge_type=badge_type,
                    plain=plain,
                )
                return
        # If badge not found, add it
        if plain:
            style = NodeBadgeStyle()
        else:
            badge_type = badge_type if badge_type in node_badge_style_map.keys() else "info"
            style = get_badge_style(badge_type)
        new_badge = NodeBadge(label=key, value=label, style=style)
        self.add_badge(new_badge)

    def remove_badge_by_key(self, key: str, silent: bool = True):
        """Removes the badge by key from the card."""
        for k, badge in self.settings.badges.items():
            if badge.label == key:
                self.remove_badge(int(k), silent=silent)
                return
        if not silent:
            raise KeyError(f"Badge with key '{key}' not found in badges.")

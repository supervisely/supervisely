from abc import abstractmethod
from typing import Callable, Dict, Literal, Optional, Union

from supervisely.app.fastapi import _MainServer
from supervisely.app.widgets.vue_flow.models import (
    NodeBadge,
    NodeBadgeStyle,
    NodeBadgeStyleMap,
    NodeSettings,
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
    # Tooltip Methods --------------------------------------------------
    # ------------------------------------------------------------------
    def update_property(self, key: str, value: str, link: str = None, highlight: bool = None):
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
        self.update_node(self)

    def remove_property_by_key(self, key: str, silent: bool = True):
        """Removes the property by key of the card."""
        for idx, prop in enumerate(self.settings.tooltip.properties):
            if prop.get("label") == key:
                self.settings.tooltip.properties.pop(idx)
                self.update_node(self)
                return
        if not silent:
            raise KeyError(f"Property with key '{key}' not found in tooltip properties.")

    # ------------------------------------------------------------------
    # Badge Methods ----------------------------------------------------
    # ------------------------------------------------------------------
    def add_badge(self, badge: Union[dict, NodeBadge]):
        """Adds a badge to the card."""
        if not isinstance(badge, (dict, NodeBadge)):
            raise TypeError("Badge must be an instance of NodeBadge or a dict")
        self.settings.badges.append(badge)
        self.update_node(self)

    def remove_badge(self, idx: int, silent: bool = True):
        """Removes the badge by index of the card."""
        if not self.settings.badges or idx >= len(self.settings.badges):
            if not silent:
                raise IndexError("Badge index out of range")
        self.settings.badges.pop(idx)
        self.update_node(self)

    def update_badge(
        self,
        idx: int,
        label: str,
        on_hover: str = None,
        badge_type: Literal["info", "success", "warning", "error"] = None,
        plain: Optional[bool] = None,
    ):
        """Updates the badge by index of the card."""
        if not self.settings.badges or idx >= len(self.settings.badges):
            raise IndexError("Badge index out of range")
        badge = self.settings.badges[idx]
        badge._label = label
        if on_hover is not None:
            badge._on_hover = on_hover
        if plain is not None and plain:
            badge.style = NodeBadgeStyle()
        elif badge_type is not None and badge_type in NodeBadgeStyleMap.__members__:
            badge.style = NodeBadgeStyleMap[badge_type].value
        self.update_node(self)

    def update_badge_by_key(
        self,
        key: str,
        label: str,
        badge_type: Literal["info", "success", "warning", "error"] = None,
        new_key: str = None,
        plain: Optional[bool] = None,  # TODO: remove
    ):
        """Updates the badge by key of the card."""
        for badge in self.settings.badges:
            if badge.label == key:
                badge.value = label
                if new_key is not None:
                    badge.label = new_key
                if plain is not None and plain:
                    badge.style = NodeBadgeStyle()
                elif badge_type is not None:
                    badge.style = NodeBadgeStyleMap[badge_type].value
                self.update_node(self)
                return
        # If badge not found, add it
        if plain:
            style = NodeBadgeStyle()
        else:
            badge_type = badge_type if badge_type in NodeBadgeStyleMap.__members__ else "info"
            style = NodeBadgeStyleMap[badge_type].value
        new_badge = NodeBadge(label=key, value=label, style=style)
        self.add_badge(new_badge)

    def remove_badge_by_key(self, key: str):
        """Removes the badge by key from the card."""
        for idx, badge in enumerate(self.settings.badges):
            if badge.label == key:
                self.settings.badges.pop(idx)
                self.update_node(self)
                return

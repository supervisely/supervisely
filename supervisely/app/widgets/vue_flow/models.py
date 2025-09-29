import re
from enum import Enum
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class NodeBadgeStyle(BaseModel):
    backgroundColor: Optional[str] = Field(default=None, alias="backgroundColor")
    color: Optional[str] = Field(default=None)

    def model_dump(self, *args, **kwargs):
        return super().model_dump(*args, **kwargs, exclude_none=True)


node_badge_style_map = {
    "info": "#1976D2",
    "warning": "#FFA000",
    "error": "#D32F2F",
    "success": "#388E3C",
}


def get_badge_style(badge_type: str) -> NodeBadgeStyle:
    return NodeBadgeStyle(
        backgroundColor=node_badge_style_map.get(badge_type),
        color="#FFFFFF",
    )


class NodeBadge(BaseModel):
    label: str = Field(..., description="Badge label")
    value: str = Field(..., description="Badge value")
    style: Optional[NodeBadgeStyle] = Field(default=None, description="Badge style")


class TooltipButton(BaseModel):
    label: str = Field(..., description="Button label")
    link: Optional[Dict[str, str]] = Field(default=None, description="Optional link for the button")
    icon: Optional[str] = Field(default=None, description="Optional icon for the button")

    def __init__(self, **data):
        super().__init__(**data)
        """Post-initialization to convert Button instances to dicts."""
        self.icon = re.sub(r'<i class="(.*?)".*?</i>', r"\1", self.icon) if self.icon else None


class TooltipProperty(BaseModel):
    label: str = Field(..., description="Property label")
    value: str = Field(..., description="Property value")
    link: Optional[Dict[str, str]] = Field(
        default=None, description="Optional link for the property value"
    )
    highlight: bool = Field(default=False, description="Whether to highlight the property")


class NodeTooltip(BaseModel):

    description: Optional[str] = Field(default=None)
    properties: List[TooltipProperty] = Field(default_factory=list)
    buttons: List[TooltipButton] = Field(default_factory=list)


class NodeQueueInfo(BaseModel):
    """Information about the queue."""

    pending: int = 0
    annotating: int = 0
    reviewing: int = 0
    finished: int = 0


class Handle(BaseModel):
    """Represents a handle in the VueFlow node."""

    id: str
    type: Literal["source", "target"] = "source"
    position: Literal["left", "right", "top", "bottom"] = "left"
    label: Optional[str] = None
    connectable: bool = True
    style: Optional[Dict[str, str]] = None


class NodeIcon(BaseModel):
    """Represents an icon in the VueFlow node."""

    name: str
    color: Optional[str] = None
    bg_color: Optional[str] = Field(default=None, alias="backgroundColor")


class NodeLink(BaseModel):
    """Represents a link in the VueFlow node."""

    url: Optional[str] = Field(default=None)
    action: Optional[str] = Field(default=None)

    def model_dump(self, *args, **kwargs):
        return super().model_dump(*args, **kwargs, exclude_none=True)


class NodeSettings(BaseModel):
    """Settings for the VueFlow node."""

    type: Literal["project", "action", "queue"] = "action"
    icon: Optional[NodeIcon] = None
    previews: List[Dict[str, str]] = Field(default_factory=list)
    badges: List[NodeBadge] = Field(default_factory=list)
    tooltip: Optional[NodeTooltip] = Field(default_factory=NodeTooltip)
    queue_info: Optional[NodeQueueInfo] = Field(default_factory=NodeQueueInfo, alias="queueInfo")
    handles: List[Handle] = Field(default_factory=list)
    link: Optional[NodeLink] = Field(default=None)
    removable: bool = True
    toolbarVisible: bool = False

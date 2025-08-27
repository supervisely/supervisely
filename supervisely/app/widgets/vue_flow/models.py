import re
from enum import Enum
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class NodeBadgeStyle(BaseModel):
    backgroundColor: Optional[str] = None
    color: Optional[str] = None

    def model_dump(self, *args, **kwargs):
        return super().model_dump(*args, **kwargs, exclude_none=True)


class NodeBadgeStyleMap(Enum):
    info = NodeBadgeStyle(backgroundColor="#1976D2", color="#FFFFFF")
    warning = NodeBadgeStyle(backgroundColor="#FFA000", color="#FFFFFF")
    error = NodeBadgeStyle(backgroundColor="#D32F2F", color="#FFFFFF")
    success = NodeBadgeStyle(backgroundColor="#388E3C", color="#FFFFFF")


class NodeBadge(BaseModel):
    label: str = ""
    value: str = None
    style: Optional[NodeBadgeStyle] = None


class TooltipButton(BaseModel):
    label: str
    link: Optional[Dict[str, str]] = None
    icon: Optional[str] = None

    def __init__(self, **data):
        super().__init__(**data)
        """Post-initialization to convert Button instances to dicts."""
        self.icon = re.sub(r'<i class="(.*?)".*?</i>', r"\1", self.icon) if self.icon else None


class TooltipProperty(BaseModel):
    label: str
    value: str
    link: Optional[Dict[str, str]] = None
    highlight: bool = False


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

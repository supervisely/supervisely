from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class Edge(BaseModel):
    """Represents an edge in the VueFlow diagram."""

    id: str
    source: str
    target: str
    marker_end: str = Field(default="arrowclosed", alias="markerEnd")
    marker_start: Optional[str] = Field(default=None, alias="markerStart")
    target_handle: Optional[str] = Field(default=None, alias="targetHandle")
    source_handle: Optional[str] = Field(default=None, alias="sourceHandle")
    type: str = "smoothstep"
    label: Optional[str] = None
    animated: bool = False
    curvature: float = 1.0
    style: Optional[Dict[str, Any]] = Field(default=None)
    updatable: bool = True

    def to_json(self):
        return self.model_dump(by_alias=True, exclude_none=True)

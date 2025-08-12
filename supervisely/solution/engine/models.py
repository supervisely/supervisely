# pydantic models for messages in publish/subscribe system
from typing import Callable, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field


class Message(BaseModel):
    """Represents a message that can be sent with an event. Subclasses should define specific fields."""

    pass


class ImportStartedMessage(Message):
    """Import has started event message."""

    task_id: int = Field(..., description="The ID of the import task")


class ImportFinishedMessage(Message):
    """Import has finished event message."""

    task_id: int = Field(..., description="The ID of the task")
    success: bool = Field(..., description="Indicates if the task was successful")
    items_count: Optional[int] = Field(None, description="Count of items processed by the task")
    image_preview_url: Optional[str] = Field(
        None, description="URL for the image preview associated with the task"
    )


class SampleFinishedMessage(Message):
    """Sample has finished event message."""

    success: bool = Field(..., description="Indicates if the sampling was successful")
    src: Dict[int, List[int]] = Field(
        ...,
        description="Dictionary with dataset IDs as keys and lists of image IDs from the source project",
    )
    dst: Dict[int, List[int]] = Field(
        ...,
        description="Dictionary with dataset IDs as keys and lists of image IDs from the destination project",
    )
    items_count: int = Field(..., description="Total number of images moved in the sample")


class MoveLabeledDataFinishedMessage(Message):
    """Move labeled data has finished event message."""

    success: bool = Field(..., description="Indicates if the move was successful")
    items: List[int] = Field(..., description="List of image IDs that were moved")
    items_count: int = Field(..., description="Total number of images moved")


class LabelingQueueRefreshInfoMessage(Message):
    """Labeling queue refresh info event message."""

    pending: int = Field(..., description="Number of pending items in the labeling queue")
    annotating: int = Field(..., description="Number of items currently being annotated")
    reviewing: int = Field(..., description="Number of items currently being reviewed")
    finished: int = Field(..., description="Number of items that have been finished")
    rejected: int = Field(..., description="Number of items that have been rejected")


class TrainValSplitMessage(Message):
    """Train/Val split event message."""

    train: List[int] = Field(..., description="List of image IDs in the training set")
    val: List[int] = Field(..., description="List of image IDs in the validation set")


class LabelingQueueAcceptedImagesMessage(Message):
    """Labeling queue accepted images event message."""

    accepted_images: List[int] = Field(
        ..., description="List of image IDs that have been accepted in the labeling queue"
    )

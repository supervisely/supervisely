from supervisely.collection.str_enum import StrEnum


class CVTask(StrEnum):
    """Enum of supported computer vision task types for benchmarking."""

    OBJECT_DETECTION: str = "object_detection"
    INSTANCE_SEGMENTATION: str = "instance_segmentation"
    SEMANTIC_SEGMENTATION: str = "semantic_segmentation"

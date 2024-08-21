from supervisely.collection.str_enum import StrEnum


class CVTask(StrEnum):

    OBJECT_DETECTION: str = "object_detection"
    INSTANCE_SEGMENTATION: str = "instance_segmentation"
    SEMANTIC_SEGMENTATION: str = "semantic_segmentation"

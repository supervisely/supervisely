from supervisely.collection.str_enum import StrEnum


class CVTask(StrEnum):

    OBJECT_DETECTION: str = "object_detection"
    SEMANTIC_SEGMENTATION: str = "semantic_segmentation"
    INSTANCE_SEGMENTATION: str = "instance_segmentation"

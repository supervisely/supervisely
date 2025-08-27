from supervisely.nn.artifacts.artifacts import BaseTrainArtifacts
from supervisely.nn.artifacts import (
    YOLOv5,
    YOLOv5v2,
    YOLOv8,
    MMClassification,
    MMPretrain,
    MMSegmentation,
    MMDetection,
    MMDetection3,
    Detectron2,
    UNet,
    RITM,
    RTDETR,
)


class FrameworkName:
    YOLOV5 = "YOLOv5"
    YOLOV5V2 = "YOLOv5 2.0"
    YOLOV8 = "YOLOv8+"
    MMCLASSIFICATION = "MMClassification"
    MMPRETRAIN = "MMPretrain"
    MMSEGMENTATION = "MMSegmentation"
    MMDETECTION = "MMDetection"
    MMDETECTION3 = "MMDetection 3.0"
    RTDETR = "RT-DETR"
    DETECTRON2 = "Detectron2"
    UNET = "UNet"
    RITM = "RITM"


class FrameworkMapper:
    _map = {
        FrameworkName.YOLOV5: YOLOv5,
        FrameworkName.YOLOV5V2: YOLOv5v2,
        FrameworkName.YOLOV8: YOLOv8,
        FrameworkName.MMCLASSIFICATION: MMClassification,
        FrameworkName.MMPRETRAIN: MMPretrain,
        FrameworkName.MMSEGMENTATION: MMSegmentation,
        FrameworkName.MMDETECTION: MMDetection,
        FrameworkName.MMDETECTION3: MMDetection3,
        FrameworkName.RTDETR: RTDETR,
        FrameworkName.DETECTRON2: Detectron2,
        FrameworkName.UNET: UNet,
        FrameworkName.RITM: RITM,
    }

    @classmethod
    def get_framework_cls(cls, name: str, team_id: int) -> BaseTrainArtifacts:
        if name not in cls._map:
            raise ValueError(f"Unknown framework: {name}")
        return cls._map[name](team_id)

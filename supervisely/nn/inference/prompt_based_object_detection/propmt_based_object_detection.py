from typing import Dict
from supervisely.geometry.bitmap import Bitmap
from supervisely.nn.prediction_dto import PredictionSegmentation
from supervisely.annotation.label import Label
from supervisely.sly_logger import logger
import numpy as np
import functools
from supervisely.geometry.rectangle import Rectangle
from supervisely.imaging import image as sly_image
from supervisely.decorators.inference import _scale_ann_to_original_size, _process_image_path
from supervisely.io.fs import silent_remove
from supervisely.decorators.inference import process_image_sliding_window
from supervisely.nn.inference.object_detection.object_detection import (
    ObjectDetection,
)


class PromptBasedObjectDetection(ObjectDetection):
    def get_info(self) -> dict:
        info = super().get_info()
        info["task type"] = "prompt-based object detection"
        return info

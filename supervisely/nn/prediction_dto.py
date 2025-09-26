from typing import List, Optional

import numpy as np

from supervisely.geometry.cuboid_3d import Cuboid3d
from supervisely.geometry.polyline_3d import Polyline3D


class Prediction:
    def __init__(self, class_name):
        self.class_name = class_name


class PredictionMask(Prediction):
    def __init__(self, class_name: str, mask: np.ndarray, score: Optional[float] = None):
        """
        class_name: Name of the class.
        mask:       Numpy array with bool or binary ([0, 1] or [0, 255]) values.
                    Will be converted to sly.Bitmap geometry.
        score:      Confidence score.
        """
        super(PredictionMask, self).__init__(class_name=class_name)
        self.mask = mask
        self.score = score


class PredictionBBox(Prediction):
    def __init__(self, class_name: str, bbox_tlbr: List[int], score: Optional[float]):
        super(PredictionBBox, self).__init__(class_name=class_name)
        self.bbox_tlbr = bbox_tlbr
        self.score = score


class PredictionSegmentation(Prediction):
    def __init__(self, mask: np.ndarray):
        self.mask = mask


class PredictionAlphaMask(Prediction):
    def __init__(self, class_name: str, mask: np.ndarray):
        """
        class_name: Name of the class.
        mask:       Numpy array with values in range [0, 255].
                    Will be converted to sly.AlphaMask geometry.
        """
        super(PredictionAlphaMask, self).__init__(class_name=class_name)
        self.mask = mask


class ProbabilityMask(Prediction):
    def __init__(self, class_name: str, mask: np.ndarray):
        """
        class_name: Name of the class.
        mask:       Numpy array with values in range [0, 255].
                    Will be converted to sly.AlphaMask geometry.
        """
        super(ProbabilityMask, self).__init__(class_name=class_name)
        self.mask = mask


class PredictionKeypoints(Prediction):
    def __init__(self, class_name: str, labels: List[str], coordinates: List[float]):
        super(PredictionKeypoints, self).__init__(class_name=class_name)
        self.labels = labels
        self.coordinates = coordinates


class PredictionPoint(Prediction):
    def __init__(self, class_name: str, col: int, row: int):
        super().__init__(class_name=class_name)
        self.col = col
        self.row = row


class PredictionCuboid3d(Prediction):
    def __init__(self, class_name: str, cuboid_3d: Cuboid3d, score: Optional[float]):
        """
        :param class_name: Predicted class name.
        :param cuboid_3d: Cuboid3d object.
        :param score: Confidence score.
        """
        super(PredictionCuboid3d, self).__init__(class_name=class_name)
        self.cuboid_3d = cuboid_3d
        self.score = score


class PredictionPolyline3D(Prediction):
    def __init__(self, class_name: str, polyline_3d: Polyline3D):
        super(PredictionPolyline3D, self).__init__(class_name=class_name)
        self.polyline_3d = polyline_3d

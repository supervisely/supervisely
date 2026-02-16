from typing import List, Optional

import numpy as np

from supervisely.geometry.cuboid_3d import Cuboid3d
from supervisely.geometry.polyline_3d import Polyline3D


class Prediction:
    def __init__(self, class_name):
        """
        :param class_name: Name of the class.
        :type class_name: str
        """
        self.class_name = class_name


class PredictionMask(Prediction):
    def __init__(self, class_name: str, mask: np.ndarray, score: Optional[float] = None):
        """
        :param class_name: Name of the class.
        :type class_name: str
        :param mask: Numpy array with bool or binary ([0, 1] or [0, 255]) values.
        :type mask: np.ndarray
        :param score: Confidence score.
        :type score: float, optional
        """
        super(PredictionMask, self).__init__(class_name=class_name)
        self.mask = mask
        self.score = score


class PredictionBBox(Prediction):
    def __init__(self, class_name: str, bbox_tlbr: List[int], score: Optional[float], angle: Optional[float] = None):
        """
        :param class_name: Predicted class name.
        :type class_name: str
        :param bbox_tlbr: Bounding box in (top, left, bottom, right) format.
        :type bbox_tlbr: list of 4 ints
        :param score: Confidence score.
        :type score: float, optional
        :param angle: Angle of rotation in radians. Positive values mean clockwise rotation.
        :type angle: int or float, optional
        """
        super(PredictionBBox, self).__init__(class_name=class_name)
        self.bbox_tlbr = bbox_tlbr
        self.score = score
        self.angle = angle


class PredictionSegmentation(Prediction):
    def __init__(self, mask: np.ndarray):
        """
        :param mask: Numpy array with bool or binary ([0, 1] or [0, 255]) values.
        :type mask: np.ndarray
        """
        self.mask = mask


class PredictionAlphaMask(Prediction):
    def __init__(self, class_name: str, mask: np.ndarray):
        """
        :param class_name: Name of the class.
        :type class_name: str
        :param mask: Numpy array with values in range [0, 255].
        :type mask: np.ndarray
        """
        super(PredictionAlphaMask, self).__init__(class_name=class_name)
        self.mask = mask


class ProbabilityMask(Prediction):
    def __init__(self, class_name: str, mask: np.ndarray):
        """
        :param class_name: Name of the class.
        :type class_name: str
        :param mask: Numpy array with values in range [0, 255].
        :type mask: np.ndarray
        """
        super(ProbabilityMask, self).__init__(class_name=class_name)
        self.mask = mask


class PredictionKeypoints(Prediction):
    def __init__(self, class_name: str, labels: List[str], coordinates: List[float]):
        """
        :param class_name: Name of the class.
        :type class_name: str
        :param labels: List of labels.
        :type labels: List[str]
        :param coordinates: List of coordinates.
        :type coordinates: List[float]
        """
        super(PredictionKeypoints, self).__init__(class_name=class_name)
        self.labels = labels
        self.coordinates = coordinates


class PredictionPoint(Prediction):
    def __init__(self, class_name: str, col: int, row: int):
        """
        :param class_name: Name of the class.
        :type class_name: str
        :param col: Column index.
        :type col: int
        :param row: Row index.
        :type row: int
        """
        super().__init__(class_name=class_name)
        self.col = col
        self.row = row


class PredictionCuboid3d(Prediction):
    def __init__(self, class_name: str, cuboid_3d: Cuboid3d, score: Optional[float]):
        """
        :param class_name: Name of the class.
        :type class_name: str
        :param cuboid_3d: :class:`~supervisely.geometry.cuboid_3d.Cuboid3d` object.
        :type cuboid_3d: :class:`~supervisely.geometry.cuboid_3d.Cuboid3d`
        :param score: Confidence score.
        :type score: float, optional
        """
        super(PredictionCuboid3d, self).__init__(class_name=class_name)
        self.cuboid_3d = cuboid_3d
        self.score = score


class PredictionPolyline3D(Prediction):
    def __init__(self, class_name: str, polyline_3d: Polyline3D):
        """
        :param class_name: Name of the class.
        :type class_name: str
        :param polyline_3d: :class:`~supervisely.geometry.polyline_3d.Polyline3D` object.
        :type polyline_3d: :class:`~supervisely.geometry.polyline_3d.Polyline3D`
        """
        super(PredictionPolyline3D, self).__init__(class_name=class_name)
        self.polyline_3d = polyline_3d

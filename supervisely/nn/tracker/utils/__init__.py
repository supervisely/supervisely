from typing import Tuple

import numpy as np

from supervisely.annotation.annotation import Annotation
from supervisely.project.project_meta import ProjectMeta


def ann_to_detections(ann: Annotation, project_meta: ProjectMeta) -> Tuple(np.ndarray, dict):
    """
    Convert annotation to detections array.
    :param ann: Supervisely Annotation object
    :type ann: Annotation
    :param project_meta: Supervisely ProjectMeta object
    :type project_meta: ProjectMeta
    :return: detections array N x (x, y, x, y, conf, label) and class to label mapping
    :rtype: Tuple(np.ndarray, dict)
    """
    # convert ann to N x (x, y, x, y, conf, cls) np.array
    cls2label = {obj_class.name: i for i, obj_class in enumerate(project_meta.obj_classes)}
    detections = []
    for label in ann.labels:
        cat = cls2label[label.obj_class.name]
        bbox = label.geometry.to_bbox()
        conf = label.tags.get("confidence").value
        detections.append([bbox.left, bbox.top, bbox.right, bbox.bottom, conf, cat])
    detections = np.array(detections)
    return detections, cls2label

import colorsys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from supervisely import logger


def read_label_file(
    label_path: str,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    :param label_path: Path to ``.label`` file.
    :type label_path: str
    :return: Semantic labels and instance IDs.
    :rtype: Tuple[Optional[np.ndarray], Optional[np.ndarray]]
    """
    if not Path(label_path).exists():
        return None, None

    try:
        labels = np.fromfile(label_path, dtype=np.uint32)
    except (OSError, ValueError) as e:
        logger.warning(f"Failed to read SemanticKITTI label file {label_path}: {e}")
        return None, None

    return labels & 0xFFFF, labels >> 16


def generate_color_for_class(class_id: int) -> List[int]:
    """
    :param class_id: Semantic class ID.
    :type class_id: int
    :return: Deterministic RGB color.
    :rtype: List[int]
    """
    golden_angle = 137.5
    hue = (class_id * golden_angle) % 360 / 360.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.8, 0.95)
    return [int(r * 255), int(g * 255), int(b * 255)]

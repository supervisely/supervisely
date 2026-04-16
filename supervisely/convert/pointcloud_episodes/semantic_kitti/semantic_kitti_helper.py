import colorsys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from supervisely import logger

# Official SemanticKITTI label mapping from semantic-kitti.yaml (PRBonn/semantic-kitti-api).
# Colors converted from BGR (original) → RGB.
# Format: {label_id: (class_name, [R, G, B])}
SEMANTIC_KITTI_LABEL_MAP: Dict[int, Tuple[str, List[int]]] = {
    0: ("unlabeled", [0, 0, 0]),
    1: ("outlier", [255, 0, 0]),
    10: ("car", [100, 150, 245]),
    11: ("bicycle", [100, 230, 245]),
    13: ("bus", [100, 80, 250]),
    15: ("motorcycle", [30, 60, 150]),
    16: ("on-rails", [0, 0, 255]),
    18: ("truck", [80, 30, 180]),
    20: ("other-vehicle", [0, 0, 255]),
    30: ("person", [255, 30, 30]),
    31: ("bicyclist", [255, 40, 200]),
    32: ("motorcyclist", [150, 30, 90]),
    40: ("road", [255, 0, 255]),
    44: ("parking", [255, 150, 255]),
    48: ("sidewalk", [75, 0, 75]),
    49: ("other-ground", [175, 0, 75]),
    50: ("building", [255, 200, 0]),
    51: ("fence", [255, 120, 50]),
    52: ("other-structure", [255, 150, 0]),
    60: ("lane-marking", [150, 255, 170]),
    70: ("vegetation", [0, 175, 0]),
    71: ("trunk", [135, 60, 0]),
    72: ("terrain", [150, 240, 80]),
    80: ("pole", [255, 240, 150]),
    81: ("traffic-sign", [255, 0, 0]),
    99: ("other-object", [50, 255, 255]),
    252: ("moving-car", [100, 150, 245]),
    253: ("moving-bicyclist", [255, 40, 200]),
    254: ("moving-person", [255, 30, 30]),
    255: ("moving-motorcyclist", [150, 30, 90]),
    256: ("moving-on-rails", [0, 0, 255]),
    257: ("moving-bus", [100, 80, 250]),
    258: ("moving-truck", [80, 30, 180]),
    259: ("moving-other-vehicle", [0, 0, 255]),
}


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

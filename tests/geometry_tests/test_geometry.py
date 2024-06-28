import os
from typing import List

import numpy as np
import pytest

import supervisely.imaging.image as sly_image
from supervisely.geometry.geometry import Geometry

# Draw Settings
default_color = [255, 255, 255]
thickness = 1


def draw_test(
    dir_name: str,
    test_name: str,
    image: np.ndarray,
    geometry: Geometry = None,
    color: List[int] = default_color,
):
    test_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = f"{test_dir}/test_results/{dir_name}/{test_name}.png"
    if geometry is not None:
        geometry.draw(image, color)
        image_path = f"{test_dir}/test_results/{dir_name}/{test_name}.png"

    sly_image.write(image_path, image)


def run_geometry_test():
    test_dir = os.path.dirname(os.path.abspath(__file__))
    pytest.main([test_dir])


if __name__ == "__main__":
    run_geometry_test()

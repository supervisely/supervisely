import random
from typing import List, Tuple, Union

import numpy as np
import pytest

from supervisely.geometry.alpha_mask import AlphaMask
from supervisely.geometry.any_geometry import AnyGeometry
from supervisely.geometry.bitmap import Bitmap
from supervisely.geometry.graph import KeypointsTemplate, Node
from supervisely.geometry.image_rotator import ImageRotator
from supervisely.geometry.point import Point
from supervisely.geometry.point_location import PointLocation, _flip_row_col_order
from supervisely.geometry.polygon import Polygon
from supervisely.geometry.polyline import Polyline
from supervisely.geometry.rectangle import Rectangle


@pytest.fixture
def random_kp_int() -> KeypointsTemplate:
    template = KeypointsTemplate()
    # keypoints
    template.add_point(label="nose", row=635, col=427)
    template.add_point(label="left_eye", row=597, col=404)
    template.add_point(label="right_eye", row=685, col=401)
    template.add_point(label="left_ear", row=575, col=431)
    template.add_point(label="right_ear", row=723, col=425)
    template.add_point(label="left_shoulder", row=502, col=614)
    template.add_point(label="right_shoulder", row=794, col=621)
    template.add_point(label="left_elbow", row=456, col=867)
    template.add_point(label="right_elbow", row=837, col=874)
    template.add_point(label="left_wrist", row=446, col=1066)
    template.add_point(label="right_wrist", row=845, col=1073)
    template.add_point(label="left_hip", row=557, col=1035)
    template.add_point(label="right_hip", row=743, col=1043)
    template.add_point(label="left_knee", row=541, col=1406)
    template.add_point(label="right_knee", row=751, col=1421)
    template.add_point(label="left_ankle", row=501, col=1760)
    template.add_point(label="right_ankle", row=774, col=1765)
    # connections
    template.add_edge(src="left_ankle", dst="left_knee")
    template.add_edge(src="left_knee", dst="left_hip")
    template.add_edge(src="right_ankle", dst="right_knee")
    template.add_edge(src="right_knee", dst="right_hip")
    template.add_edge(src="left_hip", dst="right_hip")
    template.add_edge(src="left_shoulder", dst="left_hip")
    template.add_edge(src="right_shoulder", dst="right_hip")
    template.add_edge(src="left_shoulder", dst="right_shoulder")
    template.add_edge(src="left_shoulder", dst="left_elbow")
    template.add_edge(src="right_shoulder", dst="right_elbow")
    template.add_edge(src="left_elbow", dst="left_wrist")
    template.add_edge(src="right_elbow", dst="right_wrist")
    template.add_edge(src="left_eye", dst="right_eye")
    template.add_edge(src="nose", dst="left_eye")
    template.add_edge(src="nose", dst="right_eye")
    template.add_edge(src="left_eye", dst="left_ear")
    template.add_edge(src="right_eye", dst="right_ear")
    template.add_edge(src="left_ear", dst="left_shoulder")
    template.add_edge(src="right_ear", dst="right_shoulder")
    return template


@pytest.fixture
def random_kp_float() -> KeypointsTemplate:
    template = KeypointsTemplate()
    # keypoints
    template.add_point(label="nose", row=635.353234, col=427.654433)
    template.add_point(label="left_eye", row=597.764234, col=404.123143)
    template.add_point(label="right_eye", row=685.982453, col=401.543543)
    template.add_point(label="left_ear", row=575.123321, col=431.232455)
    template.add_point(label="right_ear", row=723.654345, col=425.634311)
    template.add_point(label="left_shoulder", row=502.785754, col=614.435532)
    template.add_point(label="right_shoulder", row=794.234654, col=621.123431)
    template.add_point(label="left_elbow", row=456.231231, col=867.773421)
    template.add_point(label="right_elbow", row=837.554364, col=874.654777)
    template.add_point(label="left_wrist", row=446.654455, col=1066.654344)
    template.add_point(label="right_wrist", row=845.768666, col=1073.543221)
    template.add_point(label="left_hip", row=557.345234, col=1035.213415)
    template.add_point(label="right_hip", row=743.873254, col=1043.768812)
    template.add_point(label="left_knee", row=541.213412, col=1406.554361)
    template.add_point(label="right_knee", row=751.654723, col=1421.223332)
    template.add_point(label="left_ankle", row=501.234554, col=1760.561223)
    template.add_point(label="right_ankle", row=774.555666, col=1765.234432)
    # connections
    template.add_edge(src="left_ankle", dst="left_knee")
    template.add_edge(src="left_knee", dst="left_hip")
    template.add_edge(src="right_ankle", dst="right_knee")
    template.add_edge(src="right_knee", dst="right_hip")
    template.add_edge(src="left_hip", dst="right_hip")
    template.add_edge(src="left_shoulder", dst="left_hip")
    template.add_edge(src="right_shoulder", dst="right_hip")
    template.add_edge(src="left_shoulder", dst="right_shoulder")
    template.add_edge(src="left_shoulder", dst="left_elbow")
    template.add_edge(src="right_shoulder", dst="right_elbow")
    template.add_edge(src="left_elbow", dst="left_wrist")
    template.add_edge(src="right_elbow", dst="right_wrist")
    template.add_edge(src="left_eye", dst="right_eye")
    template.add_edge(src="nose", dst="left_eye")
    template.add_edge(src="nose", dst="right_eye")
    template.add_edge(src="left_eye", dst="left_ear")
    template.add_edge(src="right_eye", dst="right_ear")
    template.add_edge(src="left_ear", dst="left_shoulder")
    template.add_edge(src="right_ear", dst="right_shoulder")
    return template


def get_template(template: KeypointsTemplate):
    return template


def test_geometry_name(random_kp_int, random_kp_float):
    for kp_template in [random_kp_int, random_kp_float]:
        template = get_template(kp_template)
        assert template.geometry_name() == "graph"


def test_name(random_kp_int, random_kp_float):
    for kp_template in [random_kp_int, random_kp_float]:
        template = get_template(kp_template)


def test_to_json(random_kp_int, random_kp_float):
    for kp_template in [random_kp_int, random_kp_float]:
        template = get_template(kp_template)


def test_from_json(random_kp_int, random_kp_float):
    for kp_template in [random_kp_int, random_kp_float]:
        template = get_template(kp_template)


def test_crop(random_kp_int, random_kp_float):
    for kp_template in [random_kp_int, random_kp_float]:
        template = get_template(kp_template)


def test_relative_crop(random_kp_int, random_kp_float):
    for kp_template in [random_kp_int, random_kp_float]:
        template = get_template(kp_template)


def test_rotate(random_kp_int, random_kp_float):
    for kp_template in [random_kp_int, random_kp_float]:
        template = get_template(kp_template)


def test_resize(random_kp_int, random_kp_float):
    for kp_template in [random_kp_int, random_kp_float]:
        template = get_template(kp_template)


def test_scale(random_kp_int, random_kp_float):
    for kp_template in [random_kp_int, random_kp_float]:
        template = get_template(kp_template)


def test_translate(random_kp_int, random_kp_float):
    for kp_template in [random_kp_int, random_kp_float]:
        template = get_template(kp_template)


def test_fliplr(random_kp_int, random_kp_float):
    for kp_template in [random_kp_int, random_kp_float]:
        template = get_template(kp_template)


def test_flipud(random_kp_int, random_kp_float):
    for kp_template in [random_kp_int, random_kp_float]:
        template = get_template(kp_template)


def test__draw_bool_compatible(random_kp_int, random_kp_float):
    for kp_template in [random_kp_int, random_kp_float]:
        template = get_template(kp_template)


def test_draw(random_kp_int, random_kp_float):
    for kp_template in [random_kp_int, random_kp_float]:
        template = get_template(kp_template)


def test_get_mask(random_kp_int, random_kp_float):
    for kp_template in [random_kp_int, random_kp_float]:
        template = get_template(kp_template)


def test__draw_impl(random_kp_int, random_kp_float):
    for kp_template in [random_kp_int, random_kp_float]:
        template = get_template(kp_template)


def test_draw_contour(random_kp_int, random_kp_float):
    for kp_template in [random_kp_int, random_kp_float]:
        template = get_template(kp_template)


def test__draw_contour_impl(random_kp_int, random_kp_float):
    for kp_template in [random_kp_int, random_kp_float]:
        template = get_template(kp_template)


def test_area(random_kp_int, random_kp_float):
    for kp_template in [random_kp_int, random_kp_float]:
        template = get_template(kp_template)


def test_to_bbox(random_kp_int, random_kp_float):
    for kp_template in [random_kp_int, random_kp_float]:
        template = get_template(kp_template)


def test_clone(random_kp_int, random_kp_float):
    for kp_template in [random_kp_int, random_kp_float]:
        template = get_template(kp_template)


def test_validate(random_kp_int, random_kp_float):
    for kp_template in [random_kp_int, random_kp_float]:
        template = get_template(kp_template)


def test_config_from_json(random_kp_int, random_kp_float):
    for kp_template in [random_kp_int, random_kp_float]:
        template = get_template(kp_template)


def test_config_to_json(random_kp_int, random_kp_float):
    for kp_template in [random_kp_int, random_kp_float]:
        template = get_template(kp_template)


def test_allowed_transforms(random_kp_int, random_kp_float):
    for kp_template in [random_kp_int, random_kp_float]:
        template = get_template(kp_template)


def test_convert(random_kp_int, random_kp_float):
    for kp_template in [random_kp_int, random_kp_float]:
        template = get_template(kp_template)


if __name__ == "__main__":
    pytest.main([__file__])

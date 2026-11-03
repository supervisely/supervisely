# coding: utf-8
"""The Smart Tool init mask: the figure a session starts from, in any drawable shape."""

from types import SimpleNamespace

import numpy as np

import supervisely as sly
from supervisely.nn.inference import get_smart_tool_init_mask

IMAGE_SIZE = (20, 30)  # height, width


def _square(left, top, right, bottom):
    return [
        sly.PointLocation(row=top, col=left),
        sly.PointLocation(row=top, col=right),
        sly.PointLocation(row=bottom, col=right),
        sly.PointLocation(row=bottom, col=left),
    ]


def _request(geometry, **context):
    return {
        "mask": {"geometry_type": geometry.geometry_name(), "geometry": geometry.to_json()},
        **context,
    }


def test_polygon_figure_is_rasterized_in_place():
    polygon = sly.Polygon(exterior=_square(4, 2, 12, 10))

    mask = get_smart_tool_init_mask(_request(polygon), IMAGE_SIZE)

    assert mask.shape == IMAGE_SIZE
    assert mask[6, 8] == 255
    assert mask[6, 20] == 0
    assert mask[15, 8] == 0


def test_multipolygon_figure_keeps_its_parts_and_holes():
    multipolygon = sly.Multipolygon(
        parts=[
            sly.Polygon(exterior=_square(2, 2, 10, 10), interior=[_square(5, 5, 7, 7)]),
            sly.Polygon(exterior=_square(20, 12, 26, 18)),
        ]
    )

    mask = get_smart_tool_init_mask(_request(multipolygon), IMAGE_SIZE)

    assert mask[3, 3] == 255, "first part"
    assert mask[6, 6] == 0, "hole in the first part"
    assert mask[15, 23] == 255, "second part"


def test_the_figure_is_kept_for_the_rest_of_the_session():
    polygon = sly.Polygon(exterior=_square(4, 2, 12, 10))
    cache = {}

    # The Smart Tool sends the figure with the first click only; the clicks that follow
    # carry its id alone.
    first_click = get_smart_tool_init_mask(_request(polygon, figure_id=77), IMAGE_SIZE, cache=cache)
    next_click = get_smart_tool_init_mask({"figure_id": 77}, IMAGE_SIZE, cache=cache)

    assert np.array_equal(first_click, next_click)
    assert get_smart_tool_init_mask({"figure_id": 78}, IMAGE_SIZE, cache=cache) is None


def test_deprecated_figure_id_request_still_reads_the_annotation():
    bitmap = sly.Bitmap(data=np.ones((4, 5), bool), origin=sly.PointLocation(row=3, col=4))
    label = {"id": 55, **bitmap.to_json()}
    api = SimpleNamespace(
        annotation=SimpleNamespace(download_json=lambda image_id: {"objects": [label]})
    )

    mask = get_smart_tool_init_mask(
        {"figure_id": 55, "image_id": 9, "init_figure": True}, IMAGE_SIZE, api=api
    )

    assert mask[3:7, 4:9].all()
    assert mask.sum() == 255 * 4 * 5

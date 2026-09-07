"""Offline regressions for the Smart Tool initial-figure normalization path.

The shared image initial-mask path used to call ``sly.Bitmap.from_json`` on whatever label
the platform selected, so polygon/multipolygon figures (including the ones belonging to an
AnyShape class) could not be edited via Smart Tool. These tests pin the normalization
contract and the two image routes that consume it.

Everything here is CPU-only: the API and the predictor are mocked at their boundaries, no
instance is contacted and no model is loaded. Run as `python -m pytest tests/unit` from the
repo root (bare `pytest` may import an installed supervisely instead of this checkout).
"""

import threading
from types import SimpleNamespace

import numpy as np
import pytest
from cacheout import Cache
from cachetools import LRUCache
from fastapi import FastAPI

import supervisely as sly
from supervisely.nn.inference import Inference
from supervisely.nn.inference.interactive_segmentation import functional
from supervisely.nn.inference.interactive_segmentation import (
    interactive_segmentation as interactive_segmentation_module,
)
from supervisely.nn.inference.interactive_segmentation.interactive_segmentation import (
    InteractiveSegmentation,
)
from supervisely.nn.prediction_dto import PredictionSegmentation

IMG_H, IMG_W = 12, 12
FIGURE_ID = 777
IMAGE_ID = 42


# ------------------------------------------------------------------ label builders


def _ann_json(label, height=IMG_H, width=IMG_W):
    return {"size": {"height": height, "width": width}, "objects": [label], "tags": []}


def _label(geometry_json, geometry_type, figure_id=FIGURE_ID, class_title="smart"):
    return {
        "id": figure_id,
        "classTitle": class_title,
        "tags": [],
        "geometryType": geometry_type,
        "shape": geometry_type,
        **geometry_json,
    }


def _bitmap_label(mask, row=0, col=0, **kwargs):
    bitmap = sly.Bitmap(mask, origin=sly.PointLocation(row=row, col=col))
    geometry_json = {"bitmap": bitmap.to_json()["bitmap"]}
    return _label(geometry_json, "bitmap", **kwargs), bitmap


def _polygon_label(exterior, interior=(), **kwargs):
    """Builds a polygon label. Points are given as (x, y) pairs, as the platform sends them."""
    geometry_json = {
        "points": {
            "exterior": [list(point) for point in exterior],
            "interior": [[list(point) for point in contour] for contour in interior],
        }
    }
    return _label(geometry_json, "polygon", **kwargs)


def _multipolygon_label(parts, **kwargs):
    geometry_json = {
        "parts": [
            {
                "exterior": [list(point) for point in exterior],
                "interior": [[list(point) for point in contour] for contour in interior],
            }
            for exterior, interior in parts
        ]
    }
    return _label(geometry_json, "multipolygon", **kwargs)


def _rect_points(x1, y1, x2, y2):
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


def _full_mask(bitmap, height=IMG_H, width=IMG_W):
    """Boolean full-image mask of a Bitmap, built independently of the code under test."""
    mask = np.zeros((height, width), bool)
    top, left = bitmap.origin.row, bitmap.origin.col
    data = bitmap.data
    mask[top : top + data.shape[0], left : left + data.shape[1]] = data
    return mask


# ------------------------------------------------------------------ api / predictor mocks


class _FakeAnnotationApi:
    def __init__(self, ann_json):
        self._ann_json = ann_json
        self.downloads = []

    def download_json(self, image_id):
        self.downloads.append(image_id)
        return self._ann_json


class _FakeImageApi:
    def __init__(self, image_np):
        self._image_np = image_np

    def get_info_by_id(self, image_id):
        return SimpleNamespace(
            id=image_id, height=self._image_np.shape[0], width=self._image_np.shape[1]
        )

    def download_np(self, image_id):
        return self._image_np.copy()


class _FakeApi:
    def __init__(self, ann_json, image_np):
        self.annotation = _FakeAnnotationApi(ann_json)
        self.image = _FakeImageApi(image_np)


class _FakeFramesCache:
    """Stands in for :class:`~supervisely.nn.inference.cache.InferenceImageCache`."""

    def __init__(self, image_np):
        self._image_np = image_np
        self.endpoints_added = 0

    def add_cache_endpoint(self, server):
        self.endpoints_added += 1

    def download_image(self, api, image_id, **kwargs):
        return self._image_np.copy()

    def download_frame(self, api, video_id, frame_index):
        raise AssertionError("Video frames are out of scope of the image init_figure path")

    def download_image_by_hash(self, api, image_hash):
        raise AssertionError("Image hashes are not used by these tests")


class _StubSegmentation(InteractiveSegmentation):
    """InteractiveSegmentation with the heavy Inference machinery replaced.

    Only the attributes the Smart Tool routes actually touch are set up, so the production
    route bodies run unchanged against mocked API/predictor boundaries.
    """

    def __init__(self, image_np, pred_mask):
        self._pred_mask = pred_mask
        self.predict_calls = []
        self._inference_image_lock = threading.Lock()
        self._inference_image_cache = Cache(ttl=60)
        self._init_mask_cache = LRUCache(maxsize=10)
        self.cache = _FakeFramesCache(image_np)
        server = FastAPI()
        self._server = server
        self._app = SimpleNamespace(get_server=lambda: server)

    def _get_inference_settings(self, state):
        return dict(state.get("settings", {}))

    def predict(self, image_path, clicks, settings):
        init_mask = settings.get("init_mask")
        self.predict_calls.append(
            {
                "clicks": [(click.x, click.y, click.is_positive) for click in clicks],
                "init_mask": None if init_mask is None else init_mask.copy(),
            }
        )
        return PredictionSegmentation(self._pred_mask)


@pytest.fixture
def image_np():
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, size=(IMG_H, IMG_W, 3), dtype=np.uint8)


@pytest.fixture
def pred_mask():
    mask = np.zeros((10, 10), bool)
    mask[2:5, 3:6] = True
    return mask


@pytest.fixture
def serve_routes(monkeypatch, tmp_path):
    """Registers the real Smart Tool routes and returns them by path."""

    def _serve(model: _StubSegmentation):
        monkeypatch.setattr(Inference, "serve", lambda self: None)
        monkeypatch.setattr(interactive_segmentation_module, "get_data_dir", lambda: str(tmp_path))
        model.serve()
        return {
            route.path: route.endpoint
            for route in model._server.routes
            if getattr(route, "endpoint", None) is not None
        }

    return _serve


def _context(**overrides):
    context = {
        "image_id": IMAGE_ID,
        "figure_id": FIGURE_ID,
        "crop": [{"x": 1, "y": 1}, {"x": 10, "y": 10}],
        "positive": [{"x": 3, "y": 3}],
        "negative": [],
        "request_uid": "uid-1",
    }
    context.update(overrides)
    return context


def _request(context, api):
    return SimpleNamespace(
        state=SimpleNamespace(context=context, state={"settings": {}}, api=api)
    )


# ------------------------------------------------------------------ decoding / rasterization


def test_download_init_mask_keeps_bitmap_with_nonzero_origin():
    data = np.array([[1, 0, 1], [1, 1, 1]], bool)
    label, bitmap = _bitmap_label(data, row=4, col=5)
    api = _FakeApi(_ann_json(label), np.zeros((IMG_H, IMG_W, 3), np.uint8))

    result = functional.download_init_mask(api, FIGURE_ID, IMAGE_ID)

    assert isinstance(result, sly.Bitmap)
    assert (result.origin.row, result.origin.col) == (4, 5)
    np.testing.assert_array_equal(result.data, bitmap.data)


def test_download_init_mask_rasterizes_polygon_with_a_hole():
    label = _polygon_label(_rect_points(2, 2, 8, 8), [_rect_points(4, 4, 6, 6)])
    api = _FakeApi(_ann_json(label), np.zeros((IMG_H, IMG_W, 3), np.uint8))

    result = functional.download_init_mask(api, FIGURE_ID, IMAGE_ID)

    assert (result.origin.row, result.origin.col) == (2, 2)
    expected = np.zeros((IMG_H, IMG_W), bool)
    expected[2:9, 2:9] = True
    expected[4:7, 4:7] = False
    np.testing.assert_array_equal(_full_mask(result), expected)


def test_polygon_without_interior_field_is_accepted():
    label = _polygon_label(_rect_points(1, 1, 4, 4))
    del label["points"]["interior"]

    result = functional.label_to_init_bitmap(label, (IMG_H, IMG_W))

    expected = np.zeros((IMG_H, IMG_W), bool)
    expected[1:5, 1:5] = True
    np.testing.assert_array_equal(_full_mask(result), expected)


def test_multipolygon_unions_disconnected_parts_and_keeps_holes():
    label = _multipolygon_label(
        [
            (_rect_points(1, 1, 4, 4), [_rect_points(2, 2, 3, 3)]),
            (_rect_points(7, 7, 10, 10), []),
        ]
    )

    result = functional.label_to_init_bitmap(label, (IMG_H, IMG_W))

    expected = np.zeros((IMG_H, IMG_W), bool)
    expected[1:5, 1:5] = True
    expected[2:4, 2:4] = False
    expected[7:11, 7:11] = True
    np.testing.assert_array_equal(_full_mask(result), expected)
    # The bitmap spans both parts.
    assert (result.origin.row, result.origin.col) == (1, 1)
    assert result.data.shape == (10, 10)


def test_hole_of_one_part_does_not_erase_an_overlapping_part():
    """A hole is cut per part; the union with the other parts must survive it."""
    label = _multipolygon_label(
        [
            (_rect_points(1, 1, 9, 9), [_rect_points(3, 3, 7, 7)]),
            (_rect_points(4, 4, 6, 6), []),
        ]
    )

    result = functional.label_to_init_bitmap(label, (IMG_H, IMG_W))

    expected = np.zeros((IMG_H, IMG_W), bool)
    expected[1:10, 1:10] = True
    expected[3:8, 3:8] = False
    expected[4:7, 4:7] = True  # the second part fills the hole back
    np.testing.assert_array_equal(_full_mask(result), expected)


def test_polygon_at_the_image_edge_has_zero_origin():
    label = _polygon_label(_rect_points(0, 0, 3, 3))

    result = functional.label_to_init_bitmap(label, (IMG_H, IMG_W))

    assert (result.origin.row, result.origin.col) == (0, 0)
    assert result.data.shape == (4, 4)
    assert result.data.all()


def test_polygon_is_clipped_to_the_image_bounds():
    label = _polygon_label(_rect_points(-5, 8, 5, 40))

    result = functional.label_to_init_bitmap(label, (IMG_H, IMG_W))

    expected = np.zeros((IMG_H, IMG_W), bool)
    expected[8:IMG_H, 0:6] = True
    np.testing.assert_array_equal(_full_mask(result), expected)
    assert (result.origin.row, result.origin.col) == (8, 0)


def test_bitmap_is_clipped_to_the_image_bounds():
    data = np.ones((6, 6), bool)
    label, _ = _bitmap_label(data, row=IMG_H - 2, col=IMG_W - 3)

    result = functional.label_to_init_bitmap(label, (IMG_H, IMG_W))

    assert (result.origin.row, result.origin.col) == (IMG_H - 2, IMG_W - 3)
    assert result.data.shape == (2, 3)
    assert result.data.all()


def test_figure_completely_outside_of_the_image_is_reported():
    data = np.ones((2, 2), bool)
    label, _ = _bitmap_label(data, row=IMG_H + 5, col=0)

    with pytest.raises(functional.InitMaskError, match="outside of the image"):
        functional.label_to_init_bitmap(label, (IMG_H, IMG_W))


@pytest.mark.parametrize("geometry_type", ["bitmap", "polygon", "multipolygon"])
def test_any_shape_labels_are_dispatched_by_their_concrete_type(geometry_type):
    """AnyShape is a class, the label still stores its concrete geometry type."""
    if geometry_type == "bitmap":
        label, _ = _bitmap_label(np.ones((3, 3), bool), row=2, col=2, class_title="any_shape")
    elif geometry_type == "polygon":
        label = _polygon_label(_rect_points(2, 2, 4, 4), class_title="any_shape")
    else:
        label = _multipolygon_label([(_rect_points(2, 2, 4, 4), [])], class_title="any_shape")

    result = functional.label_to_init_bitmap(label, (IMG_H, IMG_W))

    expected = np.zeros((IMG_H, IMG_W), bool)
    expected[2:5, 2:5] = True
    np.testing.assert_array_equal(_full_mask(result), expected)


def test_missing_figure_id_is_reported():
    label = _polygon_label(_rect_points(1, 1, 4, 4))
    api = _FakeApi(_ann_json(label), np.zeros((IMG_H, IMG_W, 3), np.uint8))

    with pytest.raises(functional.InitMaskError, match="not provided"):
        functional.download_init_mask(api, None, IMAGE_ID)
    assert api.annotation.downloads == []


def test_unknown_figure_id_is_reported():
    label = _polygon_label(_rect_points(1, 1, 4, 4))
    api = _FakeApi(_ann_json(label), np.zeros((IMG_H, IMG_W, 3), np.uint8))

    with pytest.raises(functional.InitMaskError, match="not found"):
        functional.download_init_mask(api, FIGURE_ID + 1, IMAGE_ID)


def test_unsupported_geometry_is_reported_and_not_parsed_as_bitmap():
    label = _label({"points": {"exterior": [[1, 1], [5, 5]], "interior": []}}, "rectangle")

    with pytest.raises(functional.InitMaskError, match="rectangle"):
        functional.label_to_init_bitmap(label, (IMG_H, IMG_W))


@pytest.mark.parametrize(
    "label",
    [
        _polygon_label([(1, 1), (4, 4)]),
        _polygon_label(_rect_points(1, 1, 6, 6), [[(2, 2), (3, 3)]]),
        _label({"points": {"exterior": "nope", "interior": []}}, "polygon"),
        _label({}, "polygon"),
        _label({"parts": [{"interior": []}]}, "multipolygon"),
        _label({"parts": "nope"}, "multipolygon"),
        _label({"bitmap": {"origin": [0, 0]}}, "bitmap"),
        _label({"bitmap": {"origin": [0, 0], "data": "not-base64"}}, "bitmap"),
    ],
    ids=[
        "short-exterior",
        "short-interior",
        "bad-exterior",
        "no-points",
        "no-exterior",
        "bad-parts",
        "no-data",
        "bad-data",
    ],
)
def test_malformed_geometry_is_reported(label):
    with pytest.raises(functional.InitMaskError):
        functional.label_to_init_bitmap(label, (IMG_H, IMG_W))


def test_geometry_type_is_required_when_it_can_not_be_inferred():
    label = {"id": FIGURE_ID, "points": {"exterior": [[1, 1], [4, 1], [4, 4]], "interior": []}}

    with pytest.raises(functional.InitMaskError, match="missing"):
        functional.label_to_init_bitmap(label, (IMG_H, IMG_W))


def test_legacy_label_without_geometry_type_falls_back_to_bitmap():
    label, bitmap = _bitmap_label(np.ones((2, 2), bool), row=1, col=1)
    del label["geometryType"]
    del label["shape"]

    result = functional.label_to_init_bitmap(label, (IMG_H, IMG_W))

    np.testing.assert_array_equal(result.data, bitmap.data)
    assert (result.origin.row, result.origin.col) == (1, 1)


def test_polygon_is_rasterized_without_a_known_image_size():
    """Annotations without a "size" field must still yield a correctly positioned bitmap."""
    label = _polygon_label(_rect_points(3, 5, 6, 9))
    api = _FakeApi({"objects": [label]}, np.zeros((IMG_H, IMG_W, 3), np.uint8))

    result = functional.download_init_mask(api, FIGURE_ID, IMAGE_ID)

    assert (result.origin.row, result.origin.col) == (5, 3)
    assert result.data.shape == (5, 4)
    assert result.data.all()


# ------------------------------------------------------------------ bitmap_to_mask


def test_bitmap_to_mask_places_the_mask_on_a_full_image_canvas():
    bitmap = sly.Bitmap(np.ones((2, 3), bool), origin=sly.PointLocation(row=4, col=5))

    mask = functional.bitmap_to_mask(bitmap, IMG_H, IMG_W)

    assert mask.shape == (IMG_H, IMG_W)
    assert mask.dtype == np.uint8
    assert set(np.unique(mask)) == {0, 255}
    expected = np.zeros((IMG_H, IMG_W), np.uint8)
    expected[4:6, 5:8] = 255
    np.testing.assert_array_equal(mask, expected)


def test_bitmap_to_mask_clips_a_bitmap_hanging_over_the_image_border():
    bitmap = sly.Bitmap(np.ones((5, 5), bool), origin=sly.PointLocation(row=IMG_H - 2, col=-2))

    mask = functional.bitmap_to_mask(bitmap, IMG_H, IMG_W)

    expected = np.zeros((IMG_H, IMG_W), np.uint8)
    expected[IMG_H - 2 :, 0:3] = 255
    np.testing.assert_array_equal(mask, expected)


def test_bitmap_to_mask_of_a_fully_outside_bitmap_is_empty():
    bitmap = sly.Bitmap(np.ones((2, 2), bool), origin=sly.PointLocation(row=IMG_H + 3, col=0))

    mask = functional.bitmap_to_mask(bitmap, IMG_H, IMG_W)

    assert not mask.any()


# ------------------------------------------------------------------ smart tool routes


def _expected_cropped_init_mask(full_mask, crop):
    x1, y1 = crop[0]["x"], crop[0]["y"]
    x2, y2 = crop[1]["x"], crop[1]["y"]
    return (full_mask[y1 : y2 + 1, x1 : x2 + 1] * 255).astype(np.uint8)


def test_smart_segmentation_feeds_a_rasterized_polygon_to_the_predictor(
    image_np, pred_mask, serve_routes
):
    label = _polygon_label(_rect_points(2, 2, 8, 8), [_rect_points(4, 4, 6, 6)])
    api = _FakeApi(_ann_json(label), image_np)
    model = _StubSegmentation(image_np, pred_mask)
    route = serve_routes(model)["/smart_segmentation"]
    context = _context(init_figure=True)

    result = route(response=SimpleNamespace(status_code=200), request=_request(context, api))

    assert result["success"] is True and result["error"] is None
    # Predicted bitmap is reported in image coordinates (crop origin + local origin).
    assert result["origin"] == {"x": 1 + 3, "y": 1 + 2}
    assert result["bitmap"] is not None

    assert len(model.predict_calls) == 1
    init_mask = model.predict_calls[0]["init_mask"]
    expected_full = np.zeros((IMG_H, IMG_W), bool)
    expected_full[2:9, 2:9] = True
    expected_full[4:7, 4:7] = False
    np.testing.assert_array_equal(
        init_mask, _expected_cropped_init_mask(expected_full, context["crop"])
    )
    assert init_mask.shape == (10, 10)
    assert model.predict_calls[0]["clicks"] == [(2, 2, True)]


def test_smart_segmentation_reuses_the_cached_init_mask_for_the_next_clicks(
    image_np, pred_mask, serve_routes
):
    label = _multipolygon_label([(_rect_points(2, 2, 5, 5), []), (_rect_points(7, 7, 9, 9), [])])
    api = _FakeApi(_ann_json(label), image_np)
    model = _StubSegmentation(image_np, pred_mask)
    route = serve_routes(model)["/smart_segmentation"]

    first = route(
        response=SimpleNamespace(status_code=200),
        request=_request(_context(init_figure=True), api),
    )
    # A follow-up click does not carry init_figure any more.
    second = route(
        response=SimpleNamespace(status_code=200),
        request=_request(_context(request_uid="uid-2"), api),
    )

    assert first["success"] is True and second["success"] is True
    assert api.annotation.downloads == [IMAGE_ID], "init figure must be downloaded once"
    assert list(model._init_mask_cache.keys()) == [FIGURE_ID]
    np.testing.assert_array_equal(
        model.predict_calls[0]["init_mask"], model.predict_calls[1]["init_mask"]
    )
    assert model.predict_calls[1]["init_mask"].any()


def test_smart_segmentation_without_an_init_figure_passes_no_init_mask(
    image_np, pred_mask, serve_routes
):
    label = _polygon_label(_rect_points(2, 2, 8, 8))
    api = _FakeApi(_ann_json(label), image_np)
    model = _StubSegmentation(image_np, pred_mask)
    route = serve_routes(model)["/smart_segmentation"]

    result = route(
        response=SimpleNamespace(status_code=200),
        request=_request(_context(figure_id=None), api),
    )

    assert result["success"] is True
    assert api.annotation.downloads == []
    assert model.predict_calls[0]["init_mask"] is None


def test_smart_segmentation_reports_an_unsupported_init_figure(image_np, pred_mask, serve_routes):
    label = _label({"points": {"exterior": [[1, 1], [5, 5]], "interior": []}}, "rectangle")
    api = _FakeApi(_ann_json(label), image_np)
    model = _StubSegmentation(image_np, pred_mask)
    route = serve_routes(model)["/smart_segmentation"]

    result = route(
        response=SimpleNamespace(status_code=200),
        request=_request(_context(init_figure=True), api),
    )

    assert result["success"] is False
    assert "rectangle" in result["error"]
    assert result["origin"] is None and result["bitmap"] is None
    assert model.predict_calls == []


def test_smart_segmentation_without_clicks_returns_the_no_result_response(
    image_np, pred_mask, serve_routes
):
    label = _polygon_label(_rect_points(2, 2, 8, 8))
    api = _FakeApi(_ann_json(label), image_np)
    model = _StubSegmentation(image_np, pred_mask)
    route = serve_routes(model)["/smart_segmentation"]

    result = route(
        response=SimpleNamespace(status_code=200),
        request=_request(_context(init_figure=True, positive=[], negative=[]), api),
    )

    assert result == {"origin": None, "bitmap": None, "success": True, "error": None}
    assert model.predict_calls == []


def test_smart_segmentation_returns_the_no_result_response_for_an_empty_prediction(
    image_np, serve_routes
):
    label = _polygon_label(_rect_points(2, 2, 8, 8))
    api = _FakeApi(_ann_json(label), image_np)
    model = _StubSegmentation(image_np, np.zeros((10, 10), bool))
    route = serve_routes(model)["/smart_segmentation"]

    result = route(
        response=SimpleNamespace(status_code=200),
        request=_request(_context(init_figure=True), api),
    )

    assert result == {"origin": None, "bitmap": None, "success": True, "error": None}
    assert len(model.predict_calls) == 1


def test_smart_segmentation_batch_normalizes_every_state(image_np, pred_mask, serve_routes):
    label = _polygon_label(_rect_points(2, 2, 8, 8), [_rect_points(4, 4, 6, 6)])
    api = _FakeApi(_ann_json(label), image_np)
    model = _StubSegmentation(image_np, pred_mask)
    route = serve_routes(model)["/smart_segmentation_batch"]
    states = [
        _context(init_figure=True),
        _context(request_uid="uid-2"),  # continuation of the same figure, served from cache
    ]
    request = SimpleNamespace(
        state=SimpleNamespace(context={"states": states}, state={"settings": {}}, api=api)
    )

    results = route(response=SimpleNamespace(status_code=200), request=request)

    assert len(results) == 2
    assert all(item["success"] is True and item["error"] is None for item in results)
    assert all(item["origin"] == {"x": 4, "y": 3} for item in results)
    assert api.annotation.downloads == [IMAGE_ID]

    expected_full = np.zeros((IMG_H, IMG_W), bool)
    expected_full[2:9, 2:9] = True
    expected_full[4:7, 4:7] = False
    expected = _expected_cropped_init_mask(expected_full, states[0]["crop"])
    for call in model.predict_calls:
        np.testing.assert_array_equal(call["init_mask"], expected)


def test_smart_segmentation_batch_reports_a_broken_state_without_failing_the_rest(
    image_np, pred_mask, serve_routes
):
    good = _polygon_label(_rect_points(2, 2, 8, 8), figure_id=FIGURE_ID)
    broken = _label({"points": {"exterior": [[1, 1], [5, 5]], "interior": []}}, "rectangle")
    broken["id"] = FIGURE_ID + 1
    ann_json = _ann_json(good)
    ann_json["objects"].append(broken)
    api = _FakeApi(ann_json, image_np)
    model = _StubSegmentation(image_np, pred_mask)
    route = serve_routes(model)["/smart_segmentation_batch"]
    states = [
        _context(init_figure=True, figure_id=FIGURE_ID + 1, request_uid="uid-broken"),
        _context(init_figure=True),
    ]
    request = SimpleNamespace(
        state=SimpleNamespace(context={"states": states}, state={"settings": {}}, api=api)
    )

    results = route(response=SimpleNamespace(status_code=200), request=request)

    assert results[0]["success"] is False and "rectangle" in results[0]["error"]
    assert results[1]["success"] is True and results[1]["bitmap"] is not None
    assert len(model.predict_calls) == 1
    assert model.predict_calls[0]["init_mask"].any()

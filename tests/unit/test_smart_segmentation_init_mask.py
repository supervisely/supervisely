"""Init mask of the smart tool endpoints: taken from the request, not from the figure id."""

import base64
import threading
from types import SimpleNamespace

import numpy as np
import pytest
from cacheout import Cache
from cachetools import LRUCache
from fastapi import Response

import supervisely as sly
from supervisely.nn.inference import InteractiveSegmentation
from supervisely.nn.inference.inference import Inference
from supervisely.nn.inference.interactive_segmentation import functional
from supervisely.nn.inference.interactive_segmentation import (
    interactive_segmentation as module,
)
from supervisely.nn.prediction_dto import PredictionSegmentation

IMAGE_ID = 777
FIGURE_ID = 555
IMAGE_H, IMAGE_W = 100, 120
CROP = [{"x": 10, "y": 20}, {"x": 59, "y": 69}]


def init_bitmap() -> sly.Bitmap:
    """Bitmap of the figure being edited, overlapping the crop region."""
    data = np.zeros((30, 40), bool)
    data[5:25, 5:35] = True
    return sly.Bitmap(data=data, origin=sly.PointLocation(row=15, col=25))


def context_mask(bitmap: sly.Bitmap) -> dict:
    """`mask` context field as the platform sends it: base64 data and an {x, y} origin."""
    return {
        "data": sly.Bitmap.data_2_base64(bitmap.data),
        "origin": {"x": bitmap.origin.col, "y": bitmap.origin.row},
    }


def empty_context_mask() -> dict:
    """`mask` field of a figure whose raster ended up with no set pixels."""
    return {
        "data": sly.Bitmap.data_2_base64(np.zeros((30, 40), bool)),
        "origin": {"x": 25, "y": 15},
    }


def make_context(**overrides) -> dict:
    context = {
        "image_id": IMAGE_ID,
        "crop": CROP,
        "positive": [{"x": 30, "y": 40}],
        "negative": [],
        "request_uid": "uid-1",
    }
    context.update(overrides)
    return context


class FakeApi:
    """API that records every call and only serves the annotation it was given."""

    def __init__(self, ann_json=None):
        self.calls = []
        self._ann_json = ann_json
        self.annotation = SimpleNamespace(download_json=self._download_json)
        self.image = SimpleNamespace(get_info_by_id=self._get_info_by_id)

    def _download_json(self, image_id, *args, **kwargs):
        self.calls.append(("annotation.download_json", image_id))
        if self._ann_json is None:
            raise AssertionError("annotation download is not expected here")
        return self._ann_json

    def _get_info_by_id(self, image_id, *args, **kwargs):
        self.calls.append(("image.get_info_by_id", image_id))
        return SimpleNamespace(id=image_id, height=IMAGE_H, width=IMAGE_W)


class Service:
    """Smart tool endpoints of a served model, with the model itself stubbed out."""

    def __init__(self, routes):
        self._routes = routes
        self.init_masks = []

    def call(self, path, context, api):
        request = SimpleNamespace(
            state=SimpleNamespace(state={}, context=context, api=api),
        )
        response = Response()
        body = self._routes[path](response, request)
        return response, body


@pytest.fixture
def service(tmp_path, monkeypatch):
    monkeypatch.setattr(Inference, "serve", lambda self: None)
    monkeypatch.setattr(module, "get_data_dir", lambda: str(tmp_path))

    routes = {}

    class FakeServer:
        def post(self, path):
            def register(func):
                routes[path] = func
                return func

            return register

    served = Service(routes)
    model = InteractiveSegmentation.__new__(InteractiveSegmentation)
    model._app = SimpleNamespace(get_server=lambda: FakeServer())
    model.cache = SimpleNamespace(
        add_cache_endpoint=lambda server: None,
        download_image=None,
        download_frame=None,
        download_image_by_hash=None,
    )
    model._inference_image_lock = threading.Lock()
    model._inference_image_cache = Cache(ttl=60)
    model._init_mask_cache = LRUCache(maxsize=100)
    model._get_inference_settings = lambda state: {}

    def predict(image_path, clicks, settings):
        served.init_masks.append(settings["init_mask"])
        mask = np.zeros(sly.image.read(image_path).shape[:2], bool)
        mask[1:5, 1:5] = True
        return PredictionSegmentation(mask)

    model.predict = predict
    # The image is already cached, so a passing request needs no image download.
    model._inference_image_cache.set(str(IMAGE_ID), np.zeros((IMAGE_H, IMAGE_W, 3), np.uint8))
    model.serve()
    return served


def test_context_mask_matches_figure_id_path_without_any_api_call(service):
    bitmap = init_bitmap()
    legacy_api = FakeApi({"objects": [{"id": FIGURE_ID, **bitmap.to_json()}]})
    _, legacy_body = service.call(
        "/smart_segmentation",
        make_context(figure_id=FIGURE_ID, init_figure=True),
        legacy_api,
    )
    mask_api = FakeApi()
    response, body = service.call(
        "/smart_segmentation",
        make_context(mask=context_mask(bitmap)),
        mask_api,
    )

    assert legacy_body["success"] is True
    assert body["success"] is True
    assert response.status_code == 200
    assert mask_api.calls == []

    legacy_mask, from_context = service.init_masks
    assert from_context.shape == legacy_mask.shape
    assert from_context.dtype == legacy_mask.dtype
    assert np.array_equal(from_context, legacy_mask)
    assert set(np.unique(from_context)) == {0, 255}
    # Aligned with the cropped inference image, not with the full image.
    assert from_context.shape == (
        CROP[1]["y"] - CROP[0]["y"] + 1,
        CROP[1]["x"] - CROP[0]["x"] + 1,
    )
    assert np.array_equal(
        from_context,
        functional.crop_image(CROP, functional.bitmap_to_mask(bitmap, IMAGE_H, IMAGE_W)),
    )


def test_polygon_figure_is_served_from_the_context_mask(service):
    polygon = sly.Polygon(
        exterior=[
            sly.PointLocation(20, 30),
            sly.PointLocation(20, 60),
            sly.PointLocation(45, 60),
            sly.PointLocation(45, 30),
        ]
    )
    polygon_mask = sly.Bitmap(polygon.get_mask((IMAGE_H, IMAGE_W)))
    # The annotation holds a polygon, so the deprecated path could not rebuild it.
    api = FakeApi({"objects": [{"id": FIGURE_ID, **polygon.to_json()}]})

    response, body = service.call(
        "/smart_segmentation",
        make_context(figure_id=FIGURE_ID, init_figure=True, mask=context_mask(polygon_mask)),
        api,
    )

    assert body["success"] is True
    assert response.status_code == 200
    assert api.calls == []
    assert np.array_equal(
        service.init_masks[-1],
        functional.crop_image(CROP, functional.bitmap_to_mask(polygon_mask, IMAGE_H, IMAGE_W)),
    )


def test_undecodable_context_mask_is_a_bad_request(service):
    api = FakeApi({"objects": [{"id": FIGURE_ID, **init_bitmap().to_json()}]})
    broken = {
        "data": base64.b64encode(b"definitely not an encoded mask").decode(),
        "origin": {"x": 25, "y": 15},
    }

    response, body = service.call(
        "/smart_segmentation",
        make_context(figure_id=FIGURE_ID, init_figure=True, mask=broken),
        api,
    )

    assert response.status_code == 400
    assert body["success"] is False
    assert api.calls == []
    assert service.init_masks == []


def test_legacy_figure_id_path_is_unchanged_without_a_context_mask(service):
    bitmap = init_bitmap()
    api = FakeApi({"objects": [{"id": FIGURE_ID, **bitmap.to_json()}]})

    _, body = service.call(
        "/smart_segmentation",
        make_context(figure_id=FIGURE_ID, init_figure=True),
        api,
    )

    assert body["success"] is True
    assert api.calls == [
        ("annotation.download_json", IMAGE_ID),
        ("image.get_info_by_id", IMAGE_ID),
    ]
    assert np.array_equal(
        service.init_masks[-1],
        functional.crop_image(CROP, functional.bitmap_to_mask(bitmap, IMAGE_H, IMAGE_W)),
    )


def test_batch_endpoint_honours_the_context_mask(service):
    bitmap = init_bitmap()
    api = FakeApi()

    response, body = service.call(
        "/smart_segmentation_batch",
        {"states": [make_context(mask=context_mask(bitmap))]},
        api,
    )

    assert response.status_code == 200
    assert [item["success"] for item in body] == [True]
    assert api.calls == []
    assert np.array_equal(
        service.init_masks[-1],
        functional.crop_image(CROP, functional.bitmap_to_mask(bitmap, IMAGE_H, IMAGE_W)),
    )


def test_batch_endpoint_rejects_an_undecodable_context_mask(service):
    api = FakeApi({"objects": [{"id": FIGURE_ID, **init_bitmap().to_json()}]})
    broken = {"data": "@@ not base64 @@", "origin": {"x": 25, "y": 15}}

    response, body = service.call(
        "/smart_segmentation_batch",
        {"states": [make_context(figure_id=FIGURE_ID, init_figure=True, mask=broken)]},
        api,
    )

    assert response.status_code == 400
    assert body["success"] is False
    assert api.calls == []
    assert service.init_masks == []


def test_a_session_keeps_the_init_mask_after_the_first_click(service):
    # The platform gates `mask` and `init_figure` on sendGeometryToSmartAnnotation and
    # clears that flag right after the first request, so every later click of the same
    # session arrives with figure_id alone.
    bitmap = init_bitmap()
    api = FakeApi()

    _, first_body = service.call(
        "/smart_segmentation",
        make_context(figure_id=FIGURE_ID, init_figure=True, mask=context_mask(bitmap)),
        api,
    )
    _, second_body = service.call(
        "/smart_segmentation",
        make_context(figure_id=FIGURE_ID, request_uid="uid-2"),
        api,
    )

    assert first_body["success"] is True
    assert second_body["success"] is True
    assert ("annotation.download_json", IMAGE_ID) not in api.calls
    first_mask, second_mask = service.init_masks
    assert first_mask is not None
    assert second_mask is not None
    assert np.array_equal(second_mask, first_mask)


def test_an_all_zero_context_mask_is_served_without_an_init_mask(service):
    api = FakeApi()

    response, body = service.call(
        "/smart_segmentation",
        make_context(figure_id=FIGURE_ID, init_figure=True, mask=empty_context_mask()),
        api,
    )
    batch_response, batch_body = service.call(
        "/smart_segmentation_batch",
        {
            "states": [
                make_context(figure_id=FIGURE_ID, init_figure=True, mask=empty_context_mask())
            ]
        },
        api,
    )

    assert response.status_code == 200
    assert body["success"] is True
    assert batch_response.status_code == 200
    assert [item["success"] for item in batch_body] == [True]
    assert api.calls == []
    assert service.init_masks == [None, None]


def test_the_decode_helper_is_exported_from_supervisely_nn_inference():
    from supervisely.nn.inference import get_init_mask_from_context

    assert get_init_mask_from_context is functional.get_init_mask_from_context

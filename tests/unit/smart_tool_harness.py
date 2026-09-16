"""Offline harness shared by the Smart Tool initial-mask regressions and the reproducer.

Everything here is CPU-only: the Supervisely API and the predictor are replaced by stubs at
their boundaries, so the production Smart Tool route bodies run unchanged without an
instance, a model or a GPU.
"""

import contextlib
import threading
from types import SimpleNamespace
from unittest import mock

import numpy as np
from cacheout import Cache
from cachetools import LRUCache
from fastapi import FastAPI

import supervisely as sly
from supervisely.nn.inference import Inference
from supervisely.nn.inference.interactive_segmentation import (
    interactive_segmentation as interactive_segmentation_module,
)
from supervisely.nn.inference.interactive_segmentation.interactive_segmentation import (
    InteractiveSegmentation,
)
from supervisely.nn.prediction_dto import PredictionSegmentation

IMG_H, IMG_W = 12, 12
IMAGE_ID = 42
FIGURE_ID = 777
LOCAL_FIGURE_ID = "local-figure-1"
CROP = [{"x": 1, "y": 1}, {"x": 10, "y": 10}]


class AnnotationDownloadForbidden(AssertionError):
    """Raised by the stub API when a request downloads an annotation it must not need."""


def encode_mask(mask) -> str:
    """Encodes a boolean mask exactly like the "bitmap" field of the Supervisely format."""
    return sly.Bitmap.data_2_base64(np.asarray(mask, bool))


def holed_mask():
    """Tight raster with a hole and a disconnected component, as a rasterized polygon has."""
    mask = np.zeros((5, 6), bool)
    mask[0:3, 0:5] = True
    mask[1, 1:3] = False  # hole
    mask[4, 4:6] = True  # disconnected part
    return mask

def mask_payload(mask, x: int, y: int) -> dict:
    """Builds the contract mask payload: a tight bitmap placed at (x, y) in image coordinates."""
    return {"origin": [x, y], "data": encode_mask(mask)}


def place_mask(mask, x: int, y: int, height: int = IMG_H, width: int = IMG_W) -> np.ndarray:
    """Expected full-image mask, built independently of the code under test (clips to bounds)."""
    canvas = np.zeros((height, width), bool)
    data = np.asarray(mask, bool)
    top, left = max(y, 0), max(x, 0)
    bottom, right = min(y + data.shape[0], height), min(x + data.shape[1], width)
    if bottom > top and right > left:
        canvas[top:bottom, left:right] = data[top - y : bottom - y, left - x : right - x]
    return canvas


def crop_full_mask(full_mask: np.ndarray, crop=CROP) -> np.ndarray:
    """Expected predictor input: the full-image mask cropped like the request image."""
    x1, y1 = crop[0]["x"], crop[0]["y"]
    x2, y2 = crop[1]["x"], crop[1]["y"]
    return (full_mask[y1 : y2 + 1, x1 : x2 + 1] * 255).astype(np.uint8)


def bitmap_label(mask, x: int, y: int, figure_id: int = FIGURE_ID) -> dict:
    """Saved bitmap label, as the legacy figure-id download path receives it."""
    bitmap = sly.Bitmap(np.asarray(mask, bool), origin=sly.PointLocation(row=y, col=x))
    return {
        "id": figure_id,
        "classTitle": "smart",
        "tags": [],
        "geometryType": "bitmap",
        "shape": "bitmap",
        "bitmap": bitmap.to_json()["bitmap"],
    }


def ann_json(*labels, height: int = IMG_H, width: int = IMG_W) -> dict:
    return {"size": {"height": height, "width": width}, "objects": list(labels), "tags": []}


class _FakeAnnotationApi:
    def __init__(self, annotation=None, forbidden=False):
        self._annotation = annotation
        self._forbidden = forbidden
        self.downloads = []

    def set_annotation(self, annotation):
        """Replays a figure that changed on the instance between two requests."""
        self._annotation = annotation

    def download_json(self, image_id):
        self.downloads.append(image_id)
        if self._forbidden or self._annotation is None:
            raise AnnotationDownloadForbidden(
                f"Annotation of image {image_id} must not be downloaded for this request"
            )
        return self._annotation


class _FakeImageApi:
    def __init__(self, image_np):
        self._image_np = image_np
        self.info_requests = []

    def get_info_by_id(self, image_id):
        self.info_requests.append(image_id)
        return SimpleNamespace(
            id=image_id, height=self._image_np.shape[0], width=self._image_np.shape[1]
        )

    def download_np(self, image_id):
        return self._image_np.copy()


class FakeApi:
    """Stub of :class:`~supervisely.api.api.Api` with only the endpoints the routes touch."""

    def __init__(self, image_np, annotation=None, forbid_download=False):
        self.annotation = _FakeAnnotationApi(annotation, forbidden=forbid_download)
        self.image = _FakeImageApi(image_np)


class _FakeFramesCache:
    """Stands in for :class:`~supervisely.nn.inference.cache.InferenceImageCache`."""

    def __init__(self, image_np):
        self._image_np = image_np

    def add_cache_endpoint(self, server):
        pass

    def download_image(self, api, image_id, **kwargs):
        return self._image_np.copy()

    def download_frame(self, api, video_id, frame_index):
        return self._image_np.copy()

    def download_image_by_hash(self, api, image_hash):
        return self._image_np.copy()


class StubSegmentation(InteractiveSegmentation):
    """InteractiveSegmentation with the heavy Inference machinery replaced.

    Only the attributes the Smart Tool routes actually touch are set up, so the production
    route bodies run unchanged against the stubbed API and predictor.
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


@contextlib.contextmanager
def smart_tool_routes(model: StubSegmentation, data_dir):
    """Registers the real Smart Tool routes and yields them by path."""
    with mock.patch.object(Inference, "serve", lambda self: None), mock.patch.object(
        interactive_segmentation_module, "get_data_dir", lambda: str(data_dir)
    ):
        model.serve()
        yield {
            route.path: route.endpoint
            for route in model._server.routes
            if getattr(route, "endpoint", None) is not None
        }


def context(**overrides) -> dict:
    """Smart Tool request context; the initial figure fields are set by the caller."""
    request_context = {
        "image_id": IMAGE_ID,
        "crop": CROP,
        "positive": [{"x": 3, "y": 3}],
        "negative": [],
        "request_uid": "uid-1",
    }
    request_context.update(overrides)
    return request_context


def request(request_context: dict, api: FakeApi):
    return SimpleNamespace(
        state=SimpleNamespace(context=request_context, state={"settings": {}}, api=api)
    )


def batch_request(states, api: FakeApi):
    return SimpleNamespace(
        state=SimpleNamespace(context={"states": list(states)}, state={"settings": {}}, api=api)
    )


def response_stub():
    return SimpleNamespace(status_code=200)


def sample_image(seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 255, size=(IMG_H, IMG_W, 3), dtype=np.uint8)


def sample_prediction() -> np.ndarray:
    mask = np.zeros((10, 10), bool)
    mask[2:5, 3:6] = True
    return mask

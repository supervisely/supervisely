"""Issue reproducer: Smart Tool edit initialized by a direct mask, without a figure id.

Runs the production ``/smart_segmentation`` route against a stub API whose annotation
download always fails, so the request can only succeed if the server reads the initial
figure from the ``mask`` of the request context.

Usage: ``python tests/unit/smart_tool_direct_mask_repro.py`` from the repository root.
Exits 0 only if every checked scenario holds; prints the observed request/response facts.
"""

import os
import sys
import tempfile
import traceback

import numpy as np

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_TESTS_DIR))
# Always exercise the sources of this checkout, never an installed supervisely.
sys.path.insert(0, _TESTS_DIR)
sys.path.insert(0, _REPO_ROOT)
from smart_tool_harness import (  # noqa: E402  (path is set up above)
    CROP,
    IMG_H,
    IMG_W,
    LOCAL_FIGURE_ID,
    FakeApi,
    StubSegmentation,
    context,
    crop_full_mask,
    holed_mask,
    mask_payload,
    place_mask,
    request,
    response_stub,
    sample_image,
    sample_prediction,
    smart_tool_routes,
)

import supervisely as sly  # noqa: E402  (path is set up above)

FAILURES = []


def check(label, condition):
    print(f"[repro] {label}: {condition}")
    if not condition:
        FAILURES.append(label)



def run_scenario(name, mask, x, y, continuation):
    print(f"[repro] --- {name}: origin=(x={x}, y={y}) mask_shape={mask.shape}")
    image_np = sample_image()
    api = FakeApi(image_np, forbid_download=True)
    model = StubSegmentation(image_np, sample_prediction())
    payload = mask_payload(mask, x, y)
    first = context(init_figure=True, local_figure_id=LOCAL_FIGURE_ID, mask=payload)
    print(f"[repro] request fields: {sorted(first)}")
    print(f"[repro] figure_id in request: {'figure_id' in first}")

    with tempfile.TemporaryDirectory() as data_dir:
        with smart_tool_routes(model, data_dir) as routes:
            route = routes["/smart_segmentation"]
            response = route(response=response_stub(), request=request(first, api))
            if continuation:
                follow_up = context(request_uid="uid-2", local_figure_id=LOCAL_FIGURE_ID)
                route(response=response_stub(), request=request(follow_up, api))

    expected = crop_full_mask(place_mask(mask, x, y))
    init_mask = model.predict_calls[0]["init_mask"]
    print(f"[repro] annotation downloads attempted: {api.annotation.downloads}")
    print(
        f"[repro] response: success={response['success']} error={response['error']} "
        f"origin={response['origin']}"
    )
    print(
        f"[repro] predictor init_mask: shape={None if init_mask is None else init_mask.shape} "
        f"dtype={None if init_mask is None else init_mask.dtype} "
        f"nonzero={None if init_mask is None else int((init_mask > 0).sum())} "
        f"(expected nonzero={int((expected > 0).sum())})"
    )
    check(f"{name}: no annotation download", api.annotation.downloads == [])
    check(f"{name}: request succeeded", response["success"] is True)
    check(f"{name}: predictor received the initial mask", init_mask is not None)
    check(
        f"{name}: predictor mask equals the expected crop",
        init_mask is not None and np.array_equal(init_mask, expected),
    )
    if continuation:
        follow_up_mask = model.predict_calls[1]["init_mask"]
        check(
            f"{name}: continuation click reused the mask without resending it",
            follow_up_mask is not None and np.array_equal(follow_up_mask, expected),
        )


def main():
    print(f"[repro] supervisely package under test: {sly.__file__}")
    print(f"[repro] image size: {IMG_H}x{IMG_W}, crop: {CROP}")
    try:
        run_scenario("nonzero origin", holed_mask(), x=5, y=4, continuation=True)
        run_scenario("edge origin", holed_mask(), x=0, y=0, continuation=False)
        clipped = np.ones((4, 4), bool)
        run_scenario("clipped at the border", clipped, x=IMG_W - 2, y=IMG_H - 3, continuation=False)
    except Exception:  # noqa: BLE001 - the baseline failure mode is part of the evidence
        traceback.print_exc()
        print("[repro] FAILED: the request could not be served from the mask")
        return 1
    if FAILURES:
        print(f"[repro] FAILED: {FAILURES}")
        return 1
    print("[repro] OK: direct mask initialized Smart Tool without a figure id")
    return 0


if __name__ == "__main__":
    sys.exit(main())

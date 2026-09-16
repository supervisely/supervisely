"""Offline regressions for the Smart Tool initial-mask path.

The Smart Tool server used to learn the edited figure only from ``figure_id``, downloading
the whole annotation of the image for every initial request. Clients now send the figure
itself as ``mask``: ``{"origin": [x, y], "data": <base64 bitmap payload>}`` in image
coordinates. These tests pin the decoding contract, the precedence of the mask over the
deprecated figure-id download, the continuation cache identity and the two image routes
that consume all of it.

Everything is CPU-only: the API and the predictor are stubbed at their boundaries, no
instance is contacted and no model is loaded. Run as ``python -m pytest tests/unit`` from
the repository root (bare ``pytest`` may import an installed supervisely instead).
"""

import logging

import numpy as np
import pytest
from smart_tool_harness import (
    CROP,
    FIGURE_ID,
    IMAGE_ID,
    IMG_H,
    IMG_W,
    LOCAL_FIGURE_ID,
    AnnotationDownloadForbidden,
    FakeApi,
    StubSegmentation,
    ann_json,
    batch_request,
    bitmap_label,
    context,
    crop_full_mask,
    encode_mask,
    mask_payload,
    place_mask,
    request,
    response_stub,
    sample_image,
    sample_prediction,
    smart_tool_routes,
)

import supervisely as sly
from supervisely.nn.inference.interactive_segmentation import functional
from supervisely.sly_logger import logger as sly_logger

IMG_SIZE = (IMG_H, IMG_W)


def holed_mask():
    """Tight raster with a hole and a disconnected component, as a rasterized polygon has."""
    mask = np.zeros((5, 6), bool)
    mask[0:3, 0:5] = True
    mask[1, 1:3] = False
    mask[4, 4:6] = True
    return mask


@pytest.fixture
def image_np():
    return sample_image()


@pytest.fixture
def pred_mask():
    return sample_prediction()


@pytest.fixture
def routes(tmp_path):
    """Serves the production Smart Tool routes of a stub model."""

    def _routes(model):
        return smart_tool_routes(model, tmp_path)

    return _routes


# ------------------------------------------------------------------ mask decoding


def test_mask_with_a_nonzero_origin_keeps_holes_and_disconnected_pixels():
    mask = holed_mask()

    bitmap = functional.decode_init_mask(mask_payload(mask, x=5, y=4), IMG_SIZE)

    assert isinstance(bitmap, sly.Bitmap)
    assert (bitmap.origin.row, bitmap.origin.col) == (4, 5)
    np.testing.assert_array_equal(
        functional.bitmap_to_mask(bitmap, IMG_H, IMG_W),
        (place_mask(mask, x=5, y=4) * 255).astype(np.uint8),
    )


def test_mask_at_the_image_edge_is_placed_at_zero_origin():
    mask = holed_mask()

    bitmap = functional.decode_init_mask(mask_payload(mask, x=0, y=0), IMG_SIZE)

    assert (bitmap.origin.row, bitmap.origin.col) == (0, 0)
    np.testing.assert_array_equal(np.asarray(bitmap.data, bool), mask)


def test_padded_mask_data_keeps_its_absolute_placement():
    """A payload padded with empty rows/columns must not shift the figure."""
    padded = np.zeros((6, 6), bool)
    padded[2:4, 3:5] = True

    bitmap = functional.decode_init_mask(mask_payload(padded, x=2, y=3), IMG_SIZE)

    assert (bitmap.origin.row, bitmap.origin.col) == (3 + 2, 2 + 3)
    np.testing.assert_array_equal(
        np.asarray(functional.bitmap_to_mask(bitmap, IMG_H, IMG_W) > 0),
        place_mask(padded, x=2, y=3),
    )


def test_mask_hanging_over_the_image_border_is_clipped():
    mask = np.ones((4, 5), bool)

    bitmap = functional.decode_init_mask(mask_payload(mask, x=IMG_W - 2, y=IMG_H - 3), IMG_SIZE)

    assert (bitmap.origin.row, bitmap.origin.col) == (IMG_H - 3, IMG_W - 2)
    assert bitmap.data.shape == (3, 2)
    np.testing.assert_array_equal(
        np.asarray(functional.bitmap_to_mask(bitmap, IMG_H, IMG_W) > 0),
        place_mask(mask, x=IMG_W - 2, y=IMG_H - 3),
    )


def test_mask_with_a_negative_origin_is_clipped_to_the_image():
    mask = np.ones((4, 4), bool)

    bitmap = functional.decode_init_mask(mask_payload(mask, x=-2, y=-1), IMG_SIZE)

    assert (bitmap.origin.row, bitmap.origin.col) == (0, 0)
    assert bitmap.data.shape == (3, 2)


def test_mask_fully_outside_of_the_image_is_reported():
    with pytest.raises(functional.InitMaskError, match="outside of the image"):
        functional.decode_init_mask(mask_payload(np.ones((2, 2), bool), x=IMG_W + 4, y=0), IMG_SIZE)


def test_empty_mask_is_reported():
    empty = sly.Bitmap.data_2_base64(np.zeros((4, 4), bool))

    with pytest.raises(functional.InitMaskError, match="empty"):
        functional.decode_init_mask({"origin": [1, 1], "data": empty}, IMG_SIZE)


@pytest.mark.parametrize(
    "payload, message",
    [
        ("not-an-object", "must be an object"),
        ({"data": encode_mask(np.ones((2, 2), bool))}, "origin"),
        ({"origin": [1], "data": encode_mask(np.ones((2, 2), bool))}, "origin"),
        ({"origin": "1,2", "data": encode_mask(np.ones((2, 2), bool))}, "origin"),
        ({"origin": [1, 2.5], "data": encode_mask(np.ones((2, 2), bool))}, "origin"),
        ({"origin": [True, 2], "data": encode_mask(np.ones((2, 2), bool))}, "origin"),
        ({"origin": ["1", 2], "data": encode_mask(np.ones((2, 2), bool))}, "origin"),
        ({"origin": [1, 2]}, "data"),
        ({"origin": [1, 2], "data": ""}, "data"),
        ({"origin": [1, 2], "data": 17}, "data"),
        ({"origin": [1, 2], "data": "not-base64-at-all"}, "decode"),
    ],
    ids=[
        "not-an-object",
        "no-origin",
        "short-origin",
        "origin-not-a-pair",
        "fractional-origin",
        "boolean-origin",
        "string-origin",
        "no-data",
        "empty-data",
        "data-not-a-string",
        "broken-base64",
    ],
)
def test_malformed_mask_payload_is_reported(payload, message):
    with pytest.raises(functional.InitMaskError, match=message):
        functional.decode_init_mask(payload, IMG_SIZE)


def test_integral_float_origin_is_accepted():
    """JSON numbers may arrive as floats; whole values are still valid coordinates."""
    bitmap = functional.decode_init_mask(
        {"origin": [3.0, 2.0], "data": encode_mask(np.ones((2, 2), bool))}, IMG_SIZE
    )

    assert (bitmap.origin.row, bitmap.origin.col) == (2, 3)


# ------------------------------------------------------------------ legacy download


def test_download_init_mask_reads_a_saved_bitmap():
    label = bitmap_label(np.array([[1, 0, 1], [1, 1, 1]], bool), x=5, y=4)
    api = FakeApi(sample_image(), annotation=ann_json(label))

    bitmap = functional.download_init_mask(api, FIGURE_ID, IMAGE_ID)

    assert api.annotation.downloads == [IMAGE_ID]
    assert (bitmap.origin.row, bitmap.origin.col) == (4, 5)
    np.testing.assert_array_equal(
        np.asarray(bitmap.data, bool), np.array([[1, 0, 1], [1, 1, 1]], bool)
    )


def test_download_init_mask_without_a_figure_id_does_not_download():
    api = FakeApi(sample_image(), annotation=ann_json())

    with pytest.raises(functional.InitMaskError, match="not provided"):
        functional.download_init_mask(api, None, IMAGE_ID)
    assert api.annotation.downloads == []


def test_download_init_mask_reports_an_unknown_figure():
    label = bitmap_label(np.ones((2, 2), bool), x=1, y=1)
    api = FakeApi(sample_image(), annotation=ann_json(label))

    with pytest.raises(functional.InitMaskError, match="not found"):
        functional.download_init_mask(api, FIGURE_ID + 1, IMAGE_ID)


def test_download_init_mask_reports_a_non_bitmap_figure():
    label = {"id": FIGURE_ID, "geometryType": "polygon", "points": {"exterior": [[1, 1]]}}
    api = FakeApi(sample_image(), annotation=ann_json(label))

    with pytest.raises(functional.InitMaskError, match="initial bitmap"):
        functional.download_init_mask(api, FIGURE_ID, IMAGE_ID)


# ------------------------------------------------------------------ bitmap_to_mask


def test_bitmap_to_mask_places_the_mask_on_a_full_image_canvas():
    bitmap = sly.Bitmap(np.ones((2, 3), bool), origin=sly.PointLocation(row=4, col=5))

    mask = functional.bitmap_to_mask(bitmap, IMG_H, IMG_W)

    assert mask.dtype == np.uint8 and set(np.unique(mask)) == {0, 255}
    expected = np.zeros((IMG_H, IMG_W), np.uint8)
    expected[4:6, 5:8] = 255
    np.testing.assert_array_equal(mask, expected)


def test_bitmap_to_mask_clips_a_bitmap_hanging_over_the_image_border():
    bitmap = sly.Bitmap(np.ones((5, 5), bool), origin=sly.PointLocation(row=IMG_H - 2, col=-2))

    mask = functional.bitmap_to_mask(bitmap, IMG_H, IMG_W)

    expected = np.zeros((IMG_H, IMG_W), np.uint8)
    expected[IMG_H - 2 :, 0:3] = 255
    np.testing.assert_array_equal(mask, expected)


# ------------------------------------------------------------------ single route


def test_direct_mask_without_a_figure_id_reaches_the_predictor(image_np, pred_mask, routes):
    mask = holed_mask()
    api = FakeApi(image_np, forbid_download=True)
    model = StubSegmentation(image_np, pred_mask)
    smtool_state = context(
        init_figure=True, local_figure_id=LOCAL_FIGURE_ID, mask=mask_payload(mask, 5, 4)
    )

    with routes(model) as served:
        result = served["/smart_segmentation"](
            response=response_stub(), request=request(smtool_state, api)
        )

    assert "figure_id" not in smtool_state
    assert api.annotation.downloads == []
    assert result["success"] is True and result["error"] is None
    assert result["origin"] == {"x": 1 + 3, "y": 1 + 2}  # crop origin + prediction origin
    assert result["bitmap"] is not None
    np.testing.assert_array_equal(
        model.predict_calls[0]["init_mask"], crop_full_mask(place_mask(mask, 5, 4))
    )
    assert model.predict_calls[0]["clicks"] == [(2, 2, True)]


def test_direct_mask_hanging_over_the_border_is_clipped_for_the_predictor(
    image_np, pred_mask, routes
):
    mask = np.ones((4, 5), bool)
    api = FakeApi(image_np, forbid_download=True)
    model = StubSegmentation(image_np, pred_mask)
    smtool_state = context(
        init_figure=True,
        local_figure_id=LOCAL_FIGURE_ID,
        mask=mask_payload(mask, IMG_W - 2, IMG_H - 3),
    )

    with routes(model) as served:
        result = served["/smart_segmentation"](
            response=response_stub(), request=request(smtool_state, api)
        )

    assert result["success"] is True
    np.testing.assert_array_equal(
        model.predict_calls[0]["init_mask"],
        crop_full_mask(place_mask(mask, IMG_W - 2, IMG_H - 3)),
    )


def test_mask_wins_over_a_supplied_figure_id(image_np, pred_mask, routes):
    """The annotation download of the stub API always fails, so a download would surface."""
    mask = holed_mask()
    api = FakeApi(image_np, forbid_download=True)
    model = StubSegmentation(image_np, pred_mask)
    smtool_state = context(
        init_figure=True,
        figure_id=FIGURE_ID,
        local_figure_id=LOCAL_FIGURE_ID,
        mask=mask_payload(mask, 5, 4),
    )

    with routes(model) as served:
        result = served["/smart_segmentation"](
            response=response_stub(), request=request(smtool_state, api)
        )

    assert api.annotation.downloads == []
    assert result["success"] is True
    np.testing.assert_array_equal(
        model.predict_calls[0]["init_mask"], crop_full_mask(place_mask(mask, 5, 4))
    )


def test_continuation_click_reuses_the_mask_by_local_figure_id(image_np, pred_mask, routes):
    mask = holed_mask()
    api = FakeApi(image_np, forbid_download=True)
    model = StubSegmentation(image_np, pred_mask)

    with routes(model) as served:
        route = served["/smart_segmentation"]
        route(
            response=response_stub(),
            request=request(
                context(
                    init_figure=True,
                    figure_id=FIGURE_ID,
                    local_figure_id=LOCAL_FIGURE_ID,
                    mask=mask_payload(mask, 5, 4),
                ),
                api,
            ),
        )
        route(
            response=response_stub(),
            request=request(context(request_uid="uid-2", local_figure_id=LOCAL_FIGURE_ID), api),
        )

    assert list(model._init_mask_cache.keys()) == [LOCAL_FIGURE_ID]
    assert api.annotation.downloads == []
    np.testing.assert_array_equal(
        model.predict_calls[1]["init_mask"], crop_full_mask(place_mask(mask, 5, 4))
    )


def test_continuation_falls_back_to_the_legacy_figure_id_cache_key(image_np, pred_mask, routes):
    mask = holed_mask()
    api = FakeApi(image_np, forbid_download=True)
    model = StubSegmentation(image_np, pred_mask)

    with routes(model) as served:
        route = served["/smart_segmentation"]
        route(
            response=response_stub(),
            request=request(
                context(init_figure=True, figure_id=FIGURE_ID, mask=mask_payload(mask, 5, 4)), api
            ),
        )
        route(
            response=response_stub(),
            request=request(context(request_uid="uid-2", figure_id=FIGURE_ID), api),
        )

    assert list(model._init_mask_cache.keys()) == [FIGURE_ID]
    np.testing.assert_array_equal(
        model.predict_calls[1]["init_mask"], crop_full_mask(place_mask(mask, 5, 4))
    )


def test_legacy_figure_id_request_still_downloads_and_warns(
    image_np, pred_mask, routes, caplog, monkeypatch
):
    mask = np.ones((3, 4), bool)
    label = bitmap_label(mask, x=5, y=4)
    api = FakeApi(image_np, annotation=ann_json(label))
    model = StubSegmentation(image_np, pred_mask)
    monkeypatch.setattr(sly_logger, "propagate", True)

    with caplog.at_level(logging.WARNING, logger=sly_logger.name):
        with routes(model) as served:
            result = served["/smart_segmentation"](
                response=response_stub(),
                request=request(context(init_figure=True, figure_id=FIGURE_ID), api),
            )

    assert result["success"] is True
    assert api.annotation.downloads == [IMAGE_ID]
    assert any("deprecated" in record.getMessage() for record in caplog.records)
    np.testing.assert_array_equal(
        model.predict_calls[0]["init_mask"], crop_full_mask(place_mask(mask, 5, 4))
    )


def test_request_without_any_initial_figure_passes_no_init_mask(image_np, pred_mask, routes):
    api = FakeApi(image_np, forbid_download=True)
    model = StubSegmentation(image_np, pred_mask)

    with routes(model) as served:
        result = served["/smart_segmentation"](
            response=response_stub(), request=request(context(), api)
        )

    assert result["success"] is True
    assert api.annotation.downloads == []
    assert model.predict_calls[0]["init_mask"] is None


def test_legacy_init_figure_without_a_figure_id_is_reported(image_np, pred_mask, routes):
    api = FakeApi(image_np, annotation=ann_json(bitmap_label(np.ones((2, 2), bool), x=1, y=1)))
    model = StubSegmentation(image_np, pred_mask)

    with routes(model) as served:
        result = served["/smart_segmentation"](
            response=response_stub(), request=request(context(init_figure=True), api)
        )

    assert result["success"] is False
    assert result["origin"] is None and result["bitmap"] is None
    assert "not provided" in result["error"]
    assert api.annotation.downloads == []
    assert model.predict_calls == []


def test_malformed_mask_is_reported_and_drops_the_cached_mask(image_np, pred_mask, routes):
    mask = holed_mask()
    api = FakeApi(image_np, forbid_download=True)
    model = StubSegmentation(image_np, pred_mask)

    with routes(model) as served:
        route = served["/smart_segmentation"]
        route(
            response=response_stub(),
            request=request(
                context(
                    init_figure=True, local_figure_id=LOCAL_FIGURE_ID, mask=mask_payload(mask, 5, 4)
                ),
                api,
            ),
        )
        broken = route(
            response=response_stub(),
            request=request(
                context(
                    request_uid="uid-2",
                    init_figure=True,
                    local_figure_id=LOCAL_FIGURE_ID,
                    mask={"origin": [1, 1], "data": "not-base64-at-all"},
                ),
                api,
            ),
        )
        # A later click must not resurrect the mask of the previous prompt.
        after = route(
            response=response_stub(),
            request=request(context(request_uid="uid-3", local_figure_id=LOCAL_FIGURE_ID), api),
        )

    assert broken["success"] is False and broken["origin"] is None and broken["bitmap"] is None
    assert "decode" in broken["error"]
    assert LOCAL_FIGURE_ID not in model._init_mask_cache
    assert after["success"] is True
    assert model.predict_calls[-1]["init_mask"] is None
    assert len(model.predict_calls) == 2  # the broken request never reached the predictor


def test_empty_prediction_keeps_the_response_schema(image_np, routes):
    api = FakeApi(image_np, forbid_download=True)
    model = StubSegmentation(image_np, np.zeros((10, 10), bool))

    with routes(model) as served:
        result = served["/smart_segmentation"](
            response=response_stub(),
            request=request(context(init_figure=True, mask=mask_payload(holed_mask(), 5, 4)), api),
        )

    assert result == {"origin": None, "bitmap": None, "success": True, "error": None}
    assert len(model.predict_calls) == 1


def test_request_without_clicks_returns_the_no_result_response(image_np, pred_mask, routes):
    api = FakeApi(image_np, forbid_download=True)
    model = StubSegmentation(image_np, pred_mask)

    with routes(model) as served:
        result = served["/smart_segmentation"](
            response=response_stub(),
            request=request(
                context(init_figure=True, mask=mask_payload(holed_mask(), 5, 4), positive=[]), api
            ),
        )

    assert result == {"origin": None, "bitmap": None, "success": True, "error": None}
    assert model.predict_calls == []


# ------------------------------------------------------------------ batch route


def test_batch_route_matches_the_single_route(image_np, pred_mask, routes):
    mask = holed_mask()
    api = FakeApi(image_np, forbid_download=True)
    single_model = StubSegmentation(image_np, pred_mask)
    batch_model = StubSegmentation(image_np, pred_mask)
    states = [
        context(init_figure=True, local_figure_id=LOCAL_FIGURE_ID, mask=mask_payload(mask, 5, 4)),
        context(request_uid="uid-2", local_figure_id=LOCAL_FIGURE_ID),
    ]

    with routes(single_model) as served:
        single_results = [
            served["/smart_segmentation"](response=response_stub(), request=request(state, api))
            for state in states
        ]
    with routes(batch_model) as served:
        batch_results = served["/smart_segmentation_batch"](
            response=response_stub(), request=batch_request(states, api)
        )

    assert batch_results == single_results
    assert api.annotation.downloads == []
    expected = crop_full_mask(place_mask(mask, 5, 4))
    for call in batch_model.predict_calls:
        np.testing.assert_array_equal(call["init_mask"], expected)
    for batch_call, single_call in zip(batch_model.predict_calls, single_model.predict_calls):
        np.testing.assert_array_equal(batch_call["init_mask"], single_call["init_mask"])


def test_batch_route_does_not_leak_an_init_mask_between_states(image_np, pred_mask, routes):
    """The batch route reuses one settings dict, so a figure-less state must reset init_mask."""
    api = FakeApi(image_np, forbid_download=True)
    model = StubSegmentation(image_np, pred_mask)
    states = [
        context(
            init_figure=True, local_figure_id=LOCAL_FIGURE_ID, mask=mask_payload(holed_mask(), 5, 4)
        ),
        context(request_uid="uid-no-figure"),
    ]

    with routes(model) as served:
        results = served["/smart_segmentation_batch"](
            response=response_stub(), request=batch_request(states, api)
        )

    assert all(item["success"] is True for item in results)
    assert model.predict_calls[0]["init_mask"].any()
    assert model.predict_calls[1]["init_mask"] is None


def test_batch_route_reports_a_broken_state_without_failing_the_rest(image_np, pred_mask, routes):
    api = FakeApi(image_np, forbid_download=True)
    model = StubSegmentation(image_np, pred_mask)
    states = [
        context(
            request_uid="uid-broken",
            init_figure=True,
            local_figure_id="broken",
            mask={"origin": [0, 0], "data": "not-base64-at-all"},
        ),
        context(
            init_figure=True, local_figure_id=LOCAL_FIGURE_ID, mask=mask_payload(holed_mask(), 5, 4)
        ),
    ]

    with routes(model) as served:
        results = served["/smart_segmentation_batch"](
            response=response_stub(), request=batch_request(states, api)
        )

    assert results[0]["success"] is False and "decode" in results[0]["error"]
    assert results[1]["success"] is True and results[1]["bitmap"] is not None
    assert len(model.predict_calls) == 1
    np.testing.assert_array_equal(
        model.predict_calls[0]["init_mask"], crop_full_mask(place_mask(holed_mask(), 5, 4))
    )


def test_batch_route_still_serves_a_legacy_figure_id_state(image_np, pred_mask, routes):
    mask = np.ones((3, 4), bool)
    api = FakeApi(image_np, annotation=ann_json(bitmap_label(mask, x=5, y=4)))
    model = StubSegmentation(image_np, pred_mask)
    states = [
        context(init_figure=True, figure_id=FIGURE_ID),
        context(request_uid="uid-2", figure_id=FIGURE_ID),
    ]

    with routes(model) as served:
        results = served["/smart_segmentation_batch"](
            response=response_stub(), request=batch_request(states, api)
        )

    assert all(item["success"] is True for item in results)
    assert api.annotation.downloads == [IMAGE_ID], "the figure is downloaded once per prompt"
    expected = crop_full_mask(place_mask(mask, 5, 4))
    for call in model.predict_calls:
        np.testing.assert_array_equal(call["init_mask"], expected)


def test_stub_api_download_is_a_visible_failure():
    """Guards the other tests: a forbidden annotation download must raise, not return None."""
    api = FakeApi(sample_image(), forbid_download=True)

    with pytest.raises(AnnotationDownloadForbidden):
        api.annotation.download_json(IMAGE_ID)
    assert api.annotation.downloads == [IMAGE_ID]


def test_crop_conventions_are_unchanged():
    """Pins the crop the initial mask must match: inclusive x/y bounds of the request crop."""
    full = np.zeros((IMG_H, IMG_W), bool)
    full[1:11, 1:11] = True

    cropped = functional.crop_image(CROP, (full * 255).astype(np.uint8))

    assert cropped.shape == (10, 10)
    assert cropped.all()

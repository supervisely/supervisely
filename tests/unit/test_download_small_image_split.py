"""Tests for image sizes that the API returns as strings.

A single such image used to raise ``TypeError: '<' not supported between instances of 'str'
and 'int'`` in the small/large split and send the whole project download into the
synchronous fallback, and later the same value broke the streaming threshold with ``'>'``.
``ImageInfo.size`` is declared an int, so the value is normalized on conversion.
"""

import pytest

from supervisely.api.api import Api
from supervisely.project.project import _is_small_image

SWITCH_SIZE = 512 * 1024


class _Image:
    def __init__(self, size, id=7):  # noqa: A002 - mirrors ImageInfo
        self.size = size
        self.id = id


@pytest.mark.parametrize(
    "size, expected",
    [
        (1024, True),
        ("1024", True),  # the API returns a string for some images
        (SWITCH_SIZE, False),
        (SWITCH_SIZE + 1, False),
        (str(SWITCH_SIZE + 1), False),
        (None, False),
        ("", False),
        ("not a number", False),
    ],
)
def test_is_small_image(size, expected):
    assert _is_small_image(_Image(size), SWITCH_SIZE) is expected


def test_string_size_does_not_raise():
    """The regression: one such image was enough to abort the whole async download."""
    images = [_Image(1024), _Image("2361250"), _Image(None)]
    small = [image for image in images if _is_small_image(image, SWITCH_SIZE)]
    assert len(small) == 1


@pytest.fixture
def api():
    return Api("https://example.com", "fake-token")


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("2361250", 2361250),
        (576595, 576595),
        (None, None),
        ("", None),
        ("not a number", None),
    ],
)
def test_image_info_size_is_normalized(api, raw, expected):
    info = api.image._convert_json_info({"id": 1, "name": "a.png", "size": raw})
    assert info.size == expected


def test_streaming_threshold_survives_a_string_size(api):
    """The second site that broke: `estimated_size > size_threshold_for_streaming`."""
    info = api.image._convert_json_info({"id": 1, "name": "a.png", "size": "6291456"})
    assert info.size > 5 * 1024 * 1024

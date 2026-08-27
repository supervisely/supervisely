"""Tests for the small/large split of the async project download.

A single image whose size comes back from the API as a string used to raise
``TypeError: '<' not supported between instances of 'str' and 'int'`` and send the whole
project download into the synchronous fallback.
"""

import pytest

from supervisely.project.project import _is_small_image

SWITCH_SIZE = 512 * 1024


class _Image:
    def __init__(self, size):
        self.size = size


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

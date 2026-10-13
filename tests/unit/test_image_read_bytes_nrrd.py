"""Tests for decoding NRRD images from bytes.

pynrrd<1.0 reads raw and ascii encoded data with ``np.fromfile``, which needs a real file
descriptor, so ``read_bytes`` used to raise ``io.UnsupportedOperation: fileno`` for every NRRD
that was not compressed. That is the path ``Api.image.download_np(s)`` takes.
"""

import os
import tempfile

import nrrd
import numpy as np
import pytest

from supervisely.imaging.image import read_bytes


def _nrrd_bytes(data, **write_kwargs):
    fd, path = tempfile.mkstemp(suffix=".nrrd")
    os.close(fd)
    try:
        nrrd.write(path, data, index_order="C", **write_kwargs)
        with open(path, "rb") as f:
            return f.read()
    finally:
        os.remove(path)


def _arrays():
    rng = np.random.default_rng(0)
    return {
        "float32_2d": rng.normal(30.0, 15.0, (48, 64)).astype(np.float32),
        "uint16_2d": rng.integers(0, 65535, (48, 64), dtype=np.uint16),
        "uint8_rgb": rng.integers(0, 255, (48, 64, 3), dtype=np.uint8),
        "float64_big_endian": rng.normal(0.0, 1.0, (16, 24)).astype(">f8"),
    }


@pytest.mark.parametrize("encoding", ["raw", "ascii", "gzip", "bzip2"])
@pytest.mark.parametrize("name", sorted(_arrays()))
def test_read_bytes_nrrd_round_trip(name, encoding):
    data = _arrays()[name]
    if encoding == "ascii" and data.dtype.kind == "f":
        pytest.skip("ascii stores floats as text; exact round trip is not expected")

    result = read_bytes(_nrrd_bytes(data, header={"encoding": encoding}))

    assert result.shape == data.shape
    assert result.dtype == data.dtype
    np.testing.assert_array_equal(result, data)


def test_read_bytes_raw_nrrd_leaves_no_temp_file(tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    data = np.arange(12, dtype=np.float32).reshape(3, 4)

    read_bytes(_nrrd_bytes(data, header={"encoding": "raw"}))

    assert list(tmp_path.iterdir()) == []


def test_read_bytes_truncated_raw_nrrd_raises_and_cleans_up(tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    raw = _nrrd_bytes(np.ones((8, 8), dtype=np.float32), header={"encoding": "raw"})

    with pytest.raises(nrrd.NRRDError):
        read_bytes(raw[:-16])

    assert list(tmp_path.iterdir()) == []

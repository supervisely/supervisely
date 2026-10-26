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


@pytest.mark.parametrize("encoding", ["raw", "gzip", "bzip2"])
@pytest.mark.parametrize("name", sorted(_arrays()))
def test_read_bytes_nrrd_round_trip(name, encoding):
    data = _arrays()[name]

    result = read_bytes(_nrrd_bytes(data, header={"encoding": encoding}))

    assert result.shape == data.shape
    assert result.dtype == data.dtype
    np.testing.assert_array_equal(result, data)


@pytest.mark.parametrize("name", sorted(_arrays()))
def test_read_bytes_ascii_nrrd(name):
    data = _arrays()[name]

    result = read_bytes(_nrrd_bytes(data, header={"encoding": "ascii"}))

    # ascii data has no byte order, so it comes back in native order
    assert result.dtype == data.dtype.newbyteorder("=")
    np.testing.assert_array_equal(result, data)


@pytest.mark.parametrize("encoding", ["raw", "ascii", "ASCII", "text", "txt", "gzip", "gz", "bzip2", "bz2"])
def test_read_bytes_nrrd_encoding_aliases(encoding):
    data = np.arange(24, dtype=np.int16).reshape(4, 6)
    if encoding in ("gzip", "gz", "bzip2", "bz2"):
        payload = _nrrd_bytes(data, header={"encoding": encoding}).split(b"\n\n", 1)[1]
    elif encoding in ("ascii", "ASCII", "text", "txt"):
        payload = " ".join(str(v) for v in data.ravel()).encode() + b"\n"
    else:
        payload = data.astype("<i2").tobytes()
    header = (
        "NRRD0004\ntype: int16\ndimension: 2\nsizes: 6 4\nendian: little\n"
        "encoding: {}\n\n".format(encoding)
    ).encode()

    np.testing.assert_array_equal(read_bytes(header + payload), data)


@pytest.mark.parametrize(
    "skip_fields, prefix",
    [("byte skip: 4\n", b"\x00" * 4), ("line skip: 2\n", b"one\ntwo\n"), ("byte skip: -1\n", b"junk")],
    ids=["byte_skip", "line_skip", "byte_skip_from_end"],
)
def test_read_bytes_raw_nrrd_skips(skip_fields, prefix):
    data = np.arange(12, dtype=np.float32).reshape(3, 4)
    header = (
        "NRRD0004\ntype: float\ndimension: 2\nsizes: 4 3\nendian: little\n"
        "encoding: raw\n{}\n".format(skip_fields)
    ).encode()

    result = read_bytes(header + prefix + data.astype("<f4").tobytes())

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


@pytest.mark.parametrize("field", ["data file", "datafile"])
def test_read_bytes_relative_detached_data_file_is_not_resolved(field, tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    (tmp_path / "data.raw").write_bytes(bytes(range(16)))
    raw = (
        b"NRRD0004\ntype: uint8\ndimension: 1\nsizes: 16\n"
        b"encoding: raw\n" + field.encode() + b": data.raw\n\n"
    )

    with pytest.raises(nrrd.NRRDError):
        read_bytes(raw)

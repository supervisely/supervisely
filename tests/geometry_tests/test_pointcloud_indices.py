# coding: utf-8

import os
import sys
import unittest

import numpy as np

sdk_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, sdk_path)

from supervisely.geometry.pointcloud import Pointcloud
from supervisely.io.fs import decode_uint32_le, encode_uint32_le

UINT32_MAX = 2**32 - 1


class TestUint32Codec(unittest.TestCase):
    """Round-trip and validation tests for the little-endian uint32 index codec."""

    def test_roundtrip_basic(self):
        values = [0, 1, 2, 42, 65535, UINT32_MAX]
        self.assertEqual(decode_uint32_le(encode_uint32_le(values)), values)

    def test_roundtrip_empty(self):
        self.assertEqual(encode_uint32_le([]), b"")
        self.assertEqual(decode_uint32_le(b""), [])

    def test_roundtrip_single(self):
        self.assertEqual(decode_uint32_le(encode_uint32_le([7])), [7])

    def test_roundtrip_large(self):
        values = list(range(1_000_000))
        encoded = encode_uint32_le(values)
        self.assertEqual(len(encoded), 4 * len(values))
        self.assertEqual(decode_uint32_le(encoded), values)

    def test_roundtrip_numpy_input(self):
        values = np.array([3, 1, 4, 1, 5, 9], dtype=np.int64)
        self.assertEqual(decode_uint32_le(encode_uint32_le(values)), values.tolist())

    def test_byte_size_is_four_per_value(self):
        self.assertEqual(len(encode_uint32_le([0, 0, 0])), 12)

    def test_decode_ignores_trailing_bytes(self):
        encoded = encode_uint32_le([1, 2]) + b"\x99\x99\x99"
        self.assertEqual(decode_uint32_le(encoded), [1, 2])

    def test_negative_raises(self):
        with self.assertRaises(ValueError):
            encode_uint32_le([0, -1])

    def test_overflow_raises(self):
        with self.assertRaises(ValueError):
            encode_uint32_le([UINT32_MAX + 1])


class TestPointcloudBytes(unittest.TestCase):
    """Binary serialization of the Pointcloud geometry."""

    def test_to_from_bytes_roundtrip(self):
        pcd = Pointcloud([0, 5, 12, 23, UINT32_MAX])
        restored = Pointcloud.from_bytes(pcd.to_bytes())
        self.assertEqual(restored.indices, pcd.indices)

    def test_empty_indices_roundtrip(self):
        pcd = Pointcloud([])
        self.assertEqual(Pointcloud.from_bytes(pcd.to_bytes()).indices, [])


if __name__ == "__main__":
    unittest.main()

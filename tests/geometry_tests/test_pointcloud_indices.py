# coding: utf-8

import os
import sys
import unittest

import numpy as np

sdk_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, sdk_path)

from supervisely.api.entity_annotation.figure_api import FigureInfo
from supervisely.api.pointcloud.pointcloud_figure_api import PointcloudFigureApi
from supervisely.geometry.constants import INDICES
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


class FakePointcloudFigureApi(PointcloudFigureApi):
    def __init__(self, indices_by_figure_id):
        super().__init__(None)
        self.indices_by_figure_id = indices_by_figure_id
        self.requested_sync = []

    def download_indices_batch(self, figure_ids, progress_cb=None):
        self.requested_sync.extend(figure_ids)
        return [self.indices_by_figure_id[figure_id] for figure_id in figure_ids]


def make_figure_info(figure_id, geometry_type, geometry, entity_id=1):
    return FigureInfo(
        id=figure_id,
        class_id=1,
        updated_at="",
        created_at="",
        entity_id=entity_id,
        object_id=1,
        project_id=1,
        dataset_id=1,
        frame_index=0,
        geometry_type=geometry_type,
        geometry=geometry,
        geometry_meta=None,
        tags=[],
        meta={},
        area=None,
    )


class TestPointcloudFigureHydration(unittest.TestCase):
    def test_hydrates_missing_pointcloud_indices_in_figure_dict(self):
        api = FakePointcloudFigureApi({10: [1], 20: [2, 3]})
        inline = make_figure_info(11, Pointcloud.geometry_name(), {INDICES: [7, 8]}, entity_id=100)
        regular = make_figure_info(12, "cuboid_3d", {}, entity_id=100)
        figures_by_entity = {
            100: [
                make_figure_info(10, Pointcloud.geometry_name(), {}, entity_id=100),
                inline,
                regular,
            ],
            200: [make_figure_info(20, Pointcloud.geometry_name(), {}, entity_id=200)],
        }

        hydrated = api.hydrate_figure_infos_dict(figures_by_entity)

        self.assertEqual(api.requested_sync, [10, 20])
        self.assertEqual(hydrated[100][0].geometry, {INDICES: [1]})
        self.assertEqual(hydrated[100][1].geometry, {INDICES: [7, 8]})
        self.assertEqual(hydrated[100][2].geometry, {})
        self.assertEqual(hydrated[200][0].geometry, {INDICES: [2, 3]})
        self.assertEqual(figures_by_entity[100][0].geometry, {})

    def test_returns_same_dict_when_nothing_needs_hydration(self):
        api = FakePointcloudFigureApi({})
        figures_by_entity = {
            100: [make_figure_info(10, Pointcloud.geometry_name(), {INDICES: []})]
        }

        hydrated = api.hydrate_figure_infos_dict(figures_by_entity)

        self.assertIs(hydrated, figures_by_entity)
        self.assertEqual(api.requested_sync, [])


if __name__ == "__main__":
    unittest.main()

# coding: utf-8

import os
import sys
import unittest

import numpy as np

sdk_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, sdk_path)

import supervisely as sly
from supervisely.annotation.obj_class_collection import ObjClassCollection
from supervisely.geometry.geometry import Geometry


def _multipolygon():
    polygon_1 = sly.Polygon(
        [[10, 10], [10, 40], [40, 40], [40, 10]],
        [[[20, 20], [20, 30], [30, 30], [30, 20]]],
    )
    polygon_2 = sly.Polygon([[60, 60], [60, 80], [80, 80], [80, 60]])
    return sly.Multipolygon([polygon_1, polygon_2])


class TestMultipolygon(unittest.TestCase):
    def test_geometry_name(self):
        self.assertEqual(sly.Multipolygon.geometry_name(), "multipolygon")

    def test_to_json(self):
        geometry = _multipolygon()
        data = geometry.to_json()

        self.assertIn("parts", data)
        self.assertEqual(len(data["parts"]), 2)
        self.assertEqual(data["parts"][0]["exterior"], [[10, 10], [40, 10], [40, 40], [10, 40]])
        self.assertEqual(data["parts"][0]["interior"], [[[20, 20], [30, 20], [30, 30], [20, 30]]])

    def test_from_json_live_api_shape(self):
        data = {
            "id": 123,
            "classId": 456,
            "labelerLogin": "user",
            "createdAt": "2026-07-06T10:59:38.246Z",
            "updatedAt": "2026-07-06T11:00:14.971Z",
            "geometryType": "multipolygon",
            "parts": [
                {
                    "exterior": [[10.8, 10.2], [40.5, 10.4], [40.1, 40.7], [10.3, 40.6]],
                    "interior": [[[20, 20], [30, 20], [30, 30], [20, 30]]],
                },
                {"exterior": [[60, 60], [80, 60], [80, 80], [60, 80]], "interior": []},
            ],
        }

        geometry = sly.Multipolygon.from_json(data)

        self.assertEqual(geometry.sly_id, 123)
        self.assertEqual(geometry.class_id, 456)
        self.assertEqual(geometry.labeler_login, "user")
        self.assertEqual(len(geometry.parts), 2)
        self.assertEqual(geometry.parts[0].exterior[0].row, 10)
        self.assertEqual(geometry.parts[0].exterior[0].col, 10)

    def test_obj_class_and_annotation_roundtrip(self):
        obj_class = sly.ObjClass("multi", sly.Multipolygon, color=[255, 0, 0])
        meta = sly.ProjectMeta(obj_classes=ObjClassCollection([obj_class]))
        annotation = sly.Annotation((100, 100), [sly.Label(_multipolygon(), obj_class)])

        meta_restored = sly.ProjectMeta.from_json(meta.to_json())
        ann_restored = sly.Annotation.from_json(annotation.to_json(), meta_restored)

        self.assertEqual(meta_restored.get_obj_class("multi").geometry_type, sly.Multipolygon)
        self.assertEqual(type(ann_restored.labels[0].geometry), sly.Multipolygon)
        self.assertEqual(len(ann_restored.labels[0].geometry.parts), 2)

    def test_draw_and_mask(self):
        geometry = _multipolygon()
        bitmap = np.zeros((100, 100, 3), dtype=np.uint8)

        geometry.draw(bitmap, [255, 0, 0])
        mask = geometry.get_mask((100, 100))

        self.assertGreater(np.count_nonzero(bitmap[:, :, 0]), 0)
        self.assertTrue(mask[15, 15])
        self.assertFalse(mask[25, 25])
        self.assertTrue(mask[65, 65])

    def test_bbox_area_and_transforms(self):
        geometry = _multipolygon()

        self.assertEqual(geometry.area, 1200)
        self.assertEqual(geometry.to_bbox().to_json()["points"]["exterior"], [[10, 10], [80, 80]])
        self.assertEqual(geometry.translate(5, 7).to_bbox().to_json()["points"]["exterior"], [[17, 15], [87, 85]])
        self.assertEqual(geometry.scale(2).to_bbox().to_json()["points"]["exterior"], [[20, 20], [160, 160]])

    def test_crop(self):
        cropped = _multipolygon().crop(sly.Rectangle(0, 0, 50, 50))

        self.assertEqual(len(cropped), 1)
        self.assertEqual(type(cropped[0]), sly.Multipolygon)
        self.assertEqual(len(cropped[0].parts), 1)

    def test_convert_to_polygons(self):
        obj_class = sly.ObjClass("multi", sly.Multipolygon)
        polygon_class = sly.ObjClass("multi_polygon", sly.Polygon)
        label = sly.Label(_multipolygon(), obj_class)

        converted = label.convert(polygon_class)

        self.assertEqual(len(converted), 2)
        self.assertTrue(all(type(label.geometry) is sly.Polygon for label in converted))

    def test_convert_masks_to_multipolygon(self):
        bitmap_class = sly.ObjClass("bitmap", sly.Bitmap)
        alpha_mask_class = sly.ObjClass("alpha", sly.AlphaMask)
        multipolygon_class = sly.ObjClass("multi", sly.Multipolygon)
        mask = np.zeros((100, 100), dtype=np.bool_)
        mask[10:40, 10:40] = True
        mask[60:85, 60:85] = True

        bitmap_labels = sly.Label(sly.Bitmap(mask), bitmap_class).convert(multipolygon_class)
        alpha_labels = sly.Label(sly.AlphaMask(mask.astype(np.uint8) * 255), alpha_mask_class).convert(
            multipolygon_class
        )

        self.assertEqual(len(bitmap_labels), 1)
        self.assertEqual(type(bitmap_labels[0].geometry), sly.Multipolygon)
        self.assertEqual(len(bitmap_labels[0].geometry.parts), 2)
        self.assertEqual(len(alpha_labels), 1)
        self.assertEqual(type(alpha_labels[0].geometry), sly.Multipolygon)
        self.assertEqual(len(alpha_labels[0].geometry.parts), 2)

    def test_unknown_geometry_deserializes_as_any_geometry(self):
        meta_json = {
            "classes": [
                {
                    "id": 1,
                    "title": "future_shape",
                    "shape": "future_geometry",
                    "color": "#FF0000",
                    "geometry_config": {},
                }
            ],
            "tags": [],
        }
        ann_json = {
            "size": {"height": 100, "width": 100},
            "description": "",
            "tags": [],
            "objects": [
                {
                    "id": 10,
                    "classId": 1,
                    "classTitle": "future_shape",
                    "geometryType": "future_geometry",
                    "futurePayload": {"value": 123},
                    "tags": [],
                }
            ],
        }

        meta = sly.ProjectMeta.from_json(meta_json)
        ann = sly.Annotation.from_json(ann_json, meta)

        self.assertEqual(meta.get_obj_class("future_shape").geometry_type, sly.AnyGeometry)
        self.assertEqual(type(ann.labels[0].geometry), sly.AnyGeometry)
        self.assertEqual(ann.labels[0].geometry.raw_geometry_type, "future_geometry")

    def test_pixel_subpixel_json_helpers_support_parts(self):
        data = {
            "geometryType": "multipolygon",
            "parts": [{"exterior": [[10.4, 10.4], [20.5, 10.5], [20.5, 20.5]], "interior": []}],
        }

        pixel = Geometry._to_pixel_coordinate_system_json(data, [100, 100])
        subpixel = Geometry._to_subpixel_coordinate_system_json(pixel)

        self.assertEqual(pixel["parts"][0]["exterior"], [[10, 10], [20, 10], [20, 20]])
        self.assertEqual(subpixel["parts"][0]["exterior"], [[10.5, 10.5], [20.5, 10.5], [20.5, 20.5]])


if __name__ == "__main__":
    unittest.main()

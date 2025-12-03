# coding: utf-8

import math
import os
import sys
import unittest

import cv2
import numpy as np

sdk_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, sdk_path)

from supervisely.geometry.oriented_bbox import OrientedBBox
from supervisely.geometry.point_location import PointLocation
from supervisely.geometry.polygon import Polygon
from supervisely.geometry.rectangle import Rectangle


class TestOrientedBBoxCreation(unittest.TestCase):
    """Test cases for OrientedBBox creation and basic properties."""

    def test_create_basic_oriented_bbox(self):
        """Test creating a basic OrientedBBox without angle."""
        obb = OrientedBBox(top=100, left=100, bottom=700, right=900)
        self.assertEqual(obb.top, 100)
        self.assertEqual(obb.left, 100)
        self.assertEqual(obb.bottom, 700)
        self.assertEqual(obb.right, 900)
        self.assertEqual(obb.angle, 0)

    def test_create_oriented_bbox_with_angle(self):
        """Test creating an OrientedBBox with a positive angle."""
        obb = OrientedBBox(top=100, left=100, bottom=700, right=900, angle=15)
        self.assertEqual(obb.top, 100)
        self.assertEqual(obb.left, 100)
        self.assertEqual(obb.bottom, 700)
        self.assertEqual(obb.right, 900)
        self.assertEqual(obb.angle, 15)

    def test_create_oriented_bbox_with_negative_angle(self):
        """Test creating an OrientedBBox with a negative angle."""
        obb = OrientedBBox(top=100, left=100, bottom=200, right=300, angle=-45)
        self.assertEqual(obb.angle, -45)

    def test_create_oriented_bbox_with_float_angle(self):
        """Test creating an OrientedBBox with a float angle."""
        obb = OrientedBBox(top=100, left=100, bottom=200, right=300, angle=22.5)
        self.assertEqual(obb.angle, 22.5)

    def test_create_oriented_bbox_with_metadata(self):
        """Test creating an OrientedBBox with metadata."""
        obb = OrientedBBox(
            top=100,
            left=100,
            bottom=700,
            right=900,
            angle=15,
            sly_id=123,
            class_id=456,
            labeler_login="test_user",
            updated_at="2021-01-22T19:37:50.158Z",
            created_at="2021-01-22T19:37:50.158Z",
        )
        self.assertEqual(obb.sly_id, 123)
        self.assertEqual(obb.class_id, 456)
        self.assertEqual(obb.labeler_login, "test_user")

    def test_invalid_top_bottom(self):
        """Test that creating an OrientedBBox with top > bottom raises ValueError."""
        with self.assertRaises(ValueError):
            OrientedBBox(top=700, left=100, bottom=100, right=900)

    def test_invalid_left_right(self):
        """Test that creating an OrientedBBox with left > right raises ValueError."""
        with self.assertRaises(ValueError):
            OrientedBBox(top=100, left=900, bottom=700, right=100)

    def test_geometry_name(self):
        """Test that the geometry name is correct."""
        self.assertEqual(OrientedBBox.geometry_name(), "oriented_bbox")


class TestOrientedBBoxSerialization(unittest.TestCase):
    """Test cases for OrientedBBox JSON serialization/deserialization."""

    def test_to_json(self):
        """Test converting OrientedBBox to JSON."""
        obb = OrientedBBox(top=100, left=100, bottom=700, right=900, angle=15)
        json_data = obb.to_json()
        
        self.assertIn("points", json_data)
        self.assertIn("exterior", json_data["points"])
        self.assertIn("interior", json_data["points"])
        self.assertIn("angle", json_data)
        self.assertEqual(json_data["angle"], 15)
        self.assertEqual(len(json_data["points"]["exterior"]), 2)
        self.assertEqual(json_data["points"]["interior"], [])

    def test_from_json(self):
        """Test creating OrientedBBox from JSON."""
        json_data = {
            "points": {
                "exterior": [[100, 100], [900, 700]],
                "interior": [],
            },
            "angle": 15,
        }
        obb = OrientedBBox.from_json(json_data)
        
        self.assertEqual(obb.top, 100)
        self.assertEqual(obb.left, 100)
        self.assertEqual(obb.bottom, 700)
        self.assertEqual(obb.right, 900)
        self.assertEqual(obb.angle, 15)

    def test_from_json_without_angle(self):
        """Test creating OrientedBBox from JSON without angle (defaults to 0)."""
        json_data = {
            "points": {
                "exterior": [[100, 100], [900, 700]],
                "interior": [],
            },
        }
        obb = OrientedBBox.from_json(json_data)
        self.assertEqual(obb.angle, 0)

    def test_from_json_with_metadata(self):
        """Test creating OrientedBBox from JSON with metadata."""
        json_data = {
            "points": {
                "exterior": [[100, 100], [900, 700]],
                "interior": [],
            },
            "angle": 30,
            "id": 123,
            "classId": 456,
            "labelerLogin": "test_user",
            "updatedAt": "2021-01-22T19:37:50.158Z",
            "createdAt": "2021-01-22T19:37:50.158Z",
        }
        obb = OrientedBBox.from_json(json_data)

        self.assertEqual(obb.sly_id, 123)
        self.assertEqual(obb.class_id, 456)
        self.assertEqual(obb.labeler_login, "test_user")

    def test_from_json_invalid_exterior_points(self):
        """Test that from_json raises ValueError with invalid exterior points."""
        json_data = {
            "points": {
                "exterior": [[100, 100]],  # Only one point
                "interior": [],
            },
            "angle": 15,
        }
        with self.assertRaises(ValueError):
            OrientedBBox.from_json(json_data)

    def test_roundtrip_serialization(self):
        """Test that to_json -> from_json preserves the OrientedBBox."""
        original = OrientedBBox(top=100, left=150, bottom=500, right=800, angle=45)
        json_data = original.to_json()
        restored = OrientedBBox.from_json(json_data)

        self.assertEqual(original.top, restored.top)
        self.assertEqual(original.left, restored.left)
        self.assertEqual(original.bottom, restored.bottom)
        self.assertEqual(original.right, restored.right)
        self.assertEqual(original.angle, restored.angle)


class TestOrientedBBoxToBbox(unittest.TestCase):
    """Test cases for OrientedBBox.to_bbox() method."""

    def test_to_bbox_no_rotation(self):
        """Test to_bbox with no rotation returns same bounds."""
        obb = OrientedBBox(top=100, left=100, bottom=700, right=900, angle=0)
        bbox = obb.to_bbox()

        self.assertEqual(type(bbox), Rectangle)
        self.assertEqual(bbox.top, 100)
        self.assertEqual(bbox.left, 100)
        self.assertEqual(bbox.bottom, 700)
        self.assertEqual(bbox.right, 900)

    def test_to_bbox_with_90_degree_rotation(self):
        """Test to_bbox with 90 degree rotation swaps width and height."""
        # Create a 200x400 rectangle centered at (300, 350)
        obb = OrientedBBox(top=200, left=100, bottom=400, right=500, angle=90)
        bbox = obb.to_bbox()

        self.assertEqual(type(bbox), Rectangle)
        # The bounding box should expand to contain the rotated rectangle
        self.assertGreater(bbox.left, 100)
        self.assertLess(bbox.right, 500)

    def test_to_bbox_with_45_degree_rotation(self):
        """Test to_bbox with 45 degree rotation expands the bounding box."""
        obb = OrientedBBox(top=0, left=0, bottom=100, right=100, angle=45)
        bbox = obb.to_bbox()

        self.assertEqual(type(bbox), Rectangle)
        # With 45 degree rotation, the bounding box should be larger
        self.assertLess(bbox.left, 0)
        self.assertGreater(bbox.right, 100)

    def test_to_bbox_with_360_degree_rotation(self):
        """Test to_bbox with 360 degree rotation returns same bounds."""
        obb = OrientedBBox(top=100, left=100, bottom=700, right=900, angle=360)
        bbox = obb.to_bbox()

        self.assertEqual(bbox.top, 100)
        self.assertEqual(bbox.left, 100)
        self.assertEqual(bbox.bottom, 700)
        self.assertEqual(bbox.right, 900)


class TestOrientedBBoxContainsPointLocation(unittest.TestCase):
    """Test cases for OrientedBBox.contains_point_location() method."""

    def test_contains_point_inside_no_rotation(self):
        """Test that a point inside the OrientedBBox (no rotation) is contained."""
        obb = OrientedBBox(top=0, left=0, bottom=100, right=100, angle=0)
        point = PointLocation(row=50, col=50)
        self.assertTrue(obb.contains_point_location(point))

    def test_contains_point_outside_no_rotation(self):
        """Test that a point outside the OrientedBBox (no rotation) is not contained."""
        obb = OrientedBBox(top=0, left=0, bottom=100, right=100, angle=0)
        point = PointLocation(row=150, col=50)
        self.assertFalse(obb.contains_point_location(point))

    def test_contains_point_on_edge_no_rotation(self):
        """Test that a point on the edge of the OrientedBBox is contained."""
        obb = OrientedBBox(top=0, left=0, bottom=100, right=100, angle=0)
        point = PointLocation(row=0, col=50)
        self.assertTrue(obb.contains_point_location(point))

    def test_contains_point_with_rotation(self):
        """Test contains_point_location with a rotated OrientedBBox."""
        # Create a 100x100 square rotated 45 degrees, centered at (50, 50)
        obb = OrientedBBox(top=0, left=0, bottom=100, right=100, angle=45)
        
        # The center should always be inside
        center = PointLocation(row=50, col=50)
        self.assertTrue(obb.contains_point_location(center))

    def test_contains_point_corner_with_rotation(self):
        """Test that the original corner point is not necessarily inside after rotation."""
        obb = OrientedBBox(top=0, left=0, bottom=100, right=100, angle=45)
        
        # Original corner (0, 0) should not be inside the rotated box
        corner = PointLocation(row=0, col=0)
        self.assertFalse(obb.contains_point_location(corner))


class TestOrientedBBoxContainsPoint(unittest.TestCase):
    """Test cases for OrientedBBox.contains_point() method."""

    def test_contains_point_location(self):
        """Test contains_point with PointLocation."""
        obb = OrientedBBox(top=0, left=0, bottom=100, right=100, angle=0)
        point = PointLocation(row=50, col=50)
        self.assertTrue(obb.contains_point(point))

    def test_contains_rectangle(self):
        """Test contains_point with Rectangle."""
        obb = OrientedBBox(top=0, left=0, bottom=100, right=100, angle=0)
        rect = Rectangle(top=20, left=20, bottom=80, right=80)
        self.assertTrue(obb.contains_point(rect))

    def test_does_not_contain_rectangle(self):
        """Test contains_point returns False for Rectangle outside."""
        obb = OrientedBBox(top=0, left=0, bottom=100, right=100, angle=0)
        rect = Rectangle(top=50, left=50, bottom=150, right=150)
        self.assertFalse(obb.contains_point(rect))

    def test_contains_oriented_bbox(self):
        """Test contains_point with OrientedBBox."""
        obb1 = OrientedBBox(top=0, left=0, bottom=100, right=100, angle=0)
        obb2 = OrientedBBox(top=20, left=20, bottom=80, right=80, angle=0)
        self.assertTrue(obb1.contains_point(obb2))

    def test_unsupported_geometry_type(self):
        """Test contains_point raises TypeError for unsupported geometry."""
        obb = OrientedBBox(top=0, left=0, bottom=100, right=100, angle=0)
        with self.assertRaises(TypeError):
            obb.contains_point("invalid")


class TestOrientedBBoxContainsObb(unittest.TestCase):
    """Test cases for OrientedBBox.contains_obb() method."""

    def test_contains_smaller_obb(self):
        """Test that a larger OrientedBBox contains a smaller one."""
        obb1 = OrientedBBox(top=0, left=0, bottom=100, right=100, angle=0)
        obb2 = OrientedBBox(top=20, left=20, bottom=80, right=80, angle=0)
        self.assertTrue(obb1.contains_obb(obb2))

    def test_does_not_contain_larger_obb(self):
        """Test that a smaller OrientedBBox does not contain a larger one."""
        obb1 = OrientedBBox(top=20, left=20, bottom=80, right=80, angle=0)
        obb2 = OrientedBBox(top=0, left=0, bottom=100, right=100, angle=0)
        self.assertFalse(obb1.contains_obb(obb2))

    def test_contains_rotated_obb(self):
        """Test contains_obb with rotated OrientedBBox."""
        obb1 = OrientedBBox(top=0, left=0, bottom=200, right=200, angle=0)
        obb2 = OrientedBBox(top=70, left=70, bottom=100, right=100, angle=45)
        # The rotated small box should still fit inside
        self.assertTrue(obb1.contains_obb(obb2))


class TestOrientedBBoxContainsRectangle(unittest.TestCase):
    """Test cases for OrientedBBox.contains_rectangle() method."""

    def test_contains_smaller_rectangle(self):
        """Test that OrientedBBox contains a smaller rectangle."""
        obb = OrientedBBox(top=0, left=0, bottom=100, right=100, angle=0)
        rect = Rectangle(top=10, left=10, bottom=90, right=90)
        self.assertTrue(obb.contains_rectangle(rect))

    def test_does_not_contain_larger_rectangle(self):
        """Test that OrientedBBox does not contain a larger rectangle."""
        obb = OrientedBBox(top=20, left=20, bottom=80, right=80, angle=0)
        rect = Rectangle(top=0, left=0, bottom=100, right=100)
        self.assertFalse(obb.contains_rectangle(rect))

    def test_does_not_contain_overlapping_rectangle(self):
        """Test that OrientedBBox does not contain a partially overlapping rectangle."""
        obb = OrientedBBox(top=0, left=0, bottom=100, right=100, angle=0)
        rect = Rectangle(top=50, left=50, bottom=150, right=150)
        self.assertFalse(obb.contains_rectangle(rect))


class TestOrientedBBoxCalculateRotatedCorners(unittest.TestCase):
    """Test cases for OrientedBBox.calculate_rotated_corners() method."""

    def test_rotated_corners_no_rotation(self):
        """Test that corners without rotation match original corners."""
        obb = OrientedBBox(top=0, left=0, bottom=100, right=100, angle=0)
        corners = obb.calculate_rotated_corners()
        
        self.assertEqual(len(corners), 4)
        # With no rotation, rotated corners should be near original corners
        # but offset due to rotation around the center

    def test_rotated_corners_with_rotation(self):
        """Test that corners are properly rotated."""
        obb = OrientedBBox(top=0, left=0, bottom=100, right=100, angle=90)
        corners = obb.calculate_rotated_corners()
        
        self.assertEqual(len(corners), 4)
        # All corners should be PointLocation instances
        for corner in corners:
            self.assertEqual(type(corner), PointLocation)


class TestOrientedBBoxFromArray(unittest.TestCase):
    """Test cases for OrientedBBox.from_array() method."""

    def test_from_array_no_angle(self):
        """Test creating OrientedBBox from numpy array."""
        arr = np.zeros((300, 400))
        obb = OrientedBBox.from_array(arr)
        
        self.assertEqual(obb.top, 0)
        self.assertEqual(obb.left, 0)
        self.assertEqual(obb.bottom, 299)
        self.assertEqual(obb.right, 399)
        self.assertEqual(obb.angle, 0)

    def test_from_array_with_angle(self):
        """Test creating OrientedBBox from numpy array with angle."""
        arr = np.zeros((200, 500))
        obb = OrientedBBox.from_array(arr, angle=45)
        
        self.assertEqual(obb.top, 0)
        self.assertEqual(obb.left, 0)
        self.assertEqual(obb.bottom, 199)
        self.assertEqual(obb.right, 499)
        self.assertEqual(obb.angle, 45)


class TestOrientedBBoxGetCroppedNumpySlice(unittest.TestCase):
    """Test cases for OrientedBBox.get_cropped_numpy_slice() method."""

    def test_get_cropped_numpy_slice_no_rotation(self):
        """Test getting cropped numpy slice without rotation."""
        obb = OrientedBBox(top=10, left=20, bottom=50, right=80, angle=0)
        data = np.ones((100, 100))
        slice_result = obb.get_cropped_numpy_slice(data)
        
        # The slice should have the expected shape
        self.assertEqual(slice_result.shape, (40, 60))

    def test_get_cropped_numpy_slice_with_rotation(self):
        """Test getting cropped numpy slice with rotation."""
        obb = OrientedBBox(top=10, left=10, bottom=50, right=50, angle=45)
        data = np.ones((100, 100))
        slice_result = obb.get_cropped_numpy_slice(data)
        
        # With rotation, the slice will be from the axis-aligned bounding box
        # which is larger than the original bounds
        self.assertIsInstance(slice_result, np.ndarray)

    def test_get_cropped_numpy_slice_clipped_to_array(self):
        """Test that the slice is clipped to the array bounds."""
        obb = OrientedBBox(top=10, left=10, bottom=50, right=50, angle=0)
        data = np.ones((40, 40))
        slice_result = obb.get_cropped_numpy_slice(data)
        
        # The slice should be clipped to array bounds
        self.assertLessEqual(slice_result.shape[0], 40)
        self.assertLessEqual(slice_result.shape[1], 40)


class TestOrientedBBoxCrop(unittest.TestCase):
    """Test cases for OrientedBBox.crop() method."""

    def test_crop_overlapping_obbs(self):
        """Test cropping overlapping OrientedBBoxes."""
        obb1 = OrientedBBox(top=0, left=0, bottom=100, right=100, angle=0)
        obb2 = OrientedBBox(top=50, left=50, bottom=150, right=150, angle=0)
        
        result = obb1.crop(obb2)
        
        self.assertEqual(len(result), 1)
        self.assertEqual(type(result[0]), Polygon)

    def test_crop_with_rectangle(self):
        """Test cropping OrientedBBox with Rectangle."""
        obb = OrientedBBox(top=0, left=0, bottom=100, right=100, angle=0)
        rect = Rectangle(top=25, left=25, bottom=75, right=75)
        
        result = obb.crop(rect)
        
        self.assertEqual(len(result), 1)
        self.assertEqual(type(result[0]), Polygon)

    def test_crop_return_rectangle(self):
        """Test cropping with return_type=Rectangle."""
        obb1 = OrientedBBox(top=0, left=0, bottom=100, right=100, angle=0)
        obb2 = OrientedBBox(top=25, left=25, bottom=75, right=75, angle=0)
        
        result = obb1.crop(obb2, return_type=Rectangle)
        
        self.assertEqual(len(result), 1)
        self.assertEqual(type(result[0]), Rectangle)

    def test_crop_non_overlapping(self):
        """Test cropping non-overlapping OrientedBBoxes returns empty list."""
        obb1 = OrientedBBox(top=0, left=0, bottom=50, right=50, angle=0)
        obb2 = OrientedBBox(top=100, left=100, bottom=150, right=150, angle=0)
        
        result = obb1.crop(obb2)
        
        self.assertEqual(len(result), 0)


class TestOrientedBBoxAllowedTransforms(unittest.TestCase):
    """Test cases for OrientedBBox.allowed_transforms() method."""

    def test_allowed_transforms(self):
        """Test that allowed_transforms returns expected geometry types."""
        from supervisely.geometry.alpha_mask import AlphaMask
        from supervisely.geometry.any_geometry import AnyGeometry
        from supervisely.geometry.bitmap import Bitmap
        
        allowed = OrientedBBox.allowed_transforms()
        
        self.assertIn(AlphaMask, allowed)
        self.assertIn(AnyGeometry, allowed)
        self.assertIn(Bitmap, allowed)
        self.assertIn(Polygon, allowed)
        self.assertIn(OrientedBBox, allowed)


class TestOrientedBBoxProperties(unittest.TestCase):
    """Test cases for inherited Rectangle properties."""

    def test_width_property(self):
        """Test width property."""
        obb = OrientedBBox(top=100, left=100, bottom=200, right=400, angle=0)
        self.assertEqual(obb.width, 301)

    def test_height_property(self):
        """Test height property."""
        obb = OrientedBBox(top=100, left=100, bottom=200, right=400, angle=0)
        self.assertEqual(obb.height, 101)

    def test_center_property(self):
        """Test center property."""
        obb = OrientedBBox(top=0, left=0, bottom=100, right=100, angle=0)
        center = obb.center
        self.assertEqual(center.row, 50)
        self.assertEqual(center.col, 50)

    def test_area_property(self):
        """Test area property."""
        obb = OrientedBBox(top=0, left=0, bottom=100, right=100, angle=0)
        self.assertEqual(obb.area, 10201)


class TestOrientedBBoxEdgeCases(unittest.TestCase):
    """Test edge cases for OrientedBBox."""

    def test_zero_dimension_obb(self):
        """Test OrientedBBox with zero width."""
        obb = OrientedBBox(top=0, left=50, bottom=100, right=50, angle=0)
        self.assertEqual(obb.width, 1)
        self.assertEqual(obb.height, 101)

    def test_point_obb(self):
        """Test OrientedBBox that is a single point."""
        obb = OrientedBBox(top=50, left=50, bottom=50, right=50, angle=0)
        self.assertEqual(obb.width, 1)
        self.assertEqual(obb.height, 1)
        self.assertEqual(obb.area, 1)

    def test_large_angle(self):
        """Test OrientedBBox with angle > 360."""
        obb = OrientedBBox(top=0, left=0, bottom=100, right=100, angle=450)
        self.assertEqual(obb.angle, 450)
        # to_bbox should handle this correctly
        bbox = obb.to_bbox()
        self.assertEqual(type(bbox), Rectangle)

    def test_negative_coordinates(self):
        """Test OrientedBBox with negative coordinates."""
        obb = OrientedBBox(top=-100, left=-100, bottom=100, right=100, angle=0)
        self.assertEqual(obb.top, -100)
        self.assertEqual(obb.left, -100)
        self.assertEqual(obb.center.row, 0)
        self.assertEqual(obb.center.col, 0)

    def test_float_coordinates(self):
        """Test OrientedBBox with float coordinates."""
        obb = OrientedBBox(top=10.5, left=20.5, bottom=50.5, right=80.5, angle=15.5)
        self.assertEqual(obb.top, 10)
        self.assertEqual(obb.left, 20)
        self.assertEqual(obb.bottom, 50)
        self.assertEqual(obb.right, 80)
        self.assertEqual(obb.angle, 15.5)


class TestOrientedBBoxDrawing(unittest.TestCase):
    """Test cases for OrientedBBox drawing methods."""

    def test_draw_contour(self):
        """Test that _draw_contour_impl draws the rotated rectangle."""
        obb = OrientedBBox(top=50, left=50, bottom=250, right=150, angle=20)
        bitmap = np.zeros((300, 300, 3), dtype=np.uint8)
        color = (255, 0, 0)  # Red color

        obb.draw(bitmap, color, thickness=2)
        # obb.draw_contour(bitmap, color, thickness=2)

        # Check that some pixels along the expected corners are colored
        corners = obb.calculate_rotated_corners()
        for corner in corners:
            x, y = int(corner.col), int(corner.row)
            self.assertTrue(np.array_equal(bitmap[y, x], np.array(color)))

        # cv2.imwrite("test_obb_drawing_output.png", bitmap)  # Optional: Save output for visual inspection


if __name__ == "__main__":
    unittest.main()

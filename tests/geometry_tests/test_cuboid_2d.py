import inspect
import os
import random
from typing import Dict, List, Tuple, Union

import numpy as np
import pytest  # pylint: disable=import-error

import supervisely.imaging.image as sly_image
from supervisely.annotation.label import Label
from supervisely.annotation.obj_class import ObjClass
from supervisely.geometry.any_geometry import AnyGeometry
from supervisely.geometry.bitmap import Bitmap
from supervisely.geometry.cuboid_2d import (
    CUBOID2D_VERTICES_NAMES,
    VERTICES,
    Cuboid2d,
    Cuboid2dTemplate,
)
from supervisely.geometry.geometry import Geometry
from supervisely.geometry.graph import EDGES, GraphNodes, Node
from supervisely.geometry.image_rotator import ImageRotator
from supervisely.geometry.point import Point
from supervisely.geometry.point_location import PointLocation
from supervisely.geometry.rectangle import Rectangle
from supervisely.io.fs import get_file_name

dir_name = get_file_name(os.path.abspath(__file__))
# Draw Settings
color = [255, 255, 255]
default_color = [255, 255, 255]
thickness = 1


def get_random_image() -> np.ndarray:
    image_shape = (random.randint(801, 2000), random.randint(801, 2000), 3)
    background_color = [0, 0, 0]
    bitmap = np.full(image_shape, background_color, dtype=np.uint8)
    return bitmap


def draw_test(
    dir_name: str,
    test_name: str,
    image: np.ndarray,
    geometry: Geometry = None,
    color: List[int] = default_color,
):
    test_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = f"{test_dir}/test_results/{dir_name}/{test_name}.png"
    if geometry is not None:
        geometry.draw(image, color)
        image_path = f"{test_dir}/test_results/{dir_name}/{test_name}.png"

    sly_image.write(image_path, image)


def get_cuboid_points() -> List[Node]:
    points = [
        PointLocation(random.randint(1, 10), random.randint(1, 10)),
        PointLocation(random.randint(21, 30), random.randint(140, 150)),
        PointLocation(random.randint(151, 161), random.randint(140, 150)),
        PointLocation(random.randint(141, 151), random.randint(1, 10)),
        PointLocation(random.randint(1, 10), random.randint(91, 100)),
        PointLocation(random.randint(1, 10), random.randint(191, 200)),
        PointLocation(random.randint(81, 90), random.randint(191, 200)),
        PointLocation(random.randint(71, 80), random.randint(41, 50)),
    ]
    return points


@pytest.fixture
def random_cuboid2d_int() -> Tuple[Cuboid2d, Dict[int, Node]]:
    nodes = {}
    points = get_cuboid_points()
    for label, point in zip(CUBOID2D_VERTICES_NAMES, points):
        nodes[label] = Node(location=point, label=label)

    coords = {idx: [node.location.row, node.location.col] for idx, node in nodes.items()}
    cuboid_2d = Cuboid2d(nodes)
    return cuboid_2d, nodes, coords


@pytest.fixture
def random_cuboid2d_float() -> Tuple[Cuboid2d, Dict[int, Node]]:
    nodes = {}
    points = [
        PointLocation(round(random.uniform(1, 10), 6), round(random.uniform(1, 10), 6)),
        PointLocation(round(random.uniform(21, 30), 6), round(random.uniform(140, 150), 6)),
        PointLocation(round(random.uniform(151, 161), 6), round(random.uniform(140, 150), 6)),
        PointLocation(round(random.uniform(141, 151), 6), round(random.uniform(1, 10), 6)),
        PointLocation(round(random.uniform(1, 10), 6), round(random.uniform(91, 100), 6)),
        PointLocation(round(random.uniform(1, 10), 6), round(random.uniform(191, 200), 6)),
        PointLocation(round(random.uniform(81, 90), 6), round(random.uniform(191, 200), 6)),
        PointLocation(round(random.uniform(71, 80), 6), round(random.uniform(41, 50), 6)),
    ]
    for label, point in zip(CUBOID2D_VERTICES_NAMES, points):
        nodes[label] = Node(location=point, label=label)

    coords = {idx: [node.location.row, node.location.col] for idx, node in nodes.items()}
    cuboid_2d = Cuboid2d(nodes)
    return cuboid_2d, nodes, coords


def get_cuboid2d_nodes_coords(
    cuboid: Tuple[Cuboid2d, Dict[int, Node], Dict[int, List[Union[int, float]]]]
) -> Tuple[Cuboid2d, Dict[int, Node], Dict[int, List[Union[int, float]]]]:
    return cuboid


def check_cuboids_equal(cuboid_2d1: Cuboid2d, cuboid_2d2: Cuboid2d) -> bool:
    assert isinstance(cuboid_2d1, Cuboid2d)
    assert isinstance(cuboid_2d2, Cuboid2d)
    assert len(cuboid_2d1.nodes) == len(cuboid_2d2.nodes)
    for node1, node2 in zip(cuboid_2d1.nodes.values(), cuboid_2d2.nodes.values()):
        assert node1.location.row == node2.location.row
        assert node1.location.col == node2.location.col
    return True


def test_geometry_name(random_cuboid2d_int, random_cuboid2d_float):
    for idx, cuboid_data in enumerate([random_cuboid2d_int, random_cuboid2d_float], 1):
        cuboid_2d, nodes, coords = cuboid_data
        assert cuboid_2d.geometry_name() == "cuboid_2d"


def test_name(random_cuboid2d_int, random_cuboid2d_float):
    for idx, cuboid_data in enumerate([random_cuboid2d_int, random_cuboid2d_float], 1):
        cuboid_2d, nodes, coords = cuboid_data
        assert cuboid_2d.name() == "cuboid_2d"


def test_to_json(random_cuboid2d_int, random_cuboid2d_float):
    for idx, cuboid_data in enumerate([random_cuboid2d_int, random_cuboid2d_float], 1):
        cuboid_2d, nodes, coords = cuboid_data

        node_json = cuboid_2d.to_json()
        expected_json = {
            "vertices": {
                label: {"loc": [node.location.col, node.location.row]}
                for idx, (label, node) in enumerate(cuboid_2d.nodes.items())
            }
        }

        assert node_json == expected_json


def test_from_json(random_cuboid2d_int, random_cuboid2d_float):
    for idx, cuboid_data in enumerate([random_cuboid2d_int, random_cuboid2d_float], 1):
        cuboid_2d, nodes, coords = cuboid_data

        cuboid_2d_json = {
            "vertices": {
                idx: {"loc": [node.location.col, node.location.row]}
                for idx, node in enumerate(cuboid_2d.nodes.values())
            }
        }
        cuboid_2d_from_json = Cuboid2d.from_json(cuboid_2d_json)
        check_cuboids_equal(cuboid_2d, cuboid_2d_from_json)


def test_crop(random_cuboid2d_int, random_cuboid2d_float):
    for idx, cuboid_data in enumerate([random_cuboid2d_int, random_cuboid2d_float], 1):
        cuboid_2d, nodes, coords = cuboid_data

        min_x = min(coord[0] for coord in coords.values()) - 1
        min_y = min(coord[1] for coord in coords.values()) - 1
        max_x = max(coord[0] for coord in coords.values()) + 1
        max_y = max(coord[1] for coord in coords.values()) + 1

        all_inside_rect = Rectangle(min_x, min_y, max_x, max_y)
        cropped_all_inside = cuboid_2d.crop(all_inside_rect)
        assert len(cropped_all_inside) == 1 and check_cuboids_equal(
            cropped_all_inside[0], cuboid_2d
        ), "Crop failed when all nodes are inside the rectangle"

        no_inside_rect = Rectangle(min_x - 100, min_y - 100, min_x - 50, min_y - 50)
        cropped_no_inside = cuboid_2d.crop(no_inside_rect)
        assert len(cropped_no_inside) == 0, "Crop failed when no nodes are inside the rectangle"

        for cidx, cropped_cuboid_2d in enumerate(cropped_all_inside, 1):
            random_image = get_random_image()
            function_name = inspect.currentframe().f_code.co_name
            draw_test(
                dir_name, f"{function_name}_{cidx}_geometry_{idx}", random_image, cropped_cuboid_2d
            )


def test_relative_crop(random_cuboid2d_int, random_cuboid2d_float):
    for idx, cuboid_data in enumerate([random_cuboid2d_int, random_cuboid2d_float], 1):
        cuboid_2d, nodes, coords = cuboid_data

        min_x = min(coord[0] for coord in coords.values())
        min_y = min(coord[1] for coord in coords.values())
        max_x = max(coord[0] for coord in coords.values())
        max_y = max(coord[1] for coord in coords.values())

        crop_rect = Rectangle(min_x, min_y, max_x, max_y)
        cropped_and_shifted = cuboid_2d.relative_crop(crop_rect)
        for cuboid_2d_node in cropped_and_shifted:
            for node in cuboid_2d_node.nodes.values():
                assert (
                    node.location.col >= 0 and node.location.row >= 0
                ), "Relative crop and shift failed"

        outside_rect = Rectangle(min_x - 100, min_y - 100, min_x - 50, min_y - 50)
        cropped_outside = cuboid_2d.relative_crop(outside_rect)
        assert len(cropped_outside) == 0, "Crop outside cuboid_2d bounds failed"

        for cidx, cropped_cuboid_2d in enumerate(cropped_and_shifted, 1):
            random_image = get_random_image()
            function_name = inspect.currentframe().f_code.co_name
            draw_test(
                dir_name, f"{function_name}_{cidx}_geometry_{idx}", random_image, cropped_cuboid_2d
            )


def test_rotate(random_cuboid2d_int, random_cuboid2d_float):
    for idx, cuboid_data in enumerate([random_cuboid2d_int, random_cuboid2d_float], 1):
        cuboid_2d, nodes, coords = cuboid_data
        random_image = get_random_image()
        img_size, angle = random_image.shape[:2], random.randint(0, 360)
        rotator = ImageRotator(img_size, angle)
        rotated_cuboid_2d = cuboid_2d.rotate(rotator)
        for node, cuboid_2d_node in zip(nodes.values(), rotated_cuboid_2d.nodes.values()):
            rotated_pt = node.location.rotate(rotator)
            assert cuboid_2d_node.location.row == rotated_pt.row
            assert cuboid_2d_node.location.col == rotated_pt.col

        function_name = inspect.currentframe().f_code.co_name
        draw_test(dir_name, f"{function_name}_geometry_{idx}", random_image, rotated_cuboid_2d)


def test_resize(random_cuboid2d_int, random_cuboid2d_float):
    for idx, cuboid_data in enumerate([random_cuboid2d_int, random_cuboid2d_float], 1):
        cuboid_2d, nodes, coords = cuboid_data

        in_size = (300, 400)
        out_size = (600, 800)

        resized_cuboid_2d = cuboid_2d.resize(in_size, out_size)
        for node, cuboid_2d_node in zip(nodes.values(), resized_cuboid_2d.nodes.values()):
            resized_pt = node.location.resize(in_size, out_size)
            assert cuboid_2d_node.location.row == resized_pt.row
            assert cuboid_2d_node.location.col == resized_pt.col

        random_image = get_random_image()
        function_name = inspect.currentframe().f_code.co_name
        draw_test(dir_name, f"{function_name}_geometry_{idx}", random_image, resized_cuboid_2d)


def test_scale(random_cuboid2d_int, random_cuboid2d_float):
    for idx, cuboid_data in enumerate([random_cuboid2d_int, random_cuboid2d_float], 1):
        cuboid_2d, nodes, coords = cuboid_data

        scale_factor = 0.75
        scaled_cuboid_2d = cuboid_2d.scale(scale_factor)
        for node, cuboid_2d_node in zip(nodes.values(), scaled_cuboid_2d.nodes.values()):
            scaled_pt = node.location.scale(scale_factor)
            assert cuboid_2d_node.location.row == scaled_pt.row
            assert cuboid_2d_node.location.col == scaled_pt.col

        random_image = get_random_image()
        function_name = inspect.currentframe().f_code.co_name
        draw_test(dir_name, f"{function_name}_geometry_{idx}", random_image, scaled_cuboid_2d)


def test_translate(random_cuboid2d_int, random_cuboid2d_float):
    for idx, cuboid_data in enumerate([random_cuboid2d_int, random_cuboid2d_float], 1):
        cuboid_2d, nodes, coords = cuboid_data
        drow = 10
        dcol = 20
        translated_cuboid_2d = cuboid_2d.translate(drow, dcol)
        for node, cuboid_2d_node in zip(nodes.values(), translated_cuboid_2d.nodes.values()):
            translated_pt = node.location.translate(drow, dcol)
            assert cuboid_2d_node.location.row == translated_pt.row
            assert cuboid_2d_node.location.col == translated_pt.col

        random_image = get_random_image()
        function_name = inspect.currentframe().f_code.co_name
        draw_test(dir_name, f"{function_name}_geometry_{idx}", random_image, translated_cuboid_2d)


def test_fliplr(random_cuboid2d_int, random_cuboid2d_float):
    for idx, cuboid_data in enumerate([random_cuboid2d_int, random_cuboid2d_float], 1):
        cuboid_2d, nodes, coords = cuboid_data
        random_image = get_random_image()
        img_size = random_image.shape[:2]
        fliplr_cuboid_2d = cuboid_2d.fliplr(img_size)
        for node, cuboid_2d_node in zip(nodes.values(), fliplr_cuboid_2d.nodes.values()):
            fliplr_pt = node.location.fliplr(img_size)
            assert cuboid_2d_node.location.row == fliplr_pt.row
            assert cuboid_2d_node.location.col == fliplr_pt.col

        function_name = inspect.currentframe().f_code.co_name
        draw_test(dir_name, f"{function_name}_geometry_{idx}", random_image, fliplr_cuboid_2d)


def test_flipud(random_cuboid2d_int, random_cuboid2d_float):
    for idx, cuboid_data in enumerate([random_cuboid2d_int, random_cuboid2d_float], 1):
        cuboid_2d, nodes, coords = cuboid_data
        random_image = get_random_image()
        img_size = random_image.shape[:2]
        flipud_cuboid_2d = cuboid_2d.fliplr(img_size)
        for node, cuboid_2d_node in zip(nodes.values(), flipud_cuboid_2d.nodes.values()):
            flipud_pt = node.location.fliplr(img_size)
            assert cuboid_2d_node.location.row == flipud_pt.row
            assert cuboid_2d_node.location.col == flipud_pt.col

        function_name = inspect.currentframe().f_code.co_name
        draw_test(dir_name, f"{function_name}_geometry_{idx}", random_image, flipud_cuboid_2d)


def test__draw_bool_compatible(random_cuboid2d_int, random_cuboid2d_float):
    for idx, cuboid_data in enumerate([random_cuboid2d_int, random_cuboid2d_float], 1):
        cuboid_2d, nodes, coords = cuboid_data
        random_image = get_random_image()
        cuboid_2d._draw_bool_compatible(cuboid_2d._draw_impl, random_image, color, thickness)
        assert np.any(random_image == color)

        function_name = inspect.currentframe().f_code.co_name
        draw_test(dir_name, f"{function_name}_geometry_{idx}", random_image)


def test_draw(random_cuboid2d_int, random_cuboid2d_float):
    for idx, cuboid_data in enumerate([random_cuboid2d_int, random_cuboid2d_float], 1):
        cuboid_2d, nodes, coords = cuboid_data
        random_image = get_random_image()
        cuboid_2d.draw(random_image, color, thickness)
        assert np.any(random_image == color)

        function_name = inspect.currentframe().f_code.co_name
        draw_test(dir_name, f"{function_name}_geometry_{idx}", random_image)


def test_get_mask(random_cuboid2d_int, random_cuboid2d_float):
    for idx, cuboid_data in enumerate([random_cuboid2d_int, random_cuboid2d_float], 1):
        cuboid_2d, nodes, coords = cuboid_data
        random_image = get_random_image()
        mask = cuboid_2d.get_mask(random_image.shape[:2])
        assert mask.shape == random_image.shape[:2]
        assert mask.dtype == np.bool
        assert np.any(mask == True)

        new_bitmap = Bitmap(mask)
        function_name = inspect.currentframe().f_code.co_name
        draw_test(dir_name, f"{function_name}_geometry_{idx}", random_image, new_bitmap)


def test__draw_impl(random_cuboid2d_int, random_cuboid2d_float):
    for idx, cuboid_data in enumerate([random_cuboid2d_int, random_cuboid2d_float], 1):
        cuboid_2d, nodes, coords = cuboid_data
        random_image = get_random_image()
        cuboid_2d._draw_impl(random_image, color, thickness)
        assert np.any(random_image == color)

        function_name = inspect.currentframe().f_code.co_name
        draw_test(dir_name, f"{function_name}_geometry_{idx}", random_image)


def test_draw_contour(random_cuboid2d_int, random_cuboid2d_float):
    for idx, cuboid_data in enumerate([random_cuboid2d_int, random_cuboid2d_float], 1):
        cuboid_2d, nodes, coords = cuboid_data
        random_image = get_random_image()
        cuboid_2d.draw_contour(random_image, color, thickness)
        assert np.any(random_image == color)

        function_name = inspect.currentframe().f_code.co_name
        draw_test(dir_name, f"{function_name}_geometry_{idx}", random_image)


def test__draw_contour_impl(random_cuboid2d_int, random_cuboid2d_float):
    for idx, cuboid_data in enumerate([random_cuboid2d_int, random_cuboid2d_float], 1):
        cuboid_2d, nodes, coords = cuboid_data
        random_image = get_random_image()
        cuboid_2d._draw_contour_impl(random_image, color, thickness)
        assert np.any(random_image == color)

        function_name = inspect.currentframe().f_code.co_name
        draw_test(dir_name, f"{function_name}_geometry_{idx}", random_image)


def test_area(random_cuboid2d_int, random_cuboid2d_float):
    for idx, cuboid_data in enumerate([random_cuboid2d_int, random_cuboid2d_float], 1):
        cuboid_2d, nodes, coords = cuboid_data
        area = cuboid_2d.area
        assert isinstance(area, float)
        assert area >= 0


def test_to_bbox(random_cuboid2d_int, random_cuboid2d_float):
    for idx, cuboid_data in enumerate([random_cuboid2d_int, random_cuboid2d_float], 1):
        cuboid_2d, nodes, coords = cuboid_data
        bbox = cuboid_2d.to_bbox()

        min_x = min(coord[0] for coord in coords.values())
        min_y = min(coord[1] for coord in coords.values())
        max_x = max(coord[0] for coord in coords.values())
        max_y = max(coord[1] for coord in coords.values())

        assert bbox.top == min_x, "Top X coordinate of bbox is incorrect"
        assert bbox.left == min_y, "Left Y coordinate of bbox is incorrect"
        assert bbox.bottom == max_x, "Bottom X coordinate of bbox is incorrect"
        assert bbox.right == max_y, "Right Y coordinate of bbox is incorrect"

        random_image = get_random_image()
        function_name = inspect.currentframe().f_code.co_name
        draw_test(dir_name, f"{function_name}_geometry_{idx}", random_image, bbox)


def test_clone(random_cuboid2d_int, random_cuboid2d_float):
    for idx, cuboid_data in enumerate([random_cuboid2d_int, random_cuboid2d_float], 1):
        cuboid_2d, nodes, coords = cuboid_data
        cloned_cuboid_2d = cuboid_2d.clone()
        check_cuboids_equal(cuboid_2d, cloned_cuboid_2d)


def test_validate(random_cuboid2d_int, random_cuboid2d_float):
    for idx, cuboid_data in enumerate([random_cuboid2d_int, random_cuboid2d_float], 1):
        cuboid_2d, nodes, coords = cuboid_data
        cuboid_2d.validate("cuboid_2d", {"vertices": coords})


def test_config_from_json(random_cuboid2d_int, random_cuboid2d_float):
    for idx, cuboid_data in enumerate([random_cuboid2d_int, random_cuboid2d_float], 1):
        cuboid_2d, nodes, coords = cuboid_data
        config = {"vertices": {}}
        returned_config = cuboid_2d.config_from_json(config)
        assert returned_config == config


def test_config_to_json(random_cuboid2d_int, random_cuboid2d_float):
    for idx, cuboid_data in enumerate([random_cuboid2d_int, random_cuboid2d_float], 1):
        cuboid_2d, nodes, coords = cuboid_data
        config = {"vertices": {}}
        returned_config = cuboid_2d.config_to_json(config)
        assert returned_config == config


def test_allowed_transforms(random_cuboid2d_int, random_cuboid2d_float):
    for idx, cuboid_data in enumerate([random_cuboid2d_int, random_cuboid2d_float], 1):
        cuboid_2d, nodes, coords = cuboid_data
        allowed_transforms = cuboid_2d.allowed_transforms()
        assert set(allowed_transforms) == set([AnyGeometry, Rectangle, GraphNodes])


def test_convert(random_cuboid2d_int, random_cuboid2d_float):
    for idx, cuboid_data in enumerate([random_cuboid2d_int, random_cuboid2d_float], 1):
        cuboid_2d, nodes, coords = cuboid_data
        assert cuboid_2d.convert(type(cuboid_2d)) == [cuboid_2d]
        assert cuboid_2d.convert(AnyGeometry) == [cuboid_2d]
        for new_geometry in cuboid_2d.allowed_transforms():
            converted = cuboid_2d.convert(new_geometry)
            for g in converted:
                assert isinstance(g, new_geometry) or isinstance(g, Cuboid2d)

                random_image = get_random_image()
                function_name = inspect.currentframe().f_code.co_name
                draw_test(
                    dir_name,
                    f"{function_name}_geometry_{idx}_converted_{g.name()}",
                    random_image,
                    g,
                )

        with pytest.raises(
            NotImplementedError,
            match="from {!r} to {!r}".format(cuboid_2d.geometry_name(), Point.geometry_name()),
        ):
            cuboid_2d.convert(Point)


def test_cuboid2d_template():
    template = Cuboid2dTemplate(color)
    assert len(template._config[EDGES]) == 12
    assert len(template._config[VERTICES]) == 8


def test_cuboid2d_label(random_cuboid2d_int, random_cuboid2d_float):
    for idx, cuboid_data in enumerate([random_cuboid2d_int, random_cuboid2d_float], 1):
        cuboid_2d, nodes, coords = cuboid_data
        obj_cls = ObjClass("cuboid_2d", Cuboid2d)

        label = Label(geometry=cuboid_2d, obj_class=obj_cls)
        assert label.geometry == cuboid_2d
        assert label.geometry.geometry_name() == "cuboid_2d" == Cuboid2d.geometry_name()
        assert label.obj_class.geometry_config == Cuboid2dTemplate(obj_cls.color).config

        random_image = get_random_image()
        function_name = inspect.currentframe().f_code.co_name
        draw_test(dir_name, f"{function_name}_geometry_{idx}", random_image, label, obj_cls.color)


def test_10_vertices_cuboid2d():
    nodes = {}
    points = get_cuboid_points()

    for label, point in zip(CUBOID2D_VERTICES_NAMES, points):
        nodes[label] = Node(location=point, label=label)

    # add 2 more vertices
    nodes[8] = Node(location=PointLocation(50, 50), label=8)
    nodes[9] = Node(location=PointLocation(150, 150), label=9)
    with pytest.raises(ValueError, match="Cuboid2d must have exactly 8 vertices"):
        Cuboid2d(nodes)


if __name__ == "__main__":
    pytest.main([__file__])

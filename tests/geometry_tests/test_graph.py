import random
from typing import Dict, List, Tuple, Union

import numpy as np
import pytest

from supervisely.geometry.alpha_mask import AlphaMask
from supervisely.geometry.any_geometry import AnyGeometry
from supervisely.geometry.bitmap import Bitmap
from supervisely.geometry.graph import GraphNodes, Node
from supervisely.geometry.image_rotator import ImageRotator
from supervisely.geometry.point import Point
from supervisely.geometry.point_location import PointLocation, _flip_row_col_order
from supervisely.geometry.polygon import Polygon
from supervisely.geometry.polyline import Polyline
from supervisely.geometry.rectangle import Rectangle

# Draw Settings
color = [255, 255, 255]
thickness = 1


@pytest.fixture
def random_image() -> np.ndarray:
    image_shape = (random.randint(100, 1000), random.randint(100, 1000), 3)
    background_color = [0, 0, 0]
    bitmap = np.full(image_shape, background_color, dtype=np.uint8)
    return bitmap


@pytest.fixture
def random_kp_int() -> Tuple[GraphNodes, Dict[int, Node]]:
    vertex_1 = Node(PointLocation(random.randint(1, 10), random.randint(1, 10)))
    vertex_2 = Node(PointLocation(random.randint(11, 20), random.randint(11, 20)))
    vertex_3 = Node(PointLocation(random.randint(21, 30), random.randint(21, 30)))
    vertex_4 = Node(PointLocation(random.randint(31, 40), random.randint(31, 40)))
    vertex_5 = Node(PointLocation(random.randint(41, 50), random.randint(41, 50)))
    nodes = {0: vertex_1, 1: vertex_2, 2: vertex_3, 3: vertex_4, 4: vertex_5}
    coords = {idx: [node.location.row, node.location.col] for idx, node in nodes.items()}
    graph = GraphNodes(nodes)
    return graph, nodes, coords


@pytest.fixture
def random_kp_float() -> Tuple[GraphNodes, Dict[int, Node]]:
    vertex_1 = Node(PointLocation(round(random.uniform(1, 10), 6), round(random.uniform(1, 10), 6)))
    vertex_2 = Node(
        PointLocation(round(random.uniform(11, 20), 6), round(random.uniform(11, 20), 6))
    )
    vertex_3 = Node(
        PointLocation(round(random.uniform(21, 30), 6), round(random.uniform(21, 30), 6))
    )
    vertex_4 = Node(
        PointLocation(round(random.uniform(31, 40), 6), round(random.uniform(31, 40), 6))
    )
    vertex_5 = Node(
        PointLocation(round(random.uniform(41, 50), 6), round(random.uniform(41, 50), 6))
    )
    nodes = {0: vertex_1, 1: vertex_2, 2: vertex_3, 3: vertex_4, 4: vertex_5}
    coords = {idx: [node.location.row, node.location.col] for idx, node in nodes.items()}
    graph = GraphNodes(nodes)
    return graph, nodes, coords


def get_graph_nodes_coords(
    graph: Tuple[GraphNodes, Dict[int, Node], Dict[int, List[Union[int, float]]]]
) -> Tuple[GraphNodes, Dict[int, Node], Dict[int, List[Union[int, float]]]]:
    return graph


def check_graphs_equal(graph1: GraphNodes, graph2: GraphNodes) -> bool:
    assert isinstance(graph1, GraphNodes)
    assert isinstance(graph2, GraphNodes)
    assert len(graph1.nodes) == len(graph2.nodes)
    for node1, node2 in zip(graph1.nodes.values(), graph2.nodes.values()):
        assert node1.location.row == node2.location.row
        assert node1.location.col == node2.location.col
    return True


def test_geometry_name(random_kp_int, random_kp_float, random_image):
    for graph_nodes in [random_kp_int, random_kp_float]:
        graph, nodes, coords = get_graph_nodes_coords(graph_nodes)
        assert graph.geometry_name() == "graph"


def test_name(random_kp_int, random_kp_float, random_image):
    for graph_nodes in [random_kp_int, random_kp_float]:
        graph, nodes, coords = get_graph_nodes_coords(graph_nodes)
        assert graph.name() == "graph"


def test_to_json(random_kp_int, random_kp_float, random_image):
    for graph_nodes in [random_kp_int, random_kp_float]:
        graph, nodes, coords = get_graph_nodes_coords(graph_nodes)

        node_json = graph.to_json()
        expected_json = {
            "nodes": {
                idx: {"loc": [node.location.col, node.location.row]}
                for idx, node in enumerate(graph.nodes.values())
            }
        }

        assert node_json == expected_json


def test_from_json(random_kp_int, random_kp_float, random_image):
    for graph_nodes in [random_kp_int, random_kp_float]:
        graph, nodes, coords = get_graph_nodes_coords(graph_nodes)

        graph_json = {
            "nodes": {
                idx: {"loc": [node.location.col, node.location.row]}
                for idx, node in enumerate(graph.nodes.values())
            }
        }
        graph_from_json = GraphNodes.from_json(graph_json)
        check_graphs_equal(graph, graph_from_json)


def test_crop(random_kp_int, random_kp_float, random_image):
    for graph_nodes in [random_kp_int, random_kp_float]:
        graph, nodes, coords = get_graph_nodes_coords(graph_nodes)

        min_x = min(coord[0] for coord in coords.values()) - 1
        min_y = min(coord[1] for coord in coords.values()) - 1
        max_x = max(coord[0] for coord in coords.values()) + 1
        max_y = max(coord[1] for coord in coords.values()) + 1

        all_inside_rect = Rectangle(min_x, min_y, max_x, max_y)
        cropped_all_inside = graph.crop(all_inside_rect)
        assert len(cropped_all_inside) == 1 and check_graphs_equal(
            cropped_all_inside[0], graph
        ), "Crop failed when all nodes are inside the rectangle"

        no_inside_rect = Rectangle(min_x - 100, min_y - 100, min_x - 50, min_y - 50)
        cropped_no_inside = graph.crop(no_inside_rect)
        assert len(cropped_no_inside) == 0, "Crop failed when no nodes are inside the rectangle"


def test_relative_crop(random_kp_int, random_kp_float, random_image):
    for graph_nodes in [random_kp_int, random_kp_float]:
        graph, nodes, coords = get_graph_nodes_coords(graph_nodes)

        min_x = min(coord[0] for coord in coords.values())
        min_y = min(coord[1] for coord in coords.values())
        max_x = max(coord[0] for coord in coords.values())
        max_y = max(coord[1] for coord in coords.values())

        crop_rect = Rectangle(min_x, min_y, max_x, max_y)
        cropped_and_shifted = graph.relative_crop(crop_rect)
        for graph_node in cropped_and_shifted:
            for node in graph_node.nodes.values():
                assert (
                    node.location.col >= 0 and node.location.row >= 0
                ), "Relative crop and shift failed"

        outside_rect = Rectangle(min_x - 100, min_y - 100, min_x - 50, min_y - 50)
        cropped_outside = graph.relative_crop(outside_rect)
        assert len(cropped_outside) == 0, "Crop outside graph bounds failed"


def test_rotate(random_kp_int, random_kp_float, random_image):
    for graph_nodes in [random_kp_int, random_kp_float]:
        graph, nodes, coords = get_graph_nodes_coords(graph_nodes)
        img_size, angle = random_image.shape[:2], random.randint(0, 360)
        rotator = ImageRotator(img_size, angle)
        rotated_graph = graph.rotate(rotator)
        for node, graph_node in zip(nodes.values(), rotated_graph.nodes.values()):
            rotated_pt = node.location.rotate(rotator)
            assert graph_node.location.row == rotated_pt.row
            assert graph_node.location.col == rotated_pt.col


def test_resize(random_kp_int, random_kp_float, random_image):
    for graph_nodes in [random_kp_int, random_kp_float]:
        graph, nodes, coords = get_graph_nodes_coords(graph_nodes)

        in_size = (300, 400)
        out_size = (600, 800)

        resized_graph = graph.resize(in_size, out_size)
        for node, graph_node in zip(nodes.values(), resized_graph.nodes.values()):
            resized_pt = node.location.resize(in_size, out_size)
            assert graph_node.location.row == resized_pt.row
            assert graph_node.location.col == resized_pt.col


def test_scale(random_kp_int, random_kp_float, random_image):
    for graph_nodes in [random_kp_int, random_kp_float]:
        graph, nodes, coords = get_graph_nodes_coords(graph_nodes)

        scale_factor = 0.75
        scaled_graph = graph.scale(scale_factor)
        for node, graph_node in zip(nodes.values(), scaled_graph.nodes.values()):
            scaled_pt = node.location.scale(scale_factor)
            assert graph_node.location.row == scaled_pt.row
            assert graph_node.location.col == scaled_pt.col


def test_translate(random_kp_int, random_kp_float, random_image):
    for graph_nodes in [random_kp_int, random_kp_float]:
        graph, nodes, coords = get_graph_nodes_coords(graph_nodes)
        drow = 10
        dcol = 20
        translated_graph = graph.translate(drow, dcol)
        for node, graph_node in zip(nodes.values(), translated_graph.nodes.values()):
            translated_pt = node.location.translate(drow, dcol)
            assert graph_node.location.row == translated_pt.row
            assert graph_node.location.col == translated_pt.col


def test_fliplr(random_kp_int, random_kp_float, random_image):
    img_size = (300, 400)
    for graph_nodes in [random_kp_int, random_kp_float]:
        graph, nodes, coords = get_graph_nodes_coords(graph_nodes)
        fliplr_graph = graph.fliplr(img_size)
        for node, graph_node in zip(nodes.values(), fliplr_graph.nodes.values()):
            fliplr_pt = node.location.fliplr(img_size)
            assert graph_node.location.row == fliplr_pt.row
            assert graph_node.location.col == fliplr_pt.col


def test_flipud(random_kp_int, random_kp_float, random_image):
    img_size = (300, 400)
    for graph_nodes in [random_kp_int, random_kp_float]:
        graph, nodes, coords = get_graph_nodes_coords(graph_nodes)
        flipud_graph = graph.fliplr(img_size)
        for node, graph_node in zip(nodes.values(), flipud_graph.nodes.values()):
            flipud_pt = node.location.fliplr(img_size)
            assert graph_node.location.row == flipud_pt.row
            assert graph_node.location.col == flipud_pt.col


def test__draw_bool_compatible(random_kp_int, random_kp_float, random_image):
    for graph_nodes in [random_kp_int, random_kp_float]:
        graph, nodes, coords = get_graph_nodes_coords(graph_nodes)
        graph._draw_bool_compatible(graph._draw_impl, random_image, 255, 1)
        assert np.any(random_image == color)


def test_draw(random_kp_int, random_kp_float, random_image):
    for graph_nodes in [random_kp_int, random_kp_float]:
        graph, nodes, coords = get_graph_nodes_coords(graph_nodes)
        graph.draw(random_image, 255, 1)
        assert np.any(random_image == color)


def test_get_mask(random_kp_int, random_kp_float, random_image):
    for graph_nodes in [random_kp_int, random_kp_float]:
        graph, nodes, coords = get_graph_nodes_coords(graph_nodes)
        mask = graph.get_mask(random_image.shape[:2])
        assert mask.shape == random_image.shape[:2]
        assert mask.dtype == np.bool
        assert np.any(mask == True)


def test__draw_impl(random_kp_int, random_kp_float, random_image):
    for graph_nodes in [random_kp_int, random_kp_float]:
        graph, nodes, coords = get_graph_nodes_coords(graph_nodes)
        graph._draw_impl(random_image, color, thickness)
        assert np.any(random_image == color)


def test_draw_contour(random_kp_int, random_kp_float, random_image):
    for graph_nodes in [random_kp_int, random_kp_float]:
        graph, nodes, coords = get_graph_nodes_coords(graph_nodes)
        graph.draw_contour(random_image, color, thickness)
        assert np.any(random_image == color)


def test__draw_contour_impl(random_kp_int, random_kp_float, random_image):
    for graph_nodes in [random_kp_int, random_kp_float]:
        graph, nodes, coords = get_graph_nodes_coords(graph_nodes)
        graph._draw_contour_impl(random_image, color, thickness)
        assert np.any(random_image == color)


def test_area(random_kp_int, random_kp_float, random_image):
    for graph_nodes in [random_kp_int, random_kp_float]:
        graph, nodes, coords = get_graph_nodes_coords(graph_nodes)
        area = graph.area
        assert isinstance(area, float)
        assert area >= 0


def test_to_bbox(random_kp_int, random_kp_float, random_image):
    for graph_nodes in [random_kp_int, random_kp_float]:
        graph, nodes, coords = get_graph_nodes_coords(graph_nodes)
        bbox = graph.to_bbox()

        min_x = min(coord[0] for coord in coords.values())
        min_y = min(coord[1] for coord in coords.values())
        max_x = max(coord[0] for coord in coords.values())
        max_y = max(coord[1] for coord in coords.values())

        assert bbox.top == min_x, "Top X coordinate of bbox is incorrect"
        assert bbox.left == min_y, "Left Y coordinate of bbox is incorrect"
        assert bbox.bottom == max_x, "Bottom X coordinate of bbox is incorrect"
        assert bbox.right == max_y, "Right Y coordinate of bbox is incorrect"


def test_clone(random_kp_int, random_kp_float, random_image):
    for graph_nodes in [random_kp_int, random_kp_float]:
        graph, nodes, coords = get_graph_nodes_coords(graph_nodes)
        cloned_graph = graph.clone()
        check_graphs_equal(graph, cloned_graph)


def test_validate(random_kp_int, random_kp_float, random_image):
    for graph_nodes in [random_kp_int, random_kp_float]:
        graph, nodes, coords = get_graph_nodes_coords(graph_nodes)
        graph.validate("graph", {"nodes": coords})


def test_config_from_json(random_kp_int, random_kp_float, random_image):
    for graph_nodes in [random_kp_int, random_kp_float]:
        graph, nodes, coords = get_graph_nodes_coords(graph_nodes)
        config = {"nodes": {}}
        returned_config = graph.config_from_json(config)
        assert returned_config == config


def test_config_to_json(random_kp_int, random_kp_float, random_image):
    for graph_nodes in [random_kp_int, random_kp_float]:
        graph, nodes, coords = get_graph_nodes_coords(graph_nodes)
        config = {"nodes": {}}
        returned_config = graph.config_to_json(config)
        assert returned_config == config


def test_allowed_transforms(random_kp_int, random_kp_float, random_image):
    for graph_nodes in [random_kp_int, random_kp_float]:
        graph, nodes, coords = get_graph_nodes_coords(graph_nodes)
        allowed_transforms = graph.allowed_transforms()
        assert set(allowed_transforms) == set([AnyGeometry, Rectangle])


def test_convert(random_kp_int, random_kp_float, random_image):
    for graph_nodes in [random_kp_int, random_kp_float]:
        graph, nodes, coords = get_graph_nodes_coords(graph_nodes)
        assert graph.convert(type(graph)) == [graph]
        assert graph.convert(AnyGeometry) == [graph]
        for new_geometry in graph.allowed_transforms():
            converted = graph.convert(new_geometry)
            for g in converted:
                assert isinstance(g, new_geometry) or isinstance(g, GraphNodes)

        with pytest.raises(
            NotImplementedError,
            match="from {!r} to {!r}".format(graph.geometry_name(), Point.geometry_name()),
        ):
            graph.convert(Point)


if __name__ == "__main__":
    pytest.main([__file__])

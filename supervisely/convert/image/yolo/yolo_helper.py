from typing import List, Tuple, Union

from supervisely import AnyGeometry, GraphNodes, Polygon, Rectangle, logger
from supervisely.geometry.graph import KeypointsTemplate, Node
from supervisely.imaging.color import generate_rgb

YOLO_DETECTION_COORDS_NUM = 4
YOLO_SEGM_MIN_COORDS_NUM = 6
YOLO_KEYPOINTS_MIN_COORDS_NUM = 6

coco_classes = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


def generate_colors(count: int) -> List[List[int]]:
    colors = []
    for _ in range(count):
        new_color = generate_rgb(colors)
        colors.append(new_color)
    return colors


def get_coordinates(line: str) -> Tuple[int, List[float]]:
    """
    Parse coordinates from a line in the YOLO format.
    """
    class_index = int(line[0])
    coords = list(map(float, line[1:]))
    return class_index, coords


def convert_rectangle(
    img_height: int,
    img_width: int,
    coords: List[float],
    **kwargs,
) -> Rectangle:
    """
    Convert rectangle coordinates from relative (0-1) to absolute (px) values.
    """
    x_center, y_center, ann_width, ann_height = coords
    x_center = float(x_center)
    y_center = float(y_center)
    ann_width = float(ann_width)
    ann_height = float(ann_height)

    px_x_center = x_center * img_width
    px_y_center = y_center * img_height

    px_ann_width = ann_width * img_width
    px_ann_height = ann_height * img_height

    left = int(px_x_center - (px_ann_width / 2))
    right = int(px_x_center + (px_ann_width / 2))

    top = int(px_y_center - (px_ann_height / 2))
    bottom = int(px_y_center + (px_ann_height / 2))

    # check if the coordinates are within the image
    left, top = max(0, left), max(0, top)
    right, bottom = min(img_width, right), min(img_height, bottom)

    return Rectangle(top, left, bottom, right)


def validate_polygon_coords(coords: List[float]) -> List[float]:
    """
    Check and correct polygon coordinates:
    - remove the last point if it is the same as the first one
    """
    if coords[0] == coords[-2] and coords[1] == coords[-1]:
        return coords[:-2]
    return coords


def convert_polygon(
    img_height: int,
    img_width: int,
    coords: List[float],
    **kwargs,
) -> Union[Polygon, None]:
    """
    Convert polygon coordinates from relative (0-1) to absolute (px) values.
    """
    coords = validate_polygon_coords(coords)
    if len(coords) < 6:
        logger.warning("Polygon has less than 3 points. Skipping.")
        return None

    exterior = []
    for i in range(0, len(coords), 2):
        x = coords[i]
        y = coords[i + 1]
        px_x = min(img_width, max(0, int(x * img_width)))
        px_y = min(img_height, max(0, int(y * img_height)))
        exterior.append([px_y, px_x])
    return Polygon(exterior=exterior)


def convert_keypoints(
    img_height: int,
    img_width: int,
    num_keypoints: int,
    num_dims: int,
    coords: List[float],
    **kwargs,
) -> Union[GraphNodes, None]:
    """
    Convert keypoints coordinates from relative (0-1) to absolute (px) values.
    """
    nodes = []
    step = 3 if num_dims == 3 else 2
    shift = 4
    for i in range(shift, num_keypoints + shift, step):
        x = coords[i]
        y = coords[i + 1]
        visibility = int(coords[i + 2]) if num_dims == 3 else 2
        if visibility in [0, 1]:
            continue  # skip invisible keypoints
        px_x = min(img_width, max(0, int(x * img_width)))
        px_y = min(img_height, max(0, int(y * img_height)))
        node = Node(row=px_x, col=px_y)  # , disabled=v)
        nodes.append(node)
    if len(nodes) > 0:
        return GraphNodes(nodes)


def create_geometry_config(num_keypoints: int = None) -> KeypointsTemplate:
    """
    Create a template for keypoints with the specified number of keypoints.
    """
    i, j = 0, 0
    template = KeypointsTemplate()
    for p in list(range(num_keypoints)):
        template.add_point(label=str(p), row=i, col=j)
        j += 1
        i += 1

    return template


def is_applicable_for_rectangles(coords: List[float], **kwargs) -> bool:
    """
    Check if the coordinates are applicable for rectangles.
    """
    return len(coords) == YOLO_DETECTION_COORDS_NUM


def is_applicable_for_polygons(
    with_keypoint: bool,
    coords: List[float],
    **kwargs,
) -> bool:
    """
    Check if the coordinates are applicable for polygons.

    :param with_keypoint: Whether the YAML config file contains keypoints.
    :type with_keypoint: bool
    """
    if with_keypoint:
        return False
    return len(coords) >= YOLO_SEGM_MIN_COORDS_NUM and len(coords) % 2 == 0


def is_applicable_for_keypoints(
    with_keypoint: bool,
    num_keypoints: int,
    num_dims: int,
    coords: List[float],
    **kwargs,
) -> bool:
    """
    Check if the coordinates are applicable for keypoints.
    """
    if not with_keypoint or not num_keypoints or not num_dims:
        return False
    if len(coords) < YOLO_KEYPOINTS_MIN_COORDS_NUM:
        return False
    return len(coords) == num_keypoints * num_dims + 4


APPLICABLE_GEOMETRIES_MAP = {
    Rectangle: is_applicable_for_rectangles,
    Polygon: is_applicable_for_polygons,
    GraphNodes: is_applicable_for_keypoints,
}


def detect_geometry(
    coords: List[float],
    with_keypoint: bool,
    num_keypoints: int,
    num_dims: int,
) -> Union[Rectangle, Polygon, GraphNodes, None]:
    """
    Detect the geometry type based on the coordinates and the configuration.
    """
    for geometry, is_applicable in APPLICABLE_GEOMETRIES_MAP.items():
        if is_applicable(
            with_keypoint=with_keypoint,
            num_keypoints=num_keypoints,
            num_dims=num_dims,
            coords=coords,
        ):
            return geometry


GEOMETRY_CONVERTERS = {
    Rectangle: convert_rectangle,
    Polygon: convert_polygon,
    GraphNodes: convert_keypoints,
}


def get_geometry(
    geometry_type: Union[Rectangle, Polygon, GraphNodes, AnyGeometry],
    img_height: int,
    img_width: int,
    with_keypoint: bool,
    num_keypoints: int,
    num_dims: int,
    coords: List[float],
) -> Union[Rectangle, Polygon, GraphNodes, None]:
    """
    Get the geometry object based on the geometry type.
    """
    if geometry_type not in GEOMETRY_CONVERTERS:
        geometry_type = detect_geometry(
            coords=coords,
            with_keypoint=with_keypoint,
            num_keypoints=num_keypoints,
            num_dims=num_dims,
        )

    if geometry_type is None:
        return None

    return GEOMETRY_CONVERTERS[geometry_type](
        img_height=img_height,
        img_width=img_width,
        coords=coords,
        num_keypoints=num_keypoints,
        num_dims=num_dims,
    )

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


def generate_colors(count):
    colors = []
    for _ in range(count):
        new_color = generate_rgb(colors)
        colors.append(new_color)
    return colors


def get_coordinates(line):
    """
    Parse coordinates from a line in the YOLO format.
    """
    class_index = int(line[0])
    coords = list(map(float, line[1:]))
    return class_index, coords


def convert_rectangle(img_height, img_width, x_center, y_center, ann_width, ann_height):
    """
    Convert rectangle coordinates from relative (0-1) to absolute (px) values.
    """
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


def validate_polygon_coords(coords):
    """
    Check and correct polygon coordinates:
    - remove the last point if it is the same as the first one
    """
    if coords[0] == coords[-2] and coords[1] == coords[-1]:
        return coords[:-2]
    return coords


def convert_polygon(img_height, img_width, *coords):
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


def convert_keypoints(img_height, img_width, num_keypoints, num_dims, *coords):
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


def create_geometry_config(num_keypoints=None):
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


def get_geometry(
    geometry_type, img_height, img_width, with_keypoint, num_keypoints, num_dims, *coords
):
    """
    Convert coordinates from relative (0-1) to absolute (px) values.
    """
    geometry = None
    if geometry_type == Rectangle:
        geometry = convert_rectangle(img_height, img_width, *coords)
    elif geometry_type == Polygon:
        geometry = convert_polygon(img_height, img_width, *coords)
    elif geometry_type == GraphNodes:
        geometry = convert_keypoints(img_height, img_width, num_keypoints, num_dims, *coords)
    elif geometry_type == AnyGeometry:
        if is_applicable_for_rectangles(coords):
            geometry = convert_rectangle(img_height, img_width, *coords)
        elif is_applicable_for_polygons(with_keypoint, coords):
            geometry = convert_polygon(img_height, img_width, *coords)
        elif is_applicable_for_keypoints(with_keypoint, num_keypoints, num_dims, coords):
            geometry = convert_keypoints(img_height, img_width, num_keypoints, num_dims, *coords)
    return geometry


def is_applicable_for_rectangles(coords):
    """
    Check if the coordinates are applicable for rectangles.
    """
    return len(coords) == YOLO_DETECTION_COORDS_NUM


def is_applicable_for_polygons(with_keypoint, coords):
    """
    Check if the coordinates are applicable for polygons.

    :param with_keypoint: Whether the YAML config file contains keypoints.
    :type with_keypoint: bool
    """
    if with_keypoint:
        return False
    return len(coords) >= YOLO_SEGM_MIN_COORDS_NUM and len(coords) % 2 == 0


def is_applicable_for_keypoints(with_keypoint, num_keypoints, num_dims, coords):
    """
    Check if the coordinates are applicable for keypoints.
    """
    if not with_keypoint:
        return False
    if len(coords) < YOLO_KEYPOINTS_MIN_COORDS_NUM:
        return False
    return len(coords) == num_keypoints * num_dims + 4

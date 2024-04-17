from supervisely import GraphNodes, Polygon, Rectangle, logger
from supervisely.geometry.graph import KeypointsTemplate, Node
from supervisely.imaging.color import generate_rgb

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
    class_index = int(line[0])
    coords = list(map(float, line[1:]))
    return class_index, coords


def convert_rectangle(img_height, img_width, x_center, y_center, ann_width, ann_height):
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


def convert_polygon(img_height, img_width, *coords):
    exterior = []
    for i in range(0, len(coords), 2):
        x = coords[i]
        y = coords[i + 1]
        px_x = min(img_width, max(0, int(x * img_width)))
        px_y = min(img_height, max(0, int(y * img_height)))
        exterior.append([px_y, px_x])
    return Polygon(exterior=exterior)


def convert_keypoints(img_height, img_width, num_keypoints, num_dims, *coords):
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
    i, j = 0, 0
    template = KeypointsTemplate()
    for p in list(range(num_keypoints)):
        template.add_point(label=str(p), row=i, col=j)
        j += 1
        i += 1

    return template

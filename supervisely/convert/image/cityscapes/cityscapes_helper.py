from supervisely import Annotation, Label, PointLocation, Polygon, ProjectMeta, logger, Rectangle
from supervisely.io.json import load_json_file
from supervisely.convert.image.image_helper import validate_image_bounds

COLOR_MAP_FILE_NAME = "class_to_id.json"

IMAGE_EXT = ".png"

IMAGE_DIR_NAME = "leftImg8bit"
ANNOTATION_DIR_NAME = "gtFine"

ANNOTATIONS_NAME_ENDING = "gtFine_polygon"
MASKS_MACHINE_NAME_ENDING = "gtFine_labelIds"
MASKS_INSTANCE_NAME_ENDING = "gtFine_color"

TRAINVAL_TAG = "trainval"
TRAIN_TAG = "train"
VAL_TAG = "val"
TEST_TAG = "test"

CITYSCAPES_CLASSES_TO_COLORS_MAP = {
    "unlabeled": (0, 0, 0),
    "ego vehicle": (98, 15, 138),
    "rectification border": (15, 120, 55),
    "out of roi": (125, 138, 15),
    "static": (63, 15, 138),
    "dynamic": (111, 74, 0),
    "ground": (81, 0, 81),
    "road": (128, 64, 128),
    "sidewalk": (244, 35, 232),
    "parking": (250, 170, 160),
    "rail track": (230, 150, 140),
    "building": (70, 70, 70),
    "wall": (102, 102, 156),
    "fence": (190, 153, 153),
    "guard rail": (180, 165, 180),
    "bridge": (150, 100, 100),
    "tunnel": (150, 120, 90),
    "pole": (153, 153, 153),
    "polegroup": (153, 153, 153),
    "traffic light": (250, 170, 30),
    "traffic sign": (220, 220, 0),
    "vegetation": (107, 142, 35),
    "terrain": (152, 251, 152),
    "sky": (70, 130, 180),
    "person": (220, 20, 60),
    "rider": (255, 0, 0),
    "car": (0, 0, 142),
    "truck": (0, 0, 70),
    "bus": (0, 60, 100),
    "caravan": (0, 0, 90),
    "trailer": (0, 0, 110),
    "train": (0, 80, 100),
    "motorcycle": (0, 0, 230),
    "bicycle": (119, 11, 32),
    "license plate": (0, 0, 142),
}

CITYSCAPES_COLORS = list(CITYSCAPES_CLASSES_TO_COLORS_MAP.values())


def convert_points(simple_points):
    return [PointLocation(int(p[1]), int(p[0])) for p in simple_points]


def create_ann_from_file(
    ann: Annotation, ann_path: str, meta: ProjectMeta, renamed_classes: dict
) -> Annotation:
    ann_data = load_json_file(ann_path)
    labels = []
    for obj in ann_data["objects"]:
        class_name = obj["label"]
        class_name = renamed_classes.get(class_name, class_name)
        if class_name == "out of roi":
            polygon = obj["polygon"][:5]
            interiors = [obj["polygon"][5:]]
        else:
            polygon = obj["polygon"]
            if len(polygon) < 3:
                logger.warn(
                    "Polygon must contain at least 3 points in ann {}, obj_class {}".format(
                        ann_path, class_name
                    )
                )
                continue
            interiors = []
        interiors = [convert_points(interior) for interior in interiors]
        polygon = Polygon(convert_points(polygon), interiors)
        obj_class = meta.get_obj_class(class_name)
        labels.append(Label(polygon, obj_class))
    labels = validate_image_bounds(labels, Rectangle.from_size(ann.img_size))
    ann = ann.add_labels(labels)
    return ann

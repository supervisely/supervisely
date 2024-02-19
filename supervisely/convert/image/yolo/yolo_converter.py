import os

import yaml

from supervisely import logger
from supervisely.convert.base_converter import AvailableImageConverters
from supervisely.convert.image.image_converter import ImageConverter
from supervisely.io.fs import list_files_recursively

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


class YOLOConverter(ImageConverter):

    def __init__(self, input_data):
        self._input_data = input_data
        self._items = []
        self._meta = None

    def __str__(self):
        return AvailableImageConverters.YOLO

    @property
    def ann_ext(self):
        return ".txt"

    @property
    def key_file_ext(self):
        return ".yaml"

    def require_key_file(self):
        return True

    def validate_key_files(self):
        yamls = list_files_recursively(self._input_data, valid_extensions=[".yaml"])
        if len(yamls) == 0:
            return False
        if len(yamls) > 1:
            logger.warn("Found more than one meta file.")
        for key_file in yamls:
            with open(key_file, "r") as config_yaml_info:
                config_yaml = yaml.safe_load(config_yaml_info)
                if "names" in config_yaml:
                    logger.warn(
                        "['names'] key is empty. Class names will be taken from default coco classes names"
                    )
                classes = config_yaml.get("names", coco_classes)
                nc = config_yaml.get("nc", len(coco_classes))
                # result["names"] = config_yaml.get("names", coco_classes)
                # logger.warn("['nc'] key is empty. Number of classes will be taken from default coco classes")
                if "nc" in config_yaml:
                    if int(nc) != len(classes):
                        logger.warn("Number of classes in ['names'] and ['nc'] are different")
                        return False
                if "colors" in config_yaml:
                    if len(config_yaml["colors"]) != len(classes):
                        logger.warn("Number of classes in ['names'] and ['colors'] are different")
                        return False

                for t in ["train", "val"]:
                    if t not in config_yaml:
                        logger.warn(f"{t} path is not defined in {key_file}")
                        return False

        if self._meta is None:
            return False
        return True

    @staticmethod
    def validate_ann_format(ann_path):
        with open(ann_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5 or not all(part.replace(".", "", 1).isdigit() for part in parts):
                    return False
        return True

    def require_key_file(self):
        return True

    def validate_key_file(self, key_path):
        return os.path.isfile(key_path)

    @property
    def get_ann_ext(self):  # ?
        return ".txt"

    def get_meta(self):
        raise NotImplementedError()

    def get_items(self):
        raise NotImplementedError()

    def to_supervisely(self):
        raise NotImplementedError()

    def validate_format(self):
        return False

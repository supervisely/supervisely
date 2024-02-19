import imghdr
import os
import supervisely.convert.image.yolo.yolo_helper as yolo_helper
import yaml

from supervisely import Annotation, Label, ObjClass, Polygon, ProjectMeta, Rectangle, logger
from supervisely.convert.base_converter import AvailableImageConverters
from supervisely.convert.image.image_converter import ImageConverter
from supervisely.io.fs import JUNK_FILES, get_file_ext, get_file_name


class YOLOConverter(ImageConverter):

    def __init__(self, input_data):
        self._input_data = input_data
        self._items = []
        self._meta = None
        self.yaml_info = None
        self.class_index_to_geometry = {}
        self.coco_classes_dict = {}

    def __str__(self):
        return AvailableImageConverters.YOLO

    @property
    def ann_ext(self):
        return ".txt"

    @property
    def key_file_ext(self):
        return ".yaml"


    def validate_ann_file(self, ann_path: str, meta: ProjectMeta):
        try:
            with open(ann_path, "r") as ann_file:
                lines = ann_file.readlines()
                if len(lines) == 0:
                    logger.warn(f"Empty annotation file: {ann_path}")
                    return False
                for line in lines:
                    line = line.strip().split()
                    if len(line) > 0:
                        class_index, coords = yolo_helper.get_coordinates(line)
                        if class_index not in self.coco_classes_dict:
                            logger.warn(f"Class index {class_index} not found in the config yaml file: {ann_path}")
                            return False
                        if any([0 > c > 1 for c in coords]):
                            logger.warn(f"The bounding coordinates must be in normalized xywh format (from 0 to 1): {ann_path}")
                            return False
                        if len(coords) != 4 and (len(coords) % 2 != 0 or len(coords) < 6):
                            logger.warn(f"Invalid coordinates for rectangle or polygon geometry: {ann_path}")
                            return False
                        
                        # collect geometry types for each class
                        if len(coords) == 4:
                            geometry = Rectangle
                        elif len(coords) >= 6 and len(coords) % 2 == 0:
                            geometry = Polygon

                        if class_index not in self.class_index_to_geometry:
                            self.class_index_to_geometry[class_index] = geometry 

            return True
        except:
            return False

    def validate_key_file(self, key_path: str):
        result = {"names": None, "colors": None, "datasets": []}
        try:
            with open(key_path, "r") as config_yaml_info:
                config_yaml = yaml.safe_load(config_yaml_info)
                if "names" not in config_yaml:
                    logger.warn(
                        "['names'] key is empty. Class names will be taken from default coco classes names"
                    )
                classes = config_yaml.get("names", yolo_helper.coco_classes)
                result["names"] = classes

                nc = config_yaml.get("nc", len(classes))
                if nc is not None:
                    if int(nc) != len(classes):
                        logger.warn(
                            "Number of classes in ['names'] and ['nc'] are different. "
                            "Number of classes will be taken from number of classes in ['names']"
                        )
                        nc = len(classes)

                colors = config_yaml.get("colors", [])
                if len(colors) > 0:
                    if len(colors) != len(classes):
                        logger.warn(
                            "Number of classes in ['names'] and ['colors'] are different. "
                            "Colors will be generated automatically"
                        )
                        result["colors"] = yolo_helper.generate_colors(len(classes))
                    else:
                        result["colors"] = colors
                else:
                    result["colors"] = yolo_helper.generate_colors(len(classes))

                conf_dirname = os.path.dirname(key_path)
                for t in ["train", "val"]:
                    if t not in config_yaml:
                        logger.warn(f"{t} path is not defined in {key_path}")
                        # return False
                    if config_yaml[t].startswith(".."):
                        cur_dataset_path = os.path.normpath(
                            os.path.join(conf_dirname, "/".join(config_yaml[t].split("/")[2:]))
                        )
                    else:
                        cur_dataset_path = os.path.normpath(os.path.join(conf_dirname, config_yaml[t]))

                    if os.path.isdir(cur_dataset_path):
                        result["datasets"].append((t, cur_dataset_path))
                
                self.yaml_info = result
                self.coco_classes_dict = {i: classes[i] for i in range(len(classes))}
            return True
        except:
            return False

    def validate_format(self):
        detected_ann_cnt = 0
        config_path = None
        images_list, ann_dict = [], {}
        for root, _, files in os.walk(self._input_data):
            for file in files:
                full_path = os.path.join(root, file)
                ext = get_file_ext(full_path)
                if ext == ".yaml":
                    is_valid = self.validate_key_file(full_path)
                    if is_valid:
                        config_path = full_path
                        continue

                if file in JUNK_FILES:  # add better check
                    continue
                elif ext == self.ann_ext:
                    ann_dict[file] = full_path
                elif imghdr.what(full_path) is None:
                    # logger.info(f"Non-image file found: {full_path}")
                    return False
                else:
                    images_list.append(full_path)

        if config_path is None:
            self.yaml_info = {
                "names": yolo_helper.coco_classes,
                "colors": yolo_helper.generate_colors(len(yolo_helper.coco_classes)),
            }
        meta = ProjectMeta()
        
        # create Items
        self._items = []
        for image_path in images_list:
            item = self.Item(image_path)
            ann_name = None
            if f"{item.name}.txt" in ann_dict:
                ann_name = f"{item.name}.txt"
            elif f"{get_file_name(item.name)}.txt" in ann_dict:
                ann_name = f"{get_file_name(item.name)}.txt"
            if ann_name:
                ann_path = ann_dict[ann_name]
                is_valid = self.validate_ann_file(ann_path, meta)
                if is_valid:
                    item.set_ann_data(ann_path)
                    detected_ann_cnt += 1
            self._items.append(item)
        self._meta = self.generate_meta()
        return detected_ann_cnt > 0

    def generate_meta(self):
        meta = ProjectMeta()

        classes = []
        for class_index, class_name in self.coco_classes_dict.items():
            color = self.yaml_info["colors"][class_index]
            geometry = self.class_index_to_geometry.get(class_index)
            if geometry is None:
                geometry = Rectangle
                self.class_index_to_geometry[class_index] = geometry
            obj_cls = ObjClass(name=class_name, geometry_type=geometry, color=color)
            classes.append(obj_cls)
        meta = meta.add_obj_classes(classes)
        return meta

    def to_supervisely(self, item: ImageConverter.Item, meta: ProjectMeta = None) -> Annotation:
        """Convert to Supervisely format."""
        if meta is None:
            meta = self._meta

        if item.ann_data is None:
            return item.create_empty_annotation()

        try:
            labels = []
            height, width = item.shape
            with open(item.ann_data, "r") as ann_file:
                lines = ann_file.readlines()
                for line in lines:
                    line = line.strip().split()
                    if len(line) > 0:
                        class_index, coords = yolo_helper.get_coordinates(line)
                        if len(coords) == 4:
                            geometry = yolo_helper.convert_rectangle(height, width, *coords)
                        elif len(coords) >= 6 and len(coords) % 2 == 0:
                            geometry = yolo_helper.convert_polygon(height, width, *coords)

                        class_name = self.coco_classes_dict[class_index]
                        obj_class=meta.get_obj_class(class_name)
                        label = Label(obj_class=obj_class, geometry=geometry)
                        labels.append(label)
            return Annotation(labels=labels, img_size=(height, width))

        except Exception as e:
            logger.warn(f"Failed to convert annotation: {repr(e)}")
            return item.create_empty_annotation()

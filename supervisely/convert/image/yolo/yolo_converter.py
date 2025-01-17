import os
import yaml
from typing import Dict, List, Optional, Tuple, Union
from supervisely import (
    Api,
    Annotation,
    AnyGeometry,
    GraphNodes,
    Label,
    ObjClass,
    Polygon,
    ProjectMeta,
    Rectangle,
    logger,
    batched,
    is_development,
    ApiContext,
    generate_free_name,
)
from supervisely.io.json import load_json_file
from supervisely.convert.base_converter import AvailableImageConverters
from supervisely.convert.image.image_converter import ImageConverter
from supervisely.convert.image.yolo import yolo_helper
from supervisely.io.fs import JUNK_FILES, get_file_ext, get_file_name
from supervisely.project.project_settings import LabelingInterface
import magic
import re


class YOLOConverter(ImageConverter):

    class Item:
        def __init__(self, image_path: str):
            self.path = image_path
            self.name = os.path.splitext(os.path.basename(image_path))[0]
            self.ann_data = None
            self.shape = None

        def set_shape(self, shape=None):
            if shape is None:
                # Assuming the shape is determined by reading the image
                t = magic.from_file(self.path)
                if t != "empty":
                    self.shape = re.search("(\d+) x (\d+)", t).groups()
                else:
                    self.shape = [None, None]
            else:
                self.shape = shape

        def create_empty_annotation(self):
            return Annotation(img_size=self.shape)

    def __init__(
        self,
        input_data: str,
        labeling_interface: Optional[Union[LabelingInterface, str]],
        upload_as_links: bool,
        remote_files_map: Optional[Dict[str, str]] = None,
    ):
        super().__init__(input_data, labeling_interface, upload_as_links, remote_files_map)

        self._yaml_info: dict = None
        self._with_keypoint = False
        self._class_index_to_geometry: dict = {}
        self._coco_classes_dict: dict = {}
        self._num_kpts = None
        self._num_dims = None
        self._supports_links = True

    def __str__(self) -> str:
        return AvailableImageConverters.YOLO

    @property
    def ann_ext(self) -> str:
        return ".txt"

    @property
    def key_file_ext(self) -> str:
        return ".yaml"

    def validate_ann_file(self, ann_path: str, meta: ProjectMeta = None) -> bool:
        try:
            ann_name = os.path.basename(ann_path)
            with open(ann_path, "r") as ann_file:
                lines = ann_file.readlines()
                if len(lines) == 0:
                    logger.warn(f"Empty annotation file: {ann_path}")
                    return False
                for idx, line in enumerate(lines, start=1):
                    line = line.strip().split()
                    if len(line) > 0:
                        class_index, coords = yolo_helper.get_coordinates(line)
                        if class_index not in self._coco_classes_dict:
                            logger.warn(
                                f"Class index {class_index} not found in the config yaml file: {ann_path}"
                            )
                            return False
                        if any([0 > c > 1 for c in coords]):
                            logger.warn(
                                f"The bounding coordinates must be in normalized xywh format (from 0 to 1): {ann_path}"
                            )
                            return False
                        if (
                            len(coords) != 4
                            and (len(coords) % 2 != 0 or len(coords) < 6)
                            and not self._with_keypoint
                        ):
                            logger.warn(
                                f"Invalid coordinates for rectangle or polygon geometry: {ann_path}"
                            )
                            return False

                        # collect geometry types for each class
                        geometry = yolo_helper.detect_geometry(
                            coords, self._with_keypoint, self._num_kpts, self._num_dims
                        )
                        if geometry is None:
                            logger.warn(
                                "Invalid coordinates for the class index: "
                                f"FILE [{ann_name}], LINE [{idx}], CLASS [{class_index}]"
                            )
                            return False
                        if class_index not in self._class_index_to_geometry:
                            self._class_index_to_geometry[class_index] = geometry
                            continue
                        existing_geometry = self._class_index_to_geometry[class_index]
                        if geometry != existing_geometry:
                            self._class_index_to_geometry[class_index] = AnyGeometry

            return True
        except:
            return False

    def validate_key_file(self, key_path: str) -> bool:
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
                if "kpt_shape" in config_yaml:
                    kpt_shape = config_yaml.get("kpt_shape")
                    if not isinstance(kpt_shape, list):
                        return False
                    self._with_keypoint = True
                    self._num_kpts = int(kpt_shape[0])
                    self._num_dims = int(kpt_shape[1])
                    result["kpt_shape"] = kpt_shape

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
                self._yaml_info = result
                self._coco_classes_dict = {i: classes[i] for i in range(len(classes))}
            return True
        except:
            return False

    def validate_format(self) -> bool:
        if self.upload_as_links and self._supports_links:
            # todo: check if annotation files exist, and download images if so
            self._download_remote_files(download_images=True)
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
                elif self.is_image(full_path):
                    images_list.append(full_path)

        if config_path is None:
            self._yaml_info = {
                "names": yolo_helper.coco_classes,
                "colors": yolo_helper.generate_colors(len(yolo_helper.coco_classes)),
            }
            self._coco_classes_dict = {i: c for i, c in enumerate(yolo_helper.coco_classes)}

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
                is_valid = self.validate_ann_file(ann_path)
                if is_valid:
                    item.ann_data = ann_path
                    detected_ann_cnt += 1
            self._items.append(item)

        self._meta = self.generate_meta()
        return detected_ann_cnt > 0

    def generate_meta(self) -> ProjectMeta:

        meta = ProjectMeta()

        classes = []
        for class_index, class_name in self._coco_classes_dict.items():
            geometry_config = None
            color = self._yaml_info["colors"][class_index]
            geometry = self._class_index_to_geometry.get(class_index)
            if geometry is None:
                geometry = Rectangle
                self._class_index_to_geometry[class_index] = geometry
            if geometry == GraphNodes:
                geometry_config = yolo_helper.create_geometry_config(self._num_kpts)
            obj_cls = ObjClass(
                name=class_name,
                geometry_type=geometry,
                color=color,
                geometry_config=geometry_config,
            )
            classes.append(obj_cls)
        meta = meta.add_obj_classes(classes)
        return meta

    def upload_dataset(
        self,
        api: Api,
        dataset_id: int,
        batch_size: int = 50,
        log_progress=True,
        entities: List[Item] = None,
        progress_cb=None,
    ) -> None:
        """Upload converted data to Supervisely"""
        dataset_info = api.dataset.get_info_by_id(dataset_id, raise_error=True)
        project_id = dataset_info.project_id

        meta, renamed_classes, renamed_tags = self.merge_metas_with_conflicts(api, dataset_id)

        existing_names = set([img.name for img in api.image.get_list(dataset_id)])
        progress = None
        if progress_cb is not None:
            log_progress = True
        elif log_progress:
            progress, progress_cb = self.get_progress(self.items_count, "Uploading")

        if self.upload_as_links:
            batch_size = 1000

        for batch in batched(entities or self._items, batch_size=batch_size):
            item_names = []
            item_paths = []
            item_metas = []
            anns = []
            for item in batch:
                item.set_shape()
                item.path = self.validate_image(item.path)
                if item.path is None:
                    continue  # image has failed validation
                item.name = f"{get_file_name(item.path)}{get_file_ext(item.path).lower()}"
                if self.upload_as_links and not self.supports_links:
                    ann = None
                else:
                    ann = self.to_supervisely(item, meta, renamed_classes, renamed_tags)
                name = generate_free_name(
                    existing_names, item.name, with_ext=True, extend_used_names=True
                )
                item_names.append(name)
                item_paths.append(item.path)

                if isinstance(item.meta, str):  # path to file
                    item_metas.append(load_json_file(item.meta))
                elif isinstance(item.meta, dict):
                    item_metas.append(item.meta)
                else:
                    item_metas.append({})

                if ann is not None:
                    anns.append(ann)

            with ApiContext(
                api=api, project_id=project_id, dataset_id=dataset_id, project_meta=meta
            ):
                if self.upload_as_links:
                    img_infos = api.image.upload_links(
                        dataset_id,
                        item_names,
                        item_paths,
                        metas=item_metas,
                        batch_size=batch_size,
                        conflict_resolution="rename",
                        force_metadata_for_links=False,
                    )
                else:
                    img_infos = api.image.upload_paths(
                        dataset_id,
                        item_names,
                        item_paths,
                        metas=item_metas,
                        conflict_resolution="rename",
                    )

                img_ids = [img_info.id for img_info in img_infos]
                if len(anns) == len(img_ids):
                    api.annotation.upload_anns(
                        img_ids, anns, skip_bounds_validation=self.upload_as_links
                    )

            if log_progress:
                progress_cb(len(batch))

        if log_progress:
            if is_development() and progress is not None:
                progress.close()
        logger.info(
            f"Dataset has been successfully uploaded → {dataset_info.name}, ID:{dataset_id}"
        )

    def to_supervisely(
        self,
        item: ImageConverter.Item,
        meta: ProjectMeta = None,
        renamed_classes: dict = None,
        renamed_tags: dict = None,
    ) -> Annotation:
        """Convert to Supervisely format."""
        if meta is None:
            meta = self._meta

        if item.ann_data is None:
            # if self._upload_as_links:
            # item.set_shape([None, None])
            return item.create_empty_annotation()

        try:
            labels = []
            # item.set_shape()
            height, width = item.shape
            with open(item.ann_data, "r") as ann_file:
                lines = ann_file.readlines()
                for line in lines:
                    line = line.strip().split()
                    if len(line) > 0:
                        class_index, coords = yolo_helper.get_coordinates(line)
                        geometry_type = self._class_index_to_geometry.get(class_index)
                        if geometry_type is None:
                            continue
                        geometry = yolo_helper.get_geometry(
                            geometry_type,
                            height,
                            width,
                            self._with_keypoint,
                            self._num_kpts,
                            self._num_dims,
                            coords,
                        )
                        if geometry is None:
                            continue

                        class_name = self._coco_classes_dict[class_index]
                        if renamed_classes is not None:
                            if class_name in renamed_classes:
                                class_name = renamed_classes[class_name]
                        obj_class = meta.get_obj_class(class_name)
                        label = Label(obj_class=obj_class, geometry=geometry)
                        labels.append(label)
            return Annotation(labels=labels, img_size=(height, width))

        except Exception as e:
            logger.warn(f"Failed to convert annotation: {repr(e)}")
            return item.create_empty_annotation()

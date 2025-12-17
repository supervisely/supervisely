import os
import shutil
from pathlib import Path
from typing import Callable, Dict, List, Optional, OrderedDict, Tuple, Union

import numpy as np
from PIL import Image
from tqdm import tqdm

from supervisely._utils import generate_free_name
from supervisely.annotation.annotation import Annotation
from supervisely.annotation.label import Label
from supervisely.annotation.obj_class import ObjClass
from supervisely.annotation.obj_class_collection import ObjClassCollection
from supervisely.annotation.tag import Tag, TagValueType
from supervisely.annotation.tag_meta import TagApplicableTo, TagMeta
from supervisely.convert.image.image_helper import validate_image_bounds
from supervisely.geometry.bitmap import Bitmap
from supervisely.geometry.polygon import Polygon
from supervisely.geometry.rectangle import Rectangle
from supervisely.imaging.color import generate_rgb
from supervisely.imaging.image import read
from supervisely.io.fs import file_exists, get_file_ext, get_file_name
from supervisely.io.json import load_json_file
from supervisely.project.project import Dataset, OpenMode, Project
from supervisely.project.project_meta import ProjectMeta
from supervisely.sly_logger import logger
from supervisely.task.progress import tqdm_sly

MASKS_EXTENSION = ".png"

# Export
SUPPORTED_GEOMETRY_TYPES = {Bitmap, Polygon, Rectangle}
VALID_IMG_EXT = {".jpe", ".jpeg", ".jpg"}
TRAIN_TAG_NAME = "train"
VAL_TAG_NAME = "val"
TRAINVAL_TAG_NAME = "trainval"
DEFAULT_OBJECT_FIELDS = {"name", "class", "bndbox"}
DEFAULT_SUBCLASSES = {"pose", "truncated", "difficult", "occluded", "obstacle", "out-of-scope"}


default_classes_colors = {
    "neutral": (224, 224, 192),
    "aeroplane": (128, 0, 0),
    "bicycle": (0, 128, 0),
    "bird": (128, 128, 0),
    "boat": (0, 0, 128),
    "bottle": (128, 0, 128),
    "bus": (0, 128, 128),
    "car": (128, 128, 128),
    "cat": (64, 0, 0),
    "chair": (192, 0, 0),
    "cow": (64, 128, 0),
    "diningtable": (192, 128, 0),
    "dog": (64, 0, 128),
    "horse": (192, 0, 128),
    "motorbike": (64, 128, 128),
    "person": (192, 128, 128),
    "pottedplant": (0, 64, 0),
    "sheep": (128, 64, 0),
    "sofa": (0, 192, 0),
    "train": (128, 192, 0),
    "tvmonitor": (0, 64, 128),
}


# returns mapping: (r, g, b) color -> some (row, col) for each unique color except black
def get_col2coord(img: np.ndarray) -> dict:
    img = img.astype(np.int32)
    h, w = img.shape[:2]
    colhash = img[:, :, 0] * 256 * 256 + img[:, :, 1] * 256 + img[:, :, 2]
    unq, unq_inv, unq_cnt = np.unique(colhash, return_inverse=True, return_counts=True)
    indxs = np.split(np.argsort(unq_inv), np.cumsum(unq_cnt[:-1]))
    col2indx = {unq[i]: indxs[i][0] for i in range(len(unq))}
    return {
        (col // (256**2), (col // 256) % 256, col % 256): (indx // w, indx % w)
        for col, indx in col2indx.items()
        if col != 0
    }


def read_colors(colors_file: str) -> Tuple[ObjClassCollection, dict]:
    if os.path.isfile(colors_file):
        try:
            logger.info("Will try to read segmentation colors from provided file.")
            with open(colors_file, "r") as file:
                cls2col = {}
                for line in file:
                    parts = line.strip().split()
                    if len(parts) < 4:
                        raise ValueError("Invalid format of colors file.")
                    class_name = " ".join(parts[:-3])
                    colors = tuple(map(int, parts[-3:]))
                    cls2col[class_name] = colors
        except Exception as e:
            logger.warning(
                "Failed to read segmentation colors from provided file. "
                "Will use default PascalVOC color mapping."
            )
            cls2col = default_classes_colors
    else:
        logger.info("Will use default PascalVOC color mapping.")
        cls2col = default_classes_colors

    obj_classes_list = [
        ObjClass(name=class_name, geometry_type=Bitmap, color=color)
        for class_name, color in cls2col.items()
    ]

    logger.info(
        f"Determined {len(cls2col)} class(es).",
        extra={"classes": list(cls2col.keys())},
    )

    obj_classes = ObjClassCollection(obj_classes_list)
    color2class_name = {v: k for k, v in cls2col.items()}
    return obj_classes, color2class_name


def get_ann(
    item,
    color2class_name: dict,
    meta: ProjectMeta,
    bbox_classes_map: dict,
    renamed_classes=None,
    renamed_tags=None,
) -> Annotation:
    segm_path, inst_path = item.segm_path, item.inst_path
    height, width = item.shape
    img_rect = Rectangle.from_size(item.shape)
    ann = Annotation(img_size=(height, width))

    if item.ann_data is not None:
        bbox_labels = xml_to_sly_labels(
            item.ann_data, meta, bbox_classes_map, img_rect, renamed_classes, renamed_tags
        )
        ann = ann.add_labels(bbox_labels)

    if segm_path is None:
        return ann

    segmentation_img = read(segm_path)

    if inst_path is not None:
        instance_img = read(inst_path)
        colored_img = instance_img
        instance_img16 = instance_img.astype(np.uint16)
        col2coord = get_col2coord(instance_img16)
        curr_col2cls = []
        for col, coord in col2coord.items():
            cls_name = color2class_name.get(tuple(segmentation_img[coord]))
            if cls_name is not None:
                if renamed_classes is not None and cls_name in renamed_classes:
                    cls_name = renamed_classes[cls_name]
            curr_col2cls.append((col, cls_name))
        curr_col2cls = {
            k: v for k, v in curr_col2cls if v is not None
        }  # _instance_ color -> class name
    else:
        colored_img = segmentation_img
        segmentation_img = segmentation_img.astype(np.uint16)
        colors = list(get_col2coord(segmentation_img).keys())
        curr_col2cls = {}
        for color in colors:
            cls_name = color2class_name.get(color)
            if cls_name is not None:
                if renamed_classes is not None and cls_name in renamed_classes:
                    cls_name = renamed_classes[cls_name]
            curr_col2cls[color] = cls_name

    labels = []
    for color, class_name in curr_col2cls.items():
        mask = np.all(colored_img == color, axis=2)  # exact match (3-channel img & rgb color)
        bitmap = Bitmap(data=mask)
        obj_class = ObjClass(name=class_name, geometry_type=Bitmap)
        labels.append(Label(bitmap, obj_class))
        #  clear used pixels in mask to check missing colors, see below
        colored_img[mask] = (0, 0, 0)

    labels = validate_image_bounds(labels, img_rect)
    ann = ann.add_labels(labels)

    if np.sum(colored_img) > 0:
        logger.warning(
            f"Not all objects or classes are captured from source segmentation: {item.name}"
        )

    return ann


def xml_to_sly_labels(
    xml_path: str,
    meta: ProjectMeta,
    bbox_classes_map: dict,
    img_rect: Rectangle,
    renamed_classes=None,
    renamed_tags=None,
) -> List[Label]:
    import xml.etree.ElementTree as ET

    labels = []
    with open(xml_path, "r") as in_file:
        tree = ET.parse(in_file)
        root = tree.getroot()

    for obj in root.iter("object"):
        geometry = None
        obj_cls = None
        tags = []

        for element in obj:
            field_name, value = element.tag, element.text
            if field_name in ["name", "class"]:
                cls_name = bbox_classes_map.get(value, value)
                if renamed_classes and cls_name in renamed_classes:
                    cls_name = renamed_classes[cls_name]
                obj_cls = meta.obj_classes.get(cls_name)
                if obj_cls is None:
                    logger.warning(f"Class {cls_name} is not found in meta. Skipping.")
                    continue
            elif field_name == "bndbox":
                bbox_coords = [
                    float(element.find(x).text) for x in ("ymin", "xmin", "ymax", "xmax")
                ]
                geometry = Rectangle(*bbox_coords)
            elif field_name not in DEFAULT_OBJECT_FIELDS:
                tag_name = field_name
                if renamed_tags and tag_name in renamed_tags:
                    tag_name = renamed_tags[tag_name]
                tag_meta = meta.get_tag_meta(tag_name)
                if tag_meta is None:
                    logger.warning(f"Tag meta for '{field_name}' is not found in meta. Skipping.")
                    continue
                if tag_meta.value_type == TagValueType.ANY_STRING:
                    if not isinstance(value, str):
                        value = str(value)
                    tags.append(Tag(tag_meta, value))
                elif tag_meta.value_type == TagValueType.NONE:
                    if int(value) == 1:
                        tags.append(Tag(tag_meta))
                    else:
                        logger.debug("Tag with value '0' not added to labels.")
                elif tag_meta.value_type == TagValueType.ONEOF_STRING:
                    if value not in tag_meta.possible_values:
                        logger.warning(
                            f"Value '{value}' for tag '{tag_name}' is not in possible values: {tag_meta.possible_values}. Skipping."
                        )
                        continue
                    tags.append(Tag(tag_meta, value))

        if geometry is None or obj_cls is None:
            continue
        labels.append(Label(geometry, obj_cls, tags))
    labels = validate_image_bounds(labels, img_rect)

    return labels


def update_meta_from_xml(
    xml_path: str,
    meta: ProjectMeta,
    existing_cls_names: set,
    bbox_classes_map: dict,
    tags_to_values: Dict[str, set],
) -> ProjectMeta:
    import xml.etree.ElementTree as ET

    with open(xml_path, "r") as in_file:
        tree = ET.parse(in_file)
        root = tree.getroot()

    for obj in root.iter("object"):
        for element in obj:
            field_name = element.tag
            if field_name in ["name", "class"]:
                class_name = element.text
                original_class_name = class_name
                obj_cls = meta.obj_classes.get(class_name)
                if obj_cls is None:
                    obj_cls = ObjClass(name=class_name, geometry_type=Rectangle)
                    meta = meta.add_obj_class(obj_cls)
                    existing_cls_names.add(class_name)
                    continue
                elif obj_cls.geometry_type == Rectangle:
                    continue
                class_name = class_name + "_bbox"
                obj_cls = meta.obj_classes.get(class_name)
                if obj_cls is None:
                    obj_cls = ObjClass(name=class_name, geometry_type=Rectangle)
                    meta = meta.add_obj_class(obj_cls)
                    existing_cls_names.add(class_name)
                elif obj_cls.geometry_type == Rectangle:
                    pass
                else:
                    class_name = generate_free_name(
                        existing_cls_names, class_name, extend_used_names=True
                    )
                    obj_cls = ObjClass(name=class_name, geometry_type=Rectangle)
                    meta = meta.add_obj_class(obj_cls)
                bbox_classes_map[original_class_name] = class_name
            elif field_name not in DEFAULT_OBJECT_FIELDS:
                value = element.text
                tags_to_values[field_name].add(value)

    return meta


def sly_ann_to_pascal_voc(ann: Annotation, image_name: str) -> Tuple[dict]:
    """
    Convert Supervisely annotation to Pascal VOC format annotation.

    :param ann: Supervisely annotation.
    :type ann: :class:`Annotation<supervisely.annotation.annotation.Annotation>`
    :param image_name: Image name.
    :type image_name: :class:`str`
    :return: Tuple with xml tree and instance and class masks in PIL.Image format.
    :rtype: :class:`Tuple`

    :Usage example:

     .. code-block:: python

        import supervisely as sly
        from supervisely.convert.image.pascal_voc.pascal_voc_helper import sly_ann_to_pascal_voc

        ann = sly.Annotation.from_json(ann_json, meta)
        xml_tree, instance_mask, class_mask = sly_ann_to_pascal_voc(ann, image_name)
    """

    def from_ann_to_instance_mask(ann: Annotation, contour_thickness: int = 3):
        mask = np.zeros((ann.img_size[0], ann.img_size[1], 3), dtype=np.uint8)
        for label in ann.labels:
            if label.obj_class.geometry_type == Rectangle:
                continue

            if label.obj_class.name == "neutral":
                label.geometry.draw(mask, default_classes_colors["neutral"])
                continue

            label.geometry.draw_contour(mask, default_classes_colors["neutral"], contour_thickness)
            label.geometry.draw(mask, label.obj_class.color)

        res_mask = Image.fromarray(mask)
        res_mask = res_mask.convert("P", palette=Image.ADAPTIVE)  # pylint: disable=no-member
        return res_mask

    def from_ann_to_class_mask(ann: Annotation, contour_thickness: int = 3):
        exist_colors = [[0, 0, 0], default_classes_colors["neutral"]]
        mask = np.zeros((ann.img_size[0], ann.img_size[1], 3), dtype=np.uint8)
        for label in ann.labels:
            if label.obj_class.geometry_type == Rectangle:
                continue

            if label.obj_class.name == "neutral":
                label.geometry.draw(mask, default_classes_colors["neutral"])
                continue

            new_color = generate_rgb(exist_colors)
            exist_colors.append(new_color)
            label.geometry.draw_contour(mask, default_classes_colors["neutral"], contour_thickness)
            label.geometry.draw(mask, new_color)

        res_mask = Image.fromarray(mask)
        res_mask = res_mask.convert("P", palette=Image.ADAPTIVE)  # pylint: disable=no-member
        return res_mask

    def from_ann_to_xml(ann: Annotation, image_name: str):
        import lxml.etree as ET  # pylint: disable=import-error

        xml_root = ET.Element("annotation")

        ET.SubElement(xml_root, "folder").text = f"VOC"
        ET.SubElement(xml_root, "filename").text = image_name

        xml_root_source = ET.SubElement(xml_root, "source")
        ET.SubElement(xml_root_source, "database").text = ""

        ET.SubElement(xml_root_source, "annotation").text = "PASCAL VOC"
        ET.SubElement(xml_root_source, "image").text = ""

        xml_root_size = ET.SubElement(xml_root, "size")
        ET.SubElement(xml_root_size, "width").text = str(ann.img_size[1])
        ET.SubElement(xml_root_size, "height").text = str(ann.img_size[0])
        ET.SubElement(xml_root_size, "depth").text = "3"

        ET.SubElement(xml_root, "segmented").text = "1" if len(ann.labels) > 0 else "0"

        for label in ann.labels:
            if label.obj_class.name == "neutral":
                continue

            bitmap_to_bbox = label.geometry.to_bbox()

            xml_ann_obj = ET.SubElement(xml_root, "object")
            ET.SubElement(xml_ann_obj, "name").text = label.obj_class.name
            ET.SubElement(xml_ann_obj, "pose").text = "Unspecified"
            ET.SubElement(xml_ann_obj, "truncated").text = "0"
            ET.SubElement(xml_ann_obj, "difficult").text = "0"

            xml_ann_obj_bndbox = ET.SubElement(xml_ann_obj, "bndbox")
            ET.SubElement(xml_ann_obj_bndbox, "xmin").text = str(bitmap_to_bbox.left)
            ET.SubElement(xml_ann_obj_bndbox, "ymin").text = str(bitmap_to_bbox.top)
            ET.SubElement(xml_ann_obj_bndbox, "xmax").text = str(bitmap_to_bbox.right)
            ET.SubElement(xml_ann_obj_bndbox, "ymax").text = str(bitmap_to_bbox.bottom)

        tree = ET.ElementTree(xml_root)
        return tree

    pascal_ann = from_ann_to_xml(ann, image_name)
    instance_mask = from_ann_to_instance_mask(ann)
    class_mask = from_ann_to_class_mask(ann)
    return pascal_ann, instance_mask, class_mask


def sly_ds_to_pascal_voc(
    dataset: Dataset,
    meta: ProjectMeta,
    dest_dir: Optional[str] = None,
    train_val_split_coef: float = 0.8,
    log_progress: bool = False,
    progress_cb: Optional[Union[tqdm, Callable]] = None,
) -> None:
    """
    Convert Supervisely dataset to Pascal VOC format.

    :param meta: Project meta information.
    :type meta: :class:`ProjectMeta<supervisely.project.project_meta.ProjectMeta>`
    :param dest_dir: Destination directory.
    :type dest_dir: :class:`str`, optional
    :param train_val_split_coef: Coefficient for splitting images into train and validation sets.
    :type train_val_split_coef: :class:`float`, optional
    :param log_progress: If True, log progress.
    :type log_progress: :class:`str`, optional
    :param progress_cb: Progress callback.
    :type progress_cb: :class:`Callable`, optional
    :return: None
    :rtype: NoneType

    :Usage example:

     .. code-block:: python

        import supervisely as sly
        from supervisely.convert.image.pascal_voc.pascal_voc_helper import sly_ds_to_pascal_voc

        project_path = "/home/admin/work/supervisely/projects/lemons_annotated"
        project = sly.Project(project_path, sly.OpenMode.READ)

        for ds in project.datasets:
            dest_dir = "/home/admin/work/supervisely/projects/lemons_annotated_pascal_voc"
            sly_ds_to_pascal_voc(ds, project.meta, dest_dir=dest_dir)
    """
    import lxml.etree as ET  # pylint: disable=import-error

    def write_main_set(
        is_trainval: int,
        images_stats: dict,
        meta: ProjectMeta,
        result_main_sets_dir: str,
        result_segmentation_sets_dir: str,
    ):
        res_files = ["trainval.txt", "train.txt", "val.txt"]
        for file in os.listdir(result_segmentation_sets_dir):
            if file in res_files:
                shutil.copyfile(
                    os.path.join(result_segmentation_sets_dir, file),
                    os.path.join(result_main_sets_dir, file),
                )

        train_imgs = [i for i in images_stats if i["dataset"] == TRAIN_TAG_NAME]
        val_imgs = [i for i in images_stats if i["dataset"] == VAL_TAG_NAME]

        write_objs = [
            {"suffix": "trainval", "imgs": images_stats},
            {"suffix": "train", "imgs": train_imgs},
            {"suffix": "val", "imgs": val_imgs},
        ]

        if is_trainval == 1:
            trainval_imgs = [
                i for i in images_stats if i["dataset"] == TRAIN_TAG_NAME + VAL_TAG_NAME
            ]
            write_objs[0] = {"suffix": "trainval", "imgs": trainval_imgs}

        for obj_cls in meta.obj_classes:
            if obj_cls.geometry_type not in SUPPORTED_GEOMETRY_TYPES:
                continue
            if obj_cls.name == "neutral":
                continue
            for o in write_objs:
                with open(
                    os.path.join(result_main_sets_dir, f'{obj_cls.name}_{o["suffix"]}.txt'), "a"
                ) as f:
                    for img_stats in o["imgs"]:
                        v = "1" if obj_cls.name in img_stats["classes"] else "-1"
                        f.write(f'{img_stats["name"]} {v}\n')

    def write_segm_set(is_trainval: int, images_stats: dict, result_imgsets_dir: str):
        with open(os.path.join(result_imgsets_dir, "trainval.txt"), "a") as f:
            if is_trainval == 1:
                f.writelines(
                    i["name"] + "\n"
                    for i in images_stats
                    if i["dataset"] == TRAIN_TAG_NAME + VAL_TAG_NAME
                )
            else:
                f.writelines(i["name"] + "\n" for i in images_stats)
        with open(os.path.join(result_imgsets_dir, "train.txt"), "a") as f:
            f.writelines(i["name"] + "\n" for i in images_stats if i["dataset"] == TRAIN_TAG_NAME)
        with open(os.path.join(result_imgsets_dir, "val.txt"), "a") as f:
            f.writelines(i["name"] + "\n" for i in images_stats if i["dataset"] == VAL_TAG_NAME)

    if progress_cb is not None:
        log_progress = False

    if log_progress:
        progress_cb = tqdm_sly(
            desc=f"Converting dataset '{dataset.short_name}' to Pascal VOC format",
            total=len(dataset),
        )

    logger.info(f"Processing dataset: '{dataset.name}'")

    # Prepare Pascal VOC root directory
    if dest_dir is None:
        dest_dir = str(Path(dataset.path).parent / "pascal_voc")

    pascal_root_path = os.path.join(dest_dir, "VOCdevkit", "VOC")
    result_images_dir = os.path.join(pascal_root_path, "JPEGImages")
    result_ann_dir = os.path.join(pascal_root_path, "Annotations")
    result_obj_dir = os.path.join(pascal_root_path, "SegmentationObject")
    result_class_dir = os.path.join(pascal_root_path, "SegmentationClass")
    result_image_sets_dir = os.path.join(pascal_root_path, "ImageSets")
    result_segmentation_sets_dir = os.path.join(result_image_sets_dir, "Segmentation")
    result_main_sets_dir = os.path.join(result_image_sets_dir, "Main")
    result_colors_file_path = os.path.join(pascal_root_path, "colors.txt")

    # Create directories if not exist
    os.makedirs(result_images_dir, exist_ok=True)
    os.makedirs(result_ann_dir, exist_ok=True)
    os.makedirs(result_obj_dir, exist_ok=True)
    os.makedirs(result_class_dir, exist_ok=True)
    os.makedirs(result_image_sets_dir, exist_ok=True)
    os.makedirs(result_segmentation_sets_dir, exist_ok=True)
    os.makedirs(result_main_sets_dir, exist_ok=True)

    # Create colors.txt file
    if not file_exists(result_colors_file_path):
        with open(result_colors_file_path, "w") as f:
            f.write(
                f"neutral {default_classes_colors['neutral'][0]} {default_classes_colors['neutral'][1]} {default_classes_colors['neutral'][2]}\n"
            )

    image_stats = []
    classes_colors = {}
    for item_name, img_path, ann_path in dataset.items():
        # Assign unique name to avoid conflicts
        unique_name = f"{dataset.name}_{get_file_name(item_name)}"

        # Load annotation
        ann = Annotation.from_json(load_json_file(ann_path), meta)
        pascal_ann, instance_mask, class_mask = sly_ann_to_pascal_voc(ann, unique_name)

        # Write ann
        ann_path = os.path.join(result_ann_dir, f"{unique_name}.xml")
        ET.indent(pascal_ann, space="    ")
        pascal_ann.write(ann_path, pretty_print=True)

        # Save instance mask
        instance_mask_path = os.path.join(
            result_obj_dir, f"{unique_name}_instance{MASKS_EXTENSION}"
        )
        instance_mask.save(instance_mask_path)

        # Save class mask
        class_mask_path = os.path.join(result_class_dir, f"{unique_name}_class{MASKS_EXTENSION}")
        class_mask.save(class_mask_path)

        # Save original image
        img_ext = get_file_ext(img_path)
        if img_ext not in VALID_IMG_EXT:
            jpg_name = f"{unique_name}.jpg"
            jpg_image_path = os.path.join(result_images_dir, jpg_name)
            img = Image.open(img_path)
            img.save(jpg_image_path, "JPEG")
        else:
            jpg_name = f"{unique_name}{img_ext}"
            jpg_image_path = os.path.join(result_images_dir, jpg_name)
            shutil.copyfile(img_path, jpg_image_path)

        # Update stats
        cur_img_stats = {"classes": set(), "dataset": None, "name": jpg_name}
        image_stats.append(cur_img_stats)

        # Get classes colors
        for label in ann.labels:
            cur_img_stats["classes"].add(label.obj_class.name)
            classes_colors[label.obj_class.name] = tuple(label.obj_class.color)

        if log_progress:
            progress_cb.update(1)

    # Update colors.txt file
    classes_colors = OrderedDict((sorted(classes_colors.items(), key=lambda t: t[0])))
    with open(result_colors_file_path, "a") as cc:
        for k in classes_colors.keys():
            if k == "neutral":
                continue
            cc.write(f"{k} {classes_colors[k][0]} {classes_colors[k][1]} {classes_colors[k][2]}\n")

    # Create splits
    imgs_to_split = [i for i in image_stats if i["dataset"] is None]
    train_len = int(len(imgs_to_split) * train_val_split_coef)

    for img_stat in imgs_to_split[:train_len]:
        img_stat["dataset"] = TRAIN_TAG_NAME
    for img_stat in imgs_to_split[train_len:]:
        img_stat["dataset"] = VAL_TAG_NAME

    is_trainval = 0
    write_segm_set(is_trainval, image_stats, result_segmentation_sets_dir)
    write_main_set(
        is_trainval, image_stats, meta, result_main_sets_dir, result_segmentation_sets_dir
    )


def sly_project_to_pascal_voc(
    project: Union[Project, str],
    dest_dir: Optional[str] = None,
    train_val_split_coef: float = 0.8,
    log_progress: bool = True,
    progress_cb: Optional[Union[tqdm, Callable]] = None,
) -> None:
    """
    Convert Supervisely project to Pascal VOC format.

    :param dest_dir: Destination directory.
    :type dest_dir: :class:`str`, optional
    :param train_val_split_coef: Coefficient for splitting images into train and validation sets.
    :type train_val_split_coef: :class:`float`, optional
    :param log_progress: Show uploading progress bar.
    :type log_progress: :class:`bool`
    :param progress_cb: Function for tracking conversion progress (for all items in the project).
    :type progress_cb: callable, optional
    :return: None
    :rtype: NoneType

    :Usage example:

    .. code-block:: python

        import supervisely as sly

        # Local folder with Project
        project_directory = "/home/admin/work/supervisely/source/project"

        # Convert Project to Pascal VOC format
        sly.Project(project_directory).to_pascal_voc(log_progress=True)
    """
    if isinstance(project, str):
        project = Project(project, mode=OpenMode.READ)

    if dest_dir is None:
        dest_dir = project.directory

    Path(dest_dir).mkdir(parents=True, exist_ok=True)

    if progress_cb is not None:
        log_progress = False

    if log_progress:
        progress_cb = tqdm_sly(
            desc="Converting Supervisely project to Pascal VOC format", total=project.total_items
        )

    for dataset in project.datasets:
        dataset: Dataset
        dataset.to_pascal_voc(
            meta=project.meta,
            dest_dir=dest_dir,
            train_val_split_coef=train_val_split_coef,
            log_progress=log_progress,
            progress_cb=progress_cb,
        )
        logger.info(f"Dataset '{dataset.short_name}' has been converted to Pascal VOC format.")
    logger.info(f"Project '{project.name}' has been converted to Pascal VOC format.")


def to_pascal_voc(
    input_data: Union[Project, Dataset, str],
    dest_dir: Optional[str] = None,
    meta: Optional[ProjectMeta] = None,
    train_val_split_coef: float = 0.8,
    log_progress: bool = True,
    progress_cb: Optional[Union[tqdm, Callable]] = None,
) -> None:
    """
    Universal function to convert Supervisely project or dataset to Pascal VOC format.
    Note:
        - For better compatibility, please pass named arguments explicitly. Otherwise, the function may not work as expected.
            You can use the dedicated functions for each data type:

                - :func:`sly.convert.sly_project_to_pascal_voc()`
                - :func:`sly.convert.sly_ds_to_pascal_voc()`

        - If the input_data is a Project, the dest_dir parameters are required.
        - If the input_data is a Dataset, the meta and dest_dir parameters are required.

    :param input_data: Input data to convert (Project, Dataset, or path to the project/dataset directory).
    :type input_data: :class:`Project<supervisely.project.project.Project>`, :class:`Dataset<supervisely.dataset.dataset.Dataset>`, or :class:`str`
    :param dest_dir: Destination directory.
    :type dest_dir: :class:`str`, optional
    :param meta: Project meta information (required for Dataset conversion).
    :type meta: :class:`ProjectMeta<supervisely.project.project_meta.ProjectMeta>`, optional
    :param train_val_split_coef: Coefficient for splitting images into train and validation sets.
    :type train_val_split_coef: :class:`float`, optional
    :param log_progress: Show uploading progress bar.
    :type log_progress: :class:`bool`
    :param progress_cb: Function for tracking conversion progress (for all items in the project).
    :type progress_cb: callable, optional
    :return: None
    :rtype: NoneType

    :Usage example:

    .. code-block:: python

        import supervisely as sly

        # Local folder with Project
        project_directory = "/home/admin/work/supervisely/source/project"
        project_fs = sly.Project(project_directory, sly.OpenMode.READ)

        # Convert Project to Pascal VOC format
        sly.convert.to_pascal_voc(project_directory, dest_dir="./pascal_voc")
        # or
        sly.convert.to_pascal_voc(project_fs, dest_dir="./pascal_voc")

        # Convert Dataset to Pascal VOC format
        dataset: sly.Dataset = project_fs.datasets.get("dataset_name")
        sly.convert.to_pascal_voc(dataset, dest_dir="./pascal_voc")
    """
    if isinstance(input_data, str):
        try:
            input_data = Project(input_data, mode=OpenMode.READ)
        except Exception as e:
            try:
                input_data = Dataset(input_data, mode=OpenMode.READ)
            except Exception as e:
                raise ValueError("Please check the path or the input data.")

    if isinstance(input_data, (Project, str)):
        return sly_project_to_pascal_voc(
            project=input_data,
            dest_dir=dest_dir,
            train_val_split_coef=train_val_split_coef,
            log_progress=log_progress,
            progress_cb=progress_cb,
        )
    elif isinstance(input_data, Dataset):
        return sly_ds_to_pascal_voc(
            dataset=input_data,
            meta=meta,
            dest_dir=dest_dir,
            train_val_split_coef=train_val_split_coef,
            log_progress=log_progress,
            progress_cb=progress_cb,
        )
    else:
        raise ValueError(f"Unsupported input data type: {type(input_data)}")

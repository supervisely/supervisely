import os
from typing import List, Tuple

import numpy as np

from supervisely import (
    Annotation,
    Label,
    ObjClass,
    ObjClassCollection,
    ProjectMeta,
    generate_free_name,
    logger,
)
from supervisely.geometry.bitmap import Bitmap
from supervisely.geometry.rectangle import Rectangle
from supervisely.imaging.image import read

MASKS_EXTENSION = ".png"

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
        logger.info("Will try to read segmentation colors from provided file.")
        in_lines = filter(None, map(str.strip, open(colors_file, "r").readlines()))
        in_splitted = (x.split() for x in in_lines)
        # Format: {name: (R, G, B)}, values [0; 255]
        cls2col = {}
        for x in in_splitted:
            if len(x) != 4:
                raise ValueError("Invalid format of colors file.")
            cls2col[x[0]] = (int(x[1]), int(x[2]), int(x[3]))
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
) -> Annotation:
    segm_path, inst_path = item.segm_path, item.inst_path
    height, width = item.shape

    ann = Annotation(img_size=(height, width))
    if segm_path is None and inst_path is None:
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

    for color, class_name in curr_col2cls.items():
        mask = np.all(colored_img == color, axis=2)  # exact match (3-channel img & rgb color)
        bitmap = Bitmap(data=mask)
        obj_class = ObjClass(name=class_name, geometry_type=Bitmap)

        ann = ann.add_label(Label(bitmap, obj_class))
        #  clear used pixels in mask to check missing colors, see below
        colored_img[mask] = (0, 0, 0)

    if np.sum(colored_img) > 0:
        logger.warn(
            f"Not all objects or classes are captured from source segmentation: {item.name}"
        )

    if item.ann_data is not None:
        bbox_labels = xml_to_sly_labels(item.ann_data, meta, bbox_classes_map, renamed_classes)
        ann = ann.add_labels(bbox_labels)

    return ann


def xml_to_sly_labels(
    xml_path: str,
    meta: ProjectMeta,
    bbox_classes_map: dict,
    renamed_classes=None,
) -> List[Label]:
    import xml.etree.ElementTree as ET

    labels = []
    with open(xml_path, "r") as in_file:
        tree = ET.parse(in_file)
        root = tree.getroot()

        for obj in root.iter("object"):
            cls_name = obj.find("name").text
            cls_name = bbox_classes_map.get(cls_name, cls_name)
            if renamed_classes and cls_name in renamed_classes:
                cls_name = renamed_classes[cls_name]
            obj_cls = meta.obj_classes.get(cls_name)
            if obj_cls is None:
                logger.warn(f"Class {cls_name} is not found in meta. Skipping.")
                continue
            xmlbox = obj.find("bndbox")
            bbox_coords = [float(xmlbox.find(x).text) for x in ("ymin", "xmin", "ymax", "xmax")]
            bbox = Rectangle(*bbox_coords)
            label = Label(bbox, obj_cls)
            labels.append(label)

    return labels


def update_meta_from_xml(
    xml_path: str,
    meta: ProjectMeta,
    existing_cls_names: set,
    bbox_classes_map: dict,
) -> ProjectMeta:
    import xml.etree.ElementTree as ET

    with open(xml_path, "r") as in_file:
        tree = ET.parse(in_file)
        root = tree.getroot()

        for obj in root.iter("object"):
            class_name = obj.find("name").text
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

    return meta

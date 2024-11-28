import json
import os
from os.path import join as pjoin
from typing import Callable, List, Optional

import numpy as np

from supervisely import Bitmap
from supervisely._utils import batched
from supervisely.project.project import Dataset, OpenMode, Project


def sly2coco(
    sly_project_path: str,
    is_dt_dataset: bool,
    accepted_shapes: list = None,
    conf_threshold: float = None,
    progress: Optional[Callable] = None,
    classes_whitelist: Optional[List[str]] = None,
):
    from pycocotools import mask as maskUtils  # pylint: disable=import-error

    project = Project(sly_project_path, mode=OpenMode.READ)

    # Categories
    meta_path = pjoin(sly_project_path, "meta.json")
    with open(meta_path, "r") as f:
        meta = json.load(f)
    classes_sorted = sorted(meta["classes"], key=lambda x: x["title"])
    if accepted_shapes is not None:
        accepted_shapes = set(accepted_shapes)
        accepted_shapes.add("any")

    cat_id_map = {}
    i = 1
    for obj_cls in classes_sorted:
        if accepted_shapes is not None and obj_cls["shape"] not in accepted_shapes:
            continue
        if classes_whitelist:
            if obj_cls["title"] not in classes_whitelist:
                continue
        cat_id_map[obj_cls["title"]] = i
        i += 1

    categories = [{"id": id, "name": cat} for cat, id in cat_id_map.items()]

    # Images + Annotations
    images = []
    annotations = []
    annotation_id = 1
    total = project.total_items
    with progress(message="Evaluation: Converting to COCO format", total=total) as pbar:
        img_id = 1
        for dataset in sorted(project.datasets, key=lambda x: x.name):
            dataset: Dataset
            ann_path = dataset.ann_dir
            imginfo_path = dataset.img_info_dir
            ann_files = sorted(os.listdir(ann_path))
            for batch in batched(ann_files, 30):
                for ann_file in batch:
                    img_name = os.path.splitext(ann_file)[0]
                    with open(os.path.join(ann_path, ann_file), "r") as f:
                        ann = json.load(f)
                    with open(os.path.join(imginfo_path, ann_file), "r") as f:
                        img_info = json.load(f)
                    img_w = ann["size"]["width"]
                    img_h = ann["size"]["height"]
                    img = {
                        "id": img_id,
                        "file_name": img_name,
                        "width": img_w,
                        "height": img_h,
                        "sly_id": img_info["id"],
                        "dataset": dataset.name,
                    }
                    images.append(img)
                    for label in ann["objects"]:
                        geometry_type = label["geometryType"]
                        if accepted_shapes is not None and geometry_type not in accepted_shapes:
                            continue
                        class_name = label["classTitle"]
                        if classes_whitelist:
                            if class_name not in classes_whitelist:
                                continue
                        category_id = cat_id_map[class_name]
                        sly_id = label["id"]
                        if geometry_type == "rectangle":
                            ((left, top), (right, bottom)) = label["points"]["exterior"]
                            width = right - left + 1
                            height = bottom - top + 1
                            annotation = {
                                "id": annotation_id,
                                "image_id": img_id,
                                "category_id": category_id,
                                "bbox": [left, top, width, height],
                                "area": float(width * height),
                                "iscrowd": 0,
                                "sly_id": sly_id,
                            }
                        elif geometry_type in ["polygon", "bitmap"]:
                            if geometry_type == "bitmap":
                                bitmap = Bitmap.from_json(label)
                                mask_np = _uncrop_bitmap(bitmap, img_w, img_h)
                                segmentation = maskUtils.encode(np.asfortranarray(mask_np))
                            else:
                                polygon = label["points"]["exterior"]
                                polygon = [[coord for sublist in polygon for coord in sublist]]
                                rles = maskUtils.frPyObjects(polygon, img_h, img_w)
                                segmentation = maskUtils.merge(rles)
                            segmentation["counts"] = segmentation["counts"].decode()
                            annotation = {
                                "id": annotation_id,
                                "image_id": img_id,
                                "category_id": category_id,
                                "segmentation": segmentation,
                                "iscrowd": 0,
                                "sly_id": sly_id,
                            }
                            if not is_dt_dataset:
                                area = int(maskUtils.area(segmentation))
                                bbox = maskUtils.toBbox(segmentation)
                                bbox = [int(coord) for coord in bbox]
                                annotation["area"] = area
                                annotation["bbox"] = bbox
                        else:
                            raise NotImplementedError(
                                f"Geometry type '{geometry_type}' is not implemented."
                            )
                        # Extract confidence score from the tag
                        if is_dt_dataset:
                            conf = _extract_confidence(label)
                            annotation["score"] = conf
                            if conf_threshold is not None and conf < conf_threshold:
                                continue
                        annotations.append(annotation)
                        annotation_id += 1
                    img_id += 1

                pbar.update(len(batch))

    coco_dataset = {"images": images, "annotations": annotations, "categories": categories}
    return coco_dataset


def _extract_confidence(label: dict):
    conf_tag = [tag for tag in label["tags"] if tag["name"] in ["confidence", "confidence-model"]]
    assert len(conf_tag) == 1, f"'confidence' tag is not found."
    return float(conf_tag[0]["value"])


def _uncrop_bitmap(bitmap: Bitmap, image_width, image_height):
    data = bitmap.data
    h, w = data.shape
    o = bitmap.origin
    b = np.zeros((image_height, image_width), dtype=data.dtype)
    b[o.row : o.row + h, o.col : o.col + w] = data
    return b

import json
import os
from os.path import join as pjoin


def sly2coco(sly_project_path: str, is_dt_dataset: bool, accepted_shapes: list = None, conf_threshold: float = None):
    datasets = [name for name in os.listdir(sly_project_path) if os.path.isdir(pjoin(sly_project_path, name))]

    # Categories
    meta_path = pjoin(sly_project_path, 'meta.json')
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    classes_sorted = sorted(meta['classes'], key=lambda x: x['title'])
    if accepted_shapes is None:
        cat2id = {cat['title']: i+1 for i, cat in enumerate(classes_sorted)}
    else:
        accepted_shapes: set = set(accepted_shapes)
        accepted_shapes.add('any')
        cat2id = {cat['title']: i+1 for i, cat in enumerate(classes_sorted) if cat['shape'] in accepted_shapes}
    categories = [{"id": id, "name": cat} for cat, id in cat2id.items()]

    # Images + Annotations
    images = []
    annotations = []
    annotation_id = 1
    for dataset_name in datasets:
        ann_path = pjoin(sly_project_path, dataset_name, 'ann')
        imginfo_path = pjoin(sly_project_path, dataset_name, 'img_info')
        ann_files = sorted(os.listdir(ann_path))
        for img_id, ann_file in enumerate(ann_files):
            img_name = os.path.splitext(ann_file)[0]
            with open(os.path.join(ann_path, ann_file), 'r') as f:
                ann = json.load(f)
            with open(os.path.join(imginfo_path, ann_file), 'r') as f:
                img_info = json.load(f)
            img = {
                "id": img_id,
                "file_name": img_name,
                "width": ann['size']['width'],
                "height": ann['size']['height'],
                "sly_id": img_info['id'],
                "dataset": dataset_name,
            }
            images.append(img)
            for label in ann['objects']:
                geometry_type = label['geometryType']
                if accepted_shapes is not None and geometry_type not in accepted_shapes:
                    continue

                if geometry_type == 'rectangle':
                    class_name = label['classTitle']
                    category_id = cat2id[class_name]
                    ((left, top), (right, bottom)) = label['points']['exterior']
                    width = right - left + 1
                    height = bottom - top + 1
                    sly_id = label['id']
                    
                    annotation = {
                        "id": annotation_id,
                        "image_id": img_id,
                        "category_id": category_id,
                        "bbox": [left, top, width, height],
                        "area": float(width * height),
                        "iscrowd": 0,
                        "sly_id": sly_id,
                    }

                    # Extract confidence score from the tag
                    if is_dt_dataset:
                        conf = _extract_confidence(label)
                        annotation["score"] = conf
                        if conf_threshold is not None and conf < conf_threshold:
                            continue    
                else:
                    # TODO: Implement other geometry types (calculate area, bbox, etc)
                    raise NotImplementedError(f"Geometry type '{geometry_type}' is not implemented.")
                    
                annotations.append(annotation)
                annotation_id += 1

    coco_dataset = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }
    return coco_dataset


def _extract_confidence(label: dict):
    conf_tag = [tag for tag in label['tags'] if tag['name'] == 'confidence']
    assert len(conf_tag) == 1, f"'confidence' tag is not found."
    return float(conf_tag[0]['value'])

from copy import deepcopy
from typing import Dict


SLY_MESH_ANN_KEYS = ("labels", "tags")


def rename_in_json(ann_json: Dict, renamed_classes=None, renamed_tags=None) -> Dict:
    ann_json = deepcopy(ann_json)
    renamed_classes = renamed_classes or {}
    renamed_tags = renamed_tags or {}

    for label in ann_json.get("labels", []):
        class_title = label.get("classTitle")
        if class_title is not None:
            label["classTitle"] = renamed_classes.get(class_title, class_title)
        for tag in label.get("tags", []):
            tag_name = tag.get("name")
            if tag_name is not None:
                tag["name"] = renamed_tags.get(tag_name, tag_name)

    for tag in ann_json.get("tags", []):
        tag_name = tag.get("name")
        if tag_name is not None:
            tag["name"] = renamed_tags.get(tag_name, tag_name)

    return ann_json

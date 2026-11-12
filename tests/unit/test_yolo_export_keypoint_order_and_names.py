"""Supervisely -> YOLO export: keypoint order and output file names.

- YOLO pose reads keypoint k of every line as the same body part. Keypoints used to be written in
  the order the label stores its nodes, and a label with fewer nodes was padded at the end, so a
  missing node shifted the following ones into the wrong slots. They now follow the class's
  keypoint template, and a missing node is ``0 0 0`` in its own slot.
- YOLO pairs ``images/<split>/<stem>.<ext>`` with ``labels/<split>/<stem>.txt``. Two images that
  differ only in extension (``a.jpg``, ``a.png``) got different image names but one label file, so
  the second image's labels overwrote the first's. The second now gets the stem ``<stem>_1``, as in
  the export-to-yolov8 and convert-supervisely-to-yolov5-format apps.
"""

import os

import numpy as np
import pytest

import supervisely as sly
from supervisely.geometry.graph import KeypointsTemplate

IMG_SIZE = (100, 200)  # rows, cols
NODES = ["nose", "left", "right"]
LOC = {"nose": (10, 20), "left": (30, 40), "right": (50, 60)}  # row, col


def _person_class():
    template = KeypointsTemplate()
    for name in NODES:
        template.add_point(label=name, row=LOC[name][0], col=LOC[name][1])
    template.add_edge(src="left", dst="right")
    return sly.ObjClass("person", sly.GraphNodes, geometry_config=template)


PERSON = _person_class()


def _keypoints(node_names):
    graph = sly.GraphNodes({name: sly.Node(sly.PointLocation(*LOC[name])) for name in node_names})
    ann = sly.Annotation(img_size=IMG_SIZE, labels=[sly.Label(graph, PERSON)])
    (line,) = ann.to_yolo(["person"], "pose")
    values = line.split()[5:]
    return [values[i : i + 3] for i in range(0, len(values), 3)]


def _kp(name):
    row, col = LOC[name]
    return [f"{col / IMG_SIZE[1]:.6f}", f"{row / IMG_SIZE[0]:.6f}", "2"]


def test_keypoints_in_template_order():
    assert _keypoints(NODES) == [_kp("nose"), _kp("left"), _kp("right")]


def test_keypoints_stored_in_another_order_follow_the_template():
    assert _keypoints(["right", "nose", "left"]) == [_kp("nose"), _kp("left"), _kp("right")]


def test_missing_keypoint_keeps_its_slot():
    assert _keypoints(["nose", "right"]) == [_kp("nose"), ["0", "0", "0"], _kp("right")]


# ----------------------------------------------------------------------------- file names

BOX = sly.ObjClass("box", sly.Rectangle)


def _project(tmp_path, items):
    """items: {image name: rectangle (top, left, bottom, right)}"""
    project = sly.Project(str(tmp_path / "project"), sly.OpenMode.CREATE)
    project.set_meta(sly.ProjectMeta(obj_classes=[BOX]))
    dataset = project.create_dataset("ds")
    for name, rect in items.items():
        ann = sly.Annotation(img_size=IMG_SIZE, labels=[sly.Label(sly.Rectangle(*rect), BOX)])
        dataset.add_item_np(name, np.zeros((*IMG_SIZE, 3), dtype=np.uint8), ann=ann)
    return sly.Project(str(tmp_path / "project"), sly.OpenMode.READ)


def _read_split(dest, split):
    images = sorted(os.listdir(os.path.join(dest, "images", split)))
    labels = {}
    for name in sorted(os.listdir(os.path.join(dest, "labels", split))):
        with open(os.path.join(dest, "labels", split, name)) as f:
            labels[name] = f.read().split("\n")
    return images, labels


def test_images_that_differ_only_in_extension_keep_their_own_labels(tmp_path):
    project = _project(tmp_path, {"a.jpg": (0, 0, 9, 9), "a.png": (50, 100, 99, 199)})
    dest = str(tmp_path / "yolo")
    project.to_yolo(dest_dir=dest)

    images, labels = _read_split(dest, "train")
    assert images == ["ds_a.jpg", "ds_a_1.png"]
    assert sorted(labels) == ["ds_a.txt", "ds_a_1.txt"]
    boxes = {name: lines[0].split()[1:] for name, lines in labels.items()}
    assert len(labels["ds_a.txt"]) == len(labels["ds_a_1.txt"]) == 1
    assert boxes["ds_a.txt"] != boxes["ds_a_1.txt"]
    # a.jpg's box is in the top-left corner, a.png's in the bottom-right one
    assert [float(v) for v in boxes["ds_a.txt"][:2]] == pytest.approx([0.02, 0.04])
    assert [float(v) for v in boxes["ds_a_1.txt"][:2]] == pytest.approx([0.745, 0.74])


def test_names_without_a_clash_are_unchanged(tmp_path):
    project = _project(tmp_path, {"a.jpg": (0, 0, 9, 9), "b.png": (0, 0, 9, 9)})
    dest = str(tmp_path / "yolo")
    project.to_yolo(dest_dir=dest)

    images, labels = _read_split(dest, "train")
    assert images == ["ds_a.jpg", "ds_b.png"]
    assert sorted(labels) == ["ds_a.txt", "ds_b.txt"]

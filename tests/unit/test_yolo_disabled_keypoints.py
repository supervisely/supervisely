"""
Tests for the `disabled_keypoints` option of the Supervisely -> YOLO pose export.

A disabled graph node used to be written with visibility flag 1 unconditionally.
Ultralytics builds its keypoint mask as `gt_kpt[..., 2] != 0`
(ultralytics/utils/loss.py, v8.4.6), so 1 and 2 are both trained and only 0 is
skipped - which meant a disabled node was always trained and there was no way to
say otherwise. `disabled_keypoints` picks the flag: "include" writes 1, "ignore"
writes 0. Node coordinates are written either way, matching the older converter
in supervisely-ecosystem/yolov8 (train/src/sly_to_yolov8.py), which writes
`visibility = 0 if graph_node.disabled else 2` and keeps the coordinates.

The default is "include", so output is unchanged for every existing caller.
"""

import pytest

import supervisely as sly
from supervisely.geometry.graph import KeypointsTemplate

IMG_SIZE = (100, 100)


def _template(node_names):
    template = KeypointsTemplate()
    for i, name in enumerate(node_names):
        template.add_point(label=name, row=10 * (i + 1), col=10 * (i + 1))
    template.add_edge(src=node_names[0], dst=node_names[1])
    return template


def _graph(node_names):
    """First node enabled, second node disabled."""
    return sly.GraphNodes(
        {
            node_names[0]: sly.Node(sly.PointLocation(row=20, col=30), disabled=False),
            node_names[1]: sly.Node(sly.PointLocation(row=40, col=50), disabled=True),
        }
    )


CAT_NODES = ["head", "tail"]
DOG_NODES = ["nose", "paw"]
CAT = sly.ObjClass("cat", sly.GraphNodes, geometry_config=_template(CAT_NODES))
DOG = sly.ObjClass("dog", sly.GraphNodes, geometry_config=_template(DOG_NODES))
CLASS_NAMES = ["cat", "dog"]


@pytest.fixture(name="ann")
def _ann():
    return sly.Annotation(
        img_size=IMG_SIZE,
        labels=[sly.Label(_graph(CAT_NODES), CAT), sly.Label(_graph(DOG_NODES), DOG)],
    )


def _flags(lines):
    """{class name: (enabled node flag, disabled node flag)} from YOLO pose lines."""
    flags = {}
    for line in lines:
        parts = line.split()
        flags[CLASS_NAMES[int(parts[0])]] = (parts[7], parts[10])
    return flags


def test_default_keeps_disabled_nodes_trained(ann):
    assert _flags(ann.to_yolo(CLASS_NAMES, "pose")) == {"cat": ("2", "1"), "dog": ("2", "1")}


def test_include_matches_the_default(ann):
    assert ann.to_yolo(CLASS_NAMES, "pose", disabled_keypoints="include") == ann.to_yolo(
        CLASS_NAMES, "pose"
    )


def test_ignore_marks_disabled_nodes_as_not_labelled(ann):
    lines = ann.to_yolo(CLASS_NAMES, "pose", disabled_keypoints="ignore")
    assert _flags(lines) == {"cat": ("2", "0"), "dog": ("2", "0")}


def test_ignore_keeps_node_coordinates(ann):
    default = ann.to_yolo(CLASS_NAMES, "pose")[0].split()
    ignored = ann.to_yolo(CLASS_NAMES, "pose", disabled_keypoints="ignore")[0].split()
    assert default[:10] == ignored[:10]
    assert (default[10], ignored[10]) == ("1", "0")


def test_per_class_map(ann):
    lines = ann.to_yolo(CLASS_NAMES, "pose", disabled_keypoints={"cat": "ignore", "dog": "include"})
    assert _flags(lines) == {"cat": ("2", "0"), "dog": ("2", "1")}


def test_class_missing_from_the_map_falls_back_to_include(ann):
    lines = ann.to_yolo(CLASS_NAMES, "pose", disabled_keypoints={"cat": "ignore"})
    assert _flags(lines) == {"cat": ("2", "0"), "dog": ("2", "1")}


def test_empty_map_behaves_like_the_default(ann):
    assert ann.to_yolo(CLASS_NAMES, "pose", disabled_keypoints={}) == ann.to_yolo(
        CLASS_NAMES, "pose"
    )


@pytest.mark.parametrize("value", ["visible", "Ignore", None, {"cat": "visible"}])
def test_unsupported_mode_raises(ann, value):
    with pytest.raises(ValueError):
        ann.to_yolo(CLASS_NAMES, "pose", disabled_keypoints=value)


def test_other_task_types_are_unaffected(ann):
    assert ann.to_yolo(CLASS_NAMES, "detect", disabled_keypoints="ignore") == ann.to_yolo(
        CLASS_NAMES, "detect"
    )

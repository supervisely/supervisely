"""
Security tests: class names used as file names by the Pascal VOC export.

``sly_ds_to_pascal_voc`` (``convert/image/pascal_voc/pascal_voc_helper.py``, behind
``sly.convert.to_pascal_voc`` and ``Project.to_pascal_voc``) writes one image set per class,
``ImageSets/Main/<class name>_<split>.txt``, opened in append mode. The class name is project
data: anyone who can edit a project's classes chooses it, and export apps run the conversion on
the project they are given. A name with ``../`` segments therefore created or appended to
``*_trainval.txt`` / ``*_train.txt`` / ``*_val.txt`` files outside the export folder, and a plain
``/`` (``cars/trucks``) pointed into a folder that does not exist, so the export failed at the
end with ``FileNotFoundError``. The fix turns ``/`` and ``\\`` in the class part of the file name
into ``_``; class names in the XML, ``colors.txt`` and the set contents are unchanged.

The exploit test runs the real export with a crafted class name and asserts that nothing was
written outside the export folder; it FAILS on BASE and PASSES on FIXED. The regression tests
pass on both. Everything is written under ``tmp_path``.
"""

import os
from pathlib import Path

import numpy as np
import pytest

import supervisely as sly
from supervisely.convert.image.pascal_voc import pascal_voc_helper

IMG_SIZE = (20, 30)


def _export(tmp_path: Path, class_names):
    """Export a one-image project whose image has one box of every class; returns the folder."""
    classes = [sly.ObjClass(name, sly.Rectangle) for name in class_names]
    project = sly.Project(str(tmp_path / "project"), sly.OpenMode.CREATE)
    project.set_meta(sly.ProjectMeta(obj_classes=classes))
    dataset = project.create_dataset("ds")
    labels = [sly.Label(sly.Rectangle(1, 1, 5, 5), obj_class) for obj_class in classes]
    dataset.add_item_np(
        "a.jpg",
        np.zeros((*IMG_SIZE, 3), dtype=np.uint8),
        ann=sly.Annotation(img_size=IMG_SIZE, labels=labels),
    )
    project = sly.Project(str(tmp_path / "project"), sly.OpenMode.READ)
    dest = tmp_path / "sandbox" / "export"
    project.to_pascal_voc(dest_dir=str(dest), log_progress=False)
    return dest


@pytest.mark.parametrize(
    "class_name",
    ["../../../../../escaped", "a/../../../../../../escaped", "/abs/escaped", "..\\..\\escaped"],
)
def test_class_set_file_name_stays_in_main(class_name):
    main = os.path.join("out", "ImageSets", "Main")
    file_name = pascal_voc_helper._class_set_file_name(class_name, "trainval")
    assert "/" not in file_name and "\\" not in file_name
    assert os.path.dirname(os.path.normpath(os.path.join(main, file_name))) == main


def test_class_set_file_name_keeps_plain_names():
    assert pascal_voc_helper._class_set_file_name("car", "train") == "car_train.txt"
    assert pascal_voc_helper._class_set_file_name("cars/trucks", "val") == "cars_trucks_val.txt"


def test_export_class_name_does_not_escape_export_folder(tmp_path):
    pytest.importorskip("lxml", reason="the Pascal VOC export writes its XML with lxml")
    # ImageSets/Main is 4 levels below the export folder: 5 "../" reach the sandbox around it
    dest = _export(tmp_path, ["car", "../../../../../escaped"])

    sandbox = tmp_path / "sandbox"
    outside = [p for p in sandbox.rglob("*") if p.is_file() and dest not in p.parents]
    assert outside == []
    main = dest / "VOCdevkit" / "VOC" / "ImageSets" / "Main"
    assert (main / ".._.._.._.._.._escaped_trainval.txt").is_file()


def test_export_class_name_with_slash(tmp_path):
    pytest.importorskip("lxml", reason="the Pascal VOC export writes its XML with lxml")
    dest = _export(tmp_path, ["cars/trucks"])

    main = dest / "VOCdevkit" / "VOC" / "ImageSets" / "Main"
    with open(main / "cars_trucks_trainval.txt") as f:
        assert f.read() == "ds_a.jpg 1\n"
    with open(dest / "VOCdevkit" / "VOC" / "Annotations" / "ds_a.xml") as f:
        assert "<name>cars/trucks</name>" in f.read()

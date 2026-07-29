# coding: utf-8
"""Regression tests for Data Version restore's unpickle backward-compat.

ObjClass, TagMeta, ProjectMeta and ProjectSettings each gained a new
required attribute over time. Data Version backups are pickles of objects
built by the SDK version active at backup time, and pickle restores
__dict__ directly without calling __init__ - so unpickling an object
created before one of these attributes existed used to raise AttributeError
in to_json() (see supervisely/project/project.py Project.upload_bin()).

Each test below deletes the newly-added attribute from a fresh instance's
__dict__ before pickling, to simulate an old backup, then asserts unpickling
and to_json() still work via the class's __setstate__.
"""

import pickle

import supervisely as sly
from supervisely.annotation.obj_class import ObjClass
from supervisely.annotation.tag_meta import TagMeta, TagTargetType
from supervisely.geometry.any_geometry import AnyGeometry
from supervisely.project.project_meta import ProjectMeta
from supervisely.project.project_settings import ProjectSettings


def _roundtrip(obj):
    return pickle.loads(pickle.dumps(obj))


def test_obj_class_unpickle_missing_geometry_type_name():
    obj_class = ObjClass("lemon", sly.Rectangle, color=[255, 120, 0])
    del obj_class.__dict__["_geometry_type_name"]

    restored = _roundtrip(obj_class)

    assert restored.to_json()["shape"] == "rectangle"


def test_obj_class_unpickle_missing_geometry_type_name_any_geometry():
    obj_class = ObjClass("any_cls", AnyGeometry, color=[10, 10, 10])
    del obj_class.__dict__["_geometry_type_name"]

    restored = _roundtrip(obj_class)

    assert restored.to_json()["shape"] == AnyGeometry.geometry_name()


def test_obj_class_unpickle_new_style_unchanged():
    obj_class = ObjClass("kiwi", sly.Bitmap, color=[0, 200, 0])

    restored = _roundtrip(obj_class)

    assert restored.to_json() == obj_class.to_json()


def test_tag_meta_unpickle_missing_target_type():
    tag_meta = TagMeta("fruit", sly.TagValueType.ANY_STRING)
    del tag_meta.__dict__["_target_type"]

    restored = _roundtrip(tag_meta)

    assert restored.to_json()["target_type"] == TagTargetType.ALL


def test_tag_meta_unpickle_new_style_unchanged():
    tag_meta = TagMeta(
        "fruit", sly.TagValueType.ANY_STRING, target_type=TagTargetType.FRAME_BASED
    )

    restored = _roundtrip(tag_meta)

    assert restored.to_json() == tag_meta.to_json()


def test_project_settings_unpickle_missing_labeling_interface():
    settings = ProjectSettings(multiview_enabled=True, multiview_tag_name="group")
    del settings.__dict__["labeling_interface"]

    restored = _roundtrip(settings)

    assert restored.labeling_interface is None
    assert "labelingInterface" not in restored.to_json()


def test_project_meta_unpickle_missing_project_settings():
    meta = ProjectMeta(
        obj_classes=[ObjClass("lemon", sly.Rectangle)],
        tag_metas=[TagMeta("fruit", sly.TagValueType.ANY_STRING)],
        project_type=sly.ProjectType.IMAGES,
    )
    del meta.__dict__["_project_settings"]

    restored = _roundtrip(meta)

    assert restored.to_json()["projectSettings"] == ProjectSettings().to_json()


def test_project_meta_unpickle_all_legacy_attrs_missing_at_once():
    obj_class = ObjClass("lemon", sly.Rectangle)
    tag_meta = TagMeta("fruit", sly.TagValueType.ANY_STRING)
    meta = ProjectMeta(
        obj_classes=[obj_class], tag_metas=[tag_meta], project_type=sly.ProjectType.IMAGES
    )
    del meta.__dict__["_project_settings"]
    del obj_class.__dict__["_geometry_type_name"]
    del tag_meta.__dict__["_target_type"]

    restored = _roundtrip(meta)
    json_meta = restored.to_json()

    assert json_meta["classes"][0]["shape"] == "rectangle"
    assert json_meta["tags"][0]["target_type"] == TagTargetType.ALL
    assert json_meta["projectSettings"] == ProjectSettings().to_json()

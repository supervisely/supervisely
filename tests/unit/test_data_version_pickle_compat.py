# coding: utf-8
"""Regression tests for Data Version restore's unpickle backward-compat.

ObjClass gained _geometry_type_name (SDK v6.74.10) and TagMeta gained
_target_type (v6.73.260) after the .bin backup format already existed
(v6.73.123). Backups are pickles of objects built by the SDK active at
backup time, and pickle restores __dict__ directly without calling
__init__ - so unpickling an object created before one of these attributes
existed used to raise AttributeError in to_json() (see
supervisely/project/project.py Project.upload_bin()).

Each test deletes the newly-added attribute from a fresh instance's
__dict__ before pickling, to simulate an old backup, then asserts the
object is usable after the restore path runs restore_legacy_defaults()
over it, the way Project.upload_bin() does.
"""

import pickle

import pytest

import supervisely as sly
from supervisely.annotation.obj_class import ObjClass
from supervisely.annotation.obj_class_collection import ObjClassCollection
from supervisely.annotation.tag_meta import TagMeta, TagTargetType
from supervisely.annotation.tag_meta_collection import TagMetaCollection
from supervisely.geometry.any_geometry import AnyGeometry
from supervisely.io.pickle_compat import restore_legacy_defaults
from supervisely.project.project_meta import ProjectMeta
from supervisely.project.project_settings import ProjectSettings


def _roundtrip(obj):
    """Unpickle, then backfill the way Project.upload_bin() does."""
    restored = pickle.loads(pickle.dumps(obj))
    restore_legacy_defaults(restored)
    return restored


def _roundtrip_meta(meta):
    """Same, mirroring the exact call upload_bin() makes for a ProjectMeta."""
    restored = pickle.loads(pickle.dumps(meta))
    restore_legacy_defaults(
        restored,
        restored.project_settings,
        *restored.obj_classes,
        *restored.tag_metas,
    )
    return restored


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


def test_project_meta_unpickle_backfills_nested_objects():
    """The backfill must reach classes and tags nested inside a pickled ProjectMeta,
    which is how they actually arrive in a Data Version backup."""
    obj_class = ObjClass("lemon", sly.Rectangle)
    tag_meta = TagMeta("fruit", sly.TagValueType.ANY_STRING)
    meta = ProjectMeta(
        obj_classes=[obj_class], tag_metas=[tag_meta], project_type=sly.ProjectType.IMAGES
    )
    del obj_class.__dict__["_geometry_type_name"]
    del tag_meta.__dict__["_target_type"]

    restored = _roundtrip_meta(meta)
    json_meta = restored.to_json()

    assert json_meta["classes"][0]["shape"] == "rectangle"
    assert json_meta["tags"][0]["target_type"] == TagTargetType.ALL


def test_project_meta_unpickle_never_revalidates_against_stricter_rules():
    """
    ProjectSettings.validate() rejects multiview_tag_name for VIDEOS projects,
    but that rule was only added in #1547 (2025-11-21, video multiview support)
    - before it, validate() didn't even branch on project type. A video
    project version backed up earlier could have multiview_tag_name set and
    be perfectly valid at creation time. The restore path must never re-run
    validate(), or restoring such a (now technically non-compliant) backup
    would newly crash, even though nothing about the object itself changed.

    __init__ enforces this rule today, so such an object can no longer be
    built through the normal constructor - it is hand-crafted here via
    __new__ to stand in for a real legacy pickle.
    """
    settings = ProjectSettings.__new__(ProjectSettings)
    settings.__dict__.update(
        {
            "multiview_enabled": True,
            "multiview_tag_name": "group",
            "multiview_tag_id": None,
            "multiview_is_synced": False,
            "labeling_interface": None,
        }
    )
    meta = ProjectMeta.__new__(ProjectMeta)
    meta.__dict__.update(
        {
            "_obj_classes": ObjClassCollection(),
            "_tag_metas": TagMetaCollection(),
            "_project_type": sly.ProjectType.VIDEOS,
            "_project_settings": settings,
        }
    )

    # Confirms the scenario is realistic: today's validate() really does
    # reject it, so a crash-on-unpickle regression here would be silent otherwise.
    with pytest.raises(RuntimeError):
        meta.project_settings.validate(meta)

    restored = _roundtrip_meta(meta)  # must not raise

    assert restored.project_settings.multiview_tag_name == "group"

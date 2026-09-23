"""
Tests for `LabelingInterface` against the platform's list of labeling interfaces.

`ProjectSettings` rejects any `labelingInterface` outside `LabelingInterface.values()`,
so a value the platform stores but the enum lacks makes `get_meta(with_settings=True)`,
project download and every converter fail on that project. PLATFORM_INTERFACES is
copied from `PROJECT_INTERFACE_TYPES` in shared/lib/packages/Projects/ProjectsTypes.js
(supervisely/main, branch dev).
"""

import pytest

import supervisely as sly
from supervisely.convert.base_converter import BaseConverter
from supervisely.project.project_settings import LabelingInterface

PLATFORM_INTERFACES = [
    "default",
    "multi_view",
    "multispectral",
    "images_with_16_color",
    "nrrd",
    "medical_imaging_single",
    "medical_imaging_multiple",
    "point_cloud_episodes",
    "image_matting",
    "fisheye",
    "overlay",
    "telemetry",
]


def test_enum_matches_platform():
    assert sorted(LabelingInterface.values()) == sorted(PLATFORM_INTERFACES)


@pytest.mark.parametrize("value", PLATFORM_INTERFACES)
def test_project_meta_round_trip(value):
    meta_json = {
        "classes": [],
        "tags": [],
        "projectType": "videos",
        "projectSettings": {
            "multiView": {"enabled": False, "tagName": None, "tagId": None, "isSynced": False},
            "labelingInterface": value,
        },
    }
    meta = sly.ProjectMeta.from_json(meta_json)
    assert meta.labeling_interface == value
    assert sly.ProjectMeta.from_json(meta.to_json()).labeling_interface == value


@pytest.mark.parametrize("value", PLATFORM_INTERFACES)
def test_converter_accepts(value, tmp_path):
    assert BaseConverter(str(tmp_path), labeling_interface=value)._labeling_interface == value


def test_unknown_value_still_rejected():
    with pytest.raises(ValueError):
        sly.ProjectSettings(labeling_interface="not_an_interface")

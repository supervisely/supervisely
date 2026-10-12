"""Downloading a volume project with an ``x-job-id`` header.

With that header ``figures.list`` returns only the figures of the job's classes, while the
annotation still holds every figure on the volume. Merging ``custom_data`` used to look up each
annotation figure in that list and raise ``AttributeError: 'NoneType' object has no attribute
'custom_data'`` for the first figure outside the job's classes.
"""

import json
import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from supervisely.annotation.obj_class import ObjClass
from supervisely.geometry.rectangle import Rectangle
from supervisely.project.project_meta import ProjectMeta
from supervisely.project.volume_project import download_volume_project

VOLUME_ID = 10
IN_JOB_FIGURE_ID = 101
OUT_OF_JOB_FIGURE_ID = 102
NO_CUSTOM_DATA_FIGURE_ID = 103


def _figure(figure_id, object_id, row):
    return {
        "id": figure_id,
        "objectId": object_id,
        "geometryType": Rectangle.geometry_name(),
        "geometry": {"points": {"exterior": [[0, row], [2, row + 2]], "interior": []}},
    }


def _ann_json():
    return {
        "volumeId": VOLUME_ID,
        "volumeName": "vol.nrrd",
        "description": "",
        "tags": [],
        "objects": [
            {"id": 1, "classTitle": "in_job", "tags": []},
            {"id": 2, "classTitle": "out_of_job", "tags": []},
        ],
        "volumeMeta": {
            "ACS": "RAS",
            "intensity": {"max": 1, "min": 0},
            "windowWidth": 1,
            "rescaleSlope": 1,
            "windowCenter": 0.5,
            "channelsCount": 1,
            "dimensionsIJK": {"x": 16, "y": 16, "z": 16},
            "IJK2WorldMatrix": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
            "rescaleIntercept": 0,
        },
        "planes": [
            {
                "name": "axial",
                "normal": {"x": 0, "y": 0, "z": 1},
                "slices": [
                    {
                        "index": 1,
                        "figures": [
                            _figure(IN_JOB_FIGURE_ID, 1, 0),
                            _figure(OUT_OF_JOB_FIGURE_ID, 2, 1),
                            _figure(NO_CUSTOM_DATA_FIGURE_ID, 1, 2),
                        ],
                    }
                ],
            },
            {"name": "sagittal", "normal": {"x": 1, "y": 0, "z": 0}, "slices": []},
            {"name": "coronal", "normal": {"x": 0, "y": 1, "z": 0}, "slices": []},
        ],
        "spatialFigures": [],
    }


def _api(listed_figures):
    meta = ProjectMeta(
        obj_classes=[ObjClass("in_job", Rectangle), ObjClass("out_of_job", Rectangle)]
    )
    api = MagicMock()
    api.project.get_meta.return_value = meta.to_json()
    api.dataset.get_info_by_id.return_value = SimpleNamespace(id=1, name="ds")
    api.volume.get_list.return_value = [SimpleNamespace(id=VOLUME_ID, name="vol.nrrd", sizeb=0)]
    api.volume.annotation.download_bulk.return_value = [_ann_json()]
    api.volume.figure.download.return_value = {VOLUME_ID: listed_figures}
    api.volume.annotation._update_on_transfer.side_effect = (
        lambda transfer_type, ann, project_meta, nrrd_paths: (ann, project_meta)
    )
    return api


def _saved_custom_data(dest_dir):
    with open(os.path.join(dest_dir, "ds", "ann", "vol.nrrd.json")) as f:
        ann = json.load(f)
    with open(os.path.join(dest_dir, "key_id_map.json")) as f:
        figure_ids = json.load(f)["figures"]
    return {
        figure_ids[figure["key"]]: figure.get("customData")
        for plane in ann["planes"]
        for slice_ in plane["slices"]
        for figure in slice_["figures"]
    }


@pytest.mark.parametrize("listed_custom_data", [{"source": "job"}, {}, None])
def test_figures_outside_job_classes_are_kept_without_custom_data(tmp_path, listed_custom_data):
    """Only the job's figures are listed; the rest are downloaded as they are."""
    api = _api(
        [
            SimpleNamespace(id=IN_JOB_FIGURE_ID, custom_data=listed_custom_data),
            SimpleNamespace(id=NO_CUSTOM_DATA_FIGURE_ID, custom_data=None),
        ]
    )
    dest_dir = str(tmp_path / "project")

    download_volume_project(api, 1, dest_dir, [1], download_volumes=False, progress_cb=lambda n: None)

    custom_data = _saved_custom_data(dest_dir)
    assert set(custom_data) == {IN_JOB_FIGURE_ID, OUT_OF_JOB_FIGURE_ID, NO_CUSTOM_DATA_FIGURE_ID}
    assert custom_data[IN_JOB_FIGURE_ID] == (listed_custom_data or {})
    assert custom_data[OUT_OF_JOB_FIGURE_ID] == {}
    assert custom_data[NO_CUSTOM_DATA_FIGURE_ID] == {}


def test_custom_data_of_every_listed_figure_is_merged(tmp_path):
    """Without a job every figure is listed, and each one's custom_data is kept."""
    api = _api(
        [
            SimpleNamespace(id=IN_JOB_FIGURE_ID, custom_data={"a": 1}),
            SimpleNamespace(id=OUT_OF_JOB_FIGURE_ID, custom_data={"b": 2}),
            SimpleNamespace(id=NO_CUSTOM_DATA_FIGURE_ID, custom_data={}),
        ]
    )
    dest_dir = str(tmp_path / "project")

    download_volume_project(api, 1, dest_dir, [1], download_volumes=False, progress_cb=lambda n: None)

    assert _saved_custom_data(dest_dir) == {
        IN_JOB_FIGURE_ID: {"a": 1},
        OUT_OF_JOB_FIGURE_ID: {"b": 2},
        NO_CUSTOM_DATA_FIGURE_ID: {},
    }

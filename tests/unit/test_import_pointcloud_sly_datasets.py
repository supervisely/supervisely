"""Supervisely-format point cloud (and episode) projects with several datasets, through ImportManager.

An archive from "Export point clouds project in Supervisely format" has ``<ds>/pointcloud/<name>``
and ``<ds>/ann/<name>.json`` per dataset. The same cloud name in two datasets must keep its own
annotation and related images, and the datasets must be kept.

Runs offline: the input is a tiny project written to a temporary folder, and the upload goes to an
in-memory stand-in for the API that records what would be sent.
"""

import json
import os
import sys
import uuid
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from PIL import Image

import supervisely.convert.converter as converter_module
from supervisely.convert.converter import ImportManager
from supervisely.project.project_type import ProjectType

# ----------------------------------------------------------------------------- fake API


class _FakeDatasetApi:
    def __init__(self, store):
        self._s = store

    def _info(self, ds):
        return SimpleNamespace(id=ds["id"], name=ds["name"], project_id=1, parent_id=ds["parent_id"])

    def add(self, name, parent_id=None):
        ds_id = 100 + len(self._s.datasets)
        self._s.datasets[ds_id] = {"id": ds_id, "name": name, "parent_id": parent_id}
        self._s.items[ds_id] = []
        return self._info(self._s.datasets[ds_id])

    def get_info_by_id(self, ds_id, raise_error=False):
        return self._info(self._s.datasets[ds_id])

    def get_list(self, project_id, recursive=False, parent_id=None, **kw):
        return [
            self._info(d)
            for d in self._s.datasets.values()
            if recursive or d["parent_id"] == parent_id
        ]

    def create(self, project_id, name, parent_id=None, **kw):
        return self.add(name, parent_id)

    def update(self, ds_id, name=None, **kw):
        self._s.datasets[ds_id]["name"] = name
        return self._info(self._s.datasets[ds_id])


class _FakeItemApi:
    """Both api.pointcloud and api.pointcloud_episode."""

    def __init__(self, store):
        self._s = store

    def get_list(self, ds_id, *a, **kw):
        return [SimpleNamespace(id=i, name=n) for i, n in self._s.items[ds_id]]

    def get_list_related_images_batch(self, ds_id, ids):
        return []

    def upload_paths(self, ds_id, names, paths, *a, **kw):
        infos = []
        for name in names:
            self._s.next_id += 1
            self._s.items[ds_id].append((self._s.next_id, name))
            self._s.where[self._s.next_id] = (ds_id, name)
            infos.append(SimpleNamespace(id=self._s.next_id, name=name))
        return infos

    def upload_related_image(self, path):
        return f"hash-of:{path}"

    def upload_related_images(self, paths, *a, **kw):
        return [f"hash-of:{p}" for p in paths]

    def add_related_images(self, infos, camera_names=None):
        for info in infos:
            self._s.rimgs.setdefault(info["entityId"], []).append(info["meta"]["deviceId"])
        return [{"id": i} for i, _ in enumerate(infos)]


class FakeApi:
    def __init__(self):
        self.datasets, self.items, self.where, self.rimgs = {}, {}, {}, {}
        self.cloud_anns = {}  # pointcloud id -> class names of its objects
        self.episode_anns = {}  # dataset id -> (class names, {frame: cloud name})
        self.next_id = 0
        self.dataset = _FakeDatasetApi(self)
        self.pointcloud = _FakeItemApi(self)
        self.pointcloud_episode = _FakeItemApi(self)
        self.project = SimpleNamespace(
            get_meta=lambda pid, with_settings=False: {"classes": [], "tags": []},
            update_meta=lambda pid, meta: meta,
        )
        self.pointcloud.annotation = SimpleNamespace(append=self._append_cloud_ann)
        self.pointcloud_episode.annotation = SimpleNamespace(append=self._append_episode_ann)
        self.image = SimpleNamespace(figure=SimpleNamespace(create_bulk=lambda **kw: None))

    def _append_cloud_ann(self, pcd_id, ann, key_id_map=None):
        self.cloud_anns[pcd_id] = sorted(o.obj_class.name for o in ann.objects)

    def _append_episode_ann(self, ds_id, ann, frame_to_pointcloud_ids, key_id_map=None):
        frames = {f: self.where[i][1] for f, i in frame_to_pointcloud_ids.items()}
        self.episode_anns[ds_id] = (sorted(o.obj_class.name for o in ann.objects), frames)

    def path(self, ds_id):
        ds = self.datasets[ds_id]
        parts = [ds["name"]]
        while ds["parent_id"] is not None:
            ds = self.datasets[ds["parent_id"]]
            parts.insert(0, ds["name"])
        return "/".join(parts)

    def clouds(self):
        """{"<dataset path>/<cloud name>": (class names, related image device ids)}"""
        return {
            f"{self.path(ds_id)}/{name}": (self.cloud_anns.get(i), self.rimgs.get(i, []))
            for ds_id, items in self.items.items()
            for i, name in items
        }


# ----------------------------------------------------------------------------- inputs

CUBOID = {
    "position": {"x": 0, "y": 0, "z": 0},
    "rotation": {"x": 0, "y": 0, "z": 0},
    "dimensions": {"x": 1, "y": 1, "z": 1},
}
META = {
    "classes": [
        {"title": cls, "shape": "cuboid_3d", "color": "#FF0000", "geometry_config": {}}
        for cls in ("car", "person", "tree")
    ],
    "tags": [],
}


def _write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)


def _pcd(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(
            "VERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\nWIDTH 1\nHEIGHT 1\n"
            "VIEWPOINT 0 0 0 1 0 0 0\nPOINTS 1\nDATA ascii\n0 0 0\n"
        )


def _object_and_figure(cls):
    obj = {"key": uuid.uuid4().hex, "classTitle": cls, "tags": []}
    fig = {
        "key": uuid.uuid4().hex,
        "objectKey": obj["key"],
        "geometryType": "cuboid_3d",
        "geometry": CUBOID,
    }
    return obj, fig


def _cloud_ann(path, cls):
    obj, fig = _object_and_figure(cls)
    _write_json(
        path,
        {"description": "", "key": uuid.uuid4().hex, "tags": [], "objects": [obj], "figures": [fig]},
    )


def _related_image(rimg_dir, device_id):
    os.makedirs(rimg_dir, exist_ok=True)
    Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(os.path.join(rimg_dir, "cam.png"))
    _write_json(
        os.path.join(rimg_dir, "cam.png.json"),
        {"name": "cam.png", "meta": {"deviceId": device_id, "sensorsData": {}}},
    )


def _cloud_dataset(ds_dir, clouds):
    """clouds: {cloud name: class name or None (no annotation file)}"""
    for name, cls in clouds.items():
        _pcd(os.path.join(ds_dir, "pointcloud", name))
        if cls is not None:
            _cloud_ann(os.path.join(ds_dir, "ann", f"{name}.json"), cls)
        _related_image(
            os.path.join(ds_dir, "related_images", name.replace(".", "_")),
            f"{os.path.basename(ds_dir)}:{name}",
        )


def _episode_dataset(ds_dir, cls, n_frames):
    names = [f"frame_{i:03d}.pcd" for i in range(n_frames)]
    for name in names:
        _pcd(os.path.join(ds_dir, "pointcloud", name))
        _related_image(
            os.path.join(ds_dir, "related_images", name.replace(".", "_")),
            f"{os.path.basename(ds_dir)}:{name}",
        )
    _write_json(os.path.join(ds_dir, "frame_pointcloud_map.json"), dict(enumerate(names)))
    obj, fig = _object_and_figure(cls)
    _write_json(
        os.path.join(ds_dir, "annotation.json"),
        {
            "description": "",
            "key": uuid.uuid4().hex,
            "tags": [],
            "objects": [obj],
            "framesCount": n_frames,
            "frames": [{"index": 0, "figures": [fig]}],
        },
    )


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("SLY_APP_DATA_DIR", str(tmp_path / "app_data"))
    monkeypatch.setenv("TEAM_ID", "1")
    monkeypatch.setattr(converter_module.Api, "from_env", classmethod(lambda cls: object()))
    # format detection probes the ROS bag converter, which imports rosbag (not on PyPI); there is
    # no .bag file in these inputs, so a stand-in module is enough
    if "rosbag" not in sys.modules:
        monkeypatch.setitem(sys.modules, "rosbag", ModuleType("rosbag"))
    src = tmp_path / "src"
    src.mkdir()
    return src


TARGET = "dataset 2026-10-28 10-00-00"


def _run(src, project_type):
    """ImportManager + upload into a fresh target dataset, as main-import does."""
    importer = ImportManager(str(src), project_type)
    assert str(importer.converter) == "supervisely"
    api = FakeApi()
    target = api.dataset.add(TARGET)
    importer._api = api
    importer.upload_dataset(target.id)
    return api


# ----------------------------------------------------------------------------- point clouds


def test_same_cloud_name_in_two_datasets_keeps_its_own_annotation(env):
    project = env / "My project"
    _write_json(str(project / "meta.json"), META)
    _cloud_dataset(str(project / "ds0"), {"frame_000.pcd": "car", "frame_001.pcd": None})
    # ds1 has an annotation for frame_001.pcd too, but that one belongs to ds1's own cloud
    _cloud_dataset(str(project / "ds1"), {"frame_000.pcd": "person", "frame_001.pcd": "tree"})

    api = _run(env, ProjectType.POINT_CLOUDS)

    assert api.clouds() == {
        "ds0/frame_000.pcd": (["car"], ["ds0:frame_000.pcd"]),
        "ds0/frame_001.pcd": ([], ["ds0:frame_001.pcd"]),
        "ds1/frame_000.pcd": (["person"], ["ds1:frame_000.pcd"]),
        "ds1/frame_001.pcd": (["tree"], ["ds1:frame_001.pcd"]),
    }


def test_nested_datasets_are_kept(env):
    project = env / "My project"
    _write_json(str(project / "meta.json"), META)
    _cloud_dataset(str(project / "ds0"), {"a.pcd": "car"})
    _cloud_dataset(str(project / "ds0" / "datasets" / "child"), {"a.pcd": "person"})

    api = _run(env, ProjectType.POINT_CLOUDS)

    assert api.clouds() == {
        "ds0/a.pcd": (["car"], ["ds0:a.pcd"]),
        "ds0/child/a.pcd": (["person"], ["child:a.pcd"]),
    }


def test_single_dataset_goes_into_the_target_dataset(env):
    project = env / "My project"
    _write_json(str(project / "meta.json"), META)
    _cloud_dataset(str(project / "ds0"), {"a.pcd": "car", "b.pcd": "person"})

    api = _run(env, ProjectType.POINT_CLOUDS)

    assert api.clouds() == {
        f"{TARGET}/a.pcd": (["car"], ["ds0:a.pcd"]),
        f"{TARGET}/b.pcd": (["person"], ["ds0:b.pcd"]),
    }


def test_loose_clouds_and_annotations_are_still_matched_by_name(env):
    _write_json(str(env / "meta.json"), META)
    _pcd(str(env / "clouds" / "a.pcd"))
    _cloud_ann(str(env / "labels" / "a.pcd.json"), "car")

    api = _run(env, ProjectType.POINT_CLOUDS)

    assert api.clouds() == {f"{TARGET}/a.pcd": (["car"], [])}


# ----------------------------------------------------------------------------- episodes


def test_two_episodes_keep_their_frames_and_annotations(env):
    project = env / "My episodes"
    _write_json(str(project / "meta.json"), META)
    _episode_dataset(str(project / "ep0"), "car", 2)
    _episode_dataset(str(project / "ep1"), "person", 3)

    api = _run(env, ProjectType.POINT_CLOUD_EPISODES)

    by_path = {api.path(ds_id): ann for ds_id, ann in api.episode_anns.items()}
    assert by_path == {
        "ep0": (["car"], {0: "frame_000.pcd", 1: "frame_001.pcd"}),
        "ep1": (["person"], {0: "frame_000.pcd", 1: "frame_001.pcd", 2: "frame_002.pcd"}),
    }
    assert sorted(api.clouds().values()) == sorted(
        [(None, [f"{ep}:frame_{i:03d}.pcd"]) for ep, n in (("ep0", 2), ("ep1", 3)) for i in range(n)]
    )
    assert {k.split("/")[0] for k in api.clouds()} == {"ep0", "ep1"}
    for key, (_, rimgs) in api.clouds().items():
        assert rimgs == [key.replace("/", ":")]


def test_single_episode_goes_into_the_target_dataset(env):
    project = env / "My episodes"
    _write_json(str(project / "meta.json"), META)
    _episode_dataset(str(project / "ep0"), "car", 2)

    api = _run(env, ProjectType.POINT_CLOUD_EPISODES)

    by_path = {api.path(ds_id): ann for ds_id, ann in api.episode_anns.items()}
    assert by_path == {TARGET: (["car"], {0: "frame_000.pcd", 1: "frame_001.pcd"})}


def test_same_episode_path_in_two_projects_gives_two_datasets(env):
    for project, cls in (("A", "car"), ("B", "person")):
        _write_json(str(env / project / "meta.json"), META)
        _episode_dataset(str(env / project / "ep0"), cls, 1)

    api = _run(env, ProjectType.POINT_CLOUD_EPISODES)

    by_path = {api.path(ds_id): ann for ds_id, ann in api.episode_anns.items()}
    assert by_path == {
        "ep0": (["car"], {0: "frame_000.pcd"}),
        "ep0_01": (["person"], {0: "frame_000.pcd"}),
    }

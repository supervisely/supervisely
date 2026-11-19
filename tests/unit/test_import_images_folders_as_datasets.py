"""Images without annotations: input folders -> nested datasets (``folders_as_datasets``).

Runs ImportManager on local folders with an in-memory stand-in for the API, so the
resulting dataset tree and image names can be checked without a Supervisely instance.
"""

import json
import os
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

import supervisely.convert.converter as converter_module
import supervisely.convert.image.image_converter as image_converter_module
from supervisely.convert.converter import ImportManager
from supervisely.convert.image.image_converter import (
    FOLDER_DATASETS,
    FOLDER_ITEMS,
    ImageConverter,
    group_items_by_folders,
)
from supervisely.project.project_type import ProjectType

# ----------------------------------------------------------------------------- fake API


class _FakeDatasetApi:
    def __init__(self, store):
        self._s = store

    def _info(self, ds):
        return SimpleNamespace(
            id=ds["id"],
            name=ds["name"],
            project_id=1,
            parent_id=ds["parent_id"],
            items_count=len(self._s.images[ds["id"]]),
        )

    def add(self, name, parent_id=None):
        ds_id = 100 + len(self._s.datasets)
        self._s.datasets[ds_id] = {"id": ds_id, "name": name, "parent_id": parent_id}
        self._s.images[ds_id] = []
        return self._info(self._s.datasets[ds_id])

    def get_info_by_id(self, ds_id, raise_error=False):
        return self._info(self._s.datasets[ds_id])

    def get_list(self, project_id, filters=None, recursive=False, parent_id=None):
        res = [
            self._info(d)
            for d in self._s.datasets.values()
            if recursive or d["parent_id"] == parent_id
        ]
        for f in filters or []:
            res = [d for d in res if getattr(d, f["field"]) == f["value"]]
        return res

    def get_info_by_name(self, project_id, name, parent_id=None):
        found = self.get_list(project_id, [{"field": "name", "value": name}], parent_id=parent_id)
        return found[0] if found else None

    def _get_effective_new_name(self, project_id, name, change_name_if_conflict, parent_id=None):
        res, i = name, 1
        while change_name_if_conflict and self.get_info_by_name(project_id, res, parent_id):
            res, i = f"{name}_{i:03d}", i + 1
        return res

    def create(self, project_id, name, change_name_if_conflict=False, parent_id=None, **kw):
        name = self._get_effective_new_name(project_id, name, change_name_if_conflict, parent_id)
        self._s.calls.append(("dataset.create", name, parent_id))
        return self.add(name, parent_id)

    def update(self, ds_id, name=None, **kw):
        self._s.calls.append(("dataset.update", ds_id, name))
        self._s.datasets[ds_id]["name"] = name
        return self._info(self._s.datasets[ds_id])


class _FakeImageApi:
    def __init__(self, store):
        self._s = store

    def get_list(self, ds_id):
        return [SimpleNamespace(name=n) for n in self._s.images[ds_id]]

    def upload_paths(self, ds_id, names, paths, metas=None, conflict_resolution=None, **kw):
        self._s.images[ds_id].extend(names)
        self._s.calls.append(("image.upload_paths", ds_id, tuple(names)))
        infos = []
        for name in names:
            self._s.next_img += 1
            infos.append(SimpleNamespace(id=self._s.next_img, name=name, height=8, width=8))
        return infos


class FakeApi:
    def __init__(self):
        self.datasets = {}
        self.images = {}
        self.calls = []
        self.next_img = 0
        self.optimization_context = {}
        self.dataset = _FakeDatasetApi(self)
        self.image = _FakeImageApi(self)
        self.project = SimpleNamespace(
            get_meta=lambda pid, with_settings=False: {"classes": [], "tags": []},
            update_meta=lambda pid, meta: meta,
        )
        self.annotation = SimpleNamespace(
            upload_anns=lambda ids, anns, **kw: self.calls.append(
                ("annotation.upload_anns", tuple(ids), tuple(len(a.labels) for a in anns))
            )
        )

    def tree(self):
        """{"path/of/dataset": sorted image names}"""

        def path(ds):
            parts = [ds["name"]]
            while ds["parent_id"] is not None:
                ds = self.datasets[ds["parent_id"]]
                parts.insert(0, ds["name"])
            return "/".join(parts)

        return {path(d): sorted(self.images[i]) for i, d in self.datasets.items()}


# ----------------------------------------------------------------------------- inputs


def _img(path, ext="jpg"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arr = np.random.randint(0, 255, (8, 8, 3), dtype=np.uint8)
    Image.fromarray(arr).save(path, format="JPEG" if ext == "jpg" else ext.upper())


def _nested_parity_input(root):
    """The layout of /agents-sandbox/import-parity/images/nested/ (PARITY.md)."""
    for name in ("IMG_1836.jpeg", "IMG_3861.jpeg", "IMG_8144.jpeg"):
        _img(os.path.join(root, "nested", "lemons", name))
    _img(os.path.join(root, "nested", "fruits", "apple_2.jpg"))
    _img(os.path.join(root, "nested", "fruits", "apples", "apple_1.jpg"))
    _img(os.path.join(root, "nested", "fruits", "apples", "pear_3.jpg"))
    _img(os.path.join(root, "nested", "fruits", "kiwis", "IMG_8144.jpeg"))
    _img(os.path.join(root, "nested", "fruits", "kiwis", "pear_1.jpg"))
    return os.path.join(root, "nested")


NESTED_EXPECTED = {
    "lemons": ["IMG_1836.jpeg", "IMG_3861.jpeg", "IMG_8144.jpeg"],
    "fruits": ["apple_2.jpg"],
    "fruits/apples": ["apple_1.jpg", "pear_3.jpg"],
    "fruits/kiwis": ["IMG_8144.jpeg", "pear_1.jpg"],
}


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("SLY_APP_DATA_DIR", str(tmp_path / "app_data"))
    monkeypatch.setenv("TEAM_ID", "1")
    monkeypatch.setattr(converter_module.Api, "from_env", classmethod(lambda cls: object()))
    src = tmp_path / "src"
    src.mkdir()
    return src


TARGET = "dataset 2026-10-25 04-12-28"


def under(target, expected):
    """`expected` (paths relative to the input) as datasets nested inside the target dataset."""
    return {target: [], **{f"{target}/{path}": items for path, items in expected.items()}}


def _run(src, folders_as_datasets, target_name=TARGET, prefill=None):
    """ImportManager + upload into a fresh target dataset, as main-import does."""
    importer = ImportManager(
        str(src), ProjectType.IMAGES, folders_as_datasets=folders_as_datasets
    )
    api = FakeApi()
    target = api.dataset.add(target_name)
    if prefill:
        api.images[target.id].extend(prefill)
    importer._api = api
    importer.upload_dataset(target.id)
    return importer, api


# ----------------------------------------------------------------------------- grouping


def _items(root, rel_paths):
    return [ImageConverter.Item(os.path.join(root, p)) for p in rel_paths]


def _shape(tree):
    return {
        "items": sorted(os.path.basename(i.path) for i in tree[FOLDER_ITEMS]),
        "datasets": {k: _shape(v) for k, v in tree[FOLDER_DATASETS].items()},
    }


def test_grouping_drops_the_input_folder_and_single_folder_chains(tmp_path):
    root = str(tmp_path)
    tree = group_items_by_folders(_items(root, ["in/a/b/x.jpg", "in/a/b/c/y.jpg"]), root)
    # "in" is the input folder, "a" only holds "b"; "b" holds images, so it stays
    assert _shape(tree) == {
        "items": [],
        "datasets": {"b": {"items": ["x.jpg"], "datasets": {"c": {"items": ["y.jpg"], "datasets": {}}}}},
    }


def test_grouping_keeps_root_images_at_the_root(tmp_path):
    root = str(tmp_path)
    tree = group_items_by_folders(_items(root, ["in/x.jpg", "in/sub/y.jpg"]), root)
    assert _shape(tree) == {
        "items": ["x.jpg"],
        "datasets": {"sub": {"items": ["y.jpg"], "datasets": {}}},
    }


def test_grouping_flat_input_has_no_datasets(tmp_path):
    root = str(tmp_path)
    tree = group_items_by_folders(_items(root, ["flat/a.jpg", "flat/b.png"]), root)
    assert _shape(tree) == {"items": ["a.jpg", "b.png"], "datasets": {}}


def test_grouping_several_inputs_become_datasets(tmp_path):
    root = str(tmp_path)
    tree = group_items_by_folders(_items(root, ["A/x.jpg", "B/y.jpg"]), root)
    assert sorted(tree[FOLDER_DATASETS]) == ["A", "B"]


# ----------------------------------------------------------------------------- upload


def test_nested_on_matches_import_images_tree_and_names(env):
    _nested_parity_input(env)
    _, api = _run(env, True)
    # same tree and names as Import Images, nested inside the target dataset (which keeps its name)
    assert api.tree() == under(TARGET, NESTED_EXPECTED)
    assert not [c for c in api.calls if c[0] == "dataset.update"]
    assert ("dataset.create", "lemons", 100) in api.calls


def test_nested_off_is_one_flat_dataset_with_renamed_duplicate(env):
    _nested_parity_input(env)
    _, api = _run(env, False)
    assert api.tree() == {
        "dataset 2026-10-25 04-12-28": sorted(
            [
                "IMG_1836.jpeg",
                "IMG_3861.jpeg",
                "IMG_8144.jpeg",
                "IMG_8144_01.jpeg",
                "apple_1.jpg",
                "apple_2.jpg",
                "pear_1.jpg",
                "pear_3.jpg",
            ]
        )
    }
    assert not [c for c in api.calls if c[0].startswith("dataset.")]


def test_default_keeps_todays_behaviour(env, monkeypatch):
    _nested_parity_input(env)
    assert image_converter_module.DEFAULT_FOLDERS_AS_DATASETS is False
    _, default_api = _run(env, None)
    _, off_api = _run(env, False)
    assert default_api.tree() == off_api.tree()
    monkeypatch.setattr(image_converter_module, "DEFAULT_FOLDERS_AS_DATASETS", True)
    _, flipped_api = _run(env, None)
    assert flipped_api.tree() == under(TARGET, NESTED_EXPECTED)


def test_non_empty_target_is_root_and_never_renamed(env):
    _nested_parity_input(env)
    _, api = _run(env, True, target_name="my data", prefill=["old.jpg"])
    tree = api.tree()
    assert tree["my data"] == ["old.jpg"]
    assert tree["my data/lemons"] == NESTED_EXPECTED["lemons"]
    assert tree["my data/fruits/kiwis"] == NESTED_EXPECTED["fruits/kiwis"]
    assert not [c for c in api.calls if c[0] == "dataset.update"]


def test_root_images_go_into_target_and_folders_nest_inside(env):
    # env is the selected input folder itself (main-import downloads it into one folder)
    _img(str(env / "root.jpg"))
    _img(str(env / "sub" / "root.jpg"))
    _, api = _run(env, True)
    assert api.tree() == {
        "dataset 2026-10-25 04-12-28": ["root.jpg"],
        "dataset 2026-10-25 04-12-28/sub": ["root.jpg"],
    }


def test_single_subfolder_of_the_input_becomes_the_dataset(env):
    # as Import Images: the selected folder is the root, its only subfolder holds images and becomes a
    # dataset — nested inside the target, which keeps its name
    for name in ("a.jpg", "b.jpg"):
        _img(str(env / "photos" / name))
    _, api = _run(env, True)
    assert api.tree() == under(TARGET, {"photos": ["a.jpg", "b.jpg"]})


def test_empty_chosen_dataset_is_never_renamed_and_nothing_lands_outside(env):
    # the user picked an existing, empty dataset in the wizard: everything must go inside it
    _nested_parity_input(env)
    importer = ImportManager(str(env), ProjectType.IMAGES, folders_as_datasets=True)
    api = FakeApi()
    api.dataset.add("fruits")  # a dataset of the user's at the project root, must stay untouched
    target = api.dataset.add("my chosen dataset")
    importer._api = api
    importer.upload_dataset(target.id)
    tree = api.tree()
    assert not [c for c in api.calls if c[0] == "dataset.update"]
    assert tree["fruits"] == []
    assert {k: v for k, v in tree.items() if k != "fruits"} == under("my chosen dataset", NESTED_EXPECTED)


def test_flat_input_identical_with_option_on_and_off(env):
    for name in ("a.jpg", "b.jpg"):
        _img(str(env / name))
    _img(str(env / "c.png"), ext="png")
    (env / "notes.txt").write_text("x")
    _, on = _run(env, True)
    _, off = _run(env, False)
    assert on.calls == off.calls
    assert on.tree() == {"dataset 2026-10-25 04-12-28": ["a.jpg", "b.jpg", "c.png"]}


# ------------------------------------------------- annotation formats are not affected


def _sly_project(root):
    p = os.path.join(root, "proj")
    os.makedirs(p)
    meta = {"classes": [{"title": "box", "shape": "rectangle", "color": "#FF0000"}], "tags": []}
    with open(os.path.join(p, "meta.json"), "w") as f:
        json.dump(meta, f)
    for ds in ("ds1", "ds2"):
        _img(os.path.join(p, ds, "img", "a.jpg"))
        os.makedirs(os.path.join(p, ds, "ann"))
        ann = {
            "size": {"height": 8, "width": 8},
            "tags": [],
            "objects": [
                {
                    "classTitle": "box",
                    "tags": [],
                    "points": {"exterior": [[1, 1], [4, 4]], "interior": []},
                }
            ],
        }
        with open(os.path.join(p, ds, "ann", "a.jpg.json"), "w") as f:
            json.dump(ann, f)


def _coco(root):
    d = os.path.join(root, "coco", "part1")
    _img(os.path.join(d, "images", "a.jpg"))
    _img(os.path.join(d, "images", "b.jpg"))
    os.makedirs(os.path.join(d, "annotations"))
    data = {
        "images": [
            {"id": 1, "file_name": "a.jpg", "height": 8, "width": 8},
            {"id": 2, "file_name": "b.jpg", "height": 8, "width": 8},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [1, 1, 3, 3], "area": 9,
             "iscrowd": 0, "segmentation": []}
        ],
        "categories": [{"id": 1, "name": "box"}],
    }
    with open(os.path.join(d, "annotations", "instances.json"), "w") as f:
        json.dump(data, f)


def _yolo(root):
    d = os.path.join(root, "yolo")
    _img(os.path.join(d, "images", "train", "a.jpg"))
    _img(os.path.join(d, "images", "val", "b.jpg"))
    os.makedirs(os.path.join(d, "labels", "train"))
    os.makedirs(os.path.join(d, "labels", "val"))
    with open(os.path.join(d, "labels", "train", "a.txt"), "w") as f:
        f.write("0 0.5 0.5 0.25 0.25\n")
    with open(os.path.join(d, "labels", "val", "b.txt"), "w") as f:
        f.write("0 0.5 0.5 0.25 0.25\n")
    with open(os.path.join(d, "data.yaml"), "w") as f:
        f.write("names: ['box']\nnc: 1\ntrain: images/train\nval: images/val\n")


def _images_with_meta(root):
    d = os.path.join(root, "withmeta")
    _img(os.path.join(d, "sub", "a.jpg"))
    os.makedirs(os.path.join(d, "meta"))
    with open(os.path.join(d, "meta", "a.jpg.json"), "w") as f:
        json.dump({"k": "v"}, f)


@pytest.mark.parametrize(
    "make, fmt",
    [
        (_sly_project, "supervisely"),
        (_coco, "coco"),
        (_yolo, "yolo"),
        (_images_with_meta, None),  # images + meta/ folder: plain images, kept flat
    ],
)
def test_other_inputs_upload_exactly_as_before(env, make, fmt):
    make(str(env))
    importer_on, on = _run(env, True)
    importer_off, off = _run(env, False)
    if fmt is not None:
        assert str(importer_on.converter) == fmt
    assert str(importer_on.converter) == str(importer_off.converter)
    assert on.calls == off.calls
    assert on.tree() == off.tree()

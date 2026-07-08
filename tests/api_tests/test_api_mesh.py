# isort:skip_file
"""
Integration tests for the Mesh API.

Tests are numbered sequentially so unittest runs them in the right order:
  01 - Create nested datasets
  02 - Upload meshes (2 per dataset, each a named 3-D shape)
  03 - Create project meta (classes + tags named after shapes)
  04 - Upload annotations (entity tags + label/object tags with shape-id values)
  05 - Download full project and verify structure / annotations
  06 - Re-upload the downloaded project and verify on server

Shapes layout:
  dataset parent : 1_box.obj, 2_tetrahedron.obj
  dataset child1 : 3_octahedron.obj, 4_pyramid.obj
  dataset child2 : 5_prism.obj, 6_diamond.obj

Per shape S (example: S = box, id = 1):
  class            : box                  (Mesh geometry)
  entity-level tag : box_mesh             (AnyNumber, value = 1)
  label-level tag  : box_obj              (AnyNumber, value = 1)
  both-level tag   : box_universal        (AnyNumber, value = 1)

Run:
    WORKSPACE_ID=309 python -m pytest tests/api_tests/test_api_mesh.py -v -s
"""

import json
import os
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

sdk_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, sdk_path)

from supervisely.annotation.obj_class import ObjClass
from supervisely.annotation.tag_meta import TagApplicableTo, TagMeta, TagValueType
from supervisely.api.api import Api
from supervisely.api.mesh.mesh_api import MeshInfo
from supervisely.geometry.mesh import Mesh
from supervisely.mesh_annotation.mesh_annotation import MeshAnnotation
from supervisely.mesh_annotation.mesh_label import MeshLabel
from supervisely.mesh_annotation.mesh_tag import MeshTag
from supervisely.mesh_annotation.mesh_tag_collection import MeshTagCollection
from supervisely.project.mesh_project import MeshProject
from supervisely.project.project_meta import ProjectMeta
from supervisely.project.project_type import ProjectType


# ---------------------------------------------------------------------------
# State file (persists IDs between individual pytest runs)
# ---------------------------------------------------------------------------

_STATE_FILE = os.path.join(tempfile.gettempdir(), "sly_mesh_ut_state.json")


def _save_state(cls) -> None:
    state = {
        "project_id": cls.project_id,
        "workspace_id": cls.workspace_id,
        "ds_parent_id": cls.ds_parent_id,
        "ds_child1_id": cls.ds_child1_id,
        "ds_child2_id": cls.ds_child2_id,
        "mesh_ids_parent": cls.mesh_ids_parent,
        "mesh_ids_child1": cls.mesh_ids_child1,
        "mesh_ids_child2": cls.mesh_ids_child2,
        "download_dir": cls.download_dir,
        "reuploaded_project_id": cls.reuploaded_project_id,
    }
    with open(_STATE_FILE, "w") as f:
        json.dump(state, f)


def _load_state(cls) -> bool:
    if not os.path.exists(_STATE_FILE):
        return False
    with open(_STATE_FILE) as f:
        state = json.load(f)
    cls.project_id = state.get("project_id")
    cls.workspace_id = state.get("workspace_id")
    cls.ds_parent_id = state.get("ds_parent_id")
    cls.ds_child1_id = state.get("ds_child1_id")
    cls.ds_child2_id = state.get("ds_child2_id")
    cls.mesh_ids_parent = state.get("mesh_ids_parent", [])
    cls.mesh_ids_child1 = state.get("mesh_ids_child1", [])
    cls.mesh_ids_child2 = state.get("mesh_ids_child2", [])
    cls.download_dir = state.get("download_dir")
    cls.reuploaded_project_id = state.get("reuploaded_project_id")
    return True


# ---------------------------------------------------------------------------
# Shape catalogue
# ---------------------------------------------------------------------------

SHAPES = [
    {"id": 1, "name": "box",         "n_verts": 8, "dataset": "parent"},
    {"id": 2, "name": "tetrahedron", "n_verts": 4, "dataset": "parent"},
    {"id": 3, "name": "octahedron",  "n_verts": 6, "dataset": "child1"},
    {"id": 4, "name": "pyramid",     "n_verts": 5, "dataset": "child1"},
    {"id": 5, "name": "prism",       "n_verts": 6, "dataset": "child2"},
    {"id": 6, "name": "diamond",     "n_verts": 6, "dataset": "child2"},
]

DS_SHAPES: dict = {}
for _s in SHAPES:
    DS_SHAPES.setdefault(_s["dataset"], []).append(_s)


def _tag_mesh(name: str) -> str:
    return f"{name}_mesh"       # entity-level only

def _tag_obj(name: str) -> str:
    return f"{name}_obj"        # label-level only

def _tag_uni(name: str) -> str:
    return f"{name}_universal"  # both entity and label


# ---------------------------------------------------------------------------
# OBJ writers — one per shape
# ---------------------------------------------------------------------------

def _write_box(path: str) -> None:
    """Cube: 8 vertices, 6 quad faces."""
    with open(path, "w") as f:
        f.write("# box (cube)\n")
        f.write("v -1 -1 -1\nv  1 -1 -1\nv  1  1 -1\nv -1  1 -1\n")
        f.write("v -1 -1  1\nv  1 -1  1\nv  1  1  1\nv -1  1  1\n")
        f.write("f 1 2 3 4\nf 5 8 7 6\n")
        f.write("f 1 5 6 2\nf 2 6 7 3\nf 3 7 8 4\nf 4 8 5 1\n")


def _write_tetrahedron(path: str) -> None:
    """Regular tetrahedron: 4 vertices, 4 triangular faces."""
    with open(path, "w") as f:
        f.write("# tetrahedron\n")
        f.write("v  1  1  1\nv  1 -1 -1\nv -1  1 -1\nv -1 -1  1\n")
        f.write("f 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n")


def _write_octahedron(path: str) -> None:
    """Regular octahedron: 6 vertices, 8 triangular faces."""
    with open(path, "w") as f:
        f.write("# octahedron\n")
        f.write("v  1  0  0\nv -1  0  0\nv  0  1  0\n")
        f.write("v  0 -1  0\nv  0  0  1\nv  0  0 -1\n")
        f.write("f 1 3 5\nf 3 2 5\nf 2 4 5\nf 4 1 5\n")
        f.write("f 3 1 6\nf 2 3 6\nf 4 2 6\nf 1 4 6\n")


def _write_pyramid(path: str) -> None:
    """Square-base pyramid: 5 vertices, 5 faces (1 quad + 4 tri)."""
    with open(path, "w") as f:
        f.write("# pyramid (square base)\n")
        f.write("v -1 -1  0\nv  1 -1  0\nv  1  1  0\nv -1  1  0\nv  0  0  2\n")
        f.write("f 1 4 3 2\nf 1 2 5\nf 2 3 5\nf 3 4 5\nf 4 1 5\n")


def _write_prism(path: str) -> None:
    """Triangular prism: 6 vertices, 5 faces (2 tri + 3 quad)."""
    with open(path, "w") as f:
        f.write("# triangular prism\n")
        f.write("v -1 -1  0\nv  1 -1  0\nv  0  1  0\n")
        f.write("v -1 -1  2\nv  1 -1  2\nv  0  1  2\n")
        f.write("f 1 3 2\nf 4 5 6\n")
        f.write("f 1 2 5 4\nf 2 3 6 5\nf 3 1 4 6\n")


def _write_diamond(path: str) -> None:
    """Diamond (bipyramid): 6 vertices, 8 triangular faces."""
    with open(path, "w") as f:
        f.write("# diamond (bipyramid)\n")
        f.write("v  1  0  0\nv -1  0  0\nv  0  1  0\n")
        f.write("v  0 -1  0\nv  0  0  1.5\nv  0  0 -1.5\n")
        f.write("f 1 3 5\nf 3 2 5\nf 2 4 5\nf 4 1 5\n")
        f.write("f 3 1 6\nf 2 3 6\nf 4 2 6\nf 1 4 6\n")


SHAPE_WRITERS = {
    "box":         _write_box,
    "tetrahedron": _write_tetrahedron,
    "octahedron":  _write_octahedron,
    "pyramid":     _write_pyramid,
    "prism":       _write_prism,
    "diamond":     _write_diamond,
}


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------

class TestMeshApi(unittest.TestCase):
    """End-to-end tests for the Mesh API (runs in numeric order)."""

    api: Api = None
    workspace_id: int = None

    project_id: int = None
    ds_parent_id: int = None
    ds_child1_id: int = None
    ds_child2_id: int = None

    mesh_ids_parent: list = []
    mesh_ids_child1: list = []
    mesh_ids_child2: list = []

    download_dir: str = None
    reuploaded_project_id: int = None

    # ------------------------------------------------------------------
    # Setup / teardown
    # ------------------------------------------------------------------

    @classmethod
    def setUpClass(cls):
        from dotenv import load_dotenv
        load_dotenv(os.path.expanduser("~/supervisely.env"))

        cls.api = Api.from_env()

        if _load_state(cls):
            # Validate that the saved project still exists.
            try:
                info = cls.api.project.get_info_by_id(cls.project_id)
                if info is not None:
                    print(f"\n[setup] Restored state — project_id={cls.project_id}")
                    return
            except Exception:
                pass
            print(f"\n[setup] Saved project {cls.project_id} gone — creating fresh")

        workspace_id_str = os.environ.get("WORKSPACE_ID") or input(
            "Enter workspace ID >> "
        )
        cls.workspace_id = int(workspace_id_str)
        with patch("builtins.input", return_value="user_input_value"):
            project = cls.api.project.create(
                workspace_id=cls.workspace_id,
                name="[UT] Mesh api",
                type=ProjectType.MESHES,
                change_name_if_conflict=True,
            )
        cls.project_id = project.id
        _save_state(cls)
        print(f"\n[setup] Created project id={cls.project_id}")

    @classmethod
    def tearDownClass(cls):
        # Set MESH_UT_CLEANUP=1 to enable cleanup (default: skip for step-by-step runs).
        if not os.environ.get("MESH_UT_CLEANUP"):
            print(
                f"\n[teardown] SKIPPED (set MESH_UT_CLEANUP=1 to enable) — "
                f"project_id={cls.project_id}, "
                f"reuploaded_id={cls.reuploaded_project_id}, "
                f"download_dir={cls.download_dir}"
            )
            return
        for pid, label in [
            (cls.project_id, "project"),
            (cls.reuploaded_project_id, "re-uploaded project"),
        ]:
            if not pid:
                continue
            try:
                cls.api.project.remove_permanently(pid)
                print(f"\n[teardown] Removed {label} {pid}")
            except Exception as e:
                print(f"\n[teardown] Could not remove {label} {pid}: {e}")
        if cls.download_dir and os.path.isdir(cls.download_dir):
            shutil.rmtree(cls.download_dir, ignore_errors=True)
        if os.path.exists(_STATE_FILE):
            os.remove(_STATE_FILE)

    # ------------------------------------------------------------------
    # 01 - Nested datasets
    # ------------------------------------------------------------------

    def test_01_create_nested_datasets(self):
        """Create parent dataset and two child datasets nested under it."""
        parent = self.api.dataset.create(
            self.project_id, name="parent", change_name_if_conflict=True
        )
        TestMeshApi.ds_parent_id = parent.id

        child1 = self.api.dataset.create(
            self.project_id,
            name="child1",
            parent_id=parent.id,
            change_name_if_conflict=True,
        )
        TestMeshApi.ds_child1_id = child1.id

        child2 = self.api.dataset.create(
            self.project_id,
            name="child2",
            parent_id=parent.id,
            change_name_if_conflict=True,
        )
        TestMeshApi.ds_child2_id = child2.id

        children = self.api.dataset.get_list(self.project_id, parent_id=parent.id)
        child_ids = {ds.id for ds in children}
        self.assertIn(child1.id, child_ids)
        self.assertIn(child2.id, child_ids)

        _save_state(TestMeshApi)
        print(f"[01] parent={parent.id}, child1={child1.id}, child2={child2.id}")

    # ------------------------------------------------------------------
    # 02 - Upload meshes
    # ------------------------------------------------------------------

    def test_02_upload_meshes(self):
        """Upload two named 3-D shape meshes per dataset."""
        self.assertIsNotNone(self.ds_parent_id, "test_01 must run first")

        with tempfile.TemporaryDirectory() as tmpdir:
            def _upload_ds(ds_id: int, shapes: list) -> list:
                names, paths = [], []
                for shape in shapes:
                    fname = f"{shape['id']}_{shape['name']}.obj"
                    fpath = os.path.join(tmpdir, fname)
                    SHAPE_WRITERS[shape["name"]](fpath)
                    names.append(fname)
                    paths.append(fpath)
                infos = self.api.mesh.upload_paths(ds_id, names, paths)
                self.assertEqual(len(infos), len(shapes))
                for info in infos:
                    self.assertIsInstance(info, MeshInfo)
                    self.assertIsInstance(info.id, int)
                    self.assertEqual(info.dataset_id, ds_id)
                return [info.id for info in infos]

            TestMeshApi.mesh_ids_parent = _upload_ds(
                self.ds_parent_id, DS_SHAPES["parent"]
            )
            TestMeshApi.mesh_ids_child1 = _upload_ds(
                self.ds_child1_id, DS_SHAPES["child1"]
            )
            TestMeshApi.mesh_ids_child2 = _upload_ds(
                self.ds_child2_id, DS_SHAPES["child2"]
            )

        total = (
            len(self.mesh_ids_parent)
            + len(self.mesh_ids_child1)
            + len(self.mesh_ids_child2)
        )
        self.assertEqual(total, 6)
        _save_state(TestMeshApi)
        print(
            f"[02] uploaded 6 meshes — "
            f"parent={self.mesh_ids_parent}, "
            f"child1={self.mesh_ids_child1}, "
            f"child2={self.mesh_ids_child2}"
        )

    # ------------------------------------------------------------------
    # 03 - Project meta
    # ------------------------------------------------------------------

    def test_03_create_project_meta(self):
        """Create 6 shape classes and 3 tags per shape (mesh/obj/universal)."""
        obj_classes = [ObjClass(s["name"], Mesh) for s in SHAPES]
        tag_metas = []
        for s in SHAPES:
            tag_metas.append(TagMeta(_tag_mesh(s["name"]), TagValueType.ANY_NUMBER,
                                     applicable_to=TagApplicableTo.IMAGES_ONLY))
            tag_metas.append(TagMeta(_tag_obj(s["name"]),  TagValueType.ANY_NUMBER,
                                     applicable_to=TagApplicableTo.OBJECTS_ONLY))
            tag_metas.append(TagMeta(_tag_uni(s["name"]),  TagValueType.ANY_NUMBER,
                                     applicable_to=TagApplicableTo.ALL))

        meta = ProjectMeta(obj_classes=obj_classes, tag_metas=tag_metas)
        self.api.project.update_meta(self.project_id, meta.to_json())

        fetched = ProjectMeta.from_json(self.api.project.get_meta(self.project_id))
        class_names = {cls.name for cls in fetched.obj_classes}
        tag_names = {tm.name for tm in fetched.tag_metas}

        for s in SHAPES:
            self.assertIn(s["name"],          class_names)
            self.assertIn(_tag_mesh(s["name"]), tag_names)
            self.assertIn(_tag_obj(s["name"]),  tag_names)
            self.assertIn(_tag_uni(s["name"]),  tag_names)

        _save_state(TestMeshApi)
        print(
            f"[03] meta: {len(class_names)} classes — {sorted(class_names)}, "
            f"{len(tag_metas)} tags"
        )

    # ------------------------------------------------------------------
    # 04 - Upload annotations
    # ------------------------------------------------------------------

    def test_04_upload_annotations(self):
        """Annotate every mesh: 1 label + entity tags + label tags."""
        self.assertTrue(self.mesh_ids_parent, "test_02 must run first")

        project_meta = ProjectMeta.from_json(
            self.api.project.get_meta(self.project_id)
        )

        # Build pairs: (mesh_id, shape_dict) in upload order.
        mesh_shape_pairs = (
            list(zip(self.mesh_ids_parent, DS_SHAPES["parent"]))
            + list(zip(self.mesh_ids_child1, DS_SHAPES["child1"]))
            + list(zip(self.mesh_ids_child2, DS_SHAPES["child2"]))
        )

        for mesh_id, shape in mesh_shape_pairs:
            sname = shape["name"]
            sid   = shape["id"]
            nverts = shape["n_verts"]

            obj_class = project_meta.get_obj_class(sname)
            tm_mesh = project_meta.get_tag_meta(_tag_mesh(sname))
            tm_obj  = project_meta.get_tag_meta(_tag_obj(sname))
            tm_uni  = project_meta.get_tag_meta(_tag_uni(sname))

            label = MeshLabel(
                geometry=Mesh(list(range(nverts))),
                obj_class=obj_class,
                tags=MeshTagCollection([
                    MeshTag(tm_obj, value=sid),
                    MeshTag(tm_uni, value=sid),
                ]),
            )
            ann = MeshAnnotation(
                labels=[label],
                tags=MeshTagCollection([
                    MeshTag(tm_mesh, value=sid),
                    MeshTag(tm_uni, value=sid),
                ]),
            )
            self.api.mesh.annotation.append(mesh_id, ann)

        # Spot-check: first mesh in parent dataset → box (id=1).
        first_id = self.mesh_ids_parent[0]
        ann_json = self.api.mesh.annotation.download(first_id)
        self.assertEqual(ann_json.get("meshId"), first_id)
        # 1 label with class "box"
        self.assertEqual(len(ann_json.get("labels", [])), 1)
        self.assertEqual(ann_json["labels"][0]["classTitle"], "box")
        # 2 entity tags: box_mesh + box_universal
        self.assertEqual(len(ann_json.get("tags", [])), 2)
        # 2 label tags: box_obj + box_universal
        self.assertEqual(len(ann_json["labels"][0].get("tags", [])), 2)
        # geometry indices present
        indices = ann_json["labels"][0]["geometry"]["indices"]
        self.assertEqual(indices, list(range(8)))  # box has 8 vertices

        # Verify every mesh has correct shape-specific content.
        self._assert_project_content_correct(self.project_id)

        _save_state(TestMeshApi)
        print(f"[04] annotations uploaded for {len(mesh_shape_pairs)} meshes — content verified ✓")

    # ------------------------------------------------------------------
    # 05 - Download project
    # ------------------------------------------------------------------

    def test_05_download_project(self):
        """Download the full project to a local directory and verify structure."""
        self.assertTrue(self.mesh_ids_parent, "test_02 must run first")

        TestMeshApi.download_dir = tempfile.mkdtemp(prefix="sly_mesh_ut_")

        project_fs = MeshProject.download(
            api=self.api,
            project_id=self.project_id,
            dest_dir=self.download_dir,
            download_meshes=True,
            log_progress=False,
        )

        self.assertIsInstance(project_fs, MeshProject)
        self.assertEqual(project_fs.meta.project_type, ProjectType.MESHES.value)

        # Nested datasets use path-style names locally.
        ds_names = {ds.name for ds in project_fs.datasets}
        self.assertIn("parent",       ds_names)
        self.assertIn("parent/child1", ds_names)
        self.assertIn("parent/child2", ds_names)

        # 2 meshes × 3 datasets = 6 total.
        total_items = sum(len(ds) for ds in project_fs.datasets)
        self.assertEqual(total_items, 6)

        # All 6 shape classes present in meta.
        class_names = {cls.name for cls in project_fs.meta.obj_classes}
        for s in SHAPES:
            self.assertIn(s["name"], class_names)

        # Annotation spot-check: parent dataset first item → box.
        parent_ds = next(ds for ds in project_fs.datasets if ds.name == "parent")
        first_item_name = next(iter(parent_ds))
        ann_json = parent_ds.get_ann_json(first_item_name)
        self.assertEqual(len(ann_json.get("labels", [])), 1)
        self.assertEqual(ann_json["labels"][0]["classTitle"], "box")
        self.assertEqual(len(ann_json.get("tags", [])), 2)

        _save_state(TestMeshApi)
        print(
            f"[05] downloaded to {self.download_dir} — "
            f"datasets={ds_names}, total_items={total_items}"
        )

    # ------------------------------------------------------------------
    # 06 - Re-upload project + deep cross-project comparison
    # ------------------------------------------------------------------

    def _assert_project_content_correct(self, project_id: int) -> None:
        """Verify every mesh has the correct shape class, tags, and geometry indices."""
        # Build name → shape lookup: "1_box.obj" → SHAPES[0]
        shape_by_filename = {f"{s['id']}_{s['name']}.obj": s for s in SHAPES}

        issues = []
        datasets = self.api.dataset.get_list(project_id, recursive=True)
        for ds in datasets:
            for mesh_info in self.api.mesh.get_list(ds.id):
                shape = shape_by_filename.get(mesh_info.name)
                if shape is None:
                    issues.append(f"Unexpected mesh name '{mesh_info.name}' in dataset '{ds.name}'")
                    continue

                sname = shape["name"]
                sid   = shape["id"]
                nverts = shape["n_verts"]
                loc = f"ds='{ds.name}' mesh='{mesh_info.name}'"

                ann = self.api.mesh.annotation.download(mesh_info.id)

                # Entity tags must be {shape}_mesh=sid and {shape}_universal=sid
                etags = {t["name"]: t["value"] for t in ann.get("tags", [])}
                expected_etags = {_tag_mesh(sname): sid, _tag_uni(sname): sid}
                if etags != expected_etags:
                    issues.append(f"{loc}: entity tags {etags} != {expected_etags}")

                # Exactly 1 label
                labels = ann.get("labels", [])
                if len(labels) != 1:
                    issues.append(f"{loc}: expected 1 label, got {len(labels)}")
                    continue

                label = labels[0]

                # Class must match shape name
                if label["classTitle"] != sname:
                    issues.append(f"{loc}: classTitle '{label['classTitle']}' != '{sname}'")

                # Label tags must be {shape}_obj=sid and {shape}_universal=sid
                ltags = {t["name"]: t["value"] for t in label.get("tags", [])}
                expected_ltags = {_tag_obj(sname): sid, _tag_uni(sname): sid}
                if ltags != expected_ltags:
                    issues.append(f"{loc}: label tags {ltags} != {expected_ltags}")

                # Geometry indices must match vertex count
                indices = label.get("geometry", {}).get("indices")
                expected_indices = list(range(nverts))
                if indices != expected_indices:
                    issues.append(f"{loc}: indices {indices} != {expected_indices}")

        if issues:
            self.fail("Content errors:\n  " + "\n  ".join(issues))

    def _assert_projects_equal(self, src_id: int, dst_id: int) -> None:
        """Compare two mesh projects exhaustively: meta, datasets, mesh names, annotations."""

        # ── 1. Project meta ───────────────────────────────────────────
        src_meta = ProjectMeta.from_json(self.api.project.get_meta(src_id))
        dst_meta = ProjectMeta.from_json(self.api.project.get_meta(dst_id))

        src_classes = {c.name: c for c in src_meta.obj_classes}
        dst_classes = {c.name: c for c in dst_meta.obj_classes}
        self.assertEqual(set(src_classes), set(dst_classes), "class names differ")

        src_tags = {t.name: t for t in src_meta.tag_metas}
        dst_tags = {t.name: t for t in dst_meta.tag_metas}
        self.assertEqual(set(src_tags), set(dst_tags), "tag names differ")
        for name in src_tags:
            self.assertEqual(
                src_tags[name].value_type, dst_tags[name].value_type,
                f"tag '{name}' value_type differs"
            )
            self.assertEqual(
                src_tags[name].applicable_to, dst_tags[name].applicable_to,
                f"tag '{name}' applicable_to differs"
            )

        # ── 2. Datasets ───────────────────────────────────────────────
        src_datasets = {ds.name: ds for ds in self.api.dataset.get_list(src_id, recursive=True)}
        dst_datasets = {ds.name: ds for ds in self.api.dataset.get_list(dst_id, recursive=True)}
        self.assertEqual(set(src_datasets), set(dst_datasets), "dataset names differ")

        # ── 3. Mesh names + annotations ───────────────────────────────
        issues = []
        for ds_name in sorted(src_datasets):
            src_meshes = {m.name: m for m in self.api.mesh.get_list(src_datasets[ds_name].id)}
            dst_meshes = {m.name: m for m in self.api.mesh.get_list(dst_datasets[ds_name].id)}
            self.assertEqual(
                set(src_meshes), set(dst_meshes),
                f"mesh names differ in dataset '{ds_name}'"
            )

            for mesh_name in sorted(src_meshes):
                src_ann = self.api.mesh.annotation.download(src_meshes[mesh_name].id)
                dst_ann = self.api.mesh.annotation.download(dst_meshes[mesh_name].id)

                loc = f"ds='{ds_name}' mesh='{mesh_name}'"

                # Entity-level tags: {name: value}
                src_etags = {t["name"]: t["value"] for t in src_ann.get("tags", [])}
                dst_etags = {t["name"]: t["value"] for t in dst_ann.get("tags", [])}
                if src_etags != dst_etags:
                    issues.append(f"{loc}: entity tags {src_etags} → {dst_etags}")

                # Labels
                src_labels = src_ann.get("labels", [])
                dst_labels = dst_ann.get("labels", [])
                if len(src_labels) != len(dst_labels):
                    issues.append(f"{loc}: label count {len(src_labels)} → {len(dst_labels)}")
                    continue

                for i, (sl, dl) in enumerate(zip(src_labels, dst_labels)):
                    if sl["classTitle"] != dl["classTitle"]:
                        issues.append(f"{loc} label[{i}]: class '{sl['classTitle']}' → '{dl['classTitle']}'")

                    # Label-level tags: {name: value}
                    src_ltags = {t["name"]: t["value"] for t in sl.get("tags", [])}
                    dst_ltags = {t["name"]: t["value"] for t in dl.get("tags", [])}
                    if src_ltags != dst_ltags:
                        issues.append(f"{loc} label[{i}]: label tags {src_ltags} → {dst_ltags}")

                    # Geometry indices
                    src_idx = sl.get("geometry", {}).get("indices")
                    dst_idx = dl.get("geometry", {}).get("indices")
                    if src_idx != dst_idx:
                        issues.append(f"{loc} label[{i}]: geometry indices differ")

        if issues:
            self.fail("Projects differ:\n  " + "\n  ".join(issues))

    def test_06_reupload_project(self):
        """Re-upload the downloaded project and do a full cross-project comparison."""
        self.assertIsNotNone(self.download_dir, "test_05 must run first")

        reuploaded_id, reuploaded_name = MeshProject.upload(
            directory=self.download_dir,
            api=self.api,
            workspace_id=self.workspace_id,
            project_name="[UT] Mesh api (re-upload)",
            log_progress=False,
        )
        TestMeshApi.reuploaded_project_id = reuploaded_id

        # Basic sanity checks.
        project_info = self.api.project.get_info_by_id(reuploaded_id)
        self.assertEqual(project_info.type, str(ProjectType.MESHES))

        datasets = self.api.dataset.get_list(reuploaded_id, recursive=True)
        self.assertGreaterEqual(len(datasets), 3)
        total = sum(ds.items_count for ds in datasets)
        self.assertEqual(total, 6)

        # Verify re-uploaded project has correct shape-specific content.
        self._assert_project_content_correct(reuploaded_id)

        # Deep comparison: original vs re-uploaded.
        self._assert_projects_equal(self.project_id, reuploaded_id)

        _save_state(TestMeshApi)
        print(
            f"[06] re-uploaded id={reuploaded_id} name={reuploaded_name!r}, "
            f"datasets={len(datasets)}, items={total} — projects match ✓"
        )


if __name__ == "__main__":
    unittest.main()

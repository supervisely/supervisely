# coding: utf-8

import os
import tempfile
import unittest
import uuid
from types import SimpleNamespace
from unittest.mock import patch

import requests

from supervisely.annotation.obj_class import ObjClass
from supervisely.annotation.tag_meta import TagMeta, TagValueType
from supervisely.api.mesh import mesh_annotation_api as mesh_annotation_api_module
from supervisely.api.mesh.mesh_api import MeshApi, MeshInfo
from supervisely.api.mesh.mesh_tag_api import MeshTagApi
from supervisely.api.module_api import ApiField
from supervisely.convert.base_converter import AvailableMeshConverters
from supervisely.convert.mesh.mesh_converter import MeshConverter
from supervisely.convert.mesh.per_vertex_labels.per_vertex_labels_converter import (
    PerVertexLabelsMeshConverter,
)
from supervisely.convert.mesh.sly.sly_mesh_converter import SLYMeshConverter
from supervisely.geometry.mesh import Mesh
from supervisely.io.json import dump_json_file, load_json_file
from supervisely.mesh_annotation.mesh_annotation import MeshAnnotation
from supervisely.mesh_annotation.mesh_indices import (
    decode_mesh_indices,
    decode_mesh_indices_base64,
    decode_mesh_indices_in_json,
    encode_mesh_indices,
    encode_mesh_indices_base64,
    encode_mesh_indices_in_json,
)
from supervisely.mesh_annotation.mesh_label import MeshLabel
from supervisely.mesh_annotation.mesh_tag import MeshTag
from supervisely.mesh_annotation.mesh_tag_collection import MeshTagCollection
from supervisely.project import read_project
from supervisely.project.mesh_project import GEOMETRIES_DIR_NAME, MeshProject
from supervisely.project.project import OpenMode
from supervisely.project.project_meta import ProjectMeta
from supervisely.project.project_type import ProjectType
from supervisely.video_annotation.key_id_map import KeyIdMap


def mesh_json(mesh_id=1, name="mesh.obj", **kwargs):
    data = {
        "id": mesh_id,
        "name": name,
        "title": name,
        "description": "",
        "parentId": None,
        "workspaceId": 2,
        "projectId": 3,
        "datasetId": 4,
        "pathOriginal": None,
        "fullStorageUrl": None,
        "link": None,
        "meta": {},
        "fileMeta": {},
        "frame": None,
        "size": None,
        "customData": {},
        "objectsCount": 0,
        "tags": [],
        "createdBy": 5,
        "createdAt": "2026-01-01T00:00:00.000Z",
        "updatedAt": "2026-01-01T00:00:00.000Z",
    }
    data.update(kwargs)
    return data


def tag_info_json(tag_id=55, name="weather", **kwargs):
    data = {
        "id": tag_id,
        "projectId": 3,
        "name": name,
        "settings": {},
        "color": [255, 0, 0],
        "createdAt": "2026-01-01T00:00:00.000Z",
        "updatedAt": "2026-01-01T00:00:00.000Z",
    }
    data.update(kwargs)
    return data


def object_json(object_id=300, class_id=44, entity_id=None, **kwargs):
    data = {
        "id": object_id,
        "description": "",
        "createdAt": "2026-01-01T00:00:00.000Z",
        "updatedAt": "2026-01-01T00:00:00.000Z",
        "datasetId": 4,
        "classId": class_id,
        "entityId": entity_id,
        "tags": [],
        "meta": {},
        "createdBy": 5,
    }
    data.update(kwargs)
    return data


def figure_json(figure_id=700, mesh_id=123, object_id=300, geometry=None, **kwargs):
    data = {
        "id": figure_id,
        "classId": 44,
        "createdAt": "2026-01-01T00:00:00.000Z",
        "updatedAt": "2026-01-01T00:00:00.000Z",
        "entityId": mesh_id,
        "objectId": object_id,
        "projectId": 3,
        "datasetId": 4,
        "frameIndex": 0,
        "geometryType": "mesh",
        "geometry": geometry or {"indices": [0, 1, 2, 42, 65535]},
        "geometryMeta": {},
        "tags": [],
        "meta": {},
        "area": None,
        "priority": None,
        "customData": None,
    }
    data.update(kwargs)
    return data


class Response:
    def __init__(self, value):
        self.value = value

    def json(self):
        return self.value

    def iter_content(self, chunk_size=1024):
        if isinstance(self.value, bytes):
            yield self.value


class FakeFileApi:
    def __init__(self):
        self.free_name_calls = []
        self.upload_bulk_calls = []

    def get_free_name(self, team_id, path):
        self.free_name_calls.append((team_id, path))
        return path

    def upload_bulk(self, team_id, src_paths, dst_paths, progress_cb=None):
        self.upload_bulk_calls.append((team_id, src_paths, dst_paths, progress_cb))
        return [
            SimpleNamespace(id=1000 + idx, full_storage_url=f"https://files.example/{idx}.stl")
            for idx, _ in enumerate(src_paths)
        ]


class FakeDatasetApi:
    def get_info_by_id(self, dataset_id):
        return SimpleNamespace(id=dataset_id, team_id=77, project_id=3)


class FakeProjectApi:
    def __init__(self):
        self.updated_meta = None
        self.meta = ProjectMeta(project_type=ProjectType.MESHES)

    def get_meta(self, project_id, with_settings=False):
        return self.meta.to_json()

    def update_meta(self, project_id, meta):
        self.updated_meta = meta
        if isinstance(meta, ProjectMeta):
            self.meta = meta
            return meta
        self.meta = ProjectMeta.from_json(meta)
        return self.meta


class FakeObjectClassApi:
    def get_name_to_id_map(self, project_id):
        return {"car": 44, "any": 45}

    def get_list(self, project_id):
        return [SimpleNamespace(id=44, name="car"), SimpleNamespace(id=45, name="any")]


class FakeApi:
    def __init__(self):
        self.calls = []
        self.file = FakeFileApi()
        self.dataset = FakeDatasetApi()
        self.project = FakeProjectApi()
        self.object_class = FakeObjectClassApi()
        self.annotation_response = None
        self.object_response = []
        self.figure_response = []
        self.entity_response = []
        self.tag_response = []
        self.mesh_infos = {}
        self.fail_team_file_id_upload = False

    def post(self, method, data, **kwargs):
        self.calls.append((method, data, kwargs))
        if method == "entities.list":
            return Response(
                {
                    "total": len(self.entity_response),
                    "perPage": 100,
                    "pagesCount": 1,
                    "entities": self.entity_response,
                }
            )
        if method == "entities.info":
            return Response(self.mesh_infos.get(data["id"], mesh_json(mesh_id=data["id"])))
        if method == "entities.download":
            return Response(b"mesh-bytes")
        if method == "entities.bulk.add":
            if self.fail_team_file_id_upload and ApiField.TEAM_FILE_ID in data["entities"][0]:
                raise http_error(404, "The following entities with teamFileId were not found")
            return Response(
                [
                    mesh_json(
                        mesh_id=idx + 1,
                        name=entity["name"],
                        link=entity.get("link"),
                        pathOriginal=None,
                    )
                    for idx, entity in enumerate(data["entities"])
                ]
            )
        if method == "tags.entities.bulk.add":
            return Response([{"id": 900 + idx} for idx, _ in enumerate(data["tags"])])
        if method == "tags.list":
            return Response(
                {
                    "total": len(self.tag_response),
                    "perPage": 100,
                    "pagesCount": 1,
                    "entities": self.tag_response,
                }
            )
        if method == "annotation-objects.list":
            return Response(
                {
                    "total": len(self.object_response),
                    "perPage": 100,
                    "pagesCount": 1,
                    "entities": self.object_response,
                }
            )
        if method == "annotation-objects.bulk.add":
            return Response([{"id": 300 + idx} for idx, _ in enumerate(data["annotationObjects"])])
        if method == "figures.list":
            return Response(
                {
                    "total": len(self.figure_response),
                    "perPage": 100,
                    "pagesCount": 1,
                    "entities": self.figure_response,
                }
            )
        if method == "figures.bulk.add":
            return Response([{"id": 700 + idx} for idx, _ in enumerate(data["figures"])])
        if method == "entities.annotations.bulk.info":
            if self.annotation_response is not None:
                return Response(self.annotation_response)
            return Response(
                [
                    {
                        ApiField.ENTITY_ID: mesh_id,
                        ApiField.ANNOTATION: {
                            "meshId": mesh_id,
                            "figures": [
                                {
                                    "geometry": {
                                        "indices": [0, 1, 2, 42, 65535]
                                    }
                                }
                            ],
                        },
                    }
                    for mesh_id in data[ApiField.ENTITY_IDS]
                ]
            )
        if method == "entities.annotations.bulk.add":
            return Response([{"id": 100 + idx} for idx, _ in enumerate(data["annotations"])])
        if method == "figures.bulk.upload.geometry":
            return Response({})
        raise AssertionError(f"Unexpected method: {method}")


def http_error(status_code, message):
    response = requests.Response()
    response.status_code = status_code
    response._content = message.encode("utf-8")
    return requests.exceptions.HTTPError(message, response=response)


class TestMeshApi(unittest.TestCase):
    def test_project_type_and_mesh_info_conversion(self):
        self.assertEqual(ProjectType.MESHES.value, "meshes")
        api = MeshApi(FakeApi())
        info = api._convert_json_info(mesh_json(name=None, title="scan.stl"))
        self.assertIsInstance(info, MeshInfo)
        self.assertEqual(info.name, "scan.stl")
        self.assertEqual(info.created_by_id, 5)
        self.assertEqual(info.tags, [])

    def test_mesh_info_tags_default_fields_and_conversion(self):
        raw_tags = [
            {
                ApiField.ENTITY_ID: 123,
                ApiField.TAG_ID: 55,
                ApiField.ID: 901,
                ApiField.VALUE: "clear",
            }
        ]
        fake = FakeApi()
        fake.mesh_infos = {123: mesh_json(mesh_id=123, tags=raw_tags)}
        fake.entity_response = [mesh_json(mesh_id=123, tags=raw_tags)]
        api = MeshApi(fake)

        self.assertIn(ApiField.TAGS, api.default_fields())
        self.assertIn(ApiField.TAGS, api.info_sequence())

        info = api.get_info_by_id(123)
        self.assertEqual(info.tags, raw_tags)
        method, data, _ = fake.calls[-1]
        self.assertEqual(method, "entities.info")
        self.assertIn(ApiField.TAGS, data[ApiField.FIELDS])

        infos = api.get_list(dataset_id=4)
        self.assertEqual(infos[0].tags, raw_tags)
        method, data, _ = fake.calls[-1]
        self.assertEqual(method, "entities.list")
        self.assertIn(ApiField.TAGS, data[ApiField.FIELDS])

    def test_mesh_info_tags_missing_custom_field_is_none(self):
        api = MeshApi(FakeApi())
        data = mesh_json()
        data.pop(ApiField.TAGS)

        info = api._convert_json_info(data)

        self.assertIsNone(info.tags)

    def test_get_list_uses_generic_entities_payload(self):
        fake = FakeApi()
        api = MeshApi(fake)
        api.get_list(project_id=10, filters=[{"field": "name", "operator": "=", "value": "a.obj"}])

        method, data, _ = fake.calls[-1]
        self.assertEqual(method, "entities.list")
        self.assertEqual(data[ApiField.PROJECT_ID], 10)
        self.assertIsNone(data[ApiField.DATASET_ID])
        self.assertEqual(data[ApiField.SORT], "id")
        self.assertIn(ApiField.FIELDS, data)

    def test_info_and_download_use_generic_entities_payloads(self):
        fake = FakeApi()
        api = MeshApi(fake)

        info = api.get_info_by_id(123, fields=[ApiField.ID, ApiField.NAME])
        method, data, _ = fake.calls[-1]
        self.assertEqual(info.id, 123)
        self.assertEqual(method, "entities.info")
        self.assertEqual(data[ApiField.ID], 123)
        self.assertEqual(data[ApiField.FIELDS], [ApiField.ID, ApiField.NAME])

        fd, path = tempfile.mkstemp(suffix=".obj")
        os.close(fd)
        try:
            api.download_path(123, path)
            with open(path, "rb") as f:
                self.assertEqual(f.read(), b"mesh-bytes")
        finally:
            os.remove(path)
        method, data, kwargs = fake.calls[-1]
        self.assertEqual(method, "entities.download")
        self.assertEqual(data[ApiField.ID], 123)
        self.assertTrue(kwargs["stream"])

    def test_upload_links_do_not_use_hash(self):
        fake = FakeApi()
        api = MeshApi(fake)

        api.upload_links(4, ["a.obj"], ["https://example.com/a.obj"])
        method, data, _ = fake.calls[-1]
        self.assertEqual(method, "entities.bulk.add")
        self.assertEqual(data["entities"][0][ApiField.LINK], "https://example.com/a.obj")
        self.assertNotIn(ApiField.HASH, data["entities"][0])

        api._upload_by_team_file_ids(4, ["b.ply"], [123])
        method, data, _ = fake.calls[-1]
        self.assertEqual(data["entities"][0][ApiField.TEAM_FILE_ID], 123)
        self.assertNotIn(ApiField.HASH, data["entities"][0])

    def test_upload_paths_stages_files_in_team_files(self):
        fake = FakeApi()
        api = MeshApi(fake)

        fd, path = tempfile.mkstemp(suffix=".obj")
        os.close(fd)
        try:
            api.upload_paths(4, ["local.obj"], [path])
        finally:
            os.remove(path)

        self.assertEqual(fake.file.free_name_calls[0], (77, "/supervisely/mesh_uploads/4/local.obj"))
        self.assertEqual(fake.file.upload_bulk_calls[0][0], 77)
        method, data, _ = fake.calls[-1]
        self.assertEqual(method, "entities.bulk.add")
        self.assertEqual(data["entities"][0][ApiField.TEAM_FILE_ID], 1000)
        self.assertNotIn(ApiField.HASH, data["entities"][0])

    def test_upload_paths_reserves_duplicate_team_file_names(self):
        fake = FakeApi()
        api = MeshApi(fake)

        paths = []
        try:
            for _ in range(2):
                fd, path = tempfile.mkstemp(suffix=".obj")
                os.close(fd)
                paths.append(path)

            api.upload_paths(4, ["local.obj", "local.obj"], paths)
        finally:
            for path in paths:
                os.remove(path)

        dst_paths = fake.file.upload_bulk_calls[0][2]
        self.assertEqual(
            dst_paths,
            ["/supervisely/mesh_uploads/4/local.obj", "/supervisely/mesh_uploads/4/local_000.obj"],
        )

    def test_upload_paths_does_not_fall_back_to_link_when_team_file_id_is_unavailable(self):
        fake = FakeApi()
        fake.fail_team_file_id_upload = True
        api = MeshApi(fake)

        fd, path = tempfile.mkstemp(suffix=".stl")
        os.close(fd)
        try:
            with self.assertRaises(requests.exceptions.HTTPError):
                api.upload_paths(4, ["local.stl"], [path])
        finally:
            os.remove(path)

        bulk_add_calls = [call for call in fake.calls if call[0] == "entities.bulk.add"]
        self.assertEqual(len(bulk_add_calls), 1)
        first_entity = bulk_add_calls[0][1][ApiField.ENTITIES][0]
        self.assertEqual(first_entity[ApiField.TEAM_FILE_ID], 1000)
        self.assertNotIn(ApiField.LINK, first_entity)
        self.assertNotIn(ApiField.HASH, first_entity)

    def test_no_hash_upload_surface_and_extension_validation(self):
        api = MeshApi(FakeApi())
        self.assertFalse(hasattr(api, "upload_hash"))
        self.assertFalse(hasattr(api, "upload_hashes"))

        with self.assertRaises(ValueError):
            api.upload_links(4, ["bad.glb"], ["https://example.com/bad.glb"])

        fd, path = tempfile.mkstemp(suffix=".stl")
        os.close(fd)
        try:
            with self.assertRaises(ValueError):
                api.upload_paths(4, ["bad.glb"], [path])
            with self.assertRaises(ValueError):
                api.upload_paths(4, ["valid.obj"], [path])
        finally:
            os.remove(path)


class TestMeshProject(unittest.TestCase):
    def test_local_mesh_project_layout_and_geometry_sidecars(self):
        indices = [0, 1, 2, 42, 65535]
        ann_json = {
            "description": "",
            "key": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "tags": [],
            "labels": [
                {
                    "key": "cccccccccccccccccccccccccccccccc",
                    "classTitle": "car",
                    "tags": [],
                    "geometryType": "mesh_indices",
                    "geometry": {"indices": indices},
                }
            ],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            src_mesh_path = os.path.join(temp_dir, "source.obj")
            with open(src_mesh_path, "w") as f:
                f.write("o sample\n")

            project_dir = os.path.join(temp_dir, "mesh_project")
            project = MeshProject(project_dir, OpenMode.CREATE)
            ds = project.create_dataset("ds1")
            ds.add_item_file("sample.obj", src_mesh_path, ann=ann_json, _validate_item=False)

            self.assertTrue(os.path.isfile(os.path.join(project_dir, "meta.json")))
            self.assertTrue(os.path.isfile(os.path.join(ds.mesh_dir, "sample.obj")))
            self.assertEqual(project.meta.project_type, ProjectType.MESHES.value)

            stored_ann = load_json_file(ds.get_ann_path("sample.obj"))
            stored_geometry = stored_ann["labels"][0]["geometry"]
            self.assertIsNone(stored_geometry["indices"])
            self.assertIn("indicesPath", stored_geometry)
            self.assertTrue(stored_geometry["indicesPath"].startswith(GEOMETRIES_DIR_NAME + "/"))

            geometry_path = os.path.join(
                ds.get_annotation_dir("sample.obj"),
                *stored_geometry["indicesPath"].split("/"),
            )
            self.assertTrue(os.path.isfile(geometry_path))
            with open(geometry_path, "rb") as f:
                self.assertEqual(f.read(), encode_mesh_indices(indices))

            restored_ann = ds.get_ann_json("sample.obj")
            self.assertEqual(restored_ann["labels"][0]["geometry"]["indices"], indices)
            self.assertNotIn("indicesPath", restored_ann["labels"][0]["geometry"])

            reopened = read_project(project_dir)
            self.assertIsInstance(reopened, MeshProject)
            reopened_ds = reopened.datasets.get("ds1")
            self.assertEqual(reopened_ds.get_ann_json("sample.obj"), restored_ann)


class TestMeshConverter(unittest.TestCase):
    @staticmethod
    def _write_ascii_ply(path, vertices):
        lines = [
            "ply",
            "format ascii 1.0",
            f"element vertex {len(vertices)}",
            "property float x",
            "property float y",
            "property float z",
            "property uchar red",
            "property uchar green",
            "property uchar blue",
            "property int class_id",
            "property int object_id",
            "element face 1",
            "property list uchar int vertex_indices",
            "end_header",
        ]
        for index, (red, green, blue, class_id, object_id) in enumerate(vertices):
            lines.append(f"{index} 0 0 {red} {green} {blue} {class_id} {object_id}")
        lines.append("3 0 1 2")
        with open(path, "w", encoding="ascii") as file:
            file.write("\n".join(lines) + "\n")

    @staticmethod
    def _write_per_vertex_labels_project(project_dir):
        os.makedirs(os.path.join(project_dir, "Dental"), exist_ok=True)
        meta = ProjectMeta(
            obj_classes=[
                ObjClass("tooth", Mesh, color=[10, 20, 30], sly_id=101),
                ObjClass("gum", Mesh, color=[40, 50, 60], sly_id=102),
            ],
            project_type=ProjectType.MESHES.value,
        )
        dump_json_file(meta.to_json(), os.path.join(project_dir, "meta.json"))

        TestMeshConverter._write_ascii_ply(
            os.path.join(project_dir, "Dental", "labeled.ply"),
            [
                (10, 20, 30, 101, 500),
                (10, 20, 30, 101, 500),
                (40, 50, 60, 102, -1),
                (255, 255, 255, -1, -1),
            ],
        )
        TestMeshConverter._write_ascii_ply(
            os.path.join(project_dir, "Dental", "empty.ply"),
            [
                (255, 255, 255, -1, -1),
                (255, 255, 255, -1, -1),
                (255, 255, 255, -1, -1),
            ],
        )

    def test_per_vertex_labels_converter_detects_painted_ply_and_builds_mesh_labels(self):
        with tempfile.TemporaryDirectory() as project_dir:
            self._write_per_vertex_labels_project(project_dir)

            converter = MeshConverter(project_dir).detect_format()

            self.assertIsInstance(converter, PerVertexLabelsMeshConverter)
            self.assertEqual(str(converter), AvailableMeshConverters.PER_VERTEX_LABELS)
            self.assertEqual(converter.items_count, 2)
            self.assertEqual(converter.get_meta().project_type, ProjectType.MESHES.value)

            items_by_name = {item.name: item for item in converter.get_items()}
            ann_json = converter.to_supervisely(items_by_name["labeled.ply"], converter.get_meta())
            self.assertEqual(len(ann_json["labels"]), 2)
            self.assertNotIn("objects", ann_json)
            self.assertNotIn("figures", ann_json)

            labels_by_class = {
                label["classTitle"]: label["geometry"]["indices"]
                for label in ann_json["labels"]
            }
            self.assertEqual(labels_by_class["tooth"], [0, 1])
            self.assertEqual(labels_by_class["gum"], [2])
            tooth_label = [label for label in ann_json["labels"] if label["classTitle"] == "tooth"][0]
            self.assertEqual(tooth_label["customData"]["sourceObjectId"], 500)

            empty_ann = converter.to_supervisely(items_by_name["empty.ply"], converter.get_meta())
            self.assertEqual(empty_ann["labels"], [])

    def test_per_vertex_labels_converter_honors_renamed_classes(self):
        with tempfile.TemporaryDirectory() as project_dir:
            self._write_per_vertex_labels_project(project_dir)
            converter = PerVertexLabelsMeshConverter(project_dir)
            self.assertTrue(converter.validate_format())

            item = [item for item in converter.get_items() if item.name == "labeled.ply"][0]
            ann_json = converter.to_supervisely(
                item,
                converter.get_meta(),
                renamed_classes={"tooth": "tooth_1", "gum": "gum_1"},
            )

            self.assertEqual(
                {label["classTitle"] for label in ann_json["labels"]},
                {"tooth_1", "gum_1"},
            )

    def test_per_vertex_labels_converter_rejects_sly_geometry_sidecars(self):
        with tempfile.TemporaryDirectory() as project_dir:
            self._write_per_vertex_labels_project(project_dir)
            geometries_dir = os.path.join(project_dir, "Dental", "ann", "labeled.ply", "geometries")
            os.makedirs(geometries_dir, exist_ok=True)
            with open(os.path.join(geometries_dir, "figure.bin"), "wb") as file:
                file.write(b"\x00")

            converter = PerVertexLabelsMeshConverter(project_dir)

            self.assertFalse(converter.validate_format())

    def test_sly_mesh_converter_detects_fs_project_and_uploads_annotation_rows(self):
        indices = [0, 1, 2, 42, 65535]
        ann_json = {
            "description": "",
            "key": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "tags": [],
            "labels": [
                {
                    "key": "cccccccccccccccccccccccccccccccc",
                    "classTitle": "car",
                    "tags": [],
                    "geometryType": "mesh_indices",
                    "geometry": {"indices": indices},
                }
            ],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            src_mesh_path = os.path.join(temp_dir, "source.obj")
            with open(src_mesh_path, "w") as f:
                f.write("o sample\n")

            project_dir = os.path.join(temp_dir, "mesh_project")
            project = MeshProject(project_dir, OpenMode.CREATE)
            project.set_meta(
                ProjectMeta(obj_classes=[ObjClass("car", Mesh)], project_type=ProjectType.MESHES)
            )
            ds = project.create_dataset("ds1")
            ds.add_item_file("sample.obj", src_mesh_path, ann=ann_json, _validate_item=False)

            converter = MeshConverter(project_dir).detect_format()
            self.assertIsInstance(converter, SLYMeshConverter)
            self.assertEqual(str(converter), AvailableMeshConverters.SLY)
            self.assertEqual(converter.items_count, 1)
            self.assertEqual(converter.get_meta().project_type, ProjectType.MESHES.value)

            item = converter.get_items()[0]
            converted_ann = converter.to_supervisely(item, converter.get_meta())
            self.assertEqual(converted_ann["labels"][0]["geometry"]["indices"], indices)
            self.assertNotIn("indicesPath", converted_ann["labels"][0]["geometry"])

            fake = FakeApi()
            fake.project.meta = converter.get_meta()
            upload_api = SimpleNamespace(
                mesh=MeshApi(fake),
                dataset=fake.dataset,
                project=fake.project,
            )
            converter.upload_dataset(upload_api, 4, batch_size=2, log_progress=False)

            method_names = [method for method, _, _ in fake.calls]
            self.assertIn("entities.bulk.add", method_names)
            self.assertIn("annotation-objects.bulk.add", method_names)
            self.assertIn("figures.bulk.add", method_names)
            self.assertIn("figures.bulk.upload.geometry", method_names)
            self.assertNotIn("entities.annotations.bulk.add", method_names)
            object_call = [call for call in fake.calls if call[0] == "annotation-objects.bulk.add"][-1]
            self.assertEqual(object_call[1][ApiField.ANNOTATION_OBJECTS][0][ApiField.ENTITY_ID], 1)
            figure_call = [call for call in fake.calls if call[0] == "figures.bulk.add"][-1]
            stored_figure = figure_call[1][ApiField.FIGURES][0]
            self.assertEqual(stored_figure[ApiField.GEOMETRY_TYPE], "mesh")
            self.assertEqual(stored_figure[ApiField.OBJECT_ID], 300)
            self.assertNotIn(ApiField.GEOMETRY, stored_figure)
            geometry_call = [call for call in fake.calls if call[0] == "figures.bulk.upload.geometry"][-1]
            self.assertIn(encode_mesh_indices(indices), geometry_call[1].to_string())
            entity_call = [call for call in fake.calls if call[0] == "entities.bulk.add"][-1]
            self.assertIn(ApiField.TEAM_FILE_ID, entity_call[1][ApiField.ENTITIES][0])
            self.assertNotIn(ApiField.HASH, entity_call[1][ApiField.ENTITIES][0])


class TestMeshAnnotation(unittest.TestCase):
    def test_mesh_indices_codecs(self):
        indices = [0, 1, 2, 42, 65535]
        raw = encode_mesh_indices(indices)
        self.assertEqual(raw, b"\x00\x00\x00\x00\x01\x00\x00\x00\x02\x00\x00\x00*\x00\x00\x00\xff\xff\x00\x00")
        encoded = encode_mesh_indices_base64(indices)
        self.assertEqual(decode_mesh_indices_base64(encoded), indices)
        with self.assertRaisesRegex(ValueError, "divisible by 4"):
            decode_mesh_indices(b"\x00\x00\x00")

        ann_json = {"labels": [{"geometry": {"indices": indices}}]}
        stored_json = encode_mesh_indices_in_json(ann_json)
        self.assertIsInstance(stored_json["labels"][0]["geometry"]["indices"], str)
        self.assertEqual(decode_mesh_indices_in_json(stored_json), ann_json)

    def test_mesh_accepts_list_subclasses(self):
        class IndexList(list):
            pass

        indices = IndexList([1, 2, 3])
        self.assertEqual(Mesh(indices).indices, [1, 2, 3])

    def test_json_round_trip(self):
        obj_class = ObjClass("car", Mesh)
        tag_meta = TagMeta("scene", TagValueType.ANY_STRING)
        meta = ProjectMeta(obj_classes=[obj_class], tag_metas=[tag_meta])

        mesh_tag = MeshTag(tag_meta, value="outdoor")
        label = MeshLabel(Mesh([0, 1, 2]), obj_class)
        ann = MeshAnnotation(
            labels=[label],
            tags=MeshTagCollection([mesh_tag]),
        )

        restored = MeshAnnotation.from_json(ann.to_json(), meta)
        self.assertEqual(len(restored.labels), 1)
        self.assertEqual(len(restored.tags), 1)
        self.assertNotIn("objects", restored.to_json())
        self.assertNotIn("figures", restored.to_json())

    def test_mesh_label_serializes_class_id_from_single_source(self):
        obj_class = ObjClass("car", Mesh, sly_id=44)

        label_json = MeshLabel(Mesh([0, 1, 2]), obj_class, class_id=55).to_json()
        self.assertEqual(label_json[ApiField.CLASS_ID], 55)

        label_json = MeshLabel(Mesh([0, 1, 2]), obj_class).to_json()
        self.assertEqual(label_json[ApiField.CLASS_ID], 44)

    def test_legacy_mesh_json_is_rejected(self):
        meta = ProjectMeta(obj_classes=[ObjClass("car", Mesh)])

        with self.assertRaises(RuntimeError):
            MeshAnnotation.from_json({"objects": [], "figures": []}, meta)

    def test_download_uses_generic_annotation_rows(self):
        fake = FakeApi()
        fake.object_response = [object_json(entity_id=123)]
        fake.figure_response = [figure_json()]
        api = MeshApi(fake)

        ann_json = api.annotation.download(123)

        self.assertEqual(ann_json["meshId"], 123)
        self.assertEqual(ann_json["labels"][0]["geometry"]["indices"], [0, 1, 2, 42, 65535])
        self.assertEqual(len(ann_json["labels"]), 1)
        method_names = [method for method, _, _ in fake.calls]
        self.assertIn("annotation-objects.list", method_names)
        self.assertIn("figures.list", method_names)
        self.assertNotIn("entities.annotations.bulk.info", method_names)
        object_call = [call for call in fake.calls if call[0] == "annotation-objects.list"][-1]
        self.assertEqual(
            object_call[1][ApiField.FILTER],
            [{ApiField.FIELD: ApiField.ENTITY_ID, ApiField.OPERATOR: "in", ApiField.VALUE: [123]}],
        )

    def test_mesh_entity_tags_download_bulk_from_entity_list(self):
        tag_row = {
            ApiField.ENTITY_ID: 123,
            ApiField.TAG_ID: 55,
            ApiField.ID: 901,
            ApiField.VALUE: "clear",
            "labelerLogin": "german",
            ApiField.CREATED_AT: "2026-06-05T12:39:38.928Z",
            ApiField.UPDATED_AT: "2026-06-05T12:39:38.928Z",
        }
        fake = FakeApi()
        fake.tag_response = [tag_info_json(tag_id=55, name="weather")]
        fake.entity_response = [
            mesh_json(mesh_id=123, tags=[tag_row]),
            mesh_json(mesh_id=124, tags=[]),
        ]
        api = MeshApi(fake)

        ann_jsons = api.annotation.download_bulk(4, [123, 124])

        self.assertEqual(
            ann_jsons[0]["tags"],
            [
                {
                    ApiField.NAME: "weather",
                    ApiField.TAG_ID: 55,
                    ApiField.VALUE: "clear",
                    ApiField.ID: 901,
                    "labelerLogin": "german",
                    ApiField.CREATED_AT: "2026-06-05T12:39:38.928Z",
                    ApiField.UPDATED_AT: "2026-06-05T12:39:38.928Z",
                }
            ],
        )
        self.assertEqual(ann_jsons[1]["tags"], [])

        entity_calls = [
            call
            for call in fake.calls
            if call[0] == "entities.list"
            and call[1][ApiField.FIELDS] == [ApiField.ID, ApiField.TAGS]
        ]
        self.assertEqual(len(entity_calls), 1)
        _, data, _ = entity_calls[0]
        self.assertEqual(data[ApiField.DATASET_ID], 4)
        self.assertEqual(
            data[ApiField.FILTER],
            [{ApiField.FIELD: ApiField.ID, ApiField.OPERATOR: "in", ApiField.VALUE: [123, 124]}],
        )

    def test_upload_json_writes_indices_to_raw_figure_geometry(self):
        fake = FakeApi()
        fake.project.meta = ProjectMeta(
            obj_classes=[ObjClass("car", Mesh)], project_type=ProjectType.MESHES
        )
        api = MeshApi(fake)
        ann_json = {
            "labels": [
                {
                    "key": "cccccccccccccccccccccccccccccccc",
                    "classTitle": "car",
                    "tags": [],
                    "geometryType": "mesh",
                    "geometry": {"indices": [0, 1, 2, 42, 65535]},
                }
            ],
        }

        api.annotation.upload_json(123, ann_json, dataset_id=4)

        method_names = [method for method, _, _ in fake.calls]
        self.assertIn("annotation-objects.bulk.add", method_names)
        self.assertIn("figures.bulk.add", method_names)
        self.assertIn("figures.bulk.upload.geometry", method_names)
        self.assertNotIn("entities.annotations.bulk.add", method_names)
        figure_call = [call for call in fake.calls if call[0] == "figures.bulk.add"][-1]
        stored_figure = figure_call[1][ApiField.FIGURES][0]
        self.assertEqual(stored_figure[ApiField.GEOMETRY_TYPE], "mesh")
        self.assertEqual(stored_figure[ApiField.OBJECT_ID], 300)
        self.assertNotIn(ApiField.GEOMETRY, stored_figure)
        geometry_call = [call for call in fake.calls if call[0] == "figures.bulk.upload.geometry"][-1]
        self.assertIn(encode_mesh_indices([0, 1, 2, 42, 65535]), geometry_call[1].to_string())

    def test_download_hydrates_external_mesh_indices_geometry(self):
        fake = FakeApi()
        fake.object_response = [object_json(entity_id=123)]
        fake.figure_response = [
            figure_json(
                figure_id=987,
                geometry={ApiField.STORAGE_PATH: "figures/geometries/sample.bin"},
            )
        ]
        api = MeshApi(fake)
        api.figure.download_indices_batch = lambda ids: [[0, 1, 2, 42, 65535]]

        ann_json = api.annotation.download_bulk(4, [123])[0]

        self.assertEqual(ann_json["labels"][0]["geometry"]["indices"], [0, 1, 2, 42, 65535])

    def test_download_orphan_objects_is_strict_by_default(self):
        fake = FakeApi()
        fake.object_response = [
            object_json(object_id=300, entity_id=123),
            object_json(object_id=301, entity_id=123),
        ]
        fake.figure_response = [figure_json(object_id=300)]
        api = MeshApi(fake)

        with self.assertRaises(RuntimeError):
            api.annotation.download_bulk(4, [123])

    def test_download_orphan_objects_can_be_skipped(self):
        fake = FakeApi()
        fake.object_response = [
            object_json(object_id=300, entity_id=123),
            object_json(object_id=301, entity_id=123),
        ]
        fake.figure_response = [figure_json(object_id=300)]
        api = MeshApi(fake)

        with patch.object(mesh_annotation_api_module.logger, "warning") as warning:
            ann_json = api.annotation.download_bulk(4, [123], skip_orphan_objects=True)[0]

        warning.assert_called_once()
        self.assertEqual(len(ann_json["labels"]), 1)

    def test_upload_paths_requires_one_project_and_dataset(self):
        fake = FakeApi()
        fake.mesh_infos = {
            123: mesh_json(mesh_id=123, datasetId=4, projectId=3),
            124: mesh_json(mesh_id=124, datasetId=5, projectId=3),
        }
        api = MeshApi(fake)

        with tempfile.TemporaryDirectory() as temp_dir:
            ann_paths = []
            for idx in range(2):
                ann_path = os.path.join(temp_dir, f"ann_{idx}.json")
                dump_json_file({"labels": []}, ann_path)
                ann_paths.append(ann_path)

            with patch.object(mesh_annotation_api_module.logger, "warning") as warning:
                with self.assertRaises(RuntimeError):
                    api.annotation.upload_paths([123, 124], ann_paths)

            warning.assert_called_once()
            self.assertIn("Dataset to mesh ids", warning.call_args.args[0])
        self.assertNotIn("annotation-objects.bulk.add", [method for method, _, _ in fake.calls])

    def test_upload_json_updates_key_id_map_for_existing_label_id(self):
        fake = FakeApi()
        fake.project.meta = ProjectMeta(
            obj_classes=[ObjClass("car", Mesh)], project_type=ProjectType.MESHES
        )
        api = MeshApi(fake)
        key_id_map = KeyIdMap()
        label_key = uuid.UUID("cccccccccccccccccccccccccccccccc")
        ann_json = {
            "labels": [
                {
                    "id": 555,
                    "key": label_key.hex,
                    "classTitle": "car",
                    "tags": [],
                    "geometryType": "mesh",
                    "geometry": {"indices": [0, 1, 2]},
                }
            ],
        }

        api.annotation.upload_json(123, ann_json, dataset_id=4, key_id_map=key_id_map)

        self.assertEqual(key_id_map.get_figure_id(label_key), 700)

    def test_append_writes_annotation_rows(self):
        fake = FakeApi()
        fake.project.meta = ProjectMeta(
            obj_classes=[ObjClass("car", Mesh)], project_type=ProjectType.MESHES
        )
        api = MeshApi(fake)
        obj_class = ObjClass("car", Mesh)
        label = MeshLabel(Mesh([0, 1, 2]), obj_class)
        ann = MeshAnnotation(labels=[label])

        api.annotation.append(123, ann)

        methods = [method for method, _, _ in fake.calls]
        self.assertIn("annotation-objects.bulk.add", methods)
        self.assertNotIn("entities.annotations.bulk.add", methods)
        self.assertIn("figures.bulk.add", methods)

    def test_mesh_label_upload_indices_uses_raw_geometry_storage(self):
        fake = FakeApi()
        api = MeshApi(fake)

        api.figure.upload_indices_batch([987], [[0, 1, 2, 42, 65535]])

        method, data, _ = fake.calls[-1]
        self.assertEqual(method, "figures.bulk.upload.geometry")
        self.assertIn(b"\x00\x00\x00\x00\x01\x00\x00\x00\x02\x00\x00\x00*\x00\x00\x00\xff\xff\x00\x00", data.to_string())

    def test_download_indices_batch_reports_item_progress(self):
        api = MeshApi(FakeApi())
        api.figure._download_geometries_generator = lambda ids: [
            (11, SimpleNamespace(content=encode_mesh_indices([1, 2, 3]))),
            (12, SimpleNamespace(content=encode_mesh_indices([4, 5, 6]))),
        ]
        progress = []

        api.figure.download_indices_batch([11, 12], progress_cb=progress.append)

        self.assertEqual(progress, [1, 1])

    def test_append_entity_tag_payload(self):
        fake = FakeApi()
        tag_api = MeshTagApi(fake)
        tag_api.get_name_to_id_map = lambda project_id: {"weather": 55}

        tag_meta = TagMeta("weather", TagValueType.ANY_STRING)
        tag = MeshTag(tag_meta, value="clear")
        key_id_map = KeyIdMap()
        ids = tag_api.append_to_entity(101, 202, MeshTagCollection([tag]), key_id_map)

        self.assertEqual(ids, [900])
        method, data, _ = fake.calls[-1]
        self.assertEqual(method, "tags.entities.bulk.add")
        self.assertEqual(data[ApiField.PROJECT_ID], 202)
        self.assertEqual(data[ApiField.TAGS][0][ApiField.ENTITY_ID], 101)
        self.assertEqual(data[ApiField.TAGS][0][ApiField.TAG_ID], 55)
        self.assertNotIn(ApiField.HASH, data[ApiField.TAGS][0])

    def test_add_entity_tag_uses_generic_bulk_endpoint(self):
        fake = FakeApi()
        tag_api = MeshTagApi(fake)

        tag_id = tag_api.add(55, 101, value="clear", project_id=202)

        self.assertEqual(tag_id, 900)
        method, data, _ = fake.calls[-1]
        self.assertEqual(method, "tags.entities.bulk.add")
        self.assertEqual(data[ApiField.PROJECT_ID], 202)
        self.assertEqual(data[ApiField.TAGS][0][ApiField.ENTITY_ID], 101)
        self.assertEqual(data[ApiField.TAGS][0][ApiField.TAG_ID], 55)
        self.assertEqual(data[ApiField.TAGS][0][ApiField.VALUE], "clear")


if __name__ == "__main__":
    unittest.main()

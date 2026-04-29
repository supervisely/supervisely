# coding: utf-8

import os
import tempfile
import unittest
from types import SimpleNamespace

from supervisely.annotation.obj_class import ObjClass
from supervisely.annotation.tag_meta import TagMeta, TagValueType
from supervisely.api.mesh.mesh_api import MeshApi, MeshInfo
from supervisely.api.mesh.mesh_tag_api import MeshTagApi
from supervisely.api.module_api import ApiField
from supervisely.geometry.rectangle import Rectangle
from supervisely.mesh_annotation.mesh_annotation import MeshAnnotation
from supervisely.mesh_annotation.mesh_figure import MeshFigure
from supervisely.mesh_annotation.mesh_object import MeshObject
from supervisely.mesh_annotation.mesh_object_collection import MeshObjectCollection
from supervisely.mesh_annotation.mesh_tag import MeshTag
from supervisely.mesh_annotation.mesh_tag_collection import MeshTagCollection
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
        "createdBy": 5,
        "createdAt": "2026-01-01T00:00:00.000Z",
        "updatedAt": "2026-01-01T00:00:00.000Z",
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
        return [SimpleNamespace(id=1000 + idx) for idx, _ in enumerate(src_paths)]


class FakeDatasetApi:
    def get_info_by_id(self, dataset_id):
        return SimpleNamespace(id=dataset_id, team_id=77)


class FakeApi:
    def __init__(self):
        self.calls = []
        self.file = FakeFileApi()
        self.dataset = FakeDatasetApi()

    def post(self, method, data, **kwargs):
        self.calls.append((method, data, kwargs))
        if method == "entities.list":
            return Response({"total": 0, "perPage": 100, "pagesCount": 1, "entities": []})
        if method == "entities.info":
            return Response(mesh_json(mesh_id=data["id"]))
        if method == "entities.download":
            return Response(b"mesh-bytes")
        if method == "entities.bulk.add":
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
        raise AssertionError(f"Unexpected method: {method}")


class TestMeshApi(unittest.TestCase):
    def test_project_type_and_mesh_info_conversion(self):
        self.assertEqual(ProjectType.MESHES.value, "meshes")
        api = MeshApi(FakeApi())
        info = api._convert_json_info(mesh_json(name=None, title="scan.stl"))
        self.assertIsInstance(info, MeshInfo)
        self.assertEqual(info.name, "scan.stl")
        self.assertEqual(info.created_by_id, 5)

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

    def test_upload_links_and_team_file_ids_do_not_use_hash(self):
        fake = FakeApi()
        api = MeshApi(fake)

        api.upload_links(4, ["a.obj"], ["https://example.com/a.obj"])
        method, data, _ = fake.calls[-1]
        self.assertEqual(method, "entities.bulk.add")
        self.assertEqual(data["entities"][0][ApiField.LINK], "https://example.com/a.obj")
        self.assertNotIn(ApiField.HASH, data["entities"][0])

        api.upload_team_file_ids(4, ["b.ply"], [123])
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

    def test_no_hash_upload_surface_and_extension_validation(self):
        api = MeshApi(FakeApi())
        self.assertFalse(hasattr(api, "upload_hash"))
        self.assertFalse(hasattr(api, "upload_hashes"))

        with self.assertRaises(ValueError):
            api.upload_links(4, ["bad.glb"], ["https://example.com/bad.glb"])


class TestMeshAnnotation(unittest.TestCase):
    def test_json_round_trip(self):
        obj_class = ObjClass("car", Rectangle)
        tag_meta = TagMeta("scene", TagValueType.ANY_STRING)
        meta = ProjectMeta(obj_classes=[obj_class], tag_metas=[tag_meta])

        mesh_tag = MeshTag(tag_meta, value="outdoor")
        mesh_object = MeshObject(obj_class)
        figure = MeshFigure(mesh_object, Rectangle(0, 0, 10, 10))
        ann = MeshAnnotation(
            objects=MeshObjectCollection([mesh_object]),
            figures=[figure],
            tags=MeshTagCollection([mesh_tag]),
        )

        restored = MeshAnnotation.from_json(ann.to_json(), meta)
        self.assertEqual(len(restored.objects), 1)
        self.assertEqual(len(restored.figures), 1)
        self.assertEqual(len(restored.tags), 1)

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

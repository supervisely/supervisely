import json
import os
import sys
import unittest

sdk_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, sdk_path)

import supervisely as sly
from supervisely.api.api import Api
from supervisely.api.dataset_api import DatasetApi
from supervisely.api.entity_annotation.figure_api import FigureApi
from supervisely.api.image_api import ImageApi
from supervisely.api.module_api import ApiField


class _Response:
    def __init__(self, body=None):
        self._body = body or {
            ApiField.TOTAL: 0,
            "perPage": 100,
            "pagesCount": 1,
            "entities": [],
        }

    def json(self):
        return self._body


class _ApiStub:
    def __init__(self):
        self.calls = []

    def post(self, method, data):
        payload = Api._normalize_filter_payload(data)
        self.calls.append((method, payload))
        return _Response()


class TestApiFilter(unittest.TestCase):
    def test_endpoint_catalog_and_json_serialization(self):
        filters = sly.filters.image.width.gte(1024) & sly.filters.image.name.eq("cat.jpg")
        expected = [
            {"field": "width", "operator": ">=", "value": 1024},
            {"field": "name", "operator": "=", "value": "cat.jpg"},
        ]

        self.assertEqual(filters.to_json(), expected)
        self.assertEqual(json.loads(json.dumps(filters)), expected)
        self.assertEqual(
            json.loads(json.dumps(sly.filters.image.id.eq(1))),
            {"field": "id", "operator": "=", "value": 1},
        )

    def test_catalog_discovery_and_field_mapping(self):
        self.assertIn("image", sly.filters)
        self.assertIn("width", sly.filters.image)
        self.assertEqual(sly.filters.image.fields()[0:2], ["id", "name"])
        self.assertEqual(sly.filters.project.server_fields()["team_id"], ApiField.GROUP_ID)
        self.assertEqual(sly.filters.workspace.server_fields()["team_id"], ApiField.TEAM_ID)
        self.assertEqual(sly.filters.figure.frame_index.field, ApiField.FRAME)
        self.assertIs(sly.filters["image"]["width"], sly.filters.image.width)

    def test_generic_builder_aliases_between_and_nulls(self):
        filters = (
            sly.ApiFilter()
            .eq("customServerField", "value")
            .isin("id", [1, 2])
            .not_in("status", ["stopped"])
            .between("updatedAt", "2024-01-01", "2024-12-31")
            .is_null("deletedAt")
            .not_null("createdAt")
        )

        self.assertEqual(
            filters.to_json(),
            [
                {"field": "customServerField", "operator": "=", "value": "value"},
                {"field": "id", "operator": "in", "value": [1, 2]},
                {"field": "status", "operator": "!in", "value": ["stopped"]},
                {"field": "updatedAt", "operator": ">=", "value": "2024-01-01"},
                {"field": "updatedAt", "operator": "<=", "value": "2024-12-31"},
                {"field": "deletedAt", "operator": "=", "value": None},
                {"field": "createdAt", "operator": "!=", "value": None},
            ],
        )

    def test_normalize_raw_payloads_and_conditions(self):
        raw = {"field": "name", "operator": "=", "value": "cat"}
        old_list = [raw]
        condition = sly.filters.image.height.lt(800)

        self.assertEqual(sly.ApiFilter.normalize(None), [])
        self.assertEqual(sly.ApiFilter.normalize({}), [])
        self.assertEqual(sly.ApiFilter.normalize(raw), [raw])
        self.assertEqual(sly.ApiFilter.normalize(old_list), old_list)
        self.assertEqual(sly.ApiFilter.normalize(condition), [condition.to_json()])
        self.assertEqual(sly.ApiFilter.normalize([condition]), [condition.to_json()])

        normalized = sly.ApiFilter.normalize(old_list)
        normalized[0]["value"] = "dog"
        self.assertEqual(old_list[0]["value"], "cat")

    def test_invalid_builder_inputs(self):
        with self.assertRaises(ValueError):
            sly.ApiFilter().where("id", "contains", 1)
        with self.assertRaises(ValueError):
            sly.ApiFilter().eq("", 1)

    def test_representative_api_payloads_match_legacy_shape(self):
        expected = [{"field": "width", "operator": ">=", "value": 1024}]

        old_api = _ApiStub()
        ImageApi(old_api).get_list(1, filters=expected)
        self.assertEqual(old_api.calls[0][1][ApiField.FILTER], expected)

        new_api = _ApiStub()
        ImageApi(new_api).get_list(1, filters=sly.filters.image.width.gte(1024))
        self.assertEqual(new_api.calls[0][1][ApiField.FILTER], expected)

    def test_dataset_parent_filter_does_not_mutate_caller_filters(self):
        caller_filters = [sly.filters.dataset.name.eq("nested").to_json()]
        original = json.loads(json.dumps(caller_filters))

        api = _ApiStub()
        DatasetApi(api).get_list(10, filters=caller_filters, parent_id=20)

        self.assertEqual(caller_filters, original)
        self.assertEqual(
            api.calls[0][1][ApiField.FILTER],
            original + [{"field": ApiField.PARENT_ID, "operator": "=", "value": 20}],
        )

    def test_figure_image_filter_does_not_mutate_caller_filters(self):
        caller_filters = sly.filters.figure.class_id.eq(7)
        original = sly.ApiFilter.normalize(caller_filters)

        api = _ApiStub()
        FigureApi(api).download(1, image_ids=[10, 11], filters=caller_filters)

        self.assertEqual(sly.ApiFilter.normalize(caller_filters), original)
        self.assertEqual(
            api.calls[0][1][ApiField.FILTER],
            original + [{"field": ApiField.ENTITY_ID, "operator": "in", "value": [10, 11]}],
        )


if __name__ == "__main__":
    unittest.main()

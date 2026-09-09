import json
from types import SimpleNamespace
from unittest.mock import Mock

from click.testing import CliRunner

from supervisely.cli.common import CliState
from supervisely.cli.image_commands import IMAGE_OUTPUT_FIELDS, image_group


def _make_api():
    return SimpleNamespace(image=Mock(), annotation=Mock())


def _invoke(api, *args):
    return CliRunner().invoke(
        image_group,
        list(args),
        obj=CliState(api=api, json_output=True),
    )


def _image_info(image_id=7, name="frame.jpg"):
    return {
        "id": image_id,
        "name": name,
        "dataset_id": 11,
        "project_id": 13,
        "width": 1920,
        "height": 1080,
        "labels_count": 4,
        "created_at": "2026-08-01T10:00:00Z",
        "updated_at": "2026-08-02T10:00:00Z",
        "link": "https://storage.example/private.jpg",
        "full_storage_url": "https://storage.example/original.jpg",
        "path_original": "/private/original.jpg",
        "meta": {"camera": "secret"},
    }


def test_image_list_by_dataset_uses_safe_defaults_and_projects_output():
    api = _make_api()
    api.image.get_list.return_value = [_image_info()]

    result = _invoke(api, "list", "--dataset-id", "11")

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload == [{field: _image_info().get(field) for field in IMAGE_OUTPUT_FIELDS}]
    assert "storage.example" not in result.output
    assert "camera" not in result.output
    api.image.get_list.assert_called_once_with(
        dataset_id=11,
        sort="id",
        sort_order="asc",
        limit=100,
        force_metadata_for_links=False,
        project_id=None,
        only_labelled=False,
        recursive=False,
    )


def test_image_list_by_project_forwards_listing_options():
    api = _make_api()
    api.image.get_list.return_value = []

    result = _invoke(
        api,
        "list",
        "--project-id",
        "13",
        "--limit",
        "25",
        "--sort",
        "updatedAt",
        "--order",
        "desc",
        "--recursive",
        "--only-labelled",
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == []
    api.image.get_list.assert_called_once_with(
        dataset_id=None,
        sort="updatedAt",
        sort_order="desc",
        limit=25,
        force_metadata_for_links=False,
        project_id=13,
        only_labelled=True,
        recursive=True,
    )


def test_image_list_requires_exactly_one_container():
    api = _make_api()

    missing = _invoke(api, "list")
    both = _invoke(
        api,
        "list",
        "--dataset-id",
        "11",
        "--project-id",
        "13",
    )

    assert missing.exit_code == 2
    assert "exactly one" in missing.output
    assert both.exit_code == 2
    assert "exactly one" in both.output
    api.image.get_list.assert_not_called()


def test_image_list_validates_limit_and_order():
    api = _make_api()

    invalid_limit = _invoke(api, "list", "--dataset-id", "11", "--limit", "0")
    invalid_order = _invoke(api, "list", "--dataset-id", "11", "--order", "sideways")

    assert invalid_limit.exit_code == 2
    assert "0 is not in the range" in invalid_limit.output
    assert invalid_order.exit_code == 2
    assert "Invalid value for '--order'" in invalid_order.output
    api.image.get_list.assert_not_called()


def test_image_get_projects_output_and_reports_missing_resource():
    api = _make_api()
    api.image.get_info_by_id.side_effect = [_image_info(), None]

    result = _invoke(api, "get", "--id", "7")
    missing = _invoke(api, "get", "--id", "404")

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {
        field: _image_info().get(field) for field in IMAGE_OUTPUT_FIELDS
    }
    assert "storage.example" not in result.output
    assert missing.exit_code == 1
    assert "Image with ID=404 was not found" in missing.output
    assert api.image.get_info_by_id.call_args_list[0].args == (7,)
    assert api.image.get_info_by_id.call_args_list[0].kwargs == {
        "force_metadata_for_links": False,
    }
    assert api.image.get_info_by_id.call_args_list[1].args == (404,)
    assert api.image.get_info_by_id.call_args_list[1].kwargs == {
        "force_metadata_for_links": False,
    }


def test_image_annotation_returns_full_json_and_forwards_custom_data_flag():
    api = _make_api()
    annotation = {
        "description": "",
        "size": {"height": 1080, "width": 1920},
        "objects": [{"id": 3, "classTitle": "car"}],
        "customData": {"source": "review"},
    }
    api.annotation.download_json.return_value = annotation

    default_result = _invoke(api, "annotation", "--id", "7")
    custom_result = _invoke(api, "annotation", "--id", "8", "--custom-data")

    assert default_result.exit_code == 0, default_result.output
    assert json.loads(default_result.output) == annotation
    assert custom_result.exit_code == 0, custom_result.output
    assert json.loads(custom_result.output) == annotation
    assert api.annotation.download_json.call_args_list[0].args == (7,)
    assert api.annotation.download_json.call_args_list[0].kwargs == {
        "with_custom_data": False,
        "force_metadata_for_links": True,
    }
    assert api.annotation.download_json.call_args_list[1].args == (8,)
    assert api.annotation.download_json.call_args_list[1].kwargs == {
        "with_custom_data": True,
        "force_metadata_for_links": True,
    }

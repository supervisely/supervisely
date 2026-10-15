import json
from types import SimpleNamespace
from unittest.mock import Mock

import click
from click.testing import CliRunner

from supervisely.api.file_api import FileInfo
from supervisely.cli.common import CliState
from supervisely.cli.teamfiles_commands import register_teamfiles_commands


def _file_info(path="/models/model.onnx", is_dir=False):
    return FileInfo(
        team_id=9,
        id=41,
        user_id=7,
        name="model.onnx",
        hash="sensitive-hash",
        path=path,
        storage_path="/internal/storage/path",
        mime="application/octet-stream",
        ext="onnx",
        sizeb=1024,
        created_at="2026-08-01T10:00:00Z",
        updated_at="2026-08-02T10:00:00Z",
        full_storage_url="https://storage.example/private-object",
        is_dir=is_dir,
    )


def _make_group():
    group = click.Group(name="teamfiles")
    register_teamfiles_commands(group)
    return group


def _invoke(storage, arguments):
    return CliRunner().invoke(
        _make_group(),
        arguments,
        obj=CliState(
            api=SimpleNamespace(storage=storage),
            json_output=True,
        ),
    )


def test_list_uses_safe_defaults_and_projects_output():
    storage = Mock()
    storage.list.return_value = [_file_info()]

    result = _invoke(storage, ["list", "--team-id", "9", "--path", "/models"])

    assert result.exit_code == 0, result.output
    storage.list.assert_called_once_with(
        team_id=9,
        path="/models",
        recursive=False,
        return_type="fileinfo",
        with_metadata=True,
        include_files=True,
        include_folders=True,
        limit=100,
    )
    assert json.loads(result.output) == [
        {
            "team_id": 9,
            "id": 41,
            "user_id": 7,
            "name": "model.onnx",
            "path": "/models/model.onnx",
            "mime": "application/octet-stream",
            "ext": "onnx",
            "sizeb": 1024,
            "created_at": "2026-08-01T10:00:00Z",
            "updated_at": "2026-08-02T10:00:00Z",
            "is_dir": False,
        }
    ]
    assert "sensitive-hash" not in result.output
    assert "internal/storage" not in result.output
    assert "private-object" not in result.output


def test_list_forwards_recursive_limit_and_file_filter():
    storage = Mock()
    storage.list.return_value = []

    result = _invoke(
        storage,
        [
            "list",
            "--team-id",
            "9",
            "--path",
            "s3://bucket/models",
            "--recursive",
            "--limit",
            "5",
            "--files-only",
        ],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == []
    storage.list.assert_called_once_with(
        team_id=9,
        path="s3://bucket/models",
        recursive=True,
        return_type="fileinfo",
        with_metadata=True,
        include_files=True,
        include_folders=False,
        limit=5,
    )


def test_list_forwards_folder_filter():
    storage = Mock()
    storage.list.return_value = []

    result = _invoke(
        storage,
        [
            "list",
            "--team-id",
            "9",
            "--path",
            "/models",
            "--folders-only",
        ],
    )

    assert result.exit_code == 0, result.output
    storage.list.assert_called_once_with(
        team_id=9,
        path="/models",
        recursive=False,
        return_type="fileinfo",
        with_metadata=True,
        include_files=False,
        include_folders=True,
        limit=100,
    )


def test_list_rejects_mutually_exclusive_filters_and_nonpositive_limit():
    storage = Mock()

    conflicting = _invoke(
        storage,
        [
            "list",
            "--team-id",
            "9",
            "--path",
            "/models",
            "--files-only",
            "--folders-only",
        ],
    )
    invalid_limit = _invoke(
        storage,
        [
            "list",
            "--team-id",
            "9",
            "--path",
            "/models",
            "--limit",
            "0",
        ],
    )

    assert conflicting.exit_code == 2
    assert "mutually exclusive" in conflicting.output
    assert invalid_limit.exit_code == 2
    assert "not in the range" in invalid_limit.output
    storage.list.assert_not_called()


def test_info_uses_storage_api_and_projects_safe_output():
    storage = Mock()
    storage.get_info_by_path.return_value = _file_info()

    result = _invoke(
        storage,
        ["info", "--team-id", "9", "--path", "/models/model.onnx"],
    )

    assert result.exit_code == 0, result.output
    storage.get_info_by_path.assert_called_once_with(
        team_id=9,
        remote_path="/models/model.onnx",
    )
    payload = json.loads(result.output)
    assert payload["id"] == 41
    assert payload["path"] == "/models/model.onnx"
    assert "hash" not in payload
    assert "storage_path" not in payload
    assert "full_storage_url" not in payload


def test_info_reports_missing_storage_entry():
    storage = Mock()
    storage.get_info_by_path.return_value = None

    result = _invoke(
        storage,
        ["info", "--team-id", "9", "--path", "/missing.txt"],
    )

    assert result.exit_code == 1
    assert "Storage entry with ID=/missing.txt was not found" in result.output


def test_exists_checks_file_and_returns_false_as_success():
    storage = Mock()
    storage.exists.return_value = False

    result = _invoke(
        storage,
        [
            "exists",
            "--team-id",
            "9",
            "--path",
            "/missing.txt",
            "--kind",
            "file",
        ],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {
        "team_id": 9,
        "path": "/missing.txt",
        "kind": "file",
        "exists": False,
    }
    storage.exists.assert_called_once_with(team_id=9, remote_path="/missing.txt")
    storage.dir_exists.assert_not_called()


def test_exists_checks_directory_with_exact_arguments():
    storage = Mock()
    storage.dir_exists.return_value = True

    result = _invoke(
        storage,
        [
            "exists",
            "--team-id",
            "9",
            "--path",
            "azure://bucket/models",
            "--kind",
            "directory",
        ],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["exists"] is True
    storage.dir_exists.assert_called_once_with(
        team_id=9,
        remote_directory="azure://bucket/models",
    )
    storage.exists.assert_not_called()


def test_registration_preserves_existing_teamfiles_commands():
    group = click.Group(name="teamfiles")

    @group.command(name="download")
    def legacy_download():
        pass

    register_teamfiles_commands(group)

    assert {"download", "list", "info", "exists"} <= set(group.commands)

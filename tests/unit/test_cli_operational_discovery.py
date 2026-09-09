import importlib
import json
from types import SimpleNamespace
from unittest.mock import Mock

import click
from click.testing import CliRunner

from supervisely.cli.app_commands import app_group
from supervisely.cli.common import CliState
from supervisely.cli.task_commands import register_task_commands


def test_task_context_dispatches_and_redacts_sensitive_values():
    task_api = Mock()
    task_api.get_context.return_value = {
        "team": {"id": 9, "name": "Operators"},
        "workspace": {"id": 8, "name": "Automation"},
        "apiToken": "must-not-leak",
    }
    task_group = click.Group(name="task")
    register_task_commands(task_group)

    result = CliRunner().invoke(
        task_group,
        ["context", "--id", "17"],
        obj=CliState(
            api=SimpleNamespace(task=task_api),
            json_output=True,
        ),
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {
        "team": {"id": 9, "name": "Operators"},
        "workspace": {"id": 8, "name": "Automation"},
        "apiToken": "***",
    }
    task_api.get_context.assert_called_once_with(17)


def test_app_url_returns_relative_and_absolute_session_urls():
    app_api = Mock()
    app_api.get_url.return_value = "/apps/sessions/17"
    api = SimpleNamespace(
        app=app_api,
        server_address="https://operator:secret@supervisely.example/base?token=hidden",
    )

    result = CliRunner().invoke(
        app_group,
        ["url", "--task-id", "17"],
        obj=CliState(api=api, json_output=True),
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {
        "task_id": 17,
        "path": "/apps/sessions/17",
        "url": "https://supervisely.example/apps/sessions/17",
    }
    assert "operator" not in result.output
    assert "secret" not in result.output
    assert "hidden" not in result.output
    app_api.get_url.assert_called_once_with(17)


def test_teamfiles_removal_requires_confirmation_and_uses_shared_api(monkeypatch):
    cli_module = importlib.import_module("supervisely.cli.cli")
    fake_api = object()
    remove_file_run = Mock(return_value=True)
    monkeypatch.setattr(cli_module, "get_api", lambda: fake_api)
    monkeypatch.setattr(cli_module, "remove_file_run", remove_file_run)

    missing_confirmation = CliRunner().invoke(
        cli_module.cli,
        ["teamfiles", "remove-file", "--id", "9", "--path", "/model.onnx"],
    )

    assert missing_confirmation.exit_code == 2
    assert "requires --yes" in missing_confirmation.output
    remove_file_run.assert_not_called()

    confirmed = CliRunner().invoke(
        cli_module.cli,
        [
            "teamfiles",
            "remove-file",
            "--id",
            "9",
            "--path",
            "/model.onnx",
            "--yes",
        ],
    )

    assert confirmed.exit_code == 0, confirmed.output
    remove_file_run.assert_called_once_with(9, "/model.onnx", api=fake_api)


def test_legacy_teamfiles_download_uses_shared_api(monkeypatch):
    cli_module = importlib.import_module("supervisely.cli.cli")
    fake_api = object()
    download_directory_run = Mock(return_value=True)
    monkeypatch.setattr(cli_module, "get_api", lambda: fake_api)
    monkeypatch.setattr(
        cli_module,
        "download_directory_run",
        download_directory_run,
    )

    result = CliRunner().invoke(
        cli_module.cli,
        [
            "teamfiles",
            "download",
            "--id",
            "9",
            "--src",
            "/remote",
            "--dst",
            "/local",
        ],
    )

    assert result.exit_code == 0, result.output
    download_directory_run.assert_called_once_with(
        9,
        "/remote",
        "/local",
        None,
        ignore_if_not_exists=False,
        api=fake_api,
    )

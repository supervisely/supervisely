import json
import re
from types import SimpleNamespace
from unittest.mock import Mock

from click.testing import CliRunner

from supervisely.cli import resource_commands
from supervisely.cli.cli import cli


def _assert_help_commands(output, commands):
    for command in commands:
        assert re.search(rf"(?m)^  {re.escape(command)}(?:\s|$)", output), output


def test_root_help_keeps_legacy_groups_and_exposes_new_groups():
    result = CliRunner().invoke(cli, ["--help"])

    assert result.exit_code == 0, result.output
    _assert_help_commands(
        result.output,
        (
            "release",
            "project",
            "teamfiles",
            "task",
            "team",
            "workspace",
            "dataset",
            "agent",
            "app",
        ),
    )


def test_project_help_keeps_legacy_commands_and_exposes_new_commands():
    result = CliRunner().invoke(cli, ["project", "--help"])

    assert result.exit_code == 0, result.output
    _assert_help_commands(
        result.output,
        (
            "download",
            "upload",
            "get-name",
            "list",
            "get",
            "create",
            "move",
            "stats",
            "meta",
        ),
    )


def test_task_help_keeps_legacy_command_and_exposes_new_commands():
    result = CliRunner().invoke(cli, ["task", "--help"])

    assert result.exit_code == 0, result.output
    _assert_help_commands(
        result.output,
        ("set-output-dir", "list", "get", "status", "logs", "wait", "stop"),
    )


def test_global_json_option_reaches_new_command(monkeypatch):
    team_api = Mock()
    team_api.get_list.return_value = [
        {"id": 7, "name": "Operators", "token": "must-not-be-printed"}
    ]
    fake_api = SimpleNamespace(team=team_api)
    monkeypatch.setattr(resource_commands, "get_api", lambda: fake_api)

    result = CliRunner().invoke(cli, ["--json", "team", "list"])

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == [
        {"id": 7, "name": "Operators", "token": "***"}
    ]
    assert "must-not-be-printed" not in result.output
    team_api.get_list.assert_called_once_with()

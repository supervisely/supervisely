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
            "image",
            "role",
            "user",
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
        (
            "set-output-dir",
            "list",
            "get",
            "status",
            "context",
            "logs",
            "wait",
            "stop",
        ),
    )


def test_discovery_commands_are_registered_with_legacy_groups():
    runner = CliRunner()

    teamfiles_help = runner.invoke(cli, ["teamfiles", "--help"])
    team_help = runner.invoke(cli, ["team", "--help"])
    app_help = runner.invoke(cli, ["app", "--help"])

    assert teamfiles_help.exit_code == 0, teamfiles_help.output
    assert team_help.exit_code == 0, team_help.output
    assert app_help.exit_code == 0, app_help.output
    _assert_help_commands(
        teamfiles_help.output,
        ("download", "upload", "remove-file", "remove-dir", "list", "info", "exists"),
    )
    _assert_help_commands(team_help.output, ("list", "get", "members"))
    _assert_help_commands(app_help.output, ("params", "run", "sessions", "stop", "url"))


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

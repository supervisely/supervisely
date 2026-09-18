import json
from collections import namedtuple
from types import SimpleNamespace
from unittest.mock import MagicMock, call

import click
import pytest
from click.testing import CliRunner

from supervisely import ProjectType
from supervisely.cli.common import CliState
from supervisely.cli.dataset_commands import dataset_group
from supervisely.cli.project_commands import register_project_commands
from supervisely.cli.task_commands import TASK_STATUSES, register_task_commands
import supervisely.cli.task_commands as task_commands


def _make_api():
    return SimpleNamespace(
        project=MagicMock(),
        dataset=MagicMock(),
        task=MagicMock(),
    )


def _make_cli(api):
    @click.group()
    @click.option("--json", "json_output", is_flag=True)
    @click.pass_context
    def root(ctx, json_output):
        ctx.obj = CliState(json_output=json_output, api=api)

    project = click.Group(name="project")
    task = click.Group(name="task")
    register_project_commands(project)
    register_task_commands(task)
    root.add_command(project)
    root.add_command(dataset_group)
    root.add_command(task)
    return root


def _invoke_json(cli, *args):
    result = CliRunner().invoke(cli, ["--json", *args])
    assert result.exit_code == 0, result.output
    return json.loads(result.output)


def test_project_list_requires_exactly_one_container_and_dispatches():
    api = _make_api()
    api.project.get_list.return_value = [{"id": 7, "name": "demo"}]
    cli = _make_cli(api)

    output = _invoke_json(cli, "project", "list", "--workspace-id", "11")

    assert output == [{"id": 7, "name": "demo"}]
    api.project.get_list.assert_called_once_with(workspace_id=11, team_id=None)

    missing = CliRunner().invoke(cli, ["project", "list"])
    assert missing.exit_code == 2
    assert "exactly one" in missing.output

    both = CliRunner().invoke(
        cli,
        ["project", "list", "--workspace-id", "11", "--team-id", "12"],
    )
    assert both.exit_code == 2
    assert "exactly one" in both.output


def test_project_get_create_move_stats_and_meta_commands():
    api = _make_api()
    api.project.get_info_by_id.return_value = {"id": 7, "name": "demo"}
    api.project.create.return_value = {"id": 8, "name": "new"}
    api.project.get_stats.return_value = {"items": 21}
    api.project.get_meta.return_value = {"classes": [{"title": "car"}]}
    cli = _make_cli(api)

    assert _invoke_json(cli, "project", "get", "--id", "7")["name"] == "demo"
    assert _invoke_json(
        cli,
        "project",
        "create",
        "--workspace-id",
        "11",
        "--name",
        "new",
        "--type",
        "images",
    )["id"] == 8
    assert _invoke_json(
        cli, "project", "move", "--id", "8", "--workspace-id", "12"
    ) == {"id": 8, "moved": True, "workspace_id": 12}
    assert _invoke_json(cli, "project", "stats", "--id", "8") == {"items": 21}
    assert _invoke_json(cli, "project", "meta", "get", "--id", "8") == {
        "classes": [{"title": "car"}]
    }

    api.project.get_info_by_id.assert_called_once_with(7)
    api.project.create.assert_called_once_with(
        workspace_id=11,
        name="new",
        type=ProjectType.IMAGES,
    )
    api.project.move.assert_called_once_with(id=8, workspace_id=12)
    api.project.get_stats.assert_called_once_with(8)
    api.project.get_meta.assert_called_once_with(8)


def test_project_get_reports_missing_resource():
    api = _make_api()
    api.project.get_info_by_id.return_value = None

    result = CliRunner().invoke(_make_cli(api), ["project", "get", "--id", "404"])

    assert result.exit_code == 1
    assert "Project with ID=404 was not found" in result.output


def test_dataset_list_get_create_and_tree_commands():
    api = _make_api()
    api.dataset.get_list.return_value = [{"id": 3, "name": "root"}]
    api.dataset.get_info_by_id.return_value = {"id": 4, "name": "child"}
    api.dataset.create.return_value = {"id": 5, "name": "new"}
    Dataset = namedtuple("Dataset", ["id", "name", "parent_id", "items_count"])
    api.dataset.tree.return_value = iter(
        [
            ([], Dataset(3, "root", None, 10)),
            (["root"], Dataset(4, "child", 3, 2)),
        ]
    )
    cli = _make_cli(api)

    assert _invoke_json(cli, "dataset", "list", "--project-id", "2") == [
        {"id": 3, "name": "root"}
    ]
    assert _invoke_json(cli, "dataset", "get", "--id", "4")["name"] == "child"
    assert _invoke_json(
        cli,
        "dataset",
        "create",
        "--project-id",
        "2",
        "--name",
        "new",
        "--parent-id",
        "3",
    )["id"] == 5
    tree = _invoke_json(cli, "dataset", "tree", "--project-id", "2")

    assert tree[0]["path"] == ["root"]
    assert tree[0]["depth"] == 0
    assert tree[1]["path"] == ["root", "child"]
    assert tree[1]["depth"] == 1
    api.dataset.get_list.assert_called_once_with(project_id=2)
    api.dataset.get_info_by_id.assert_called_once_with(4)
    api.dataset.create.assert_called_once_with(project_id=2, name="new", parent_id=3)
    api.dataset.tree.assert_called_once_with(2)


def test_dataset_get_reports_missing_resource():
    api = _make_api()
    api.dataset.get_info_by_id.return_value = None

    result = CliRunner().invoke(_make_cli(api), ["dataset", "get", "--id", "404"])

    assert result.exit_code == 1
    assert "Dataset with ID=404 was not found" in result.output


def test_task_inspection_commands_dispatch():
    api = _make_api()
    api.task.get_list.return_value = [{"id": 13, "status": "started"}]
    api.task.get_info_by_id.return_value = {"id": 13, "status": "started"}
    api.task.get_status.return_value = "started"
    api.task.get_logs.return_value = [{"message": "ready"}]
    cli = _make_cli(api)

    assert _invoke_json(cli, "task", "list", "--workspace-id", "11")[0]["id"] == 13
    assert _invoke_json(cli, "task", "get", "--id", "13")["status"] == "started"
    assert _invoke_json(cli, "task", "status", "--id", "13") == {
        "id": 13,
        "status": "started",
    }
    assert _invoke_json(
        cli,
        "task",
        "logs",
        "--id",
        "13",
        "--since",
        "2026-08-04T10:00:00Z",
        "--limit",
        "10",
    ) == [{"message": "ready"}]

    api.task.get_list.assert_called_once_with(workspace_id=11)
    api.task.get_info_by_id.assert_called_once_with(13)
    api.task.get_status.assert_called_once_with(13)
    api.task.get_logs.assert_called_once_with(
        task_id=13,
        since_time="2026-08-04T10:00:00Z",
        limit=10,
    )


def test_task_wait_polls_until_target_status(monkeypatch):
    api = _make_api()
    api.task.get_status.side_effect = ["queued", "started", "finished"]
    sleep = MagicMock()
    monkeypatch.setattr(task_commands.time, "sleep", sleep)

    output = _invoke_json(_make_cli(api), "task", "wait", "--id", "13")

    assert output == {"id": 13, "reached": True, "status": "finished"}
    assert api.task.get_status.call_args_list == [call(13), call(13), call(13)]
    assert sleep.call_count == 2
    assert all(args[0][0] <= 1.0 for args in sleep.call_args_list)


def test_task_wait_accepts_sdk_status_values_and_target_terminal_status():
    assert "consumed" in TASK_STATUSES
    api = _make_api()
    api.task.get_status.return_value = "deployed"

    output = _invoke_json(
        _make_cli(api),
        "task",
        "wait",
        "--id",
        "13",
        "--target-status",
        "deployed",
    )

    assert output == {"id": 13, "reached": True, "status": "deployed"}


@pytest.mark.parametrize("terminal_status", ["error", "finished", "deployed", "stopped"])
def test_task_wait_fails_on_terminal_target_mismatch(terminal_status):
    api = _make_api()
    api.task.get_status.return_value = terminal_status
    cli = _make_cli(api)

    terminal = CliRunner().invoke(
        cli,
        ["task", "wait", "--id", "13", "--target-status", "started"],
    )
    assert terminal.exit_code == 1
    assert f"terminal status '{terminal_status}'" in terminal.output


def test_task_wait_times_out_by_elapsed_seconds():
    api = _make_api()
    api.task.get_status.return_value = "queued"
    cli = _make_cli(api)

    timed_out = CliRunner().invoke(
        cli,
        ["task", "wait", "--id", "13", "--timeout", "0"],
    )
    assert timed_out.exit_code == 1
    assert "Timed out after 0 seconds" in timed_out.output


def test_task_stop_requires_confirmation():
    api = _make_api()
    api.task.stop.return_value = "stopped"
    cli = _make_cli(api)

    refused = CliRunner().invoke(cli, ["task", "stop", "--id", "13"])
    assert refused.exit_code == 2
    assert "requires --yes" in refused.output
    api.task.stop.assert_not_called()

    output = _invoke_json(cli, "task", "stop", "--id", "13", "--yes")
    assert output == {"id": 13, "status": "stopped", "stopped": True}
    api.task.stop.assert_called_once_with(13)


def test_new_commands_coexist_with_legacy_project_and_task_commands():
    from supervisely.cli.cli import cli

    project_command_map = cli.commands["project"].commands
    task_command_map = cli.commands["task"].commands

    assert {"download", "get-name", "upload"} <= project_command_map.keys()
    assert {"list", "get", "create", "move", "stats", "meta"} <= project_command_map.keys()
    assert "set-output-dir" in task_command_map
    assert {"list", "get", "status", "logs", "wait", "stop"} <= task_command_map.keys()

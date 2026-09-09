import json
from types import SimpleNamespace
from unittest.mock import Mock

from click.testing import CliRunner

from supervisely.cli.common import CliState
from supervisely.cli.resource_commands import agent_group, team_group, workspace_group


def _api(**resources):
    return SimpleNamespace(**resources)


def _invoke(group, args, api):
    return CliRunner().invoke(group, args, obj=CliState(api=api, json_output=True))


def test_team_list_and_get_use_team_api():
    team_api = Mock()
    team_api.get_list.return_value = [{"id": 1, "name": "Core", "role": "admin"}]
    team_api.get_info_by_id.return_value = {"id": 1, "name": "Core"}
    api = _api(team=team_api)

    list_result = _invoke(team_group, ["list"], api)
    get_result = _invoke(team_group, ["get", "--id", "1"], api)

    assert list_result.exit_code == 0, list_result.output
    assert json.loads(list_result.output) == [{"id": 1, "name": "Core", "role": "admin"}]
    team_api.get_list.assert_called_once_with()
    assert get_result.exit_code == 0, get_result.output
    assert json.loads(get_result.output) == {"id": 1, "name": "Core"}
    team_api.get_info_by_id.assert_called_once_with(1)


def test_team_get_reports_missing_resource():
    team_api = Mock()
    team_api.get_info_by_id.return_value = None

    result = _invoke(team_group, ["get", "--id", "404"], _api(team=team_api))

    assert result.exit_code != 0
    assert "Team with ID=404 was not found or is inaccessible" in result.output


def test_workspace_list_and_get_use_workspace_api():
    workspace_api = Mock()
    workspace_api.get_list.return_value = [{"id": 11, "name": "Research", "team_id": 7}]
    workspace_api.get_info_by_id.return_value = {
        "id": 11,
        "name": "Research",
        "team_id": 7,
    }
    api = _api(workspace=workspace_api)

    list_result = _invoke(workspace_group, ["list", "--team-id", "7"], api)
    get_result = _invoke(workspace_group, ["get", "--id", "11"], api)

    assert list_result.exit_code == 0, list_result.output
    assert json.loads(list_result.output)[0]["id"] == 11
    workspace_api.get_list.assert_called_once_with(team_id=7)
    assert get_result.exit_code == 0, get_result.output
    assert json.loads(get_result.output)["name"] == "Research"
    workspace_api.get_info_by_id.assert_called_once_with(11)


def test_workspace_create_forwards_all_configuration():
    workspace_api = Mock()
    workspace_api.create.return_value = {
        "id": 12,
        "name": "Hidden Workspace",
        "team_id": 7,
    }

    result = _invoke(
        workspace_group,
        [
            "create",
            "--team-id",
            "7",
            "--name",
            "Hidden Workspace",
            "--description",
            "CLI-created",
            "--hidden",
        ],
        _api(workspace=workspace_api),
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["id"] == 12
    workspace_api.create.assert_called_once_with(
        team_id=7,
        name="Hidden Workspace",
        description="CLI-created",
        hidden=True,
    )


def test_agent_regular_list_requests_gpu_information_and_redacts_token():
    agent_api = Mock()
    agent_api.get_list.return_value = [
        {
            "id": 5,
            "name": "Runner",
            "team_id": 7,
            "status": "running",
            "token": "secret-agent-token",
            "gpu_info": {"name": "GPU"},
        }
    ]

    result = _invoke(
        agent_group,
        ["list", "--team-id", "7", "--gpu-info"],
        _api(agent=agent_api),
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload[0]["gpu_info"] == {"name": "GPU"}
    assert payload[0]["token"] == "***"
    assert "secret-agent-token" not in result.output
    agent_api.get_list.assert_called_once_with(team_id=7, with_gpu_info=True)
    agent_api.get_list_available.assert_not_called()


def test_agent_available_list_can_require_gpu():
    agent_api = Mock()
    agent_api.get_list_available.return_value = [{"id": 6, "name": "GPU Runner"}]

    result = _invoke(
        agent_group,
        ["list", "--team-id", "7", "--available", "--has-gpu"],
        _api(agent=agent_api),
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)[0]["id"] == 6
    agent_api.get_list_available.assert_called_once_with(team_id=7, has_gpu=True)
    agent_api.get_list.assert_not_called()


def test_agent_has_gpu_requires_available_mode():
    agent_api = Mock()

    result = _invoke(
        agent_group,
        ["list", "--team-id", "7", "--has-gpu"],
        _api(agent=agent_api),
    )

    assert result.exit_code != 0
    assert "--has-gpu requires --available" in result.output
    agent_api.get_list.assert_not_called()
    agent_api.get_list_available.assert_not_called()


def test_agent_get_and_status_use_agent_api():
    agent_api = Mock()
    agent_api.get_info_by_id.return_value = {
        "id": 5,
        "name": "Runner",
        "token": "secret-agent-token",
    }
    agent_api.get_status.return_value = "running"
    api = _api(agent=agent_api)

    get_result = _invoke(agent_group, ["get", "--id", "5"], api)
    status_result = _invoke(agent_group, ["status", "--id", "5"], api)

    assert get_result.exit_code == 0, get_result.output
    assert json.loads(get_result.output)["token"] == "***"
    assert "secret-agent-token" not in get_result.output
    agent_api.get_info_by_id.assert_called_once_with(5)
    assert status_result.exit_code == 0, status_result.output
    assert json.loads(status_result.output) == {"id": 5, "status": "running"}
    agent_api.get_status.assert_called_once_with(5)

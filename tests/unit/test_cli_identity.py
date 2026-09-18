import json
from collections import namedtuple
from types import SimpleNamespace
from unittest.mock import Mock

import click
from click.testing import CliRunner

from supervisely.cli.common import CliState
from supervisely.cli.identity_commands import (
    register_identity_commands,
    role_group,
    user_group,
)


UserRecord = namedtuple(
    "UserRecord",
    [
        "id",
        "login",
        "role",
        "role_id",
        "name",
        "email",
        "logins",
        "disabled",
        "last_login",
        "created_at",
        "updated_at",
    ],
)
RoleRecord = namedtuple("RoleRecord", ["id", "role", "created_at", "updated_at"])


def _user_record(user_id=4, login="alice", role="admin", role_id=1):
    return UserRecord(
        id=user_id,
        login=login,
        role=role,
        role_id=role_id,
        name="Alice Operator",
        email="private@example.com",
        logins=17,
        disabled=False,
        last_login="2026-08-07T10:00:00Z",
        created_at="2025-01-01T00:00:00Z",
        updated_at="2026-08-07T10:00:00Z",
    )


def _invoke(command, args, api, json_output=True):
    return CliRunner().invoke(
        command,
        args,
        obj=CliState(api=api, json_output=json_output),
    )


def test_user_me_uses_current_user_api_and_projects_json_output():
    user_api = Mock()
    user_api.get_my_info.return_value = _user_record()

    result = _invoke(user_group, ["me"], SimpleNamespace(user=user_api))

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {
        "disabled": False,
        "id": 4,
        "login": "alice",
        "name": "Alice Operator",
        "role": "admin",
        "role_id": 1,
    }
    assert "private@example.com" not in result.output
    assert "last_login" not in result.output
    user_api.get_my_info.assert_called_once_with()


def test_user_me_table_output_does_not_include_private_fields():
    user_api = Mock()
    user_api.get_my_info.return_value = _user_record()

    result = _invoke(
        user_group,
        ["me"],
        SimpleNamespace(user=user_api),
        json_output=False,
    )

    assert result.exit_code == 0, result.output
    assert "alice" in result.output
    assert "Alice Operator" in result.output
    assert "private@example.com" not in result.output
    assert "2026-08-07T10:00:00Z" not in result.output


def test_role_list_uses_role_api_and_projects_output():
    role_api = Mock()
    role_api.get_list.return_value = [
        RoleRecord(1, "admin", "2025-01-01", "2026-01-01"),
        RoleRecord(4, "viewer", "2025-01-01", "2026-01-01"),
    ]

    result = _invoke(role_group, ["list"], SimpleNamespace(role=role_api))

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == [
        {"id": 1, "role": "admin"},
        {"id": 4, "role": "viewer"},
    ]
    assert "created_at" not in result.output
    role_api.get_list.assert_called_once_with()


def test_register_identity_commands_attaches_team_members_command():
    @click.group(name="team")
    def team_group():
        pass

    register_identity_commands(team_group)
    user_api = Mock()
    user_api.get_team_members.return_value = [
        _user_record(),
        _user_record(user_id=8, login="bob", role="viewer", role_id=4),
    ]

    result = _invoke(
        team_group,
        ["members", "--team-id", "7"],
        SimpleNamespace(user=user_api),
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == [
        {
            "disabled": False,
            "id": 4,
            "login": "alice",
            "name": "Alice Operator",
            "role": "admin",
            "role_id": 1,
        },
        {
            "disabled": False,
            "id": 8,
            "login": "bob",
            "name": "Alice Operator",
            "role": "viewer",
            "role_id": 4,
        },
    ]
    assert "private@example.com" not in result.output
    user_api.get_team_members.assert_called_once_with(team_id=7)

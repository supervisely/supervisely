"""Read-only CLI commands for user identity and team roles."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Dict, Sequence

import click

from supervisely.cli.common import command_handler, emit, get_api, to_jsonable


_USER_SAFE_FIELDS = ("id", "login", "name", "role", "role_id", "disabled")
_ROLE_SAFE_FIELDS = ("id", "role")


def _project(value: Any, fields: Sequence[str]) -> Dict[str, Any]:
    """Return only fields intentionally exposed by an identity command."""

    data = to_jsonable(value)
    if not isinstance(data, Mapping):
        raise click.ClickException("The API returned an invalid identity record")
    return {field: data.get(field) for field in fields}


def _project_many(values: Iterable[Any], fields: Sequence[str]):
    return [_project(value, fields) for value in values]


@click.group(name="user")
def user_group() -> None:
    """Inspect the authenticated Supervisely user."""


@user_group.command(name="me")
@command_handler
def get_current_user() -> None:
    """Show the identity associated with the current API token."""

    user = get_api().user.get_my_info()
    if user is None:
        raise click.ClickException("Current user information was not returned")
    emit(
        _project(user, _USER_SAFE_FIELDS),
        columns=_USER_SAFE_FIELDS,
        title="Current user",
    )


@click.group(name="role")
def role_group() -> None:
    """List roles available on the Supervisely instance."""


@role_group.command(name="list")
@command_handler
def list_roles() -> None:
    """List role IDs and names."""

    roles = get_api().role.get_list()
    emit(
        _project_many(roles, _ROLE_SAFE_FIELDS),
        columns=_ROLE_SAFE_FIELDS,
        title="Roles",
    )


@click.command(name="members")
@click.option("--team-id", required=True, type=int, help="Supervisely team ID.")
@command_handler
def team_members_command(team_id: int) -> None:
    """List members and roles in a team."""

    members = get_api().user.get_team_members(team_id=team_id)
    emit(
        _project_many(members, _USER_SAFE_FIELDS),
        columns=_USER_SAFE_FIELDS,
        title="Team members",
    )


def register_identity_commands(team_group: click.Group) -> None:
    """Attach identity commands that belong under an existing team group."""

    team_group.add_command(team_members_command)


__all__ = ["register_identity_commands", "role_group", "user_group"]

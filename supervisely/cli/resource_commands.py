"""CLI commands for common Supervisely platform resources."""

from __future__ import annotations

import click

from supervisely.cli.common import command_handler, emit, get_api, require_resource


@click.group(name="team")
def team_group() -> None:
    """List teams and inspect team details."""


@team_group.command(name="list")
@command_handler
def list_teams() -> None:
    """List teams available to the current user."""

    teams = get_api().team.get_list()
    emit(teams, columns=("id", "name", "role", "description"), title="Teams")


@team_group.command(name="get")
@click.option("--id", "team_id", required=True, type=int, help="Supervisely team ID.")
@command_handler
def get_team(team_id: int) -> None:
    """Get one team by ID."""

    team = require_resource(get_api().team.get_info_by_id(team_id), "Team", team_id)
    emit(team, title="Team")


@click.group(name="workspace")
def workspace_group() -> None:
    """List, inspect, and create workspaces."""


@workspace_group.command(name="list")
@click.option("--team-id", required=True, type=int, help="Parent Supervisely team ID.")
@command_handler
def list_workspaces(team_id: int) -> None:
    """List workspaces in a team."""

    workspaces = get_api().workspace.get_list(team_id=team_id)
    emit(
        workspaces,
        columns=("id", "name", "team_id", "description"),
        title="Workspaces",
    )


@workspace_group.command(name="get")
@click.option("--id", "workspace_id", required=True, type=int, help="Supervisely workspace ID.")
@command_handler
def get_workspace(workspace_id: int) -> None:
    """Get one workspace by ID."""

    workspace = require_resource(
        get_api().workspace.get_info_by_id(workspace_id), "Workspace", workspace_id
    )
    emit(workspace, title="Workspace")


@workspace_group.command(name="create")
@click.option("--team-id", required=True, type=int, help="Parent Supervisely team ID.")
@click.option("--name", required=True, type=str, help="Workspace name.")
@click.option("--description", default="", show_default=True, help="Workspace description.")
@click.option("--hidden", is_flag=True, help="Create the workspace as hidden.")
@command_handler
def create_workspace(team_id: int, name: str, description: str, hidden: bool) -> None:
    """Create a workspace in a team."""

    workspace = get_api().workspace.create(
        team_id=team_id,
        name=name,
        description=description,
        hidden=hidden,
    )
    emit(workspace, title="Workspace created")


@click.group(name="agent")
def agent_group() -> None:
    """List and inspect Supervisely agents."""


@agent_group.command(name="list")
@click.option("--team-id", required=True, type=int, help="Supervisely team ID.")
@click.option(
    "--available",
    is_flag=True,
    help="List agents currently eligible to run applications.",
)
@click.option(
    "--gpu-info",
    is_flag=True,
    help="Include GPU details (available-agent responses already provide supported details).",
)
@click.option(
    "--has-gpu",
    is_flag=True,
    help="With --available, return only agents that have a GPU.",
)
@command_handler
def list_agents(team_id: int, available: bool, gpu_info: bool, has_gpu: bool) -> None:
    """List all or currently available agents in a team."""

    if has_gpu and not available:
        raise click.UsageError("--has-gpu requires --available")

    agent_api = get_api().agent
    if available:
        agents = agent_api.get_list_available(team_id=team_id, has_gpu=has_gpu)
    else:
        agents = agent_api.get_list(team_id=team_id, with_gpu_info=gpu_info)

    columns = ["id", "name", "status", "team_id", "type", "version"]
    if gpu_info or has_gpu:
        columns.append("gpu_info")
    emit(agents, columns=columns, title="Agents")


@agent_group.command(name="get")
@click.option("--id", "agent_id", required=True, type=int, help="Supervisely agent ID.")
@command_handler
def get_agent(agent_id: int) -> None:
    """Get one agent by ID."""

    agent = require_resource(get_api().agent.get_info_by_id(agent_id), "Agent", agent_id)
    emit(agent, title="Agent")


@agent_group.command(name="status")
@click.option("--id", "agent_id", required=True, type=int, help="Supervisely agent ID.")
@command_handler
def get_agent_status(agent_id: int) -> None:
    """Get an agent's runtime status."""

    status = require_resource(get_api().agent.get_status(agent_id), "Agent", agent_id)
    emit({"id": agent_id, "status": status}, title="Agent status")


__all__ = ["team_group", "workspace_group", "agent_group"]

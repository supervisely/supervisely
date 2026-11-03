"""Operational project commands for the Supervisely CLI."""

from __future__ import annotations

from typing import Optional

import click

from supervisely.cli.common import command_handler, emit, get_api, require_resource
from supervisely.project.project_type import ProjectType


PROJECT_TYPES = tuple(ProjectType.values())


def register_project_commands(project_group: click.Group) -> None:
    """Attach API-backed commands to the legacy ``project`` group."""

    @project_group.command(name="list", help="List projects in a workspace or team.")
    @click.option("--workspace-id", type=int, help="Supervisely workspace ID.")
    @click.option("--team-id", type=int, help="Supervisely team ID.")
    @command_handler
    def list_projects(workspace_id: Optional[int], team_id: Optional[int]) -> None:
        if (workspace_id is None) == (team_id is None):
            raise click.UsageError("Provide exactly one of --workspace-id or --team-id")

        projects = get_api().project.get_list(workspace_id=workspace_id, team_id=team_id)
        emit(
            projects,
            columns=(
                "id",
                "name",
                "type",
                "workspace_id",
                "items_count",
                "datasets_count",
                "updated_at",
            ),
            title="Projects",
        )

    @project_group.command(name="get", help="Get project information by ID.")
    @click.option("--id", "project_id", required=True, type=int, help="Supervisely project ID.")
    @command_handler
    def get_project(project_id: int) -> None:
        project = get_api().project.get_info_by_id(project_id)
        emit(require_resource(project, "Project", project_id), title="Project")

    @project_group.command(name="create", help="Create a project.")
    @click.option("--workspace-id", required=True, type=int, help="Destination workspace ID.")
    @click.option("--name", required=True, type=str, help="Project name.")
    @click.option(
        "--type",
        "project_type",
        required=True,
        type=click.Choice(PROJECT_TYPES, case_sensitive=False),
        help="Project modality.",
    )
    @command_handler
    def create_project(workspace_id: int, name: str, project_type: str) -> None:
        project = get_api().project.create(
            workspace_id=workspace_id,
            name=name,
            type=ProjectType(project_type),
        )
        emit(project, title="Created project")

    @project_group.command(name="move", help="Move a project to another workspace.")
    @click.option("--id", "project_id", required=True, type=int, help="Supervisely project ID.")
    @click.option("--workspace-id", required=True, type=int, help="Destination workspace ID.")
    @command_handler
    def move_project(project_id: int, workspace_id: int) -> None:
        get_api().project.move(id=project_id, workspace_id=workspace_id)
        emit({"id": project_id, "workspace_id": workspace_id, "moved": True})

    @project_group.command(name="stats", help="Get project statistics.")
    @click.option("--id", "project_id", required=True, type=int, help="Supervisely project ID.")
    @command_handler
    def project_stats(project_id: int) -> None:
        emit(get_api().project.get_stats(project_id), title="Project statistics")

    @project_group.group(name="meta")
    def project_meta() -> None:
        """Inspect project metadata."""

    @project_meta.command(name="get", help="Get project metadata.")
    @click.option("--id", "project_id", required=True, type=int, help="Supervisely project ID.")
    @command_handler
    def get_project_meta(project_id: int) -> None:
        emit(get_api().project.get_meta(project_id), title="Project metadata")

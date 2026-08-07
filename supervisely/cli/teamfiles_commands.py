"""Read-only Team Files and cloud-storage discovery commands."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Dict

import click

from supervisely.cli.common import (
    command_handler,
    emit,
    get_api,
    require_resource,
    to_jsonable,
)


_SAFE_FILE_FIELDS = (
    "team_id",
    "id",
    "user_id",
    "name",
    "path",
    "mime",
    "ext",
    "sizeb",
    "created_at",
    "updated_at",
    "is_dir",
)


def _file_summary(file_info: Any) -> Dict[str, Any]:
    """Return a stable file shape without storage internals or hashes."""

    data = to_jsonable(file_info)
    if not isinstance(data, Mapping):
        return {"value": data}
    return {field: data.get(field) for field in _SAFE_FILE_FIELDS}


def register_teamfiles_commands(teamfiles_group: click.Group) -> None:
    """Attach read-only storage commands to the legacy ``teamfiles`` group."""

    @teamfiles_group.command(name="list", help="List Team Files or cloud-storage entries.")
    @click.option("--team-id", required=True, type=int, help="Supervisely team ID.")
    @click.option("--path", required=True, type=str, help="Directory or storage path.")
    @click.option(
        "--recursive",
        is_flag=True,
        default=False,
        help="Include entries from nested directories.",
    )
    @click.option(
        "--limit",
        type=click.IntRange(min=1),
        default=100,
        show_default=True,
        help="Maximum number of entries to return.",
    )
    @click.option("--files-only", is_flag=True, help="Return files and omit folders.")
    @click.option("--folders-only", is_flag=True, help="Return folders and omit files.")
    @command_handler
    def list_storage_entries(
        team_id: int,
        path: str,
        recursive: bool,
        limit: int,
        files_only: bool,
        folders_only: bool,
    ) -> None:
        if files_only and folders_only:
            raise click.UsageError("--files-only and --folders-only are mutually exclusive")

        entries = get_api().storage.list(
            team_id=team_id,
            path=path,
            recursive=recursive,
            return_type="fileinfo",
            with_metadata=True,
            include_files=not folders_only,
            include_folders=not files_only,
            limit=limit,
        )
        emit(
            [_file_summary(entry) for entry in entries],
            columns=_SAFE_FILE_FIELDS,
            title="Storage entries",
        )

    @teamfiles_group.command(name="info", help="Get a Team Files or cloud-storage entry.")
    @click.option("--team-id", required=True, type=int, help="Supervisely team ID.")
    @click.option("--path", required=True, type=str, help="File or folder path.")
    @command_handler
    def get_storage_entry(team_id: int, path: str) -> None:
        entry = get_api().storage.get_info_by_path(
            team_id=team_id,
            remote_path=path,
        )
        entry = require_resource(entry, "Storage entry", path)
        emit(_file_summary(entry), title="Storage entry")

    @teamfiles_group.command(name="exists", help="Check whether a storage path exists.")
    @click.option("--team-id", required=True, type=int, help="Supervisely team ID.")
    @click.option("--path", required=True, type=str, help="File or folder path.")
    @click.option(
        "--kind",
        required=True,
        type=click.Choice(("file", "directory"), case_sensitive=False),
        help="Type of storage entry to check.",
    )
    @command_handler
    def storage_entry_exists(team_id: int, path: str, kind: str) -> None:
        storage = get_api().storage
        if kind == "file":
            exists = storage.exists(team_id=team_id, remote_path=path)
        else:
            exists = storage.dir_exists(team_id=team_id, remote_directory=path)

        emit(
            {
                "team_id": team_id,
                "path": path,
                "kind": kind,
                "exists": bool(exists),
            }
        )


__all__ = ["register_teamfiles_commands"]

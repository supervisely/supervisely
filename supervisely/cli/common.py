"""Shared helpers for the Supervisely command-line interface."""

from __future__ import annotations

import dataclasses
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Iterable, List, Optional, Sequence

import click
from rich.console import Console
from rich.table import Table


_SENSITIVE_KEYS = {
    "access_token",
    "authorization",
    "api_key",
    "api_token",
    "apitoken",
    "password",
    "refresh_token",
    "secret",
    "session_token",
    "sessiontoken",
    "token",
    "x_api_key",
    "x-api-key",
}


@dataclass
class CliState:
    """Configuration shared by all commands in one CLI invocation."""

    json_output: bool = False
    server_address: Optional[str] = None
    api_token: Optional[str] = None
    api: Any = None


def get_state() -> CliState:
    """Return the root CLI state, creating it for direct command tests if needed."""

    context = click.get_current_context().find_root()
    if not isinstance(context.obj, CliState):
        context.obj = CliState()
    return context.obj


def get_api():
    """Create and cache an SDK API client for the current CLI invocation."""

    state = get_state()
    if state.api is not None:
        return state.api

    # Import lazily to keep ``supervisely --help`` quick and avoid changing the
    # import behavior of legacy CLI commands.
    import supervisely as sly

    if state.server_address is None and state.api_token is None:
        state.api = sly.Api.from_env()
    elif state.server_address is not None and state.api_token is not None:
        state.api = sly.Api(
            server_address=state.server_address,
            token=state.api_token,
        )
    else:
        # Load the normal development credentials before resolving the one
        # value that was not explicitly overridden. Calling ``from_env`` here
        # would incorrectly require the overridden value to exist in the env.
        if sly.is_development():
            from dotenv import load_dotenv

            from supervisely.api.api import SUPERVISELY_ENV_FILE

            env_path = os.path.expanduser(SUPERVISELY_ENV_FILE)
            if os.path.exists(env_path):
                load_dotenv(env_path)

        server_address = (
            state.server_address
            if state.server_address is not None
            else sly.env.server_address(raise_not_found=False)
        )
        api_token = (
            state.api_token
            if state.api_token is not None
            else sly.env.api_token(raise_not_found=False)
        )
        state.api = sly.Api(
            server_address=server_address,
            token=api_token,
        )
    return state.api


def command_handler(func: Callable) -> Callable:
    """Convert SDK/runtime failures into concise Click errors."""

    @wraps(func)
    def wrapped(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except (click.ClickException, click.Abort):
            raise
        except KeyboardInterrupt as exc:
            raise click.Abort() from exc
        except Exception as exc:
            message = str(exc).strip() or exc.__class__.__name__
            raise click.ClickException(message) from exc

    return wrapped


def _key_is_sensitive(key: Any) -> bool:
    normalized = str(key).strip().lower().replace("-", "_")
    compact = normalized.replace("_", "")
    return (
        normalized in _SENSITIVE_KEYS
        or compact in _SENSITIVE_KEYS
        or compact.endswith("token")
        or compact.endswith("apikey")
    )


def to_jsonable(value: Any) -> Any:
    """Convert SDK return values to JSON-compatible, credential-safe data."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if dataclasses.is_dataclass(value):
        return to_jsonable(dataclasses.asdict(value))
    if hasattr(value, "_asdict"):
        return to_jsonable(value._asdict())
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            result[str(key)] = "***" if _key_is_sensitive(key) else to_jsonable(item)
        return result
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]
    if hasattr(value, "to_json") and callable(value.to_json):
        return to_jsonable(value.to_json())
    if hasattr(value, "__dict__"):
        return to_jsonable(vars(value))
    return str(value)


def _display_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def emit(
    value: Any,
    *,
    columns: Optional[Sequence[str]] = None,
    title: Optional[str] = None,
) -> None:
    """Render a command result as stable JSON or a compact Rich table."""

    data = to_jsonable(value)
    if get_state().json_output:
        click.echo(json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2))
        return

    console = Console()
    if isinstance(data, list):
        if not data:
            console.print("No results.")
            return
        rows = [item if isinstance(item, dict) else {"value": item} for item in data]
        selected = list(columns or _ordered_keys(rows))
        table = Table(title=title, show_lines=False)
        for column in selected:
            table.add_column(column)
        for row in rows:
            table.add_row(*[_display_value(row.get(column)) for column in selected])
        console.print(table)
        return

    if isinstance(data, dict):
        selected = list(columns or data.keys())
        table = Table(title=title, show_header=False)
        table.add_column("Field", style="bold")
        table.add_column("Value")
        for key in selected:
            if key in data:
                table.add_row(key, _display_value(data[key]))
        console.print(table)
        return

    click.echo(_display_value(data))


def _ordered_keys(rows: Iterable[Mapping[str, Any]]) -> List[str]:
    result = []
    for row in rows:
        for key in row:
            if key not in result:
                result.append(key)
    return result


def require_resource(value: Any, resource: str, resource_id: Any) -> Any:
    if value is None:
        raise click.ClickException(
            f"{resource} with ID={resource_id} was not found or is inaccessible"
        )
    return value


def require_yes(yes: bool, action: str) -> None:
    if not yes:
        raise click.UsageError(f"{action} requires --yes")


def read_json_object(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as stream:
            value = json.load(stream)
    except OSError as exc:
        raise click.ClickException(f"Cannot read JSON file '{path}': {exc}") from exc
    except json.JSONDecodeError as exc:
        raise click.ClickException(f"Invalid JSON in '{path}': {exc}") from exc
    if not isinstance(value, dict):
        raise click.ClickException(f"JSON file '{path}' must contain an object")
    return value

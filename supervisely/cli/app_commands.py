"""Commands for discovering and running Supervisely Ecosystem applications."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin, urlsplit, urlunsplit

import click

from supervisely.api.task_api import TaskApi
from supervisely.cli.common import (
    command_handler,
    emit,
    get_api,
    read_json_object,
    require_resource,
    require_yes,
    to_jsonable,
)


# Keep this public CLI description independent from the SDK module's private
# mapping. These values mirror ModuleInfo.get_arguments(), which is the source
# of truth for context-menu launch payloads.
_CONTEXT_TARGETS = {
    "files_folder": {
        "help": "Context menu of folder in Team Files. Target value is directory path.",
        "type": str,
        "key": "slyFolder",
    },
    "files_file": {
        "help": "Context menu of file in Team Files. Target value is file path.",
        "type": str,
        "key": "slyFile",
    },
    "images_project": {
        "help": "Context menu of images project. Target value is project id.",
        "type": int,
        "key": "slyProjectId",
    },
    "images_dataset": {
        "help": "Context menu of images dataset. Target value is dataset id.",
        "type": int,
        "key": "slyDatasetId",
    },
    "videos_project": {
        "help": "Context menu of videos project. Target value is project id.",
        "type": int,
        "key": "slyProjectId",
    },
    "videos_dataset": {
        "help": "Context menu of videos dataset. Target value is dataset id.",
        "type": int,
        "key": "slyDatasetId",
    },
    "point_cloud_episodes_project": {
        "help": "Context menu of pointcloud episodes project. Target value is project id.",
        "type": int,
        "key": "slyProjectId",
    },
    "point_cloud_episodes_dataset": {
        "help": "Context menu of pointcloud episodes dataset. Target value is dataset id.",
        "type": int,
        "key": "slyDatasetId",
    },
    "point_cloud_project": {
        "help": "Context menu of pointclouds project. Target value is project id.",
        "type": int,
        "key": "slyProjectId",
    },
    "point_cloud_dataset": {
        "help": "Context menu of pointclouds dataset. Target value is dataset id.",
        "type": int,
        "key": "slyDatasetId",
    },
    "mesh_project": {
        "help": "Context menu of meshes project. Target value is project id.",
        "type": int,
        "key": "slyProjectId",
    },
    "mesh_dataset": {
        "help": "Context menu of meshes dataset. Target value is dataset id.",
        "type": int,
        "key": "slyDatasetId",
    },
    "volumes_project": {
        "help": "Context menu of volumes project (DICOMs). Target value is project id.",
        "type": int,
        "key": "slyProjectId",
    },
    "volumes_dataset": {
        "help": "Context menu of volumes dataset (DICOMs). Target value is dataset id.",
        "type": int,
        "key": "slyDatasetId",
    },
    "team": {
        "help": "Context menu of team. Target value is team id.",
        "type": int,
        "key": "slyTeamId",
    },
    "team_member": {
        "help": "Context menu of team member. Target value is user id.",
        "type": int,
        "key": "slyMemberId",
    },
    "labeling_job": {
        "help": "Context menu of labeling job. Target value is labeling job id.",
        "type": int,
        "key": "slyJobId",
    },
    "ecosystem": {
        "help": "Run button in ecosystem. It is not needed to define any target",
        "type": None,
        "key": "nothing",
    },
}

_PARAMS_LIMITATION = (
    "Only parameters declared by the app are discoverable. Undeclared fields and "
    "app-specific validation rules cannot be inferred or validated generically."
)

_TASK_STATUSES = tuple(status.value for status in TaskApi.Status)


def _absolute_session_url(server_address: str, path: str) -> str:
    """Join a session path without exposing URL credentials or query values."""

    parsed = urlsplit(server_address)
    hostname = parsed.hostname or ""
    if ":" in hostname:
        hostname = "[{}]".format(hostname)
    netloc = hostname
    if parsed.port is not None:
        netloc = "{}:{}".format(netloc, parsed.port)
    origin = urlunsplit((parsed.scheme, netloc, "/", "", ""))
    return urljoin(origin, path)


def _module_data(module_info: Any) -> Dict[str, Any]:
    data = to_jsonable(module_info)
    return data if isinstance(data, dict) else {"value": data}


def _config(module_info: Any) -> Dict[str, Any]:
    config = getattr(module_info, "config", None)
    if config is None:
        config = _module_data(module_info).get("config")
    return config if isinstance(config, dict) else {}


def _modal_template_state(module_info: Any) -> Dict[str, Any]:
    getter = getattr(module_info, "get_modal_window_arguments", None)
    state = getter() if callable(getter) else _config(module_info).get("modalTemplateState", {})
    return copy.deepcopy(state) if isinstance(state, dict) else {}


def _context_target_names(module_info: Any) -> List[str]:
    getter = getattr(module_info, "get_context_menu_targets", None)
    if callable(getter):
        targets = getter()
    else:
        context_menu = _config(module_info).get("context_menu")
        if context_menu is None:
            context_menu = _config(module_info).get("contextMenu", {})
        targets = context_menu.get("target", []) if isinstance(context_menu, dict) else []
    if isinstance(targets, str):
        return [targets]
    return list(targets or [])


def _type_name(value_type: Any) -> Optional[str]:
    if value_type is int:
        return "integer"
    if value_type is str:
        return "string"
    return None


def _target_spec(target: str, include_placeholder: bool = False) -> Dict[str, Any]:
    target_info = _CONTEXT_TARGETS.get(target)
    if target_info is None:
        result = {
            "target": target,
            "type": None,
            "key": None,
            "help": "No generic CLI metadata is available for this context target.",
        }
    else:
        result = {
            "target": target,
            "type": _type_name(target_info.get("type")),
            "key": target_info.get("key"),
            "help": target_info.get("help"),
        }
    if include_placeholder and result["key"] is not None:
        result["placeholder"] = (
            "<integer>" if result["type"] == "integer" else "<string>"
        )
    return result


def _context_target_specs(module_info: Any) -> List[Dict[str, Any]]:
    return [_target_spec(target) for target in _context_target_names(module_info)]


def _release_version(release: Any) -> Optional[str]:
    if isinstance(release, Mapping):
        value = release.get("version")
        return str(value) if value is not None else None
    if release in (None, ""):
        return None
    return str(release)


def _latest_version(module_info: Any) -> Optional[str]:
    getter = getattr(module_info, "get_latest_release", None)
    if callable(getter):
        try:
            return _release_version(getter())
        except (AttributeError, IndexError, KeyError, TypeError):
            pass

    data = _module_data(module_info)
    meta = data.get("meta") or {}
    releases = meta.get("releases", []) if isinstance(meta, dict) else []
    return _release_version(releases[0]) if releases else None


def _release_versions(module: Any) -> List[str]:
    data = to_jsonable(module)
    if not isinstance(data, dict):
        return []
    meta = data.get("meta") or {}
    releases = meta.get("releases", []) if isinstance(meta, dict) else []
    versions = [_release_version(release) for release in releases]
    return [version for version in versions if version is not None]


def _module_summary(module: Any) -> Dict[str, Any]:
    data = to_jsonable(module)
    if not isinstance(data, dict):
        return {"value": data}
    return {
        "id": data.get("id"),
        "slug": data.get("slug"),
        "name": data.get("name"),
        "type": data.get("type"),
        "releases": _release_versions(data),
    }


def _choose_version(branch: Optional[str], version: Optional[str]) -> Optional[str]:
    if branch is not None and version is not None:
        raise click.UsageError("--branch and --version are mutually exclusive")
    return branch if branch is not None else version


def _resolve_module(
    slug: str, branch: Optional[str] = None, version: Optional[str] = None
) -> Any:
    selected_version = _choose_version(branch, version)
    module_info = get_api().app.get_ecosystem_module_info(
        slug=slug, version=selected_version
    )
    return require_resource(module_info, "Ecosystem app", slug)


def _required_target_description(
    targets: List[str], modal_state: Dict[str, Any]
) -> Any:
    target_specs = [
        _target_spec(target, include_placeholder=True)
        for target in targets
        if target != "ecosystem"
    ]
    if not target_specs:
        return None, None

    templates = []
    for spec in target_specs:
        template = copy.deepcopy(modal_state)
        if spec.get("key") is not None:
            template[spec["key"]] = spec.get("placeholder")
        templates.append({"target": spec["target"], "params": template})

    requirement = target_specs[0] if len(target_specs) == 1 else {"one_of": target_specs}
    template = templates[0]["params"] if len(templates) == 1 else {"one_of": templates}
    return requirement, template


def _validate_context_target(module_info: Any, params: Optional[Dict[str, Any]]) -> None:
    targets = _context_target_names(module_info)
    if not targets or "ecosystem" in targets:
        return

    if params is None:
        raise click.ClickException(
            "This app must be launched from a context target; provide --params-file "
            "using a template from 'app params'."
        )

    known_targets = []
    for target in targets:
        target_info = _CONTEXT_TARGETS.get(target)
        if target_info is not None and target_info.get("type") is not None:
            known_targets.append((target, target_info))

    if not known_targets:
        raise click.ClickException(
            "The app requires an unsupported context target: " + ", ".join(targets)
        )

    def find_matching_keys(container: Dict[str, Any]) -> Dict[str, Any]:
        result = {}
        for target, target_info in known_targets:
            key = target_info["key"]
            if key in container:
                result.setdefault(key, []).append((target, target_info))
        return result

    # ModuleInfo.get_arguments() adds the context key to the top level of the
    # modal template object. If that template already contains ``state``, the
    # context key remains its sibling, so the top-level object is authoritative.
    matching_keys = find_matching_keys(params)
    if not matching_keys and isinstance(params.get("state"), dict):
        # Accept explicitly wrapped payloads as a compatibility convenience.
        matching_keys = find_matching_keys(params["state"])

    if not matching_keys:
        expected = sorted({target_info["key"] for _, target_info in known_targets})
        raise click.ClickException(
            "Missing required context target in --params-file; set exactly one of: "
            + ", ".join(expected)
        )
    if len(matching_keys) > 1:
        raise click.ClickException(
            "Only one context target may be supplied; found: "
            + ", ".join(sorted(matching_keys))
        )

    key, matching_targets = next(iter(matching_keys.items()))
    expected_type = matching_targets[0][1]["type"]
    if key in params:
        value = params[key]
    else:
        value = params["state"][key]
    if type(value) is not expected_type:
        raise click.ClickException(
            "Context target '{}' must be {}, got {}.".format(
                key, _type_name(expected_type), type(value).__name__
            )
        )


@click.group(name="app")
def app_group() -> None:
    """Discover, inspect, run, and stop Supervisely applications."""


@app_group.command(name="list")
@click.option("--search", type=str, help="Filter Ecosystem applications by text.")
@command_handler
def list_apps(search: Optional[str]) -> None:
    """List applications available in the Ecosystem."""

    modules = get_api().app.get_list_ecosystem_modules(search=search)
    emit(
        [_module_summary(module) for module in modules],
        columns=["id", "slug", "name", "type", "releases"],
        title="Ecosystem applications",
    )


@app_group.command(name="describe")
@click.option("--slug", required=True, help="Ecosystem application slug.")
@click.option("--branch", help="Inspect configuration from this repository branch.")
@click.option("--version", help="Inspect configuration from this released version.")
@command_handler
def describe_app(slug: str, branch: Optional[str], version: Optional[str]) -> None:
    """Show application metadata and its declared launch configuration."""

    selected_version = _choose_version(branch, version)
    module_info = _resolve_module(slug, branch=branch, version=version)
    data = _module_data(module_info)
    data.update(
        {
            "selected_version": selected_version or _latest_version(module_info),
            "is_branch": branch is not None,
            "modal_template_state": _modal_template_state(module_info),
            "context_targets": _context_target_specs(module_info),
        }
    )
    emit(data, title="Application")


@app_group.command(name="params")
@click.option("--slug", required=True, help="Ecosystem application slug.")
@click.option("--branch", help="Inspect configuration from this repository branch.")
@click.option("--version", help="Inspect configuration from this released version.")
@command_handler
def app_params(slug: str, branch: Optional[str], version: Optional[str]) -> None:
    """Print the app's declared parameters and a launch template."""

    selected_version = _choose_version(branch, version)
    module_info = _resolve_module(slug, branch=branch, version=version)
    modal_state = _modal_template_state(module_info)
    target_names = _context_target_names(module_info)
    can_run_from_ecosystem = not target_names or "ecosystem" in target_names

    required_target = None
    params_template = copy.deepcopy(modal_state)
    if not can_run_from_ecosystem:
        required_target, params_template = _required_target_description(
            target_names, modal_state
        )

    emit(
        {
            "module_id": getattr(module_info, "id", _module_data(module_info).get("id")),
            "slug": getattr(module_info, "slug", slug),
            "selected_version": selected_version or _latest_version(module_info),
            "is_branch": branch is not None,
            "modal_template_state": modal_state,
            "context_targets": _context_target_specs(module_info),
            "can_run_from_ecosystem": can_run_from_ecosystem,
            "params_template": params_template,
            "required_context_target": required_target,
            "limitations": [_PARAMS_LIMITATION],
        },
        title="Application parameters",
    )


@app_group.command(name="sessions")
@click.option("--team-id", required=True, type=int, help="Supervisely team ID.")
@click.option("--slug", required=True, help="Ecosystem application slug.")
@click.option(
    "--status",
    "statuses",
    multiple=True,
    type=click.Choice(_TASK_STATUSES, case_sensitive=False),
    help="Filter by task status; repeatable.",
)
@click.option("--show-disabled", is_flag=True, help="Include disabled sessions.")
@click.option("--with-shared", is_flag=True, help="Include shared applications.")
@command_handler
def app_sessions(
    team_id: int,
    slug: str,
    statuses: List[str],
    show_disabled: bool,
    with_shared: bool,
) -> None:
    """List sessions for an application selected by slug."""

    module_info = _resolve_module(slug)
    sessions = get_api().app.get_sessions(
        team_id=team_id,
        module_id=module_info.id,
        statuses=list(statuses) or None,
        show_disabled=show_disabled,
        with_shared=with_shared,
    )
    emit(
        sessions,
        columns=["task_id", "user_id", "module_id", "app_id", "details"],
        title="Application sessions",
    )


@app_group.command(name="run")
@click.option("--slug", required=True, help="Ecosystem application slug.")
@click.option("--workspace-id", required=True, type=int, help="Destination workspace ID.")
@click.option("--agent-id", type=int, help="Agent ID; omit for automatic selection.")
@click.option("--params-file", type=str, help="Path to a JSON object with app parameters.")
@click.option("--task-name", default="run-from-cli", show_default=True, help="Task name.")
@click.option("--description", default="", help="Task description.")
@click.option(
    "--log-level",
    type=click.Choice(["info", "debug", "warning", "error"], case_sensitive=False),
    default="info",
    show_default=True,
)
@click.option(
    "--restart-policy",
    type=click.Choice(["never", "on_error"], case_sensitive=False),
    default="never",
    show_default=True,
)
@click.option("--branch", help="Run this repository branch.")
@click.option("--version", help="Run this released version.")
@command_handler
def run_app(
    slug: str,
    workspace_id: int,
    agent_id: Optional[int],
    params_file: Optional[str],
    task_name: str,
    description: str,
    log_level: str,
    restart_policy: str,
    branch: Optional[str],
    version: Optional[str],
) -> None:
    """Start an application selected by slug."""

    app_version = _choose_version(branch, version)
    module_info = _resolve_module(slug, branch=branch, version=version)
    supplied_params = read_json_object(params_file) if params_file is not None else None
    _validate_context_target(module_info, supplied_params)
    params = (
        supplied_params
        if supplied_params is not None
        else _modal_template_state(module_info)
    )

    session = get_api().app.start(
        agent_id=agent_id,
        module_id=module_info.id,
        workspace_id=workspace_id,
        description=description,
        params=params,
        log_level=log_level,
        app_version=app_version,
        is_branch=branch is not None,
        task_name=task_name,
        restart_policy=restart_policy,
    )
    emit(session, title="Application session started")


@app_group.command(name="stop")
@click.option("--task-id", required=True, type=int, help="Application task ID.")
@click.option("--yes", is_flag=True, help="Confirm stopping the application.")
@command_handler
def stop_app(task_id: int, yes: bool) -> None:
    """Stop an application session."""

    require_yes(yes, f"Stopping application task {task_id}")
    status = get_api().app.stop(task_id)
    emit({"task_id": task_id, "status": status}, title="Application session stopped")


@app_group.command(name="url")
@click.option("--task-id", required=True, type=int, help="Application task ID.")
@command_handler
def app_url(task_id: int) -> None:
    """Get the browser URL for an application session."""

    api = get_api()
    path = api.app.get_url(task_id)
    server_address = getattr(api, "server_address", None)
    url = _absolute_session_url(server_address, path) if server_address else path
    emit({"task_id": task_id, "path": path, "url": url}, title="Application session URL")


__all__ = ["app_group"]

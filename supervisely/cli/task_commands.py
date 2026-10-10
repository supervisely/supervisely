"""Operational task commands for the Supervisely CLI."""

from __future__ import annotations

import time
from enum import Enum
from typing import Optional

import click

from supervisely.api.task_api import TaskApi
from supervisely.cli.common import (
    command_handler,
    emit,
    get_api,
    require_resource,
    require_yes,
)


TASK_STATUSES = tuple(status.value for status in TaskApi.Status)
TERMINAL_STATUSES = frozenset(
    (
        TaskApi.Status.ERROR.value,
        TaskApi.Status.FINISHED.value,
        TaskApi.Status.DEPLOYED.value,
        TaskApi.Status.STOPPED.value,
    )
)
POLL_INTERVAL_SECONDS = 1.0


def _status_value(status: object) -> str:
    if isinstance(status, Enum):
        return str(status.value)
    return str(status)


def register_task_commands(task_group: click.Group) -> None:
    """Attach API-backed commands to the legacy ``task`` group."""

    @task_group.command(name="list", help="List tasks in a workspace.")
    @click.option("--workspace-id", required=True, type=int, help="Supervisely workspace ID.")
    @command_handler
    def list_tasks(workspace_id: int) -> None:
        tasks = get_api().task.get_list(workspace_id=workspace_id)
        emit(
            tasks,
            columns=("id", "status", "type", "description", "startedAt", "finishedAt"),
            title="Tasks",
        )

    @task_group.command(name="get", help="Get task information by ID.")
    @click.option("--id", "task_id", required=True, type=int, help="Supervisely task ID.")
    @command_handler
    def get_task(task_id: int) -> None:
        task = get_api().task.get_info_by_id(task_id)
        emit(require_resource(task, "Task", task_id), title="Task")

    @task_group.command(name="status", help="Get a task's current status.")
    @click.option("--id", "task_id", required=True, type=int, help="Supervisely task ID.")
    @command_handler
    def task_status(task_id: int) -> None:
        status = _status_value(get_api().task.get_status(task_id))
        emit({"id": task_id, "status": status})

    @task_group.command(
        name="context", help="Get the team and workspace context for a task."
    )
    @click.option("--id", "task_id", required=True, type=int, help="Supervisely task ID.")
    @command_handler
    def task_context(task_id: int) -> None:
        context = get_api().task.get_context(task_id)
        emit(context, title="Task context")

    @task_group.command(name="logs", help="Get task logs.")
    @click.option("--id", "task_id", required=True, type=int, help="Supervisely task ID.")
    @click.option("--since", type=str, help="Return logs after this ISO 8601 timestamp.")
    @click.option("--limit", type=click.IntRange(min=1), help="Maximum number of log records.")
    @command_handler
    def task_logs(task_id: int, since: Optional[str], limit: Optional[int]) -> None:
        logs = get_api().task.get_logs(task_id=task_id, since_time=since, limit=limit)
        emit(logs, title="Task logs")

    @task_group.command(name="wait", help="Wait until a task reaches a target status.")
    @click.option("--id", "task_id", required=True, type=int, help="Supervisely task ID.")
    @click.option(
        "--target-status",
        type=click.Choice(TASK_STATUSES, case_sensitive=False),
        default="finished",
        show_default=True,
        help="Status to wait for.",
    )
    @click.option(
        "--timeout",
        type=click.FloatRange(min=0.0),
        default=300.0,
        show_default=True,
        help="Maximum elapsed time in seconds.",
    )
    @command_handler
    def wait_for_task(task_id: int, target_status: str, timeout: float) -> None:
        task_api = get_api().task
        deadline = time.monotonic() + timeout

        while True:
            status = _status_value(task_api.get_status(task_id))
            if status == target_status:
                emit({"id": task_id, "status": status, "reached": True})
                return
            if status in TERMINAL_STATUSES:
                raise click.ClickException(
                    f"Task {task_id} reached terminal status '{status}' before "
                    f"target status '{target_status}'"
                )

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise click.ClickException(
                    f"Timed out after {timeout:g} seconds waiting for task {task_id} "
                    f"to reach status '{target_status}' (current status: '{status}')"
                )
            time.sleep(min(POLL_INTERVAL_SECONDS, remaining))

    @task_group.command(name="stop", help="Stop a running task.")
    @click.option("--id", "task_id", required=True, type=int, help="Supervisely task ID.")
    @click.option("--yes", is_flag=True, help="Confirm stopping the task.")
    @command_handler
    def stop_task(task_id: int, yes: bool) -> None:
        require_yes(yes, f"Stopping task {task_id}")
        status = _status_value(get_api().task.stop(task_id))
        emit({"id": task_id, "status": status, "stopped": True})

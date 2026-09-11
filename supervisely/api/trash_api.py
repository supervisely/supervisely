# coding: utf-8
"""API for working with the Trash Bin"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Callable, Dict, List, NamedTuple, Optional

from supervisely.sly_logger import logger

if TYPE_CHECKING:
    from supervisely.api.api import Api


TEAM = "team"
WORKSPACE = "workspace"
PROJECT = "project"
DATASET = "dataset"


class TrashItem(NamedTuple):
    """A single archived entity sitting in the Trash Bin."""

    type: str
    """Entity kind: ``team``, ``workspace``, ``project`` or ``dataset``."""
    id: int
    """Entity ID in Supervisely."""
    name: str
    """Entity name."""
    team_id: Optional[int]
    """ID of the Team the entity belongs to, if known."""
    workspace_id: Optional[int]
    """ID of the Workspace the entity belongs to, for workspaces and projects."""
    project_id: Optional[int]
    """ID of the Project the entity belongs to, for datasets."""


class TrashApi:
    """
    API for listing and emptying the Trash Bin (the ``/server-trash`` page).

    Removal in Supervisely is a two-step flow. ``*.archive`` (exposed in the SDK as
    :func:`ProjectApi.remove`, :func:`DatasetApi.remove`, :func:`TeamApi.archive` and
    :func:`WorkspaceApi.archive`) is a soft, reversible removal that moves an entity to the
    Trash Bin. ``*.remove.permanently`` is the second, irreversible step. This class walks the
    instance, finds everything already on the first step, and applies the second one.

    Only Teams, Workspaces, Projects and Datasets are reachable over the public API. The Trash
    Bin page also lists models, checkpoints, python notebooks and DTL archives; those have no
    public API and are left untouched by :func:`clear`.
    """

    def __init__(self, api: "Api"):
        self._api = api

    def get_list(
        self,
        team_id: Optional[int] = None,
        include_datasets: bool = True,
    ) -> List[TrashItem]:
        """
        List everything currently in the Trash Bin.

        The instance is walked top-down: archived Teams, then archived Workspaces inside the
        Teams that remain, then archived Projects, then archived Datasets that are still
        sitting inside live Projects. Nested entities are not listed separately from an
        archived parent that already contains them — removing the parent removes them too.

        :param team_id: Restrict the walk to a single Team. ``None`` walks every Team.
        :type team_id: int, optional
        :param include_datasets: Also look for archived Datasets inside live Projects. This
                                 costs one API call per live Project, so pass ``False`` on a
                                 large instance when only whole Projects matter.
        :type include_datasets: bool, optional
        :returns: Archived entities, parents before children.
        :rtype: List[:class:`~supervisely.api.trash_api.TrashItem`]

        :Usage Example:

            .. code-block:: python

                import os
                from dotenv import load_dotenv

                import supervisely as sly

                # Load secrets and create API object from .env file (recommended)
                # Learn more here: https://developer.supervisely.com/getting-started/basics-of-authentication
                if sly.is_development():
                    load_dotenv(os.path.expanduser("~/supervisely.env"))

                api = sly.Api.from_env()

                for item in api.trash.get_list():
                    print(item.type, item.id, item.name)
        """
        items: List[TrashItem] = []

        archived_teams = [
            team
            for team in self._api.team.get_list(archived=True)
            if team.id != 1 and (team_id is None or team.id == team_id)
        ]
        for team in archived_teams:
            items.append(TrashItem(TEAM, team.id, team.name, team.id, None, None))

        removed_team_ids = {team.id for team in archived_teams}
        live_teams = [
            team
            for team in self._api.team.get_list()
            if team.id not in removed_team_ids and (team_id is None or team.id == team_id)
        ]

        for team in live_teams:
            archived_workspaces = self._api.workspace.get_list(team.id, archived=True)
            for workspace in archived_workspaces:
                items.append(
                    TrashItem(WORKSPACE, workspace.id, workspace.name, team.id, workspace.id, None)
                )

            removed_workspace_ids = {workspace.id for workspace in archived_workspaces}
            for workspace in self._api.workspace.get_list(team.id):
                if workspace.id in removed_workspace_ids:
                    continue

                archived_projects = self._api.project.get_list(workspace.id, archived=True)
                for project in archived_projects:
                    items.append(
                        TrashItem(PROJECT, project.id, project.name, team.id, workspace.id, None)
                    )

                if not include_datasets:
                    continue

                removed_project_ids = {project.id for project in archived_projects}
                for project in self._api.project.get_list(workspace.id):
                    if project.id in removed_project_ids:
                        continue
                    for dataset in self._api.dataset.get_list(
                        project.id, recursive=True, archived=True
                    ):
                        items.append(
                            TrashItem(
                                DATASET, dataset.id, dataset.name, team.id, workspace.id, project.id
                            )
                        )

        return items

    def clear(
        self,
        team_id: Optional[int] = None,
        include_datasets: bool = True,
        cleanup_unused: bool = True,
        progress_cb: Optional[Callable] = None,
    ) -> Dict[str, int]:
        """
        !!! WARNING !!!
        Be careful, this method deletes data from the database, recovery is not possible.

        Permanently remove everything in the Trash Bin. Available only to the instance
        administrator (root user); a regular user token is not enough. Call :func:`get_list`
        first and read what it returns — there is no second Trash Bin behind this one.

        Team and Workspace removal runs in the background, so this method waits for each of
        those tasks to reach a terminal status before moving on to the entities nested below.

        :param team_id: Restrict the removal to a single Team. ``None`` empties the whole
                        instance Trash Bin.
        :type team_id: int, optional
        :param include_datasets: Also remove archived Datasets sitting inside live Projects.
        :type include_datasets: bool, optional
        :param cleanup_unused: Trigger ``instance.data.cleanup-unused`` afterwards. Image and
                               video data is reference-counted, so removing an entity drops
                               its references and the storage behind them is reclaimed by the
                               garbage collector, which also runs daily on its own.
        :type cleanup_unused: bool, optional
        :param progress_cb: Function for tracking removal progress, called with the number of
                            entities removed by each API call.
        :type progress_cb: Callable, optional
        :returns: Number of entities removed, keyed by kind.
        :rtype: Dict[str, int]

        :Usage Example:

            .. code-block:: python

                import os
                from dotenv import load_dotenv

                import supervisely as sly

                # Load secrets and create API object from .env file (recommended)
                # Learn more here: https://developer.supervisely.com/getting-started/basics-of-authentication
                if sly.is_development():
                    load_dotenv(os.path.expanduser("~/supervisely.env"))

                api = sly.Api.from_env()

                print(api.trash.clear())
                # Output: {'team': 1, 'workspace': 0, 'project': 4, 'dataset': 2}
        """
        items = self.get_list(team_id=team_id, include_datasets=include_datasets)
        removed = {TEAM: 0, WORKSPACE: 0, PROJECT: 0, DATASET: 0}

        by_type: Dict[str, List[TrashItem]] = {TEAM: [], WORKSPACE: [], PROJECT: [], DATASET: []}
        for item in items:
            by_type[item.type].append(item)

        # Teams and Workspaces take everything nested inside them, and drain in the background.
        for kind, api_module in ((TEAM, self._api.team), (WORKSPACE, self._api.workspace)):
            ids = [item.id for item in by_type[kind]]
            if not ids:
                continue
            for response in api_module.remove_permanently(ids, progress_cb=progress_cb):
                self._wait_for_task(response.get("taskId"))
            removed[kind] = len(ids)

        # `preserveProjectCard` defaults to True on the server, which keeps the project row in
        # the Trash Bin and only drops its data. `ProjectApi.remove_permanently` sends False.
        project_ids = [item.id for item in by_type[PROJECT]]
        if project_ids:
            self._api.project.remove_permanently(project_ids, progress_cb=progress_cb)
            removed[PROJECT] = len(project_ids)

        dataset_ids = [item.id for item in by_type[DATASET]]
        if dataset_ids:
            self._api.dataset.remove_permanently(dataset_ids, progress_cb=progress_cb)
            removed[DATASET] = len(dataset_ids)

        if cleanup_unused and any(removed.values()):
            self.cleanup_unused()

        return removed

    def cleanup_unused(self) -> Optional[int]:
        """
        Trigger the instance-wide garbage collector for unreferenced data. Root only.

        Storage is not released the moment a removal finishes: data is reference-counted by
        content hash, and the collector applies a grace window before reclaiming an object.
        Running this twice in a row does not shorten that window.

        :returns: ID of the background task, or ``None`` if the instance does not expose the
                  method (added in Supervisely 6.17.17).
        :rtype: Optional[int]
        """
        try:
            response = self._api.post("instance.data.cleanup-unused", {})
        except Exception as e:
            logger.warning(f"Could not trigger cleanup of unused data: {e}")
            return None
        return response.json().get("taskId")

    def _wait_for_task(
        self, task_id: Optional[int], poll_sec: int = 5, timeout_sec: int = 3600
    ) -> None:
        """Block until a background removal task reaches a terminal status."""
        if task_id is None:
            return
        deadline = time.monotonic() + timeout_sec
        while True:
            status = self._api.task.get_status(task_id)
            if status in (self._api.task.Status.FINISHED, self._api.task.Status.ERROR):
                if status is self._api.task.Status.ERROR:
                    logger.warning(f"Removal task {task_id} finished with status 'error'")
                return
            if time.monotonic() >= deadline:
                logger.warning(
                    f"Removal task {task_id} did not finish within {timeout_sec}s, "
                    f"last status '{status}'. It keeps running on the server."
                )
                return
            time.sleep(poll_sec)

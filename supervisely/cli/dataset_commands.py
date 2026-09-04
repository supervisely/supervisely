"""Dataset commands for the Supervisely CLI."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Optional

import click

from supervisely.cli.common import (
    command_handler,
    emit,
    get_api,
    require_resource,
    to_jsonable,
)


@click.group(name="dataset")
def dataset_group() -> None:
    """Inspect and create Supervisely datasets."""


@dataset_group.command(name="list", help="List top-level datasets in a project.")
@click.option("--project-id", required=True, type=int, help="Supervisely project ID.")
@command_handler
def list_datasets(project_id: int) -> None:
    datasets = get_api().dataset.get_list(project_id=project_id)
    emit(
        datasets,
        columns=("id", "name", "parent_id", "items_count", "size", "updated_at"),
        title="Datasets",
    )


@dataset_group.command(name="tree", help="List the complete dataset hierarchy in a project.")
@click.option("--project-id", required=True, type=int, help="Supervisely project ID.")
@command_handler
def dataset_tree(project_id: int) -> None:
    rows = []
    for parents, dataset in get_api().dataset.tree(project_id):
        data = to_jsonable(dataset)
        row = dict(data) if isinstance(data, Mapping) else {"value": data}
        name = row.get("name", str(dataset))
        row["path"] = [*parents, name]
        row["depth"] = len(parents)
        rows.append(row)

    emit(
        rows,
        columns=("id", "path", "parent_id", "items_count"),
        title="Dataset tree",
    )


@dataset_group.command(name="get", help="Get dataset information by ID.")
@click.option("--id", "dataset_id", required=True, type=int, help="Supervisely dataset ID.")
@command_handler
def get_dataset(dataset_id: int) -> None:
    dataset = get_api().dataset.get_info_by_id(dataset_id)
    emit(require_resource(dataset, "Dataset", dataset_id), title="Dataset")


@dataset_group.command(name="create", help="Create a dataset.")
@click.option("--project-id", required=True, type=int, help="Destination project ID.")
@click.option("--name", required=True, type=str, help="Dataset name.")
@click.option("--parent-id", type=int, help="Optional parent dataset ID.")
@command_handler
def create_dataset(project_id: int, name: str, parent_id: Optional[int]) -> None:
    dataset = get_api().dataset.create(
        project_id=project_id,
        name=name,
        parent_id=parent_id,
    )
    emit(dataset, title="Created dataset")

"""Read-only image commands for the Supervisely CLI."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Dict, Optional

import click

from supervisely.cli.common import (
    command_handler,
    emit,
    get_api,
    require_resource,
    to_jsonable,
)


IMAGE_OUTPUT_FIELDS = (
    "id",
    "name",
    "dataset_id",
    "project_id",
    "width",
    "height",
    "labels_count",
    "created_at",
    "updated_at",
)
IMAGE_SORT_FIELDS = (
    "id",
    "name",
    "description",
    "labelsCount",
    "createdAt",
    "updatedAt",
    "customSort",
)


def _image_output(image: Any) -> Dict[str, Any]:
    """Return the stable, non-storage subset exposed by image commands."""

    data = to_jsonable(image)
    if not isinstance(data, Mapping):
        raise click.ClickException("Unexpected image information returned by the API")
    return {field: data.get(field) for field in IMAGE_OUTPUT_FIELDS}


@click.group(name="image")
def image_group() -> None:
    """Inspect Supervisely images and their annotations."""


@image_group.command(name="list", help="List images in a dataset or project.")
@click.option("--dataset-id", type=int, help="Supervisely dataset ID.")
@click.option("--project-id", type=int, help="Supervisely project ID.")
@click.option(
    "--limit",
    default=100,
    show_default=True,
    type=click.IntRange(min=1),
    help="Maximum number of images to return.",
)
@click.option(
    "--sort",
    default="id",
    show_default=True,
    type=click.Choice(IMAGE_SORT_FIELDS),
    help="Image field to sort by.",
)
@click.option(
    "--order",
    default="asc",
    show_default=True,
    type=click.Choice(("asc", "desc"), case_sensitive=False),
    help="Sort order.",
)
@click.option("--recursive", is_flag=True, help="Include images from nested datasets.")
@click.option("--only-labelled", is_flag=True, help="Return only labelled images.")
@command_handler
def list_images(
    dataset_id: Optional[int],
    project_id: Optional[int],
    limit: int,
    sort: str,
    order: str,
    recursive: bool,
    only_labelled: bool,
) -> None:
    if (dataset_id is None) == (project_id is None):
        raise click.UsageError("Provide exactly one of --dataset-id or --project-id")

    images = get_api().image.get_list(
        dataset_id=dataset_id,
        sort=sort,
        sort_order=order,
        limit=limit,
        force_metadata_for_links=False,
        project_id=project_id,
        only_labelled=only_labelled,
        recursive=recursive,
    )
    emit(
        [_image_output(image) for image in images],
        columns=IMAGE_OUTPUT_FIELDS,
        title="Images",
    )


@image_group.command(name="get", help="Get image information by ID.")
@click.option("--id", "image_id", required=True, type=int, help="Supervisely image ID.")
@command_handler
def get_image(image_id: int) -> None:
    image = get_api().image.get_info_by_id(
        image_id,
        force_metadata_for_links=False,
    )
    image = require_resource(image, "Image", image_id)
    emit(_image_output(image), columns=IMAGE_OUTPUT_FIELDS, title="Image")


@image_group.command(name="annotation", help="Get an image annotation as JSON.")
@click.option("--id", "image_id", required=True, type=int, help="Supervisely image ID.")
@click.option(
    "--custom-data",
    is_flag=True,
    help="Include annotation custom data in the response.",
)
@command_handler
def get_image_annotation(image_id: int, custom_data: bool) -> None:
    annotation = get_api().annotation.download_json(
        image_id,
        with_custom_data=custom_data,
        force_metadata_for_links=True,
    )
    emit(annotation, title="Image annotation")

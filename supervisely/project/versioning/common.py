from supervisely.project.versioning.image_schema import (
    _IMAGE_SCHEMAS,
    ImageSnapshotSchema,
)
from supervisely.project.versioning.video_schema import (
    _VIDEO_SCHEMAS,
    VideoSnapshotSchema,
)
from supervisely.project.versioning.volume_schema import (
    _VOLUME_SCHEMAS,
    VolumeSnapshotSchema,
)

# Pickle. Predates the Parquet container. The public image backup API keeps writing it
# by default for callers without the optional Parquet dependency; Data Versioning asks
# for the current format explicitly. Everything still reads it: old archives have to
# stay restorable for as long as their versions exist.
IMAGE_SCHEMA_VERSION_V1 = "v1.0.0"

# Parquet, same container as video and volume snapshots.
IMAGE_SCHEMA_VERSION_V2 = "v2.0.0"

# Same layout as v2.0.0, renumbered so that the version string carries one meaning in
# every modality: columnar, ids are the server's, comparable with another version.
IMAGE_SCHEMA_VERSION_V2_1 = "v2.1.0"

# Which format new image versions are written in. Changing it is safe in both
# directions: a snapshot says what it is (a zstd frame with a manifest, or a pickle),
# so the reader picks the right path per archive rather than per SDK version.
DEFAULT_IMAGE_SCHEMA_VERSION = IMAGE_SCHEMA_VERSION_V2_1


def default_image_schema_version() -> str:
    """The format to write a new image version in, given what is installed.

    The columnar format needs pyarrow, which lives in the ``versioning`` extra rather than
    in the base install. Answering ``v2.1.0`` regardless would turn a plain
    ``pip install supervisely`` into one where creating a version raises instead of writing
    a pickle, so an install without the extra keeps the old format and everything that
    reads a snapshot still reads both.
    """
    try:
        from supervisely.project.versioning.image_snapshot_io import import_pyarrow

        import_pyarrow()
    except Exception:
        return IMAGE_SCHEMA_VERSION_V1

    return DEFAULT_IMAGE_SCHEMA_VERSION
VOLUME_SCHEMA_VERSION_V2 = "v2.0.0"
VOLUME_SCHEMA_VERSION_V2_1 = "v2.1.0"

DEFAULT_VOLUME_SCHEMA_VERSION = VOLUME_SCHEMA_VERSION_V2_1
VIDEO_SCHEMA_VERSION_V2 = "v2.0.0"
VIDEO_SCHEMA_VERSION_V2_1 = "v2.1.0"

DEFAULT_VIDEO_SCHEMA_VERSION = VIDEO_SCHEMA_VERSION_V2_1
HIDDEN_WORKSPACE_NAME = "[Do Not Modify] Instant Versions Storage"
PREVIEW_NAME_TEMPLATE = "{project_name}, preview for ver. {version_num}"
PREVIEW_DESCRIPTION_TEMPLATE = (
    "Preview for version {version_num}. "
    "Source project ID: {project_id}, version ID: {version_id}"
)
CUSTOM_DATA_VERSION_PREVIEW_KEY = "sly_version_preview"
CUSTOM_DATA_VERSION_RESTORED_KEY = "restored_from"


def get_image_snapshot_schema(schema_version: str) -> ImageSnapshotSchema:
    schema = _IMAGE_SCHEMAS.get(schema_version)
    if schema is None:
        raise RuntimeError(f"Unsupported image snapshot schema_version: {schema_version!r}")
    return schema


def get_video_snapshot_schema(schema_version: str) -> VideoSnapshotSchema:
    schema = _VIDEO_SCHEMAS.get(schema_version)
    if schema is None:
        raise RuntimeError(f"Unsupported video snapshot schema_version: {schema_version!r}")
    return schema


def get_volume_snapshot_schema(schema_version: str) -> VolumeSnapshotSchema:
    schema = _VOLUME_SCHEMAS.get(schema_version)
    if schema is None:
        raise RuntimeError(f"Unsupported volume snapshot schema_version: {schema_version!r}")
    return schema


def update_custom_data_with_version_preview(
    custom_data: dict, version_id: int, source_project_id: int, preview_created_at: str
) -> dict:
    """Update custom data for version preview project with the information about the version and preview project.

    :param custom_data: The original custom data of the preview project.
    :type custom_data: dict
    :param version_id: The ID of the version for which the preview project was created.
    :type version_id: int
    :param source_project_id: The ID of the source project for which the version was created.
    :type source_project_id: int
    :param preview_created_at: The timestamp when the preview project was created.
    :type preview_created_at: str
    :return: The updated custom data with the version preview information.
    :rtype: dict
    """

    custom_data[CUSTOM_DATA_VERSION_PREVIEW_KEY] = {
        "version_id": version_id,
        "source_project_id": source_project_id,
        "preview_created_at": preview_created_at,
    }
    return custom_data

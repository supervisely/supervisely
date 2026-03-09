from supervisely.project.versioning.video_schema import (
    _VIDEO_SCHEMAS,
    VideoSnapshotSchema,
)
from supervisely.project.versioning.volume_schema import (
    _VOLUME_SCHEMAS,
    VolumeSnapshotSchema,
)

DEFAULT_IMAGE_SCHEMA_VERSION = "v1.0.0"
DEFAULT_VOLUME_SCHEMA_VERSION = "v2.0.0"
DEFAULT_VIDEO_SCHEMA_VERSION = "v2.0.0"
HIDDEN_WORKSPACE_NAME = "[Do Not Modify] Instant Versions Storage"
PREVIEW_NAME_SUFFIX = " (Preview)"
PREVIEW_DESCRIPTION_SUFFIX = " (Preview Snapshot)"


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

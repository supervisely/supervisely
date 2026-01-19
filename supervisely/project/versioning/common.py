from supervisely.project.versioning.video_schema import VideoSnapshotSchema, _VIDEO_SCHEMAS
from supervisely.project.versioning.volume_schema import VolumeSnapshotSchema, _VOLUME_SCHEMAS

DEFAULT_IMAGE_SCHEMA_VERSION = "v1.0.0"
DEFAULT_VOLUME_SCHEMA_VERSION = "v2.0.0"
DEFAULT_VIDEO_SCHEMA_VERSION = "v2.0.0"


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

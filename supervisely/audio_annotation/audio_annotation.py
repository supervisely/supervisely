# coding: utf-8
"""Annotation of a single audio recording in a local project directory.

An audio annotation is a list of :class:`~supervisely.audio.audio_segment.AudioSegment`
and a list of :class:`~supervisely.audio.audio_recording_tag.AudioRecordingTag`,
plus the shape of the recording they refer to. There are no objects and no
figures: a label is a tag, either on a range of samples or on the whole
recording. On disk both kinds share the ``tags`` list; a recording tag is the
one with ``frameRange: null``, as on the platform.

The recording's shape is stored because the platform does not keep it -- an
audio entity's ``fileMeta`` is only ``{mime, size}`` -- and a sample range
cannot be turned into seconds without it.
"""

from __future__ import annotations

from typing import Any, Optional

from supervisely.audio.audio_recording_tag import AudioRecordingTag
from supervisely.audio.audio_segment import AudioSegment
from supervisely.io.json import load_json_file
from supervisely.project.project_meta import ProjectMeta

DESCRIPTION = "description"
TAGS = "tags"
SAMPLE_COUNT = "sampleCount"
SAMPLE_RATE = "sampleRate"
CHANNELS = "channels"


class AudioAnnotation:
    """Labels of one recording, as stored next to it on disk.

    :param sample_count: Samples per channel in the recording.
    :param sample_rate: Sample rate of the recording.
    :param channels: Number of channels.
    :param tags: Segment labels on the recording.
    :param description: Free-text description.
    :param recording_tags: Labels on the whole recording.

    :Usage example:

     .. code-block:: python

        import supervisely as sly

        ann = sly.AudioAnnotation(
            sample_count=32000,
            sample_rate=16000,
            channels=1,
            tags=[sly.AudioSegment(name="Event", start=0, end=15999)],
            recording_tags=[sly.AudioRecordingTag(name="Scene", value="indoor")],
        )
        ann.to_json()
    """

    def __init__(
        self,
        sample_count: Optional[int] = None,
        sample_rate: Optional[int] = None,
        channels: Optional[int] = None,
        tags: Optional[list[AudioSegment]] = None,
        description: str = "",
        recording_tags: Optional[list[AudioRecordingTag]] = None,
    ):
        self.sample_count = sample_count
        self.sample_rate = sample_rate
        self.channels = channels
        self.tags: list[AudioSegment] = list(tags) if tags is not None else []
        self.description = description
        self.recording_tags: list[AudioRecordingTag] = (
            list(recording_tags) if recording_tags is not None else []
        )

    @property
    def duration_seconds(self) -> Optional[float]:
        """Duration of the recording, when the shape is known."""
        if not self.sample_count or not self.sample_rate:
            return None
        return self.sample_count / float(self.sample_rate)

    def to_json(self, key_id_map=None) -> dict[str, Any]:
        """Serialize to the local project format.

        :param key_id_map: Accepted for signature compatibility with the other
            modalities; audio has no objects, so nothing is keyed.
        """
        return {
            DESCRIPTION: self.description,
            SAMPLE_COUNT: self.sample_count,
            SAMPLE_RATE: self.sample_rate,
            CHANNELS: self.channels,
            TAGS: [tag.to_json() for tag in self.recording_tags]
            + [tag.to_json() for tag in self.tags],
        }

    @classmethod
    def from_json(
        cls, data: dict[str, Any], project_meta: Optional[ProjectMeta] = None, key_id_map=None
    ) -> "AudioAnnotation":
        """Build from the local project format.

        :param project_meta: When given, every tag name is checked against it,
            the same guarantee the other modalities give.
        """
        tags, recording_tags = [], []
        for tag in data.get(TAGS, []):
            if AudioRecordingTag.is_recording_tag(tag):
                recording_tags.append(AudioRecordingTag.from_json(tag))
            else:
                tags.append(AudioSegment.from_json(tag))
        if project_meta is not None:
            for tag in recording_tags + tags:
                if tag.name is not None and project_meta.get_tag_meta(tag.name) is None:
                    raise RuntimeError(
                        f"Tag {tag.name!r} is not found in the project meta"
                    )
        return cls(
            sample_count=data.get(SAMPLE_COUNT),
            sample_rate=data.get(SAMPLE_RATE),
            channels=data.get(CHANNELS),
            tags=tags,
            description=data.get(DESCRIPTION, ""),
            recording_tags=recording_tags,
        )

    @classmethod
    def load_json_file(
        cls, path: str, project_meta: Optional[ProjectMeta] = None, key_id_map=None
    ) -> "AudioAnnotation":
        """Read an annotation json file."""
        return cls.from_json(load_json_file(path), project_meta, key_id_map)

    def clone(
        self,
        sample_count: Optional[int] = None,
        sample_rate: Optional[int] = None,
        channels: Optional[int] = None,
        tags: Optional[list[AudioSegment]] = None,
        description: Optional[str] = None,
        recording_tags: Optional[list[AudioRecordingTag]] = None,
    ) -> "AudioAnnotation":
        """Return a copy with the given fields replaced."""
        return AudioAnnotation(
            sample_count=sample_count if sample_count is not None else self.sample_count,
            sample_rate=sample_rate if sample_rate is not None else self.sample_rate,
            channels=channels if channels is not None else self.channels,
            tags=tags if tags is not None else self.tags,
            description=description if description is not None else self.description,
            recording_tags=(
                recording_tags if recording_tags is not None else self.recording_tags
            ),
        )

    def __repr__(self) -> str:
        return (
            f"AudioAnnotation(sample_count={self.sample_count}, "
            f"sample_rate={self.sample_rate}, channels={self.channels}, "
            f"tags={len(self.tags)}, recording_tags={len(self.recording_tags)})"
        )

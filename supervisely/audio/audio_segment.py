# coding: utf-8
"""An audio segment label: a tag applied to an inclusive range of samples.

The platform models an audio label as a tag assignment carrying a
``frameRange``. Despite the name inherited from video, the two numbers are
**inclusive, zero-based indices into the original source samples** -- not
frames and not milliseconds. They are 64-bit, so a long recording at a high
sample rate does not overflow.

Sample indices are exact; seconds are not. Conversions are provided for
convenience but the sample index is always the source of truth.
"""

from typing import Any, Dict, Optional

from supervisely.audio.spectrogram_settings import SpectrogramSettings


def samples_to_seconds(sample: int, sample_rate: int) -> float:
    """Convert a sample index to seconds.

    :param sample: Zero-based sample index.
    :param sample_rate: Sample rate of the *original* recording.
    """
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be > 0, got {sample_rate}")
    return sample / float(sample_rate)


def seconds_to_samples(seconds: float, sample_rate: int) -> int:
    """Convert seconds to a sample index, rounding to nearest.

    Rounding is to nearest rather than truncating so that a round trip through
    seconds does not systematically drift earlier.
    """
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be > 0, got {sample_rate}")
    if seconds < 0:
        raise ValueError(f"seconds must be >= 0, got {seconds}")
    return int(round(seconds * sample_rate))


class AudioSegment:
    """A labeled range of an audio recording.

    :param tag_id: Id of the tag meta being applied.
    :param start: First sample of the segment, inclusive.
    :param end: Last sample of the segment, **inclusive**.
    :param value: Tag value, for tags that carry one.
    :param channel: Zero-based channel this label refers to, or ``None`` for
        the mixdown.
    :param settings: Spectrogram settings active when the label was made, or
        ``None`` when it was made on the waveform alone. ``None`` is meaningful
        -- it records "labeled by ear", which is different provenance, not
        missing data.
    :param id: Server id of the tag assignment, when it came from the platform.
    :param entity_id: Id of the recording this label belongs to.
    :param labeler_login: Login of whoever created it.

    :Usage example:

     .. code-block:: python

        import supervisely as sly

        segment = sly.AudioSegment(tag_id=53295, start=16000, end=32000, channel=0)
        segment.duration_seconds(16000)  # 1.0000625
    """

    def __init__(
        self,
        tag_id: int,
        start: int,
        end: int,
        value: Any = None,
        channel: Optional[int] = None,
        settings: Optional[SpectrogramSettings] = None,
        id: Optional[int] = None,
        entity_id: Optional[int] = None,
        labeler_login: Optional[str] = None,
    ):
        if not isinstance(start, int) or isinstance(start, bool):
            raise ValueError(f"start must be an int, got {type(start).__name__}")
        if not isinstance(end, int) or isinstance(end, bool):
            raise ValueError(f"end must be an int, got {type(end).__name__}")
        if start < 0:
            raise ValueError(f"start must be >= 0, got {start}")
        if end < start:
            raise ValueError(f"end must be >= start, got start={start} end={end}")
        if channel is not None and channel < 0:
            raise ValueError(f"channel must be >= 0 or None, got {channel}")

        self.tag_id = tag_id
        self.start = start
        self.end = end
        self.value = value
        self.channel = channel
        self.settings = settings
        self.id = id
        self.entity_id = entity_id
        self.labeler_login = labeler_login

    @property
    def sample_count(self) -> int:
        """Number of samples covered, counting both endpoints."""
        return self.end - self.start + 1

    def duration_seconds(self, sample_rate: int) -> float:
        """Duration in seconds at the given sample rate."""
        return samples_to_seconds(self.sample_count, sample_rate)

    def start_seconds(self, sample_rate: int) -> float:
        """Start position in seconds."""
        return samples_to_seconds(self.start, sample_rate)

    def end_seconds(self, sample_rate: int) -> float:
        """End position in seconds, inclusive."""
        return samples_to_seconds(self.end, sample_rate)

    @classmethod
    def from_seconds(
        cls,
        tag_id: int,
        start_sec: float,
        end_sec: float,
        sample_rate: int,
        **kwargs,
    ) -> "AudioSegment":
        """Build a segment from a start/end expressed in seconds.

        ``end_sec`` is treated as exclusive, matching how people describe a
        span ("from 1.0 s to 2.0 s"), and converted to the platform's inclusive
        last sample.
        """
        start = seconds_to_samples(start_sec, sample_rate)
        end = seconds_to_samples(end_sec, sample_rate) - 1
        if end < start:
            end = start
        return cls(tag_id=tag_id, start=start, end=end, **kwargs)

    def overlaps(self, other: "AudioSegment") -> bool:
        """Whether two segments share at least one sample.

        Channel-aware: labels on different explicit channels never overlap.
        A mixdown label (``channel=None``) overlaps any channel.
        """
        if (
            self.channel is not None
            and other.channel is not None
            and self.channel != other.channel
        ):
            return False
        return self.start <= other.end and other.start <= self.end

    def to_meta_json(self) -> Dict[str, Any]:
        """Build the tag assignment's ``meta`` object.

        Returns an empty dict when there is nothing to record: the platform
        accepts a missing or empty ``meta``, but rejects a partial
        ``spectrogram`` object outright.
        """
        meta: Dict[str, Any] = {}
        if self.channel is not None:
            meta["channel"] = self.channel
        if self.settings is not None:
            # `channel` is duplicated inside the settings object -- the API
            # rejects the whole payload without it.
            meta["spectrogram"] = self.settings.clone(channel=self.channel).to_json()
        return meta

    def to_api_json(self, entity_id: Optional[int] = None) -> Dict[str, Any]:
        """Build the payload for one entry of ``entities.tags.bulk.add``."""
        payload: Dict[str, Any] = {
            "tagId": self.tag_id,
            "entityId": entity_id if entity_id is not None else self.entity_id,
            "frameRange": [self.start, self.end],
        }
        if self.value is not None:
            payload["value"] = self.value
        meta = self.to_meta_json()
        if meta:
            payload["meta"] = meta
        return payload

    @classmethod
    def from_api_json(cls, data: Dict[str, Any]) -> "AudioSegment":
        """Build from a tag assignment as returned by ``entities.list``.

        Reads ``frameRange`` when present and falls back to the
        ``startFrame``/``endFrame`` mirror.
        """
        rng = data.get("frameRange")
        if rng is None:
            start, end = data.get("startFrame"), data.get("endFrame")
            if start is None or end is None:
                raise ValueError(
                    f"tag assignment {data.get('id')} has no frameRange; "
                    "it is a recording-level tag, not a segment"
                )
        else:
            start, end = rng

        meta = data.get("meta") or {}
        spec = meta.get("spectrogram")
        settings = SpectrogramSettings.from_json(spec) if spec else None
        channel = meta.get("channel")
        if channel is None and spec is not None:
            channel = spec.get("channel")

        return cls(
            tag_id=data["tagId"],
            start=int(start),
            end=int(end),
            value=data.get("value"),
            channel=channel,
            settings=settings,
            id=data.get("id"),
            entity_id=data.get("entityId"),
            labeler_login=data.get("labelerLogin"),
        )

    def __repr__(self) -> str:
        return (
            f"AudioSegment(tag_id={self.tag_id}, start={self.start}, end={self.end}, "
            f"channel={self.channel}, settings={'yes' if self.settings else 'none'})"
        )

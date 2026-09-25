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

import numbers
from typing import Any, Dict, Optional


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

    :param tag_id: Id of the tag meta being applied. Not needed for a segment
        that only lives in a local project directory, where tags are addressed
        by ``name``.
    :param start: First sample of the segment, inclusive.
    :param end: Last sample of the segment, **inclusive**.
    :param value: Tag value, for tags that carry one.
    :param channel: Zero-based channel this label refers to, or ``None`` for
        the mixdown.
    :param meta: Any other keys to keep in the tag assignment's ``meta``. The
        platform treats ``meta`` as a free-form options object, so keys written
        by another client are preserved rather than dropped on read and write.
    :param id: Server id of the tag assignment, when it came from the platform.
    :param entity_id: Id of the recording this label belongs to.
    :param labeler_login: Login of whoever created it.
    :param name: Name of the tag meta being applied. Local project directories
        store the name; the API works with ``tag_id``.
    :param custom_data: The tag assignment's ``customData``, an object any
        client may attach. Kept as-is so a round trip does not lose it.

    :Usage example:

     .. code-block:: python

        import supervisely as sly

        segment = sly.AudioSegment(tag_id=53295, start=16000, end=32000, channel=0)
        segment.duration_seconds(16000)  # 1.0000625
    """

    def __init__(
        self,
        tag_id: Optional[int] = None,
        start: int = 0,
        end: int = 0,
        value: Any = None,
        channel: Optional[int] = None,
        meta: Optional[Dict[str, Any]] = None,
        id: Optional[int] = None,
        entity_id: Optional[int] = None,
        labeler_login: Optional[str] = None,
        name: Optional[str] = None,
        custom_data: Optional[Dict[str, Any]] = None,
    ):
        # numbers.Integral, not int: sample indices routinely arrive as numpy
        # integers from np.argmax / np.flatnonzero. bool is an Integral too.
        if not isinstance(start, numbers.Integral) or isinstance(start, bool):
            raise ValueError(f"start must be an integer, got {type(start).__name__}")
        if not isinstance(end, numbers.Integral) or isinstance(end, bool):
            raise ValueError(f"end must be an integer, got {type(end).__name__}")
        start, end = int(start), int(end)
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
        self.meta = dict(meta) if meta else {}
        self.id = id
        self.entity_id = entity_id
        self.labeler_login = labeler_login
        self.name = name
        self.custom_data = dict(custom_data) if custom_data else {}

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

    def _meta_json(self) -> Dict[str, Any]:
        """Build the tag assignment's ``meta`` object -- the platform's
        ``meta`` on a tag, unrelated to a project's ``meta.json``.

        The only key the platform validates for audio is ``channel``, the
        channel the *label* is about. The spectrogram is not recorded here:
        it is project configuration, the same for every label in the project.

        Returns an empty dict when there is nothing to record; the platform
        accepts a missing or empty ``meta``.
        """
        meta: Dict[str, Any] = dict(self.meta)
        if self.channel is not None:
            meta["channel"] = self.channel
        return meta

    def _to_api_json(self, entity_id: Optional[int] = None) -> Dict[str, Any]:
        """Build the payload for one entry of ``entities.tags.bulk.add``."""
        payload: Dict[str, Any] = {
            "tagId": self.tag_id,
            "entityId": entity_id if entity_id is not None else self.entity_id,
            "frameRange": [self.start, self.end],
        }
        if self.value is not None:
            payload["value"] = self.value
        meta = self._meta_json()
        if meta:
            payload["meta"] = meta
        if self.custom_data:
            payload["customData"] = dict(self.custom_data)
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

        # A label with no `meta.channel` is a mixdown label. Everything else in
        # `meta` is kept as-is: it is a free-form options object and may carry
        # keys this SDK does not know about, including the `spectrogram` object
        # older labels were written with, before the settings moved onto the
        # project.
        meta = dict(data.get("meta") or {})
        channel = meta.pop("channel", None)

        return cls(
            tag_id=data["tagId"],
            start=int(start),
            end=int(end),
            value=data.get("value"),
            channel=channel,
            meta=meta,
            id=data.get("id"),
            entity_id=data.get("entityId"),
            labeler_login=data.get("labelerLogin"),
            custom_data=data.get("customData"),
        )

    def to_json(self) -> Dict[str, Any]:
        """Serialize for a local project directory.

        Unlike :meth:`_to_api_json`, this identifies the tag by ``name``: a
        downloaded project is readable without the server that issued the ids.
        """
        data: Dict[str, Any] = {
            "name": self.name,
            "frameRange": [self.start, self.end],
            "channel": self.channel,
        }
        if self.value is not None:
            data["value"] = self.value
        if self.meta:
            data["meta"] = dict(self.meta)
        if self.custom_data:
            data["customData"] = dict(self.custom_data)
        if self.tag_id is not None:
            data["tagId"] = self.tag_id
        if self.id is not None:
            data["id"] = self.id
        if self.labeler_login is not None:
            data["labelerLogin"] = self.labeler_login
        return data

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "AudioSegment":
        """Build from the local project format written by :meth:`to_json`."""
        start, end = data["frameRange"]
        return cls(
            tag_id=data.get("tagId"),
            start=int(start),
            end=int(end),
            value=data.get("value"),
            channel=data.get("channel"),
            meta=data.get("meta"),
            id=data.get("id"),
            labeler_login=data.get("labelerLogin"),
            name=data.get("name"),
            custom_data=data.get("customData"),
        )

    def __repr__(self) -> str:
        return (
            f"AudioSegment(name={self.name!r}, tag_id={self.tag_id}, "
            f"start={self.start}, end={self.end}, "
            f"channel={self.channel})"
        )

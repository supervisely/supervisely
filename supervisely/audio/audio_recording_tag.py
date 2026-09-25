# coding: utf-8
"""A recording-level label: a tag applied to a whole audio recording.

The labeling tool has two kinds of audio label. A segment
(:class:`~supervisely.audio.audio_segment.AudioSegment`) covers a range of
samples. A recording tag covers the whole file: it is the same tag assignment
with **no** ``frameRange``, created when a label is applied with the scope
"Entire recording". It has no channel -- the tool always writes
``meta.channel: null`` for it.
"""

from typing import Any, Dict, Optional


class AudioRecordingTag:
    """A label on a whole audio recording.

    :param tag_id: Id of the tag meta being applied. Not needed for a tag that
        only lives in a local project directory, where tags are addressed by
        ``name``.
    :param value: Tag value, for tags that carry one.
    :param meta: Any keys to keep in the tag assignment's ``meta``, preserved
        on read and write. ``channel`` is not one of them: a recording tag is
        about every channel.
    :param id: Server id of the tag assignment, when it came from the platform.
    :param entity_id: Id of the recording this label belongs to.
    :param labeler_login: Login of whoever created it.
    :param name: Name of the tag meta being applied. Local project directories
        store the name; the API works with ``tag_id``.
    :param custom_data: The tag assignment's ``customData``, kept as-is.

    :Usage example:

     .. code-block:: python

        import supervisely as sly

        tag = sly.AudioRecordingTag(tag_id=53295, value="indoor")
        api.audio.add_recording_tag(project_id, entity_id, tag)
    """

    def __init__(
        self,
        tag_id: Optional[int] = None,
        value: Any = None,
        meta: Optional[Dict[str, Any]] = None,
        id: Optional[int] = None,
        entity_id: Optional[int] = None,
        labeler_login: Optional[str] = None,
        name: Optional[str] = None,
        custom_data: Optional[Dict[str, Any]] = None,
    ):
        self.tag_id = tag_id
        self.value = value
        self.meta = dict(meta) if meta else {}
        self.id = id
        self.entity_id = entity_id
        self.labeler_login = labeler_login
        self.name = name
        self.custom_data = dict(custom_data) if custom_data else {}

    @staticmethod
    def is_recording_tag(data: Dict[str, Any]) -> bool:
        """Whether a tag assignment, from the API or a local annotation file,
        labels the whole recording rather than a range.

        The API leaves ``frameRange`` out for such a tag; the local format
        writes it as ``null``.
        """
        return data.get("frameRange") is None and data.get("startFrame") is None

    def _to_api_json(self, entity_id: Optional[int] = None) -> Dict[str, Any]:
        """Build the payload for one entry of ``entities.tags.bulk.add``.

        ``frameRange`` is sent as ``null`` and ``meta.channel`` as ``null``,
        exactly what the labeling tool writes for "Entire recording".
        """
        payload: Dict[str, Any] = {
            "tagId": self.tag_id,
            "entityId": entity_id if entity_id is not None else self.entity_id,
            "frameRange": None,
            "meta": {**self.meta, "channel": None},
        }
        if self.value is not None:
            payload["value"] = self.value
        if self.custom_data:
            payload["customData"] = dict(self.custom_data)
        return payload

    @classmethod
    def from_api_json(cls, data: Dict[str, Any]) -> "AudioRecordingTag":
        """Build from a tag assignment as returned by ``entities.list``."""
        if not cls.is_recording_tag(data):
            raise ValueError(
                f"tag assignment {data.get('id')} has a frameRange; "
                "it is a segment, read it with AudioSegment"
            )
        meta = dict(data.get("meta") or {})
        meta.pop("channel", None)
        return cls(
            tag_id=data["tagId"],
            value=data.get("value"),
            meta=meta,
            id=data.get("id"),
            entity_id=data.get("entityId"),
            labeler_login=data.get("labelerLogin"),
            custom_data=data.get("customData"),
        )

    def to_json(self) -> Dict[str, Any]:
        """Serialize for a local project directory.

        Written into the annotation's ``tags`` list next to the segments, with
        ``frameRange: null`` -- the same shape the platform uses.
        """
        data: Dict[str, Any] = {"name": self.name, "frameRange": None}
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
    def from_json(cls, data: Dict[str, Any]) -> "AudioRecordingTag":
        """Build from the local project format written by :meth:`to_json`."""
        if data.get("frameRange") is not None:
            raise ValueError("a recording tag has no frameRange; this is a segment")
        return cls(
            tag_id=data.get("tagId"),
            value=data.get("value"),
            meta=data.get("meta"),
            id=data.get("id"),
            labeler_login=data.get("labelerLogin"),
            name=data.get("name"),
            custom_data=data.get("customData"),
        )

    def __repr__(self) -> str:
        return (
            f"AudioRecordingTag(name={self.name!r}, tag_id={self.tag_id}, "
            f"value={self.value!r})"
        )

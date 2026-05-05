import pytest

from supervisely.annotation.tag import Tag
from supervisely.annotation.tag_meta import TagMeta, TagValueType
from supervisely.annotation.tag_meta_collection import TagMetaCollection


def test_date_tag_meta_creation():
    tag_meta = TagMeta("created_at", TagValueType.DATE)

    assert tag_meta.name == "created_at"
    assert tag_meta.value_type == "date"


def test_date_tag_meta_json_round_trip():
    tag_meta = TagMeta("created_at", TagValueType.DATE, color=[10, 20, 30])

    tag_meta_json = tag_meta.to_json()
    restored_tag_meta = TagMeta.from_json(tag_meta_json)

    assert tag_meta_json["value_type"] == "date"
    assert restored_tag_meta == tag_meta


def test_date_tag_json_round_trip():
    tag_meta = TagMeta("created_at", TagValueType.DATE)
    tag_metas = TagMetaCollection([tag_meta])
    tag = Tag(tag_meta, "2026-04-23T15:15:48")

    tag_json = tag.to_json()
    restored_tag = Tag.from_json(tag_json, tag_metas)

    assert tag_json == {"name": "created_at", "value": "2026-04-23T15:15:48"}
    assert restored_tag == tag


def test_date_tag_accepts_server_iso_datetime_with_milliseconds():
    tag_meta = TagMeta("date_taken", TagValueType.DATE)
    tag_metas = TagMetaCollection([tag_meta])
    tag = Tag(tag_meta, "2026-05-12T21:14:12.000Z")

    tag_json = tag.to_json()
    restored_tag = Tag.from_json(tag_json, tag_metas)

    assert tag_json == {"name": "date_taken", "value": "2026-05-12T21:14:12.000Z"}
    assert restored_tag == tag


def test_date_tag_accepts_utc_iso_datetime_without_milliseconds():
    tag_meta = TagMeta("date_taken", TagValueType.DATE)
    tag = Tag(tag_meta, "2026-05-12T21:14:12Z")

    assert tag.to_json() == {"name": "date_taken", "value": "2026-05-12T21:14:12Z"}


def test_date_tag_accepts_datetime_with_space_separator():
    tag_meta = TagMeta("date_taken", TagValueType.DATE)
    tag = Tag(tag_meta, "2026-04-27 11:00:46")

    assert tag.to_json() == {"name": "date_taken", "value": "2026-04-27 11:00:46"}


def test_date_tag_accepts_iso_datetime_with_timezone_offset():
    tag_meta = TagMeta("date_taken", TagValueType.DATE)
    tag = Tag(tag_meta, "2026-05-12T21:14:12+00:00")

    assert tag.to_json() == {"name": "date_taken", "value": "2026-05-12T21:14:12+00:00"}


@pytest.mark.parametrize(
    "invalid_value",
    [
        None,
        20260423151548,
        "not a date",
        "2026-02-30T10:00:00",
    ],
)
def test_date_tag_rejects_values_that_are_not_iso_datetimes(invalid_value):
    tag_meta = TagMeta("created_at", TagValueType.DATE)

    assert tag_meta.is_valid_value(invalid_value) is False

    with pytest.raises(ValueError):
        Tag(tag_meta, invalid_value)


def test_date_tag_rejects_possible_values():
    with pytest.raises(ValueError):
        TagMeta("created_at", TagValueType.DATE, possible_values=["2026-04-23T15:15:48"])

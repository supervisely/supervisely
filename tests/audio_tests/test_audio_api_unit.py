# coding: utf-8
"""AudioApi without a server: info conversion and segment filtering."""

from unittest.mock import MagicMock

import pytest

from supervisely.api.audio_api import AudioApi, AudioInfo


@pytest.fixture
def audio_api():
    return AudioApi(MagicMock())


def test_info_is_a_named_tuple_like_the_other_modalities(audio_api):
    raw = {
        "id": 5,
        "name": "rain.wav",
        "hash": "h",
        "link": None,
        "datasetId": 9,
        "projectId": 3,
        "workspaceId": 1,
        "createdAt": "t",
        "updatedAt": "t",
        "createdBy": 7,
        "meta": {},
        "fileMeta": {"mime": "audio/wav", "size": 10},
        "size": 10,
        "objectsCount": 0,
        "pathOriginal": "/p",
        "fullStorageUrl": "u",
        "tags": [],
    }
    info = audio_api._convert_json_info(raw)
    assert isinstance(info, AudioInfo)
    assert (info.id, info.name, info.created_by_id) == (5, "rain.wav", 7)
    assert info.file_meta["mime"] == "audio/wav"


def test_bulk_add_projection_names_the_field_title(audio_api):
    """`entities.bulk.add` answers with `title`, the list endpoints with `name`."""
    info = audio_api._convert_json_info({"id": 5, "title": "rain.wav"})
    assert info.name == "rain.wav"
    assert info.tags is None  # not part of the default projection


def test_get_segments_skips_recording_level_tags(audio_api, monkeypatch):
    info = audio_api._convert_json_info(
        {
            "id": 5,
            "name": "rain.wav",
            "tags": [
                {"id": 1, "tagId": 10, "value": "Speech"},  # whole recording
                {"id": 2, "tagId": 11, "frameRange": [0, 99], "meta": {"channel": 1}},
            ],
        }
    )
    monkeypatch.setattr(audio_api, "get_info_by_id", lambda _id: info)

    segments = audio_api.get_segments(5)
    assert len(segments) == 1
    assert (segments[0].start, segments[0].end, segments[0].channel) == (0, 99, 1)


def test_get_segments_reports_a_missing_recording(audio_api, monkeypatch):
    monkeypatch.setattr(audio_api, "get_info_by_id", lambda _id: None)
    with pytest.raises(KeyError):
        audio_api.get_segments(404)


def test_upload_paths_checks_its_arguments(audio_api):
    with pytest.raises(ValueError):
        audio_api.upload_paths(1, ["a.wav", "b.wav"], ["/tmp/a.wav"])
    with pytest.raises(ValueError):
        audio_api.download_paths([1, 2], ["/tmp/a.wav"])

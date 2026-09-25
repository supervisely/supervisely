# coding: utf-8
"""AudioApi without a server: info conversion and segment filtering."""

from unittest.mock import MagicMock

import pytest

from supervisely.api.audio_api import AudioApi, AudioInfo
from supervisely.audio.spectrogram_settings import PROJECT_FIELDS, SpectrogramSettings


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


def test_project_settings_are_read_with_platform_defaults(audio_api):
    """A project that has never been configured has no `spectrogram` key, and
    the platform renders it with the defaults -- so do we, rather than
    answering None."""
    audio_api._api.project.get_settings.return_value = {"labelingInterface": "default"}
    settings = audio_api.get_spectrogram_settings(31)
    assert settings == SpectrogramSettings()


def test_project_settings_are_read_back(audio_api):
    audio_api._api.project.get_settings.return_value = {
        "spectrogram": {"scale": "mel", "fftSize": 1024, "melBands": 64}
    }
    settings = audio_api.get_spectrogram_settings(31)
    assert (settings.scale, settings.fft_size, settings.mel_bands) == ("mel", 1024, 64)
    assert settings.hop_length == 512  # a field the project never set


def test_settings_are_written_under_the_spectrogram_key(audio_api):
    settings = SpectrogramSettings(scale="mel", fft_size=1024)
    audio_api.set_spectrogram_settings(31, settings)
    audio_api._api.project.update_settings.assert_called_once_with(
        31, {"spectrogram": settings.to_json()}, merge_with_current=True
    )
    # merge_with_current, so writing the spectrogram does not blank the rest of
    # the project's settings.
    written = audio_api._api.project.update_settings.call_args[0][1]["spectrogram"]
    assert set(written) == set(PROJECT_FIELDS)


def test_settings_must_be_a_settings_object(audio_api):
    """A raw dict would skip validation and reach the API unchecked."""
    with pytest.raises(TypeError):
        audio_api.set_spectrogram_settings(31, {"scale": "mel"})
    audio_api._api.project.update_settings.assert_not_called()


def test_remove_segment_sends_all_three_ids(audio_api):
    """`image-tags.remove-from-image` answers 400 unless it gets the assignment
    id, the tag meta id and the recording id together."""
    from supervisely.audio.audio_segment import AudioSegment

    segment = AudioSegment(tag_id=53636, start=0, end=99, id=2122848, entity_id=7133008)
    audio_api.remove_segment(segment)
    audio_api._api.post.assert_called_once_with(
        "image-tags.remove-from-image", {"id": 2122848, "tagId": 53636, "imageId": 7133008}
    )


def test_remove_segment_refuses_a_segment_that_never_reached_the_server(audio_api):
    from supervisely.audio.audio_segment import AudioSegment

    with pytest.raises(ValueError, match="id, entity_id"):
        audio_api.remove_segment(AudioSegment(tag_id=1, start=0, end=9))
    audio_api._api.post.assert_not_called()


def test_get_tags_splits_segments_from_recording_tags(audio_api, monkeypatch):
    """What `entities.info` answers on dev: a recording tag has no `frameRange`
    key at all, and `meta.channel: null` as the labeling tool writes it."""
    info = audio_api._convert_json_info(
        {
            "id": 5,
            "name": "rain.wav",
            "tags": [
                {
                    "id": 1,
                    "tagId": 10,
                    "entityId": 5,
                    "value": "calm",
                    "meta": {"channel": None},
                    "customData": {"k": 1},
                    "labelerLogin": "anna",
                },
                {"id": 2, "tagId": 11, "frameRange": [0, 99], "meta": {"channel": 1}},
            ],
        }
    )
    monkeypatch.setattr(audio_api, "get_info_by_id", lambda _id: info)

    segments, recording_tags = audio_api.get_tags(5)
    assert [(s.start, s.end) for s in segments] == [(0, 99)]
    (tag,) = recording_tags
    assert (tag.id, tag.tag_id, tag.entity_id, tag.value) == (1, 10, 5, "calm")
    assert tag.meta == {}  # channel is not a recording tag's
    assert tag.custom_data == {"k": 1}
    assert tag.labeler_login == "anna"
    assert audio_api.get_recording_tags(5)[0].id == 1


def test_recording_tag_payload_is_what_the_tool_writes():
    from supervisely.audio.audio_recording_tag import AudioRecordingTag

    tag = AudioRecordingTag(tag_id=10, value="indoor", meta={"note": "x"}, custom_data={"k": 1})
    assert tag._to_api_json(5) == {
        "tagId": 10,
        "entityId": 5,
        "frameRange": None,
        "meta": {"note": "x", "channel": None},
        "value": "indoor",
        "customData": {"k": 1},
    }


def test_add_tags_sends_segments_and_recording_tags_in_one_request(audio_api):
    from supervisely.audio.audio_recording_tag import AudioRecordingTag
    from supervisely.audio.audio_segment import AudioSegment

    audio_api.add_tags(
        3,
        5,
        segments=[AudioSegment(tag_id=11, start=0, end=9)],
        recording_tags=[AudioRecordingTag(tag_id=10)],
    )
    ((endpoint, body), _), = audio_api._api.post.call_args_list
    assert endpoint == "entities.tags.bulk.add"
    assert body["projectId"] == 3
    assert [t.get("frameRange") for t in body["tags"]] == [None, [0, 9]]


def test_add_tags_with_nothing_makes_no_request(audio_api):
    assert audio_api.add_tags(3, 5) == []
    audio_api._api.post.assert_not_called()


def test_update_segment_sends_the_whole_meta(audio_api):
    """`image-tags.update-tag-value` replaces `meta`, so clearing a channel
    has to be sent as an explicit null."""
    from supervisely.audio.audio_segment import AudioSegment

    segment = AudioSegment(tag_id=11, start=20, end=300, id=7, meta={"x": 1}, value=0.5)
    audio_api.update_segment(segment)
    audio_api._api.post.assert_called_once_with(
        "image-tags.update-tag-value",
        {"id": 7, "value": 0.5, "frameRange": [20, 300], "meta": {"x": 1, "channel": None}},
    )


def test_update_recording_tag_never_sends_a_range(audio_api):
    from supervisely.audio.audio_recording_tag import AudioRecordingTag

    audio_api.update_recording_tag(AudioRecordingTag(tag_id=10, id=8, value="tense"))
    audio_api._api.post.assert_called_once_with(
        "image-tags.update-tag-value", {"id": 8, "value": "tense", "meta": {"channel": None}}
    )


def test_update_refuses_labels_that_never_reached_the_server(audio_api):
    from supervisely.audio.audio_recording_tag import AudioRecordingTag
    from supervisely.audio.audio_segment import AudioSegment

    with pytest.raises(ValueError):
        audio_api.update_segment(AudioSegment(tag_id=1, start=0, end=9))
    with pytest.raises(ValueError):
        audio_api.update_recording_tag(AudioRecordingTag(tag_id=1))
    audio_api._api.post.assert_not_called()


def test_remove_recording_tag_sends_all_three_ids(audio_api):
    from supervisely.audio.audio_recording_tag import AudioRecordingTag

    audio_api.remove_recording_tag(AudioRecordingTag(tag_id=10, id=8, entity_id=5))
    audio_api._api.post.assert_called_once_with(
        "image-tags.remove-from-image", {"id": 8, "tagId": 10, "imageId": 5}
    )
    with pytest.raises(ValueError, match="get_recording_tags"):
        audio_api.remove_recording_tag(AudioRecordingTag(tag_id=10))


def test_segment_custom_data_survives_the_api_shape():
    from supervisely.audio.audio_segment import AudioSegment

    segment = AudioSegment.from_api_json(
        {"id": 2, "tagId": 11, "frameRange": [0, 99], "customData": {"k": 1}}
    )
    assert segment.custom_data == {"k": 1}
    assert segment._to_api_json(5)["customData"] == {"k": 1}
    assert AudioSegment.from_json(segment.to_json()).custom_data == {"k": 1}

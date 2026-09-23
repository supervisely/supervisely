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

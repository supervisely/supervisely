# coding: utf-8
"""Live round-trip against a real instance, driven through the SDK.

Skipped unless SERVER_ADDRESS and API_TOKEN are set. Creates a throwaway audio
project and removes it afterwards, so it leaves nothing behind.

Run: pytest tests/audio_tests/test_audio_api_live.py -q -s
"""

import math
import os
import struct
import wave

import pytest

import supervisely as sly
from supervisely.audio.audio_segment import AudioSegment
from supervisely.audio.spectrogram_settings import SpectrogramSettings

pytestmark = pytest.mark.skipif(
    not (os.environ.get("SERVER_ADDRESS") and os.environ.get("API_TOKEN")),
    reason="needs SERVER_ADDRESS and API_TOKEN",
)

SR = 16000


def _write_wav(path, seconds=2.0, channels=2):
    n = int(SR * seconds)
    frames = bytearray()
    for i in range(n):
        for ch in range(channels):
            frames += struct.pack(
                "<h", int(18000 * math.sin(2 * math.pi * 440 * (ch + 1) * i / SR))
            )
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(bytes(frames))
    return str(path)


@pytest.fixture(scope="module")
def api():
    return sly.Api.from_env()


@pytest.fixture(scope="module")
def project(api):
    workspace_id = int(os.environ.get("SLY_WORKSPACE_ID", "1"))
    info = api.project.create(
        workspace_id,
        "sdk-audio-roundtrip",
        type=sly.ProjectType.AUDIO,
        change_name_if_conflict=True,
    )
    yield info
    api.project.remove(info.id)


def test_round_trip(api, project, tmp_path):
    dataset = api.dataset.create(project.id, "roundtrip")

    # --- upload -----------------------------------------------------------
    path = _write_wav(tmp_path / "rt.wav")
    uploaded = api.audio.upload_path(dataset.id, "rt.wav", path)
    entity_id = uploaded.id
    assert entity_id
    assert uploaded.name == "rt.wav"

    # --- the platform stores no audio metadata; decode locally -------------
    info = sly.audio.get_audio_info(path)
    assert info.sample_rate == SR
    assert info.channels == 2
    assert info.sample_count == int(SR * 2.0)

    # --- define a tag and write segments ----------------------------------
    meta = sly.ProjectMeta(
        tag_metas=sly.TagMetaCollection([sly.TagMeta("Event", sly.TagValueType.NONE)])
    )
    api.project.update_meta(project.id, meta.to_json())
    tag_id = next(
        t["id"] for t in api.project.get_meta(project.id)["tags"] if t["name"] == "Event"
    )

    written = [
        AudioSegment(tag_id=tag_id, start=0, end=15999, channel=None),
        AudioSegment(tag_id=tag_id, start=16000, end=24000, channel=1),
        AudioSegment(tag_id=tag_id, start=24001, end=31999, meta={"reviewedBy": "anna"}),
    ]
    api.audio.add_segments(project.id, entity_id, written)

    # --- read back --------------------------------------------------------
    got = sorted(api.audio.get_segments(entity_id), key=lambda s: s.start)
    assert len(got) == 3

    for expected, actual in zip(written, got):
        assert (actual.start, actual.end) == (expected.start, expected.end)
        assert actual.channel == expected.channel
        assert actual.meta == expected.meta, "free-form meta keys must survive"

    # --- download and render under the project's settings -----------------
    local = str(tmp_path / "downloaded.wav")
    api.audio.download_path(entity_id, local)
    assert os.path.getsize(local) == os.path.getsize(path)

    samples, rate = sly.audio.read_audio(local)
    settings = api.audio.get_spectrogram_settings(project.id)
    segment = got[0]
    spec = sly.audio.render_segment(
        samples, rate, segment.start, segment.end, settings, channel=segment.channel
    )
    assert spec.shape[0] > 0


def test_project_spectrogram_round_trip(api, project):
    """Settings are configured once for the project and read back by everyone
    labeling in it."""
    settings = SpectrogramSettings(scale="mel", fft_size=1024, hop_length=256, mel_bands=64)
    api.audio.set_spectrogram_settings(project.id, settings)

    stored = api.project.get_settings(project.id).get("spectrogram")
    if stored is None:
        pytest.skip(
            "instance predates projects.settings.spectrogram (platform MR !2098); "
            "unknown settings keys are dropped silently"
        )
    assert stored == settings.to_json()

    read_back = api.audio.get_spectrogram_settings(project.id)
    assert read_back == settings
    assert read_back.fingerprint == settings.fingerprint

    # Writing the spectrogram must not blank the rest of the project settings.
    assert api.project.get_settings(project.id).get("labelingInterface") is not None


def test_settings_outside_the_api_schema_are_refused_locally(api, project):
    """Rejected before the request, with a specific message, rather than as a
    400 from the API's Joi schema."""
    for kwargs in ({"mel_bands": 1}, {"mel_bands": 513}, {"fft_size": 65536}):
        with pytest.raises(ValueError):
            SpectrogramSettings(**kwargs)

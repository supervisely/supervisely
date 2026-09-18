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
        AudioSegment(
            tag_id=tag_id, start=0, end=15999, channel=None,
            settings=SpectrogramSettings(scale="mel", fft_size=1024, hop_length=256),
        ),
        AudioSegment(
            tag_id=tag_id, start=16000, end=24000, channel=1,
            settings=SpectrogramSettings(scale="log"),
        ),
        AudioSegment(tag_id=tag_id, start=24001, end=31999),  # labeled by ear
    ]
    api.audio.add_segments(project.id, entity_id, written)

    # --- read back --------------------------------------------------------
    got = sorted(api.audio.get_segments(entity_id), key=lambda s: s.start)
    assert len(got) == 3

    for expected, actual in zip(written, got):
        assert (actual.start, actual.end) == (expected.start, expected.end)
        assert actual.channel == expected.channel
        if expected.settings is None:
            assert actual.settings is None, "labeled-by-ear provenance must survive"
        else:
            # The settings' own `channel` is the viewed channel and is written
            # as-is; the segment's channel is a separate field.
            assert actual.settings.fingerprint == expected.settings.fingerprint
            assert actual.settings.to_json() == expected.settings.to_json()

    # --- download and render under the recorded settings ------------------
    local = str(tmp_path / "downloaded.wav")
    api.audio.download_path(entity_id, local)
    assert os.path.getsize(local) == os.path.getsize(path)

    samples, rate = sly.audio.read_audio(local)
    segment = got[0]
    spec = sly.audio.render_segment(samples, rate, segment.start, segment.end, segment.settings)
    assert spec.shape[0] == segment.settings.mel_bands


def test_invalid_settings_never_reach_the_api(api, project):
    """The SDK refuses values the API would accept but the toolbox cannot open."""
    with pytest.raises(ValueError):
        SpectrogramSettings(mel_bands=1)
    with pytest.raises(ValueError):
        SpectrogramSettings(fft_size=65536)

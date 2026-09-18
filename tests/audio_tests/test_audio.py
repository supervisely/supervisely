# coding: utf-8
"""Unit tests for the audio modality. No network access required.

Run: pytest tests/audio_tests/test_audio.py
"""

import io
import math
import struct
import wave

import numpy as np
import pytest

from supervisely.audio.audio_io import get_audio_info, read_audio, select_channel
from supervisely.audio.audio_segment import (
    AudioSegment,
    samples_to_seconds,
    seconds_to_samples,
)
from supervisely.audio.spectrogram import (
    COLOR_STOPS,
    hz_to_mel,
    mel_to_hz,
    render_segment,
    render_spectrogram,
    stft_power,
    to_image,
    window_function,
)
from supervisely.audio.spectrogram_settings import SpectrogramSettings
from supervisely.project.project_type import ProjectType

SR = 16000


def write_wav(path, seconds=1.0, channels=1, freq=440.0, sample_rate=SR):
    n = int(sample_rate * seconds)
    frames = bytearray()
    for i in range(n):
        for ch in range(channels):
            value = int(18000 * math.sin(2 * math.pi * freq * (ch + 1) * i / sample_rate))
            frames += struct.pack("<h", value)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(bytes(frames))
    return path


# --------------------------------------------------------------- project type


def test_audio_is_a_project_type():
    assert ProjectType.AUDIO.value == "audio"
    assert "audio" in [t.value for t in ProjectType]


# ---------------------------------------------------------- settings: schema


def test_settings_round_trip_through_json():
    settings = SpectrogramSettings(scale="mel", fft_size=1024, hop_length=256, channel=1)
    restored = SpectrogramSettings.from_json(settings.to_json())
    assert restored == settings


def test_colormap_excluded_by_default_to_match_the_toolbox():
    settings = SpectrogramSettings(colormap="viridis")
    assert "colormap" not in settings.to_json()
    assert settings.to_json(include_colormap=True)["colormap"] == "viridis"


def test_channel_is_emitted_inside_the_settings_object():
    # The API rejects the whole payload when the inner `channel` key is missing,
    # though null is a valid value for it.
    assert "channel" in SpectrogramSettings(channel=2).to_json()
    segment = AudioSegment(
        tag_id=1, start=0, end=10, channel=2, settings=SpectrogramSettings(channel=2)
    )
    meta = segment._meta_json()
    assert meta["channel"] == 2
    assert meta["spectrogram"]["channel"] == 2


def test_labeled_channel_and_viewed_channel_are_independent():
    """The tool writes `meta.channel` (what the label is about) and
    `meta.spectrogram.channel` (what was on screen) separately."""
    segment = AudioSegment(
        tag_id=1, start=0, end=10, channel=None, settings=SpectrogramSettings(channel=1)
    )
    meta = segment._meta_json()
    assert meta["channel"] is None
    assert meta["spectrogram"]["channel"] == 1

    restored = AudioSegment.from_api_json(
        {"tagId": 1, "frameRange": [0, 10], "meta": meta}
    )
    assert restored.channel is None
    assert restored.settings.channel == 1


def test_from_json_tolerates_missing_optional_keys():
    settings = SpectrogramSettings.from_json({"scale": "mel"})
    assert settings.scale == "mel"
    assert settings.fft_size == 2048  # platform default


# ------------------------------------------------------ settings: validation


@pytest.mark.parametrize(
    "kwargs",
    [
        {"fft_size": 1000},      # not a power of two
        {"fft_size": 16},        # below the toolbox minimum
        {"fft_size": 65536},     # above the toolbox maximum
        {"mel_bands": 1},        # below the toolbox minimum
        {"mel_bands": 513},      # above the toolbox maximum
        {"hop_length": 0},
        {"min_db": 0.0, "max_db": -100.0},
        {"scale": "bark"},
        {"window": "kaiser"},
        {"colormap": "jet"},
        {"interpolation": "bilinear"},
        {"channel": -1},
    ],
)
def test_invalid_settings_are_rejected(kwargs):
    with pytest.raises(ValueError):
        SpectrogramSettings(**kwargs)


def test_bounds_the_api_accepts_but_the_toolbox_rejects():
    """The API accepts melBands=1 and fftSize=65536; the toolbox then refuses
    to open the recording. The SDK must not be able to author that."""
    for kwargs in ({"mel_bands": 1}, {"mel_bands": 100000}, {"fft_size": 65536}):
        with pytest.raises(ValueError):
            SpectrogramSettings(**kwargs)


# --------------------------------------------------------------- fingerprint


def test_fingerprint_ignores_cosmetic_changes():
    base = SpectrogramSettings()
    assert base.fingerprint == base.clone(colormap="viridis").fingerprint
    assert base.fingerprint == base.clone(interpolation="smooth").fingerprint


@pytest.mark.parametrize(
    "override",
    [
        {"scale": "mel"},
        {"fft_size": 1024},
        {"hop_length": 256},
        {"window": "hamming"},
        {"mel_bands": 64},
        {"min_db": -80.0},
        {"max_db": -1.0},
        {"channel": 0},
    ],
)
def test_fingerprint_changes_when_the_numbers_change(override):
    base = SpectrogramSettings()
    assert base.fingerprint != base.clone(**override).fingerprint


def test_fingerprint_is_stable_across_key_order():
    a = SpectrogramSettings(scale="mel", fft_size=1024)
    b = SpectrogramSettings(fft_size=1024, scale="mel")
    assert a.fingerprint == b.fingerprint


# ------------------------------------------------------------------ segments


def test_sample_range_is_inclusive():
    segment = AudioSegment(tag_id=1, start=100, end=100)
    assert segment.sample_count == 1
    assert AudioSegment(tag_id=1, start=0, end=15999).sample_count == 16000


def test_segment_rejects_reversed_or_negative_ranges():
    with pytest.raises(ValueError):
        AudioSegment(tag_id=1, start=10, end=5)
    with pytest.raises(ValueError):
        AudioSegment(tag_id=1, start=-1, end=5)


def test_segment_rejects_float_indices():
    # Sample indices are exact; silently truncating a float loses that.
    with pytest.raises(ValueError):
        AudioSegment(tag_id=1, start=0.5, end=10)


def test_seconds_conversions_round_trip():
    assert samples_to_seconds(16000, SR) == 1.0
    assert seconds_to_samples(1.0, SR) == 16000
    assert seconds_to_samples(samples_to_seconds(12345, SR), SR) == 12345


def test_from_seconds_uses_exclusive_end():
    segment = AudioSegment.from_seconds(tag_id=1, start_sec=1.0, end_sec=2.0, sample_rate=SR)
    assert segment.start == 16000
    assert segment.end == 31999
    assert segment.sample_count == 16000


def test_bigint_sample_indices_survive():
    big = 2_200_000_000
    segment = AudioSegment(tag_id=1, start=big, end=big + 1000)
    assert segment._to_api_json(entity_id=7)["frameRange"] == [big, big + 1000]


def test_overlap_is_channel_aware():
    a = AudioSegment(tag_id=1, start=0, end=100, channel=0)
    b = AudioSegment(tag_id=1, start=50, end=150, channel=1)
    c = AudioSegment(tag_id=1, start=50, end=150, channel=0)
    mixdown = AudioSegment(tag_id=1, start=50, end=150, channel=None)
    assert not a.overlaps(b)     # different explicit channels
    assert a.overlaps(c)
    assert a.overlaps(mixdown)   # mixdown spans every channel
    assert not a.overlaps(AudioSegment(tag_id=1, start=101, end=200, channel=0))


def test_api_payload_shape():
    settings = SpectrogramSettings(scale="mel", channel=1)
    segment = AudioSegment(tag_id=42, start=10, end=20, channel=1, settings=settings)
    payload = segment._to_api_json(entity_id=7)
    assert payload["tagId"] == 42
    assert payload["entityId"] == 7
    assert payload["frameRange"] == [10, 20]
    assert payload["meta"]["spectrogram"]["scale"] == "mel"


def test_segment_without_settings_emits_no_meta():
    # A partial `spectrogram` object is rejected outright by the API, so the
    # choice is all-or-nothing.
    segment = AudioSegment(tag_id=1, start=0, end=10)
    assert segment._meta_json() == {}
    assert "meta" not in segment._to_api_json(entity_id=1)


def test_parse_tag_assignment_from_api():
    raw = {
        "id": 2117459,
        "tagId": 53306,
        "entityId": 7112583,
        "frameRange": [4000, 8000],
        "startFrame": 4000,
        "endFrame": 8000,
        "labelerLogin": "iwatkot",
        "meta": {
            "channel": 0,
            "spectrogram": {
                "maxDb": 0, "minDb": -100, "scale": "linear", "window": "hann",
                "channel": 0, "fftSize": 2048, "melBands": 128, "hopLength": 512,
                "interpolation": "sharp",
            },
        },
    }
    segment = AudioSegment.from_api_json(raw)
    assert segment.start == 4000 and segment.end == 8000
    assert segment.channel == 0
    assert segment.settings.scale == "linear"
    assert segment.labeler_login == "iwatkot"


def test_parse_falls_back_to_start_end_frame():
    segment = AudioSegment.from_api_json(
        {"id": 1, "tagId": 2, "startFrame": 5, "endFrame": 9}
    )
    assert (segment.start, segment.end) == (5, 9)
    assert segment.settings is None


def test_recording_level_tag_is_not_a_segment():
    with pytest.raises(ValueError):
        AudioSegment.from_api_json({"id": 1, "tagId": 2, "value": "Speech"})


# -------------------------------------------------------------------- decode


def test_read_wav_mono(tmp_path):
    path = write_wav(tmp_path / "mono.wav", seconds=0.5)
    samples, rate = read_audio(str(path))
    assert rate == SR
    assert samples.shape == (8000, 1)
    assert samples.dtype == np.float32
    assert np.abs(samples).max() <= 1.0


def test_read_wav_stereo_and_info(tmp_path):
    path = write_wav(tmp_path / "stereo.wav", seconds=1.0, channels=2)
    info = get_audio_info(str(path))
    assert (info.sample_rate, info.sample_count, info.channels) == (SR, SR, 2)
    assert info.duration_seconds == 1.0


def test_select_channel(tmp_path):
    path = write_wav(tmp_path / "stereo.wav", seconds=0.2, channels=2)
    samples, _ = read_audio(str(path))
    assert select_channel(samples, 0).shape == (3200,)
    assert select_channel(samples, None).shape == (3200,)   # mixdown
    with pytest.raises(ValueError):
        select_channel(samples, 5)


# --------------------------------------------------------------- spectrogram


def test_mel_scale_round_trips():
    for hz in (0.0, 100.0, 1000.0, 8000.0):
        assert mel_to_hz(hz_to_mel(hz)) == pytest.approx(hz, abs=1e-6)


def test_stft_shape_matches_settings():
    signal = np.sin(2 * np.pi * 440 * np.arange(SR) / SR).astype(np.float32)
    power = stft_power(signal, fft_size=1024, hop_length=256, window="hann")
    assert power.shape[0] == 1024 // 2 + 1
    # Frames are centred on k*hop, so every hop that falls inside the recording
    # produces one -- the tool's `lastCenter = floor((sampleCount-1)/hop)*hop`.
    assert power.shape[1] == (SR - 1) // 256 + 1


def test_stft_finds_the_right_frequency():
    signal = np.sin(2 * np.pi * 1000 * np.arange(SR) / SR).astype(np.float32)
    power = stft_power(signal, 2048, 512, "hann")
    peak_bin = int(np.argmax(power[:, power.shape[1] // 2]))
    assert peak_bin * (SR / 2048) == pytest.approx(1000, abs=SR / 2048)


def test_render_respects_scale():
    signal = np.sin(2 * np.pi * 440 * np.arange(SR) / SR).astype(np.float32)
    mel = render_spectrogram(signal, SR, SpectrogramSettings(scale="mel", mel_bands=64))
    linear = render_spectrogram(signal, SR, SpectrogramSettings(scale="linear"))
    assert mel.shape[0] == 64
    assert linear.shape[0] == 2048 // 2 + 1


def test_render_clips_to_db_range():
    signal = np.random.default_rng(0).standard_normal(SR).astype(np.float32)
    settings = SpectrogramSettings(min_db=-60.0, max_db=0.0)
    spec = render_spectrogram(signal, SR, settings)
    assert spec.min() >= -60.0 - 1e-6
    assert spec.max() <= 0.0 + 1e-6


def test_render_is_deterministic():
    signal = np.sin(2 * np.pi * 440 * np.arange(SR) / SR).astype(np.float32)
    a = render_spectrogram(signal, SR, SpectrogramSettings())
    b = render_spectrogram(signal, SR, SpectrogramSettings())
    assert np.array_equal(a, b)


def test_settings_change_the_render():
    """The whole reason settings are stored: they change the picture."""
    signal = np.sin(2 * np.pi * 440 * np.arange(SR) / SR).astype(np.float32)
    coarse = render_spectrogram(signal, SR, SpectrogramSettings(fft_size=256))
    fine = render_spectrogram(signal, SR, SpectrogramSettings(fft_size=4096))
    assert coarse.shape != fine.shape


def test_render_segment_matches_manual_slice():
    signal = np.sin(2 * np.pi * 440 * np.arange(SR) / SR).astype(np.float32)
    settings = SpectrogramSettings(fft_size=512, hop_length=128)
    direct = render_spectrogram(signal[4000:8001], SR, settings)
    viaseg = render_segment(signal, SR, 4000, 8000, settings)
    assert np.array_equal(direct, viaseg)


def test_render_segment_rejects_reversed_range():
    signal = np.zeros(SR, dtype=np.float32)
    with pytest.raises(ValueError):
        render_segment(signal, SR, 100, 50)


def test_render_shorter_than_one_window_still_works():
    signal = np.sin(2 * np.pi * 440 * np.arange(100) / SR).astype(np.float32)
    spec = render_spectrogram(signal, SR, SpectrogramSettings(fft_size=2048))
    assert spec.shape[1] >= 1


def test_render_selects_the_named_channel(tmp_path):
    path = write_wav(tmp_path / "stereo.wav", seconds=1.0, channels=2)
    samples, rate = read_audio(str(path))
    ch0 = render_spectrogram(samples, rate, SpectrogramSettings(channel=0))
    ch1 = render_spectrogram(samples, rate, SpectrogramSettings(channel=1))
    # channel 1 is generated at twice the frequency, so the pictures differ
    assert not np.array_equal(ch0, ch1)


def test_to_image_shape_and_dtype():
    signal = np.sin(2 * np.pi * 440 * np.arange(SR) / SR).astype(np.float32)
    settings = SpectrogramSettings(scale="mel", mel_bands=64)
    img = to_image(render_spectrogram(signal, SR, settings), settings)
    assert img.shape[0] == 64 and img.shape[2] == 3
    assert img.dtype == np.uint8


# ------------------------------------------- toolbox-equivalence of the render


def test_mel_edges_match_the_toolbox_formula():
    """The tool computes edges as 700*expm1(a/(bands+1)*log1p(sr/1400))."""
    from supervisely.audio.spectrogram import _mel_edges

    bands = 64
    a = np.arange(bands + 2, dtype=float)
    expected = 700.0 * np.expm1(a / (bands + 1) * np.log1p(SR / 1400.0))
    assert np.allclose(_mel_edges(2048, SR, bands), expected, atol=1e-9)


def test_mel_edges_span_zero_to_nyquist():
    from supervisely.audio.spectrogram import _mel_edges

    edges = _mel_edges(2048, SR, 64)
    assert edges[0] == pytest.approx(0.0, abs=1e-9)
    assert edges[-1] == pytest.approx(SR / 2, rel=1e-9)


def test_mel_projection_is_an_average_not_a_sum():
    """The tool divides by the weight sum. A sum would scale with band width,
    making the wide high-frequency bands systematically brighter."""
    from supervisely.audio.spectrogram import _mel_project

    mag = np.ones((1025, 4), dtype=np.float32)
    out = _mel_project(mag, 2048, SR, 64)
    # averaging constant input must give back the constant, in every band
    assert np.allclose(out, 1.0, atol=1e-5)


def test_mel_projection_never_exceeds_input_maximum():
    rng = np.random.default_rng(1)
    mag = rng.random((1025, 8)).astype(np.float32)
    from supervisely.audio.spectrogram import _mel_project

    out = _mel_project(mag, 2048, SR, 64)
    assert out.max() <= mag.max() + 1e-6


@pytest.mark.parametrize("scale", ["linear", "log", "mel"])
def test_scale_position_maps_endpoints(scale):
    from supervisely.audio.spectrogram import scale_position_to_hz

    settings = SpectrogramSettings(scale=scale)
    assert float(scale_position_to_hz(0.0, SR, settings)) == pytest.approx(0.0, abs=1e-6)
    assert float(scale_position_to_hz(1.0, SR, settings)) == pytest.approx(SR / 2, rel=1e-9)


def test_scale_position_matches_toolbox_mel_formula():
    from supervisely.audio.spectrogram import scale_position_to_hz

    settings = SpectrogramSettings(scale="mel")
    for pos in (0.1, 0.35, 0.5, 0.9):
        expected = 700.0 * (math.exp(pos * math.log1p((SR / 2) / 700.0)) - 1.0)
        assert float(scale_position_to_hz(pos, SR, settings)) == pytest.approx(expected)


def test_scale_position_matches_toolbox_log_formula():
    from supervisely.audio.spectrogram import scale_position_to_hz

    settings = SpectrogramSettings(scale="log", fft_size=2048)
    width = SR / 2048
    for pos in (0.1, 0.5, 0.9):
        expected = width * math.expm1(pos * math.log1p((SR / 2) / width))
        assert float(scale_position_to_hz(pos, SR, settings)) == pytest.approx(expected)


def test_scale_position_is_monotonic():
    from supervisely.audio.spectrogram import scale_position_to_hz

    for scale in ("linear", "log", "mel"):
        settings = SpectrogramSettings(scale=scale)
        values = scale_position_to_hz(np.linspace(0, 1, 50), SR, settings)
        assert np.all(np.diff(values) > 0)


@pytest.mark.parametrize("scale", ["linear", "log", "mel"])
def test_rows_argument_sets_the_output_height(scale):
    signal = np.sin(2 * np.pi * 1000 * np.arange(SR) / SR).astype(np.float32)
    spec = render_spectrogram(signal, SR, SpectrogramSettings(scale=scale), rows=200)
    assert spec.shape[0] == 200


def test_rows_is_not_derivable_from_settings():
    """Display height is not part of the stored settings, so two heights are
    both legitimate renders of the same recorded analysis."""
    signal = np.sin(2 * np.pi * 1000 * np.arange(SR) / SR).astype(np.float32)
    settings = SpectrogramSettings(scale="mel")
    assert render_spectrogram(signal, SR, settings, rows=128).shape[0] == 128
    assert render_spectrogram(signal, SR, settings, rows=512).shape[0] == 512


def test_row_projection_preserves_a_narrow_peak():
    """Rows max-pool rather than average, so a single loud bin survives."""
    signal = np.sin(2 * np.pi * 3000 * np.arange(SR) / SR).astype(np.float32)
    settings = SpectrogramSettings(scale="linear", fft_size=2048)
    full = render_spectrogram(signal, SR, settings, as_db=False)
    rowed = render_spectrogram(signal, SR, settings, as_db=False, rows=64)
    assert rowed.max() == pytest.approx(full.max(), rel=1e-5)


# ------------------------------------ analysis parity with the labeling tool
#
# The tool's kernel is `shared/wasm/audio-fft-rs/audio-fft/src/lib.rs` and its
# projection is `labeling-tool/src/tools/audio/engine/spectrum.ts` on the
# platform side. These tests pin the four things that make the numbers match.


def test_windows_are_periodic_like_the_tool():
    """The tool builds `2*pi*i/N` windows; numpy's divide by `N - 1`."""
    size = 64
    phase = 2 * np.pi * np.arange(size) / size
    assert np.allclose(window_function("hann", size), 0.5 - 0.5 * np.cos(phase), atol=1e-7)
    assert np.allclose(
        window_function("hamming", size), 0.54 - 0.46 * np.cos(phase), atol=1e-7
    )
    assert np.allclose(
        window_function("blackman", size),
        0.42 - 0.5 * np.cos(phase) + 0.08 * np.cos(2 * phase),
        atol=1e-7,
    )
    assert not np.allclose(window_function("hann", size), np.hanning(size), atol=1e-3)


def test_full_scale_sine_reads_zero_db():
    """`1 / sum(window)^2` with the one-sided doubling calibrates 0 dB to a
    full-scale sine, which is what `maxDb = 0` means in the tool."""
    signal = np.sin(2 * np.pi * 1000 * np.arange(SR) / SR).astype(np.float32)
    spec = render_spectrogram(
        signal, SR, SpectrogramSettings(min_db=-120.0, max_db=20.0)
    )
    assert spec.max() == pytest.approx(0.0, abs=0.5)


def test_constant_signal_puts_unit_power_in_dc():
    """DC and Nyquist are not doubled: a constant 1.0 reads exactly 0 dB."""
    signal = np.ones(4096, dtype=np.float32)
    power = stft_power(signal, 1024, 256, "hann")
    middle = power.shape[1] // 2
    assert power[0, middle] == pytest.approx(1.0, rel=1e-4)


def test_decibels_are_absolute_so_crops_stay_comparable():
    """The regression that matters: a segment render must not renormalise to
    its own loudest point, or the same event reads differently in a crop."""
    t = np.arange(2 * SR) / SR
    signal = np.zeros(2 * SR, dtype=np.float32)
    signal[:SR] = 0.9 * np.sin(2 * np.pi * 440 * t[:SR])
    signal[SR:] = 0.05 * np.sin(2 * np.pi * 1200 * t[SR:])

    settings = SpectrogramSettings(fft_size=512, hop_length=256, min_db=-120.0)
    full = render_spectrogram(signal, SR, settings)
    crop = render_segment(signal, SR, SR, 2 * SR - 1, settings)

    quiet_in_full = full[:, SR // 256 + 2 : -2].max()
    quiet_in_crop = crop[:, 2:-2].max()
    assert quiet_in_crop == pytest.approx(quiet_in_full, abs=0.5)


def test_frames_are_centred_on_the_hop_grid():
    """Frame k covers [k*hop - fft/2, k*hop + fft/2): an impulse at sample 0
    shows up in frame 0. Starting frames at k*hop instead would move every
    event half a window -- 64 ms at fft_size=2048."""
    signal = np.zeros(4096, dtype=np.float32)
    signal[0] = 1.0
    power = stft_power(signal, 1024, 256, "hann")
    assert power[:, 0].sum() > 0
    assert power[:, 0].sum() > power[:, 3].sum()


def test_render_without_db_returns_power():
    """`as_db=False` hands out power, the quantity the tool projects and the
    one training pipelines normalise themselves."""
    signal = np.sin(2 * np.pi * 1000 * np.arange(SR) / SR).astype(np.float32)
    settings = SpectrogramSettings(fft_size=1024, hop_length=256)
    quiet = render_spectrogram(0.5 * signal, SR, settings, as_db=False)
    loud = render_spectrogram(signal, SR, settings, as_db=False)
    assert loud.max() == pytest.approx(4.0 * quiet.max(), rel=1e-3)


def test_to_image_uses_the_tools_colour_stops():
    settings = SpectrogramSettings(colormap="magma", min_db=-100.0, max_db=0.0)
    spec = np.array([[-100.0], [0.0]], dtype=np.float32)
    img = to_image(spec, settings)
    stops = COLOR_STOPS["magma"]
    # to_image flips: the first row of the output is the highest frequency,
    # which is the last row of the input.
    assert tuple(img[0, 0]) == tuple(stops[-1])
    assert tuple(img[-1, 0]) == tuple(stops[0])


def test_to_image_interpolates_between_stops():
    settings = SpectrogramSettings(colormap="viridis", min_db=-100.0, max_db=0.0)
    spec = np.array([[-75.0]], dtype=np.float32)  # level 0.25 -> exactly stop 1
    img = to_image(spec, settings)
    assert tuple(img[0, 0]) == tuple(COLOR_STOPS["viridis"][1])


def test_get_audio_info_reads_the_header_only(tmp_path, monkeypatch):
    """A one-hour WAV must not be decoded to answer three header fields."""
    import supervisely.audio.audio_io as audio_io

    path = write_wav(tmp_path / "hdr.wav", seconds=0.25, channels=2)
    monkeypatch.setattr(
        audio_io,
        "read_audio",
        lambda *a, **kw: pytest.fail("get_audio_info must not decode the samples"),
    )
    info = audio_io.get_audio_info(str(path))
    assert (info.sample_rate, info.sample_count, info.channels) == (SR, SR // 4, 2)

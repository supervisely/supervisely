# coding: utf-8
"""The render must equal the labeling tool's, not merely resemble it.

`reference_tool_port.py` is a naive, loop-for-loop transliteration of the
platform's own code -- the Rust FFT kernel and the TypeScript projection and
colouriser. This module runs it next to the SDK and compares the numbers and
the painted pixels.

It catches exactly the class of bug this modality shipped with once already: a
render that looks plausible on its own and disagrees with what the annotator
saw.
"""

import math

import numpy as np
import pytest

import supervisely as sly

from . import reference_tool_port as ref

SR = 16000
HEIGHT = 48


@pytest.fixture(scope="module")
def signal():
    """Two tones plus a click, in stereo: enough structure to separate bands."""
    t = np.arange(int(SR * 0.15)) / SR
    left = 0.6 * np.sin(2 * np.pi * 440 * t) + 0.2 * np.sin(2 * np.pi * 3000 * t)
    right = 0.3 * np.sin(2 * np.pi * 1200 * t)
    stereo = np.stack([left, right], axis=1).astype(np.float32)
    stereo[len(t) // 2, :] += 0.9  # a click, to exercise the max-pooling rows
    return stereo


CASES = [
    dict(scale="linear", fftSize=256, hopLength=64, window="hann", melBands=32,
         minDb=-100.0, maxDb=0.0, colormap="magma", interpolation="sharp", channel=None),
    dict(scale="mel", fftSize=512, hopLength=128, window="hamming", melBands=24,
         minDb=-90.0, maxDb=0.0, colormap="viridis", interpolation="sharp", channel=0),
    dict(scale="log", fftSize=256, hopLength=64, window="blackman", melBands=32,
         minDb=-100.0, maxDb=-10.0, colormap="grayscale", interpolation="smooth", channel=1),
    dict(scale="mel", fftSize=256, hopLength=64, window="hann", melBands=16,
         minDb=-100.0, maxDb=0.0, colormap="magma", interpolation="smooth", channel=None),
]


def _settings(case):
    return sly.SpectrogramSettings(
        scale=case["scale"], fft_size=case["fftSize"], hop_length=case["hopLength"],
        window=case["window"], mel_bands=case["melBands"], min_db=case["minDb"],
        max_db=case["maxDb"], colormap=case["colormap"],
        interpolation=case["interpolation"],
    )


def _reference_rows(signal, case):
    """Power rows exactly as the tool computes them, top row = high frequency."""
    mono = ref.mixdown(signal, case["channel"])
    frames = ref.compute_powers(
        np.ascontiguousarray(mono, dtype=np.float32),
        case["fftSize"], case["hopLength"], case["window"],
    )
    rows = ref.SpectrumRows(case, SR, HEIGHT)
    return np.array([rows.project(frame) for frame in frames]).T


@pytest.mark.parametrize("case", CASES, ids=lambda c: f"{c['scale']}-{c['window']}-{c['interpolation']}")
def test_power_matches_the_tool(signal, case):
    expected = _reference_rows(signal, case)
    ours = np.flipud(
        sly.audio.render_spectrogram(
            signal, SR, _settings(case), as_db=False, rows=HEIGHT, channel=case["channel"]
        )
    )
    assert ours.shape == expected.shape

    # float32 windows and samples against the reference's float64 arithmetic:
    # the residual is numerical noise, not a different transform. It only shows
    # up in near-null bins 60 dB below `min_db`, which neither the tool nor a
    # training pipeline ever sees, so the comparison is made over the settings'
    # own dB range.
    lo, hi = case["minDb"], case["maxDb"]
    db_ours = np.clip(10 * np.log10(np.maximum(ours, 1e-20)), lo, hi)
    db_expected = np.clip(10 * np.log10(np.maximum(expected, 1e-20)), lo, hi)
    assert np.abs(db_ours - db_expected).max() < 0.05


@pytest.mark.parametrize("case", CASES, ids=lambda c: f"{c['scale']}-{c['colormap']}")
def test_picture_matches_the_tool(signal, case):
    expected_power = _reference_rows(signal, case)
    expected_pixels = np.array(
        [ref.colorize(column, case) for column in expected_power.T]
    ).transpose(1, 0, 2)

    spec = sly.audio.render_spectrogram(
        signal, SR, _settings(case), rows=HEIGHT, channel=case["channel"]
    )
    ours = sly.audio.to_image(spec, _settings(case))

    assert ours.shape == expected_pixels.shape
    assert np.abs(ours.astype(int) - expected_pixels.astype(int)).max() == 0


def test_stft_frame_count_matches_the_tool(signal):
    """`lastCenter = floor((sampleCount - 1) / hop) * hop`, one frame per hop."""
    mono = signal.mean(axis=1)
    frames = ref.compute_powers(np.ascontiguousarray(mono, dtype=np.float32), 256, 64, "hann")
    ours = sly.audio.stft_power(np.ascontiguousarray(mono, dtype=np.float32), 256, 64, "hann")
    assert ours.shape[1] == len(frames)


def test_window_matches_the_tool():
    for name in ("hann", "hamming", "blackman"):
        expected, _ = ref.build_window(name, 64)
        assert np.allclose(sly.audio.window_function(name, 64), np.array(expected), atol=1e-7)

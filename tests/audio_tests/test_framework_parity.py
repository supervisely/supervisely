# coding: utf-8
"""The stored settings must be reproducible outside this SDK.

The confirmed workflow stores only the audio and the settings -- never a
rendered spectrogram -- and regenerates the picture in whatever framework does
the training. So `SpectrogramSettings` is the whole contract, and the thing
worth testing is not that our renderer matches a golden file, but that someone
holding nothing but the settings object lands on the same decibels with stock
`torch` / `tf.signal`.

Five conventions are not implied by the field names, and each one is a visible
error rather than a rounding error if it is missed: periodic windows, centred
frames with *zero* padding, the `1/sum(window)^2` and x4 scaling, HTK mel edges
with a weighted *average*, and absolute dB with no `top_db`. This module is the
executable form of that list; `supervisely/audio/README.md` is the prose form.

Neither framework is a dependency of the SDK -- these skip when absent.

Run: pytest tests/audio_tests/test_framework_parity.py -q
"""

import numpy as np
import pytest

import supervisely as sly
from supervisely.audio.spectrogram import render_spectrogram

SR = 16000

#: float32 (both frameworks) against float64 (numpy's rfft) over a 100 dB
#: range quantised to 256 display levels, i.e. 0.4 dB per level.
TOLERANCE_DB = 0.1

CASES = [
    dict(scale="linear", fft_size=1024, hop_length=256),
    dict(scale="log", fft_size=2048, hop_length=512, window="hamming"),
    dict(scale="mel", fft_size=1024, hop_length=256, mel_bands=64),
    dict(scale="mel", fft_size=2048, hop_length=128, mel_bands=128, window="blackman",
         min_db=-80.0, max_db=-10.0),
]


@pytest.fixture(scope="module")
def signal():
    """Two tones and a click: narrowband and wideband content in one array."""
    t = np.arange(int(SR * 0.5)) / SR
    x = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.2 * np.sin(2 * np.pi * 3100 * t)
    x[8000:8020] += 0.6
    return x.astype(np.float32)


def periodic_window(name, size):
    """The window the settings name, built the way the labeling tool builds it."""
    a0, a1, a2 = {
        "hann": (0.5, 0.5, 0.0),
        "hamming": (0.54, 0.46, 0.0),
        "blackman": (0.42, 0.5, 0.08),
    }[name]
    phase = 2 * np.pi * np.arange(size) / size
    return (a0 - a1 * np.cos(phase) + a2 * np.cos(2 * phase)).astype(np.float32)


@pytest.mark.parametrize("case", CASES, ids=lambda c: f"{c['scale']}-{c['fft_size']}")
def test_torch_reproduces_the_stored_analysis(signal, case):
    torch = pytest.importorskip("torch")
    torchaudio = pytest.importorskip("torchaudio")
    settings = sly.SpectrogramSettings(**case)

    win = torch.from_numpy(periodic_window(settings.window, settings.fft_size))
    spec = torch.stft(
        torch.from_numpy(signal),
        n_fft=settings.fft_size,
        hop_length=settings.hop_length,
        win_length=settings.fft_size,
        window=win,
        center=True,
        pad_mode="constant",  # torch defaults to "reflect", which is not what the tool does
        normalized=False,
        return_complex=True,
    )
    power = spec.abs().double() ** 2 / float(win.double().sum()) ** 2
    power[1:-1] *= 4.0

    if settings.scale == "mel":
        fb = torchaudio.functional.melscale_fbanks(
            n_freqs=settings.fft_size // 2 + 1,
            f_min=0.0,
            f_max=SR / 2.0,
            n_mels=settings.mel_bands,
            sample_rate=SR,
            norm=None,
            mel_scale="htk",
        ).double()
        power = (fb.T @ power) / fb.sum(dim=0)[:, None]  # average, not sum

    db = 10.0 * torch.log10(power.clamp_min(1e-20))
    theirs = db.clamp(settings.min_db, settings.max_db).numpy()

    ours = render_spectrogram(signal, SR, settings).astype(np.float64)
    assert ours.shape == theirs.shape
    assert np.max(np.abs(ours - theirs)) < TOLERANCE_DB


@pytest.mark.parametrize("case", CASES, ids=lambda c: f"{c['scale']}-{c['fft_size']}")
def test_tensorflow_reproduces_the_stored_analysis(signal, case):
    tf = pytest.importorskip("tensorflow")
    settings = sly.SpectrogramSettings(**case)

    # tf.signal.stft has no `center`, so the padding is explicit: half a window
    # of zeros in front, and enough behind for the last frame.
    n = signal.size
    half = settings.fft_size // 2
    n_frames = max(1, (max(n, 1) - 1) // settings.hop_length + 1)
    needed = (n_frames - 1) * settings.hop_length + settings.fft_size
    padded = np.zeros(max(needed, half + n), dtype=np.float32)
    padded[half : half + n] = signal

    win = periodic_window(settings.window, settings.fft_size)
    spec = tf.signal.stft(
        tf.constant(padded),
        frame_length=settings.fft_size,
        frame_step=settings.hop_length,
        fft_length=settings.fft_size,
        window_fn=lambda length, dtype: tf.constant(win),
        pad_end=False,
    )[:n_frames]

    power = tf.cast(tf.abs(spec), tf.float64) ** 2 / float(np.sum(win, dtype=np.float64)) ** 2
    one_sided = np.full(power.shape[-1], 4.0)
    one_sided[0] = 1.0
    one_sided[-1] = 1.0
    power = tf.transpose(power * tf.constant(one_sided))

    if settings.scale == "mel":
        # tf.signal zeroes the DC row of the filterbank and the tool does not,
        # which makes no difference: band 0's lower edge is 0 Hz, so its
        # triangle is already zero at DC.
        fb = tf.cast(
            tf.signal.linear_to_mel_weight_matrix(
                num_mel_bins=settings.mel_bands,
                num_spectrogram_bins=settings.fft_size // 2 + 1,
                sample_rate=SR,
                lower_edge_hertz=0.0,
                upper_edge_hertz=SR / 2.0,
            ),
            tf.float64,
        )
        power = tf.transpose(fb) @ power / tf.reduce_sum(fb, axis=0)[:, None]

    db = 10.0 * tf.math.log(tf.maximum(power, 1e-20)) / np.log(10.0)
    theirs = tf.clip_by_value(db, settings.min_db, settings.max_db).numpy()

    ours = render_spectrogram(signal, SR, settings).astype(np.float64)
    assert ours.shape == theirs.shape
    assert np.max(np.abs(ours - theirs)) < TOLERANCE_DB


def test_reflect_padding_would_be_a_visible_error(signal):
    """The one convention most likely to be missed, quantified.

    `torch.stft` and `librosa.stft` both default to reflect padding. Using it
    changes the first frames by tens of decibels, which is a different event at
    the start of the recording, not a rounding difference.
    """
    torch = pytest.importorskip("torch")
    settings = sly.SpectrogramSettings(scale="linear", fft_size=1024, hop_length=256)
    win = torch.from_numpy(periodic_window(settings.window, settings.fft_size))
    kwargs = dict(
        n_fft=settings.fft_size,
        hop_length=settings.hop_length,
        win_length=settings.fft_size,
        window=win,
        center=True,
        return_complex=True,
    )
    x = torch.from_numpy(signal)
    reflected = torch.stft(x, pad_mode="reflect", **kwargs)[:, 0].abs()
    zeroed = torch.stft(x, pad_mode="constant", **kwargs)[:, 0].abs()
    difference = 20 * torch.log10((reflected + 1e-20) / (zeroed + 1e-20))
    assert float(difference.abs().max()) > 5.0

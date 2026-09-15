# coding: utf-8
"""Rendering spectrograms from a recorded :class:`SpectrogramSettings`.

The point of storing the settings on a label is to be able to reproduce the
picture the annotator was looking at, and to feed the same transform to
training. Both come from this module, so the audit image and the model input
cannot drift apart.

.. warning::

   This implementation matches the labeling tool **semantically** -- same
   window, same hop, same frequency mapping, same dB range. Whether it is
   bit-identical to the tool's WASM STFT has not been established; that needs
   a golden-file comparison against the platform renderer. Until then, treat
   the output as "the same analysis", not "the same bytes".
"""

from typing import Optional, Tuple

import numpy as np

from supervisely.audio.audio_io import read_audio, select_channel
from supervisely.audio.spectrogram_settings import SpectrogramSettings

_WINDOWS = {
    "hann": np.hanning,
    "hamming": np.hamming,
    "blackman": np.blackman,
}


def hz_to_mel(freq):
    """HTK mel scale, matching the labeling tool."""
    return 2595.0 * np.log10(1.0 + np.asarray(freq, dtype=float) / 700.0)


def mel_to_hz(mel):
    """Inverse of :func:`hz_to_mel`."""
    return 700.0 * (10.0 ** (np.asarray(mel, dtype=float) / 2595.0) - 1.0)


def stft_magnitude(samples: np.ndarray, fft_size: int, hop_length: int, window: str) -> np.ndarray:
    """Short-time Fourier transform magnitude.

    Frames start at ``k * hop_length`` with no centering padding, matching the
    labeling tool. The padding convention matters: centering shifts every frame
    by half a window, which at ``fft_size=2048`` moves an event by 64 ms.

    :return: Array of shape ``(fft_size // 2 + 1, n_frames)``.
    """
    if window not in _WINDOWS:
        raise ValueError(f"unknown window {window!r}")
    if samples.size < fft_size:
        samples = np.pad(samples, (0, fft_size - samples.size), mode="constant")

    win = _WINDOWS[window](fft_size).astype(np.float32)
    n_frames = 1 + (samples.size - fft_size) // hop_length
    if n_frames < 1:
        n_frames = 1

    # Strided view avoids materialising n_frames copies of the window.
    stride = samples.strides[0]
    frames = np.lib.stride_tricks.as_strided(
        samples,
        shape=(n_frames, fft_size),
        strides=(stride * hop_length, stride),
        writeable=False,
    )
    return np.abs(np.fft.rfft(frames * win, axis=1)).T.astype(np.float32)


def _mel_filterbank(fft_size: int, sample_rate: int, mel_bands: int) -> np.ndarray:
    n_bins = fft_size // 2 + 1
    fft_freqs = np.linspace(0.0, sample_rate / 2.0, n_bins)
    edges = mel_to_hz(np.linspace(hz_to_mel(0.0), hz_to_mel(sample_rate / 2.0), mel_bands + 2))

    fb = np.zeros((mel_bands, n_bins), dtype=np.float32)
    for i in range(mel_bands):
        lo, mid, hi = edges[i], edges[i + 1], edges[i + 2]
        if hi <= lo:
            continue
        rising = (fft_freqs - lo) / max(mid - lo, 1e-9)
        falling = (hi - fft_freqs) / max(hi - mid, 1e-9)
        fb[i] = np.clip(np.minimum(rising, falling), 0.0, None)
    return fb


def _log_rows(magnitude: np.ndarray, sample_rate: int, rows: int) -> np.ndarray:
    """Resample linear frequency bins onto a logarithmic axis."""
    n_bins = magnitude.shape[0]
    nyquist = sample_rate / 2.0
    f_min = max(nyquist / n_bins, 1.0)
    targets = np.geomspace(f_min, nyquist, rows)
    idx = np.clip((targets / nyquist * (n_bins - 1)).astype(int), 0, n_bins - 1)
    return magnitude[idx]


def render_spectrogram(
    samples: np.ndarray,
    sample_rate: int,
    settings: Optional[SpectrogramSettings] = None,
    as_db: bool = True,
) -> np.ndarray:
    """Render a spectrogram from decoded samples.

    :param samples: ``(n,)`` mono or ``(n, channels)``; the channel named in
        ``settings`` is selected, or all channels are mixed down.
    :param sample_rate: Sample rate of the recording.
    :param settings: Settings to render under. Defaults to the platform default.
    :param as_db: Return decibels clipped to the settings' range. Set ``False``
        for raw magnitude, which is what most training pipelines want before
        their own normalisation.
    :return: ``(frequency_rows, n_frames)``, low frequency first.

    :Usage example:

     .. code-block:: python

        import supervisely as sly

        samples, rate = sly.audio.read_audio("recording.wav")
        spec = sly.audio.render_spectrogram(samples, rate, segment.settings)
    """
    settings = settings or SpectrogramSettings()
    mono = select_channel(samples, settings.channel)
    mono = np.ascontiguousarray(mono, dtype=np.float32)

    mag = stft_magnitude(mono, settings.fft_size, settings.hop_length, settings.window)

    if settings.scale == "mel":
        mag = _mel_filterbank(settings.fft_size, sample_rate, settings.mel_bands) @ mag
    elif settings.scale == "log":
        mag = _log_rows(mag, sample_rate, mag.shape[0])
    # "linear" leaves the bins untouched.

    if not as_db:
        return mag

    db = 20.0 * np.log10(np.maximum(mag, 1e-10))
    db -= db.max()  # 0 dB at the loudest point, as the tool shows it
    return np.clip(db, settings.min_db, settings.max_db)


def render_segment(
    samples: np.ndarray,
    sample_rate: int,
    start: int,
    end: int,
    settings: Optional[SpectrogramSettings] = None,
    as_db: bool = True,
) -> np.ndarray:
    """Render only an inclusive sample range, for per-segment training crops.

    ``end`` is inclusive, matching the platform's ``frameRange``.
    """
    if end < start:
        raise ValueError(f"end must be >= start, got start={start} end={end}")
    return render_spectrogram(samples[start : end + 1], sample_rate, settings, as_db=as_db)


def render_from_file(
    path: str,
    settings: Optional[SpectrogramSettings] = None,
    sample_range: Optional[Tuple[int, int]] = None,
    as_db: bool = True,
) -> np.ndarray:
    """Decode a file and render it, optionally only an inclusive sample range."""
    samples, rate = read_audio(path)
    if sample_range is None:
        return render_spectrogram(samples, rate, settings, as_db=as_db)
    return render_segment(samples, rate, sample_range[0], sample_range[1], settings, as_db=as_db)


def to_image(spectrogram: np.ndarray, settings: Optional[SpectrogramSettings] = None) -> np.ndarray:
    """Paint a dB spectrogram as an RGB uint8 image, low frequency at the bottom.

    :return: ``(rows, frames, 3)`` uint8, ready for ``sly.image.write``.
    """
    settings = settings or SpectrogramSettings()
    lo, hi = settings.min_db, settings.max_db
    norm = np.clip((spectrogram - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
    norm = np.flipud(norm)  # low frequency at the bottom, as the tool draws it

    if settings.colormap == "grayscale":
        rgb = np.stack([norm] * 3, axis=-1)
    elif settings.colormap == "magma":
        rgb = np.stack([norm**0.5, norm**1.7, np.clip(1.2 * norm**2.4, 0, 1)], axis=-1)
    else:  # viridis
        rgb = np.stack([np.clip(1.1 * norm**2, 0, 1), norm**0.8, np.clip(0.9 - 0.5 * norm, 0, 1)], axis=-1)
    return (rgb * 255).astype(np.uint8)

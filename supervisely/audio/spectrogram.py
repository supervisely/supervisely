# coding: utf-8
"""Rendering spectrograms from a recorded :class:`SpectrogramSettings`.

The point of storing the settings on a label is to be able to reproduce the
picture the annotator was looking at, and to feed the same transform to
training. Both come from this module, so the audit image and the model input
cannot drift apart.

What the stored settings do and do not determine:

* The **analysis** -- STFT, window, hop, mel band edges and weighting, dB range
  -- is fully determined by them. The mel edges and the weighted-average
  projection here were read from the labeling tool and match it.
* The **picture** additionally depends on the display height, which is not
  among the stored settings. Pass ``rows=`` to reproduce a particular on-screen
  grid; without it you get the natural resolution.

Bit-exactness against the tool's WASM STFT is still unverified -- that needs a
golden-file fixture from the platform renderer. Float arithmetic will differ in
the last places regardless.
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


def _mel_edges(fft_size: int, sample_rate: int, mel_bands: int) -> np.ndarray:
    """Mel band edges, matching the labeling tool exactly.

    The tool computes ``700 * expm1(a / (melBands + 1) * log1p(sampleRate / 1400))``,
    which is the HTK mel scale with edges spaced linearly in the mel domain from
    0 to Nyquist. Verified equal to that formula to 1e-11.
    """
    a = np.arange(mel_bands + 2, dtype=float)
    return 700.0 * np.expm1(a / (mel_bands + 1) * np.log1p(sample_rate / 1400.0))


def _mel_weights(fft_size: int, sample_rate: int, mel_bands: int) -> np.ndarray:
    """Triangular mel weights, one row per band, over ``fft_size // 2 + 1`` bins."""
    edges = _mel_edges(fft_size, sample_rate, mel_bands)
    n_bins = fft_size // 2 + 1
    freqs = np.arange(n_bins) * sample_rate / fft_size

    lo = edges[:-2, None]
    mid = edges[1:-1, None]
    hi = edges[2:, None]
    rising = (freqs[None, :] - lo) / np.maximum(mid - lo, 1e-12)
    falling = (hi - freqs[None, :]) / np.maximum(hi - mid, 1e-12)
    weights = np.clip(np.minimum(rising, falling), 0.0, None)

    # The tool iterates bins from ceil(lo*n/sr) to floor(hi*n/sr); outside that
    # span the triangle is zero anyway, so clipping is equivalent.
    return weights.astype(np.float32)


def _mel_project(magnitude: np.ndarray, fft_size: int, sample_rate: int,
                 mel_bands: int) -> np.ndarray:
    """Project linear bins onto mel bands as a weighted **average**.

    The labeling tool divides by the sum of the triangular weights
    (``melPower = sum(mag * h) / sum(h)``) rather than taking a weighted sum.
    That is the difference between matching what the annotator saw and being
    systematically brighter in the wide high-frequency bands, so it matters.

    Bands narrower than one FFT bin get zero total weight; the tool interpolates
    linearly at the band centre there, and so does this.
    """
    weights = _mel_weights(fft_size, sample_rate, mel_bands)
    totals = weights.sum(axis=1)
    out = weights @ magnitude
    nonzero = totals > 0
    out[nonzero] /= totals[nonzero, None]

    if not nonzero.all():
        edges = _mel_edges(fft_size, sample_rate, mel_bands)
        n_bins = magnitude.shape[0]
        for band in np.flatnonzero(~nonzero):
            pos = edges[band + 1] * fft_size / sample_rate
            low = min(max(int(np.floor(pos)), 0), n_bins - 1)
            frac = pos - np.floor(pos)
            high = min(low + 1, n_bins - 1)
            out[band] = magnitude[low] * (1 - frac) + magnitude[high] * frac
    return out.astype(np.float32)


def scale_position_to_hz(position, sample_rate: int,
                         settings: "SpectrogramSettings") -> np.ndarray:
    """Frequency shown at a normalised vertical position, 0 (bottom) to 1 (top).

    Mirrors the labeling tool's own mapping, so a position read off the
    displayed spectrogram converts to the frequency the annotator saw:

    * ``linear`` -- ``position * nyquist``
    * ``mel``    -- ``700 * (exp(position * log1p(nyquist / 700)) - 1)``
    * ``log``    -- anchored at one bin width, ``w * expm1(position * log1p(nyquist / w))``
    """
    position = np.asarray(position, dtype=float)
    nyquist = sample_rate / 2.0
    if settings.scale == "mel":
        return 700.0 * (np.exp(position * np.log1p(nyquist / 700.0)) - 1.0)
    if settings.scale == "log":
        width = sample_rate / settings.fft_size
        return width * np.expm1(position * np.log1p(nyquist / width))
    return position * nyquist


def _project_rows(magnitude: np.ndarray, sample_rate: int,
                  settings: "SpectrogramSettings", rows: int) -> np.ndarray:
    """Resample frequency bins onto a fixed number of display rows.

    Each row takes the **maximum** over the bins it spans, which is what the
    labeling tool does -- averaging would hide a narrow peak that the annotator
    could plainly see. Returned low frequency first; the tool draws rows
    top-down, and :func:`to_image` flips for that.
    """
    n_bins = magnitude.shape[0]
    out = np.zeros((rows, magnitude.shape[1]), dtype=np.float32)
    for row in range(rows):
        low_pos, high_pos = row / rows, (row + 1) / rows
        lo_hz = scale_position_to_hz(low_pos, sample_rate, settings)
        hi_hz = scale_position_to_hz(high_pos, sample_rate, settings)
        start = int(min(n_bins - 1, np.floor(float(lo_hz) * settings.fft_size / sample_rate)))
        end = int(min(n_bins - 1, np.ceil(float(hi_hz) * settings.fft_size / sample_rate)))
        start = max(start, 0)
        end = max(end, start)
        out[row] = magnitude[start : end + 1].max(axis=0)
    return out


def render_spectrogram(
    samples: np.ndarray,
    sample_rate: int,
    settings: Optional[SpectrogramSettings] = None,
    as_db: bool = True,
    rows: Optional[int] = None,
) -> np.ndarray:
    """Render a spectrogram from decoded samples.

    :param samples: ``(n,)`` mono or ``(n, channels)``; the channel named in
        ``settings`` is selected, or all channels are mixed down.
    :param sample_rate: Sample rate of the recording.
    :param settings: Settings to render under. Defaults to the platform default.
    :param as_db: Return decibels clipped to the settings' range. Set ``False``
        for raw magnitude, which is what most training pipelines want before
        their own normalisation.
    :param rows: Resample onto this many display rows, reproducing the grid the
        labeling tool draws at that height. Leave ``None`` for the natural
        resolution -- mel bands, or FFT bins for linear and log. The row count
        is **not** part of the stored settings, so it has to be supplied if you
        are matching a particular on-screen render.
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

    if rows is not None:
        # Display grid: the tool maps rows through its scale and max-pools the
        # bins each row covers. Mel is projected to bands first.
        if settings.scale == "mel":
            mag = _mel_project(mag, settings.fft_size, sample_rate, settings.mel_bands)
            mag = _resample_bands(mag, rows)
        else:
            mag = _project_rows(mag, sample_rate, settings, rows)
    elif settings.scale == "mel":
        mag = _mel_project(mag, settings.fft_size, sample_rate, settings.mel_bands)
    # "linear" and "log" at natural resolution leave the bins untouched; the
    # scale only decides where they land on screen, which is `rows` territory.

    if not as_db:
        return mag

    db = 20.0 * np.log10(np.maximum(mag, 1e-10))
    db -= db.max()  # 0 dB at the loudest point, as the tool shows it
    return np.clip(db, settings.min_db, settings.max_db)


def _resample_bands(bands: np.ndarray, rows: int) -> np.ndarray:
    """Nearest-neighbour resample of mel bands onto display rows."""
    idx = np.clip((np.arange(rows) / rows * bands.shape[0]).astype(int), 0, bands.shape[0] - 1)
    return bands[idx]


def render_segment(
    samples: np.ndarray,
    sample_rate: int,
    start: int,
    end: int,
    settings: Optional[SpectrogramSettings] = None,
    as_db: bool = True,
    rows: Optional[int] = None,
) -> np.ndarray:
    """Render only an inclusive sample range, for per-segment training crops.

    ``end`` is inclusive, matching the platform's ``frameRange``.
    """
    if end < start:
        raise ValueError(f"end must be >= start, got start={start} end={end}")
    return render_spectrogram(
        samples[start : end + 1], sample_rate, settings, as_db=as_db, rows=rows
    )


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

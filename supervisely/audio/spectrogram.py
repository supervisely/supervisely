# coding: utf-8
"""Rendering spectrograms from a recorded :class:`SpectrogramSettings`.

The point of storing the settings on the project is to be able to reproduce
the picture the annotator was looking at, and to feed the same transform to
training. Both come from this module, so the audit image and the model input
cannot drift apart.

Every step here is matched to the labeling tool, which computes its STFT and
projection in a GPU compute kernel (WebGPU, or WebGL2 as a fallback):

* periodic windows, ``1 - cos(2*pi*i/N)`` family, not the symmetric
  ``numpy`` ones that divide by ``N - 1``;
* **power**, not magnitude: ``(re^2 + im^2) / sum(window)^2``, doubled in
  amplitude (``x4`` in power) for every bin except DC and Nyquist, so a
  full-scale sine reads 0 dB;
* frames **centred** on ``k * hop_length`` and zero-padded past the ends of the
  recording;
* absolute decibels, ``10 * log10(power)``. The tool never normalises by the
  loudest point, which is what makes two renders of the same recording -- a
  crop and the whole file -- comparable at all.

What the stored settings do and do not determine:

* The **analysis** -- STFT, window, hop, mel band edges and weighting, dB range
  -- is fully determined by them.
* The **picture** additionally depends on the display height, which is not
  among the stored settings. Pass ``rows=`` to reproduce a particular on-screen
  grid; without it you get the natural resolution.

Bit-exactness against the tool's GPU kernel is not a goal and is not claimed.
The confirmed workflow renders the training input in the consumer's own
framework from the stored parameters, so what has to hold is that those
parameters pin the analysis unambiguously -- and float arithmetic differs in
the last places between any two implementations regardless (the tool transforms
in f32 on the GPU, numpy in f64).

The conventions above are therefore the contract, not an implementation detail.
``tests/audio_tests/test_framework_parity.py`` rebuilds this analysis in
``torch`` and in ``tf.signal`` from a :class:`SpectrogramSettings` alone and
checks it lands on the same decibels; ``supervisely/audio/README.md`` has the
same recipe in prose.
"""

from typing import Optional, Tuple

import numpy as np

from supervisely.audio.audio_io import read_audio, select_channel
from supervisely.audio.spectrogram_settings import SpectrogramSettings

#: Floor applied before taking the logarithm, matching the labeling tool.
POWER_FLOOR = 1e-20

#: Colour stops of each palette, matching the labeling tool exactly.
COLOR_STOPS = {
    "viridis": [(68, 1, 84), (59, 82, 139), (33, 145, 140), (94, 201, 98), (253, 231, 37)],
    "magma": [(0, 0, 4), (81, 18, 124), (183, 55, 121), (252, 137, 97), (252, 253, 191)],
    "grayscale": [(0, 0, 0), (255, 255, 255)],
}

_WINDOW_COEFFS = {
    # a0 - a1*cos(phase) + a2*cos(2*phase), phase = 2*pi*i/N (periodic).
    "hann": (0.5, 0.5, 0.0),
    "hamming": (0.54, 0.46, 0.0),
    "blackman": (0.42, 0.5, 0.08),
}


def hz_to_mel(freq):
    """HTK mel scale, matching the labeling tool."""
    return 2595.0 * np.log10(1.0 + np.asarray(freq, dtype=float) / 700.0)


def mel_to_hz(mel):
    """Inverse of :func:`hz_to_mel`."""
    return 700.0 * (10.0 ** (np.asarray(mel, dtype=float) / 2595.0) - 1.0)


def window_function(name: str, size: int) -> np.ndarray:
    """Periodic window of ``size`` samples, as float32.

    ``numpy``'s :func:`~numpy.hanning` and friends are *symmetric* -- they
    divide the phase by ``size - 1``. The tool builds periodic windows, and the
    difference shows up as a small but systematic tilt in every frame.
    """
    if name not in _WINDOW_COEFFS:
        raise ValueError(f"unknown window {name!r}")
    a0, a1, a2 = _WINDOW_COEFFS[name]
    phase = 2.0 * np.pi * np.arange(size, dtype=np.float64) / size
    return (a0 - a1 * np.cos(phase) + a2 * np.cos(2.0 * phase)).astype(np.float32)


def stft_power(samples: np.ndarray, fft_size: int, hop_length: int, window: str) -> np.ndarray:
    """Short-time Fourier transform power spectrum, as the labeling tool computes it.

    Frame ``k`` is centred on sample ``k * hop_length`` and covers
    ``[k * hop_length - fft_size // 2, ... + fft_size)``, zero-padded where it
    runs past either end of the recording. The centring matters: starting
    frames at ``k * hop_length`` instead would move every event half a window,
    which at ``fft_size=2048`` is 64 ms.

    Each bin is scaled by ``1 / sum(window)^2`` and, except at DC and Nyquist,
    doubled in amplitude to account for the discarded negative frequencies. A
    full-scale sine therefore reads ``0`` dB.

    :return: Array of shape ``(fft_size // 2 + 1, n_frames)``.
    """
    win = window_function(window, fft_size)
    # Match the tool: sum the stored float32 window values in float64.
    normalization = 1.0 / float(np.sum(win.astype(np.float64)) ** 2)

    n_samples = int(samples.size)
    n_frames = max(1, (max(n_samples, 1) - 1) // hop_length + 1)

    half = fft_size // 2
    needed = (n_frames - 1) * hop_length + fft_size
    padded = np.zeros(max(needed, half + n_samples), dtype=np.float32)
    padded[half : half + n_samples] = samples

    # `padded` is a fresh contiguous 1-D array, so its only stride is the item
    # size. Reading `itemsize` rather than `strides[0]` also keeps pylint from
    # misreading numpy's `strides` as unsubscriptable (E1136).
    stride = padded.itemsize
    frames = np.lib.stride_tricks.as_strided(
        padded,
        shape=(n_frames, fft_size),
        strides=(stride * hop_length, stride),
        writeable=False,
    )
    spectrum = np.fft.rfft(frames * win, axis=1)
    power = (spectrum.real**2 + spectrum.imag**2) * normalization

    one_sided = np.full(power.shape[1], 4.0)
    one_sided[0] = 1.0
    one_sided[-1] = 1.0
    return (power * one_sided).T.astype(np.float32)


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


def _mel_project(power: np.ndarray, fft_size: int, sample_rate: int, mel_bands: int) -> np.ndarray:
    """Project linear bins onto mel bands as a weighted **average** of power.

    The labeling tool divides by the sum of the triangular weights
    (``melPower = sum(power * h) / sum(h)``) rather than taking a weighted sum.
    That is the difference between matching what the annotator saw and being
    systematically brighter in the wide high-frequency bands, so it matters.

    Bands narrower than one FFT bin get zero total weight; the tool interpolates
    linearly at the band centre there, and so does this.
    """
    weights = _mel_weights(fft_size, sample_rate, mel_bands)
    totals = weights.sum(axis=1)
    out = weights @ power
    nonzero = totals > 0
    out[nonzero] /= totals[nonzero, None]

    if not nonzero.all():
        edges = _mel_edges(fft_size, sample_rate, mel_bands)
        n_bins = power.shape[0]
        for band in np.flatnonzero(~nonzero):
            pos = edges[band + 1] * fft_size / sample_rate
            low = min(max(int(np.floor(pos)), 0), n_bins - 1)
            frac = pos - np.floor(pos)
            high = min(low + 1, n_bins - 1)
            out[band] = power[low] * (1 - frac) + power[high] * frac
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


def _project_rows(power: np.ndarray, sample_rate: int, settings: "SpectrogramSettings",
                  rows: int, mel_power: Optional[np.ndarray] = None) -> np.ndarray:
    """Resample frequency bins onto a fixed number of display rows.

    Follows the tool row for row. Linear and log rows take the **maximum** over
    the bins they span -- averaging would hide a narrow peak the annotator could
    plainly see. Mel rows take the band their midpoint falls in. With
    ``interpolation="smooth"`` a row that covers less than one bin (or one mel
    band) interpolates instead of snapping, which is why ``interpolation``
    changes the numbers here even though it is cosmetic at natural resolution.

    Rows are returned low frequency first; the tool draws them top-down and
    :func:`to_image` flips for that.
    """
    n_bins = power.shape[0]
    half = settings.fft_size // 2
    smooth = settings.interpolation == "smooth"
    out = np.zeros((rows, power.shape[1]), dtype=np.float32)

    for y in range(rows):
        # The tool's row `y` counts from the top; ours counts from the bottom.
        bottom, top = y / rows, (y + 1) / rows
        low_bin = float(scale_position_to_hz(bottom, sample_rate, settings)) * settings.fft_size / sample_rate
        high_bin = float(scale_position_to_hz(top, sample_rate, settings)) * settings.fft_size / sample_rate
        middle = (bottom + top) / 2.0

        position = -1.0
        if smooth:
            if settings.scale == "mel" and rows > settings.mel_bands:
                position = min(max(middle * settings.mel_bands - 0.5, 0.0), settings.mel_bands - 1)
            elif settings.scale != "mel" and high_bin - low_bin < 1:
                position = min(
                    float(half),
                    float(scale_position_to_hz(middle, sample_rate, settings)) * settings.fft_size / sample_rate,
                )

        if position >= 0:
            values = mel_power if settings.scale == "mel" else power
            low = int(np.floor(position))
            frac = position - low
            high = min(low + 1, values.shape[0] - 1)
            out[y] = values[low] * (1 - frac) + values[high] * frac
        elif settings.scale == "mel":
            band = min(settings.mel_bands - 1, int(np.floor(middle * settings.mel_bands)))
            out[y] = mel_power[band]
        else:
            start = min(half, int(np.floor(low_bin)))
            end = min(half, int(np.ceil(high_bin)))
            start = min(max(start, 0), n_bins - 1)
            end = min(max(end, start), n_bins - 1)
            out[y] = power[start : end + 1].max(axis=0)
    return out


def render_spectrogram(
    samples: np.ndarray,
    sample_rate: int,
    settings: Optional[SpectrogramSettings] = None,
    as_db: bool = True,
    rows: Optional[int] = None,
    channel: Optional[int] = None,
) -> np.ndarray:
    """Render a spectrogram from decoded samples.

    :param samples: ``(n,)`` mono or ``(n, channels)``; ``channel`` selects one,
        or all channels are mixed down.
    :param sample_rate: Sample rate of the recording.
    :param settings: Settings to render under -- normally the project's, from
        :meth:`~supervisely.api.audio_api.AudioApi.get_spectrogram_settings`.
        Defaults to the platform default.
    :param as_db: Return absolute decibels clipped to the settings' range. Set
        ``False`` for raw power, which is what most training pipelines want
        before their own normalisation. Decibels are absolute: a crop and a full
        render of the same recording give the same numbers for the same event.
    :param rows: Resample onto this many display rows, reproducing the grid the
        labeling tool draws at that height. Leave ``None`` for the natural
        resolution -- mel bands, or FFT bins for linear and log. The row count
        is **not** part of the stored settings, so it has to be supplied if you
        are matching a particular on-screen render.
    :param channel: Zero-based channel to analyse, or ``None`` to mix down.
        Not part of the stored settings: which channel is on screen is
        navigation, not project configuration.
    :return: ``(frequency_rows, n_frames)``, low frequency first.

    :Usage example:

     .. code-block:: python

        import supervisely as sly

        samples, rate = sly.audio.read_audio("recording.wav")
        settings = api.audio.get_spectrogram_settings(project_id)
        spec = sly.audio.render_spectrogram(samples, rate, settings)
    """
    settings = settings or SpectrogramSettings()
    mono = select_channel(samples, channel)
    mono = np.ascontiguousarray(mono, dtype=np.float32)

    power = stft_power(mono, settings.fft_size, settings.hop_length, settings.window)

    mel_power = None
    if settings.scale == "mel":
        mel_power = _mel_project(power, settings.fft_size, sample_rate, settings.mel_bands)

    if rows is not None:
        power = _project_rows(power, sample_rate, settings, rows, mel_power)
    elif mel_power is not None:
        power = mel_power
    # "linear" and "log" at natural resolution leave the bins untouched; the
    # scale only decides where they land on screen, which is `rows` territory.

    if not as_db:
        return power

    db = 10.0 * np.log10(np.maximum(power, POWER_FLOOR))
    return np.clip(db, settings.min_db, settings.max_db)


def render_segment(
    samples: np.ndarray,
    sample_rate: int,
    start: int,
    end: int,
    settings: Optional[SpectrogramSettings] = None,
    as_db: bool = True,
    rows: Optional[int] = None,
    channel: Optional[int] = None,
) -> np.ndarray:
    """Render only an inclusive sample range, for per-segment training crops.

    ``end`` is inclusive, matching the platform's ``frameRange``. Decibels are
    absolute, so a crop is directly comparable with the full render and with
    crops of other recordings. Pass ``channel=segment.channel`` to analyse the
    channel the label is about.
    """
    if end < start:
        raise ValueError(f"end must be >= start, got start={start} end={end}")
    return render_spectrogram(
        samples[start : end + 1], sample_rate, settings, as_db=as_db, rows=rows, channel=channel
    )


def render_from_file(
    path: str,
    settings: Optional[SpectrogramSettings] = None,
    sample_range: Optional[Tuple[int, int]] = None,
    as_db: bool = True,
    channel: Optional[int] = None,
) -> np.ndarray:
    """Decode a file and render it, optionally only an inclusive sample range."""
    samples, rate = read_audio(path)
    if sample_range is None:
        return render_spectrogram(samples, rate, settings, as_db=as_db, channel=channel)
    return render_segment(
        samples, rate, sample_range[0], sample_range[1], settings, as_db=as_db, channel=channel
    )


def to_image(spectrogram: np.ndarray, settings: Optional[SpectrogramSettings] = None) -> np.ndarray:
    """Paint a dB spectrogram as an RGB uint8 image, low frequency at the bottom.

    Uses the labeling tool's own colour stops with the same linear
    interpolation between them, so the picture matches what the annotator saw.

    :return: ``(rows, frames, 3)`` uint8, ready for ``sly.image.write``.
    """
    settings = settings or SpectrogramSettings()
    lo, hi = settings.min_db, settings.max_db
    level = np.clip((spectrogram - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
    level = np.flipud(level)  # low frequency at the bottom, as the tool draws it

    stops = np.asarray(COLOR_STOPS[settings.colormap], dtype=np.float64)
    position = level * (len(stops) - 1)
    low = np.clip(np.floor(position), 0, len(stops) - 2).astype(int)
    frac = (position - low)[..., None]
    rgb = stops[low] * (1 - frac) + stops[low + 1] * frac
    return np.clip(np.round(rgb), 0, 255).astype(np.uint8)

"""Line-by-line transliteration of the platform's own audio analysis.

Sources (platform monorepo, feature/6151-audio at 291bcde0):
  shared/wasm/audio-fft-rs/audio-fft/src/lib.rs     -- BatchFft::new / compute
  labeling-tool/src/tools/audio/engine/spectrum.ts  -- SpectrumRows, colorize
  labeling-tool/src/tools/audio/engine/types.ts     -- frequencyAtPosition, stops
  labeling-tool/src/tools/audio/engine/reader.ts    -- copy(): zero padding, mixdown

e1536b6b, merged with the feature, moved the transform and projection to a GPU
kernel (engine/fft.ts, engine/spectrogram.ts) with the same windows, 1/sum^2
and x4 scaling, zero-padded centred frames, mel edges and weighted average,
max-pooled rows and 1e-20 dB floor; the stops and frequencyAtPosition are
unchanged. The formulas below still describe it.

Deliberately naive: plain loops, no numpy tricks, so it can be read next to the
original and compared by eye.
"""
import math

import numpy as np

STOPS = {
    "viridis": [[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]],
    "magma": [[0, 0, 4], [81, 18, 124], [183, 55, 121], [252, 137, 97], [252, 253, 191]],
    "grayscale": [[0, 0, 0], [255, 255, 255]],
}


def build_window(name, fft_size):
    """lib.rs: periodic window, values stored as f32, summed in f64."""
    window = []
    total = 0.0
    for i in range(fft_size):
        phase = 2.0 * math.pi * i / fft_size
        if name == "hamming":
            value = 0.54 - 0.46 * math.cos(phase)
        elif name == "blackman":
            value = 0.42 - 0.5 * math.cos(phase) + 0.08 * math.cos(2.0 * phase)
        else:
            value = 0.5 - 0.5 * math.cos(phase)
        value = np.float32(value)
        total += float(value)
        window.append(value)
    return window, 1.0 / (total * total)


def compute_powers(samples, fft_size, hop_length, window_name):
    """lib.rs compute(), fed the way audio.worker.ts feeds it: frame k is
    centred on k*hop and read through PcmReader.copy(), which zero-fills."""
    window, normalization = build_window(window_name, fft_size)
    bins = fft_size // 2 + 1
    sample_count = len(samples)
    last_center = (sample_count - 1) // hop_length * hop_length

    frames = []
    for center in range(0, last_center + 1, hop_length):
        frame = np.zeros(fft_size, dtype=np.float64)
        start = center - fft_size // 2
        for i in range(fft_size):
            index = start + i
            if 0 <= index < sample_count:
                frame[i] = float(samples[index]) * float(window[i])
        spectrum = np.fft.fft(frame)
        powers = []
        for b in range(bins):
            re, im = spectrum[b].real, spectrum[b].imag
            one_sided = 1.0 if (b == 0 or b == fft_size // 2) else 4.0
            powers.append((re * re + im * im) * normalization * one_sided)
        frames.append(powers)
    return frames  # [frame][bin]


def frequency_at_position(position, sample_rate, settings):
    """types.ts frequencyAtPosition."""
    nyquist = sample_rate / 2
    if settings["scale"] == "mel":
        return 700 * (math.exp(position * math.log1p(nyquist / 700)) - 1)
    if settings["scale"] == "log":
        first_bin = sample_rate / settings["fftSize"]
        return first_bin * math.expm1(position * math.log1p(nyquist / first_bin))
    return position * nyquist


class SpectrumRows:
    """spectrum.ts SpectrumRows."""

    def __init__(self, settings, sample_rate, height):
        size = settings["fftSize"]
        self.settings, self.sample_rate = settings, sample_rate
        self.rows = [0.0] * height
        self.row_start, self.row_end, self.row_bands = [0] * height, [0] * height, [0] * height
        self.mel_edges = [0.0] * (settings["melBands"] + 2)
        self.mel_power = [0.0] * settings["melBands"]
        self.row_position = [-1.0] * height if settings.get("interpolation") == "smooth" else None
        for y in range(height):
            bottom = 1 - (y + 1) / height
            top = 1 - y / height
            low_bin = frequency_at_position(bottom, sample_rate, settings) * size / sample_rate
            high_bin = frequency_at_position(top, sample_rate, settings) * size / sample_rate
            self.row_start[y] = int(min(size / 2, math.floor(low_bin)))
            self.row_end[y] = int(min(size / 2, math.ceil(high_bin)))
            self.row_bands[y] = int(
                min(settings["melBands"] - 1, math.floor((bottom + top) / 2 * settings["melBands"]))
            )
            if self.row_position is not None:
                if settings["scale"] == "mel" and height > settings["melBands"]:
                    self.row_position[y] = max(
                        0, min(settings["melBands"] - 1,
                               (bottom + top) / 2 * settings["melBands"] - 0.5)
                    )
                elif settings["scale"] != "mel" and high_bin - low_bin < 1:
                    self.row_position[y] = min(
                        size / 2,
                        frequency_at_position((bottom + top) / 2, sample_rate, settings)
                        * size / sample_rate,
                    )
        for band in range(len(self.mel_edges)):
            self.mel_edges[band] = 700 * math.expm1(
                band / (settings["melBands"] + 1) * math.log1p(sample_rate / 1400)
            )

    def project(self, powers):
        settings, size = self.settings, self.settings["fftSize"]
        if settings["scale"] == "mel":
            for band in range(len(self.mel_power)):
                left, center, right = (
                    self.mel_edges[band], self.mel_edges[band + 1], self.mel_edges[band + 2]
                )
                total, weights = 0.0, 0.0
                lo = math.ceil(left * size / self.sample_rate)
                hi = min(size // 2, math.floor(right * size / self.sample_rate))
                for b in range(lo, hi + 1):
                    frequency = b * self.sample_rate / size
                    weight = ((frequency - left) / (center - left) if frequency < center
                              else (right - frequency) / (right - center))
                    total += powers[b] * weight
                    weights += weight
                position = center * size / self.sample_rate
                low = int(math.floor(position))
                fraction = position - low
                self.mel_power[band] = (
                    total / weights if weights > 0
                    else powers[low] * (1 - fraction) + powers[min(size // 2, low + 1)] * fraction
                )
        for y in range(len(self.rows)):
            power = 0.0
            position = self.row_position[y] if self.row_position is not None else -1
            if position >= 0:
                values = self.mel_power if settings["scale"] == "mel" else powers
                low = int(math.floor(position))
                fraction = position - low
                power = (values[low] * (1 - fraction)
                         + values[min(len(values) - 1, low + 1)] * fraction)
            elif settings["scale"] == "mel":
                power = self.mel_power[self.row_bands[y]]
            else:
                for b in range(self.row_start[y], self.row_end[y] + 1):
                    power = max(power, powers[b])
            self.rows[y] = power
        return list(self.rows)


def colorize(power_rows, settings):
    """spectrum.ts colorize(): absolute dB, then the palette stops."""
    stops = STOPS[settings["colormap"]]
    pixels = []
    for value in power_rows:
        db = 10 * math.log10(max(value, 1e-20))
        level = max(0.0, min(1.0, (db - settings["minDb"]) / (settings["maxDb"] - settings["minDb"])))
        position = level * (len(stops) - 1)
        low = min(len(stops) - 2, int(math.floor(position)))
        fraction = position - low
        pixels.append(
            tuple(
                # Uint8ClampedArray assignment rounds to nearest, ties to even.
                int(np.float64(stops[low][c] * (1 - fraction) + stops[low + 1][c] * fraction).round())
                for c in range(3)
            )
        )
    return pixels


def mixdown(samples, channel):
    """reader.ts: a mixdown is the mean of every channel."""
    if samples.ndim == 1:
        return samples
    if channel is None:
        return samples.mean(axis=1)
    return samples[:, channel]

# coding: utf-8
"""Spectrogram view settings recorded alongside an audio segment label.

An audio segment is labeled against a *picture* of the sound, and that picture
is one of many possible transforms of it. Two annotators using different
settings are not looking at the same evidence, so the settings a label was
drawn under are part of the label, not a user preference.

The platform stores them in the tag assignment's ``meta`` under the
``spectrogram`` key. This module is the typed, validated Python view of that
object.
"""

import hashlib
import json
from typing import Any, Dict, Optional

# Allowed enum values, mirroring the labeling tool.
SCALES = ("linear", "log", "mel")
WINDOWS = ("hann", "hamming", "blackman")
COLORMAPS = ("viridis", "magma", "grayscale")
INTERPOLATIONS = ("sharp", "smooth")

# Numeric bounds enforced by the labeling tool. The API is currently looser
# (see :func:`validate`), so we validate here to the stricter rule: a segment
# the toolbox cannot open is worse than a rejected write.
FFT_SIZE_MIN, FFT_SIZE_MAX = 32, 32768
MEL_BANDS_MIN, MEL_BANDS_MAX = 2, 512

#: Keys that change the numbers. ``colormap`` and ``interpolation`` only change
#: how the array is painted, so they are excluded from the fingerprint.
_SEMANTIC_KEYS = ("scale", "fftSize", "hopLength", "window", "melBands", "minDb", "maxDb", "channel")


class SpectrogramSettings:
    """Settings describing how a spectrogram was rendered for an audio label.

    :param scale: Frequency axis. One of ``linear``, ``log``, ``mel``.
    :param fft_size: FFT window length in samples. Power of two, 32..32768.
    :param hop_length: Samples between consecutive frames. >= 1.
    :param window: Window function. One of ``hann``, ``hamming``, ``blackman``.
    :param mel_bands: Number of mel filters, 2..512. Used when ``scale="mel"``.
    :param min_db: Lower edge of the displayed dB range.
    :param max_db: Upper edge of the displayed dB range. Must exceed ``min_db``.
    :param colormap: Palette. View-only; excluded from :attr:`fingerprint`.
    :param interpolation: ``sharp`` or ``smooth``. View-only.
    :param channel: Zero-based channel index, or ``None`` for a mixdown.

    :Usage example:

     .. code-block:: python

        import supervisely as sly

        settings = sly.SpectrogramSettings(scale="mel", fft_size=1024, hop_length=256)
        settings.to_json()
    """

    def __init__(
        self,
        scale: str = "linear",
        fft_size: int = 2048,
        hop_length: int = 512,
        window: str = "hann",
        mel_bands: int = 128,
        min_db: float = -100.0,
        max_db: float = 0.0,
        colormap: str = "magma",
        interpolation: str = "sharp",
        channel: Optional[int] = None,
    ):
        self.scale = scale
        self.fft_size = int(fft_size)
        self.hop_length = int(hop_length)
        self.window = window
        self.mel_bands = int(mel_bands)
        self.min_db = float(min_db)
        self.max_db = float(max_db)
        self.colormap = colormap
        self.interpolation = interpolation
        self.channel = None if channel is None else int(channel)
        self.validate()

    def validate(self) -> None:
        """Raise :class:`ValueError` if any field is outside what the labeling
        tool accepts.

        Validated here rather than relying on the API, which currently accepts
        values the toolbox then refuses to load (``melBands`` of 1 or 100000,
        ``fftSize`` of 16 or 65536). Writing one of those produces a recording
        that cannot be opened in the UI, so the SDK refuses to author it.
        """
        if self.scale not in SCALES:
            raise ValueError(f"scale must be one of {SCALES}, got {self.scale!r}")
        if self.window not in WINDOWS:
            raise ValueError(f"window must be one of {WINDOWS}, got {self.window!r}")
        if self.colormap not in COLORMAPS:
            raise ValueError(f"colormap must be one of {COLORMAPS}, got {self.colormap!r}")
        if self.interpolation not in INTERPOLATIONS:
            raise ValueError(
                f"interpolation must be one of {INTERPOLATIONS}, got {self.interpolation!r}"
            )
        if not FFT_SIZE_MIN <= self.fft_size <= FFT_SIZE_MAX:
            raise ValueError(
                f"fft_size must be within {FFT_SIZE_MIN}..{FFT_SIZE_MAX}, got {self.fft_size}"
            )
        if self.fft_size & (self.fft_size - 1) != 0:
            raise ValueError(f"fft_size must be a power of two, got {self.fft_size}")
        if self.hop_length < 1:
            raise ValueError(f"hop_length must be >= 1, got {self.hop_length}")
        if not MEL_BANDS_MIN <= self.mel_bands <= MEL_BANDS_MAX:
            raise ValueError(
                f"mel_bands must be within {MEL_BANDS_MIN}..{MEL_BANDS_MAX}, got {self.mel_bands}"
            )
        if self.min_db >= self.max_db:
            raise ValueError(f"min_db must be < max_db, got {self.min_db} >= {self.max_db}")
        if self.channel is not None and self.channel < 0:
            raise ValueError(f"channel must be >= 0 or None, got {self.channel}")

    def to_json(self, include_colormap: bool = False) -> Dict[str, Any]:
        """Serialize to the platform's ``meta.spectrogram`` object.

        :param include_colormap: Whether to emit ``colormap``. Defaults to
            ``False`` to match the labeling tool, which strips it before saving
            because it does not affect which events are visible. Two labels made
            under the same analysis then compare equal regardless of palette.
        """
        data = {
            "scale": self.scale,
            "fftSize": self.fft_size,
            "hopLength": self.hop_length,
            "window": self.window,
            "melBands": self.mel_bands,
            "minDb": self.min_db,
            "maxDb": self.max_db,
            "interpolation": self.interpolation,
            "channel": self.channel,
        }
        if include_colormap:
            data["colormap"] = self.colormap
        return data

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "SpectrogramSettings":
        """Build from a platform ``meta.spectrogram`` object.

        Missing optional keys fall back to the platform defaults, so settings
        written by an older toolbox build still load.
        """
        if not isinstance(data, dict):
            raise ValueError(f"spectrogram settings must be a dict, got {type(data).__name__}")
        return cls(
            scale=data.get("scale", "linear"),
            fft_size=data.get("fftSize", 2048),
            hop_length=data.get("hopLength", 512),
            window=data.get("window", "hann"),
            mel_bands=data.get("melBands", 128),
            min_db=data.get("minDb", -100.0),
            max_db=data.get("maxDb", 0.0),
            colormap=data.get("colormap", "magma"),
            interpolation=data.get("interpolation", "sharp"),
            channel=data.get("channel"),
        )

    @property
    def fingerprint(self) -> str:
        """Stable id for "is this the same transform".

        A canonical SHA-256 over the fields that change the numbers. Cosmetic
        changes (``colormap``, ``interpolation``) leave it untouched, so it
        answers three questions with one string comparison: can these datasets
        be merged, were these labels made under the same view, and does a
        training render match what the annotator saw.
        """
        core = {k: v for k, v in self.to_json().items() if k in _SEMANTIC_KEYS}
        canon = json.dumps(core, sort_keys=True, separators=(",", ":"))
        return "sha256:" + hashlib.sha256(canon.encode()).hexdigest()[:16]

    def clone(self, **overrides) -> "SpectrogramSettings":
        """Return a copy with the given fields replaced."""
        fields = {
            "scale": self.scale,
            "fft_size": self.fft_size,
            "hop_length": self.hop_length,
            "window": self.window,
            "mel_bands": self.mel_bands,
            "min_db": self.min_db,
            "max_db": self.max_db,
            "colormap": self.colormap,
            "interpolation": self.interpolation,
            "channel": self.channel,
        }
        fields.update(overrides)
        return SpectrogramSettings(**fields)

    def __eq__(self, other) -> bool:
        if not isinstance(other, SpectrogramSettings):
            return NotImplemented
        return self.to_json(include_colormap=True) == other.to_json(include_colormap=True)

    def __repr__(self) -> str:
        return (
            f"SpectrogramSettings(scale={self.scale!r}, fft_size={self.fft_size}, "
            f"hop_length={self.hop_length}, window={self.window!r}, "
            f"mel_bands={self.mel_bands}, min_db={self.min_db}, max_db={self.max_db}, "
            f"channel={self.channel})"
        )

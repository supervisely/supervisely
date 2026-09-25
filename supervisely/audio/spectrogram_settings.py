# coding: utf-8
"""Spectrogram analysis settings, configured once per audio project.

An audio segment is labeled against a *picture* of the sound, and that picture
is one of many possible transforms of it. Two annotators using different
settings are not looking at the same evidence, so the settings have to be
knowable for every label in the project.

The platform makes them project configuration rather than annotation data:
they live in ``projects.settings.spectrogram``, are applied to every recording
in the project, and changing them needs the permission to edit the project
rather than the permission to label in it. Read and write them with
:meth:`~supervisely.api.audio_api.AudioApi.get_spectrogram_settings` and
:meth:`~supervisely.api.audio_api.AudioApi.set_spectrogram_settings`.

``channel`` is deliberately *not* one of these settings. Which channel an
annotator is looking at is navigation, not configuration, so it is passed to
the render functions instead. The channel a *label* is about is
:attr:`AudioSegment.channel`, which is a different thing again.
"""

import hashlib
import json
from typing import Any, Dict

# Allowed enum values, mirroring the labeling tool and the API's Joi schema.
SCALES = ("linear", "log", "mel")
WINDOWS = ("hann", "hamming", "blackman")
COLORMAPS = ("viridis", "magma", "grayscale")
INTERPOLATIONS = ("sharp", "smooth")

# Numeric bounds. These are the API's bounds and the labeling tool's bounds --
# the two agree since the settings moved onto the project.
FFT_SIZE_MIN, FFT_SIZE_MAX = 32, 32768
MEL_BANDS_MIN, MEL_BANDS_MAX = 2, 512

#: The nine fields the platform stores, in the order the API declares them.
PROJECT_FIELDS = (
    "scale",
    "fftSize",
    "hopLength",
    "window",
    "melBands",
    "minDb",
    "maxDb",
    "colormap",
    "interpolation",
)

#: Keys that change the numbers. ``colormap`` and ``interpolation`` only change
#: how the array is painted, so they are excluded from the fingerprint.
_SEMANTIC_KEYS = ("scale", "fftSize", "hopLength", "window", "melBands", "minDb", "maxDb")


class SpectrogramSettings:
    """How every recording in an audio project is analysed and drawn.

    The defaults are the platform's defaults, so ``SpectrogramSettings()``
    describes a project that has never been configured.

    :param scale: Frequency axis. One of ``linear``, ``log``, ``mel``.
    :param fft_size: FFT window length in samples. Power of two, 32..32768.
    :param hop_length: Samples between consecutive frames. >= 1.
    :param window: Window function. One of ``hann``, ``hamming``, ``blackman``.
    :param mel_bands: Number of mel filters, 2..512. Used when ``scale="mel"``.
    :param min_db: Lower edge of the displayed dB range.
    :param max_db: Upper edge of the displayed dB range. Must exceed ``min_db``.
    :param colormap: Palette. Display-only; excluded from :attr:`fingerprint`.
    :param interpolation: ``sharp`` or ``smooth``. Display-only.

    :Usage example:

     .. code-block:: python

        import supervisely as sly

        api = sly.Api.from_env()
        settings = sly.SpectrogramSettings(scale="mel", fft_size=1024, hop_length=256)
        api.audio.set_spectrogram_settings(project_id, settings)
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
        self.validate()

    def validate(self) -> None:
        """Raise :class:`ValueError` if any field is outside what the platform
        accepts.

        The bounds are the API's own (``UpdateProjectSettings``): an enumerated
        set of FFT sizes, 2..512 mel bands, a hop of at least one sample and a
        dB range that is the right way round. Checking here turns a rejected
        request into an immediate, specific error.
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

    def to_json(self) -> Dict[str, Any]:
        """Serialize to the platform's ``projects.settings.spectrogram`` object.

        All nine fields are always present: the API requires every one of them,
        and a partial object is rejected.
        """
        return {
            "scale": self.scale,
            "fftSize": self.fft_size,
            "hopLength": self.hop_length,
            "window": self.window,
            "melBands": self.mel_bands,
            "minDb": self.min_db,
            "maxDb": self.max_db,
            "colormap": self.colormap,
            "interpolation": self.interpolation,
        }

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "SpectrogramSettings":
        """Build from a stored settings object.

        A missing key falls back to the platform default for it, which is what
        the labeling tool does: a project configured before a field existed
        keeps rendering exactly as it did. Keys that are not settings --
        ``channel`` on a legacy annotation-level object -- are ignored.
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
        )

    @property
    def fingerprint(self) -> str:
        """Stable id for "is this the same transform".

        A canonical SHA-256 over the fields that change the numbers. Display
        changes (``colormap``, ``interpolation``) leave it untouched, so it
        answers three questions with one string comparison: can these datasets
        be merged, were these labels made under the same analysis, and does a
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
        }
        fields.update(overrides)
        return SpectrogramSettings(**fields)

    def _canonical(self) -> str:
        return json.dumps(self.to_json(), sort_keys=True, separators=(",", ":"))

    def __eq__(self, other) -> bool:
        if not isinstance(other, SpectrogramSettings):
            return NotImplemented
        return self.to_json() == other.to_json()

    def __hash__(self) -> int:
        """Defining ``__eq__`` alone would set ``__hash__`` to ``None``, and
        grouping projects by analysis is the obvious thing to want to do with
        these."""
        return hash(self._canonical())

    def __repr__(self) -> str:
        return (
            f"SpectrogramSettings(scale={self.scale!r}, fft_size={self.fft_size}, "
            f"hop_length={self.hop_length}, window={self.window!r}, "
            f"mel_bands={self.mel_bands}, min_db={self.min_db}, max_db={self.max_db}, "
            f"colormap={self.colormap!r}, interpolation={self.interpolation!r})"
        )

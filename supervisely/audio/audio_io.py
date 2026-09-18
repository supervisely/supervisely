# coding: utf-8
"""Reading audio files into numpy.

The platform does not store sample rate, sample count or channel count -- an
audio entity's ``fileMeta`` is only ``{mime, size}``. This is deliberate and
confirmed: unlike other modalities, audio metadata is not forced server-side,
and clients derive what they need. The labeling tool decodes the file in the
browser, and the SDK does the same. That is why every function here takes a
local path rather than an entity id.

WAV is handled with the standard library so the common case needs no extra
dependency. Anything else needs ``soundfile`` (``pip install soundfile``),
which is an optional install.
"""

import wave
from typing import Optional, Tuple

import numpy as np


class AudioFileInfo:
    """Shape of a recording as stored in the file.

    Not to be confused with :class:`~supervisely.api.audio_api.AudioInfo`,
    which describes a recording as the *platform* sees it.

    :param sample_rate: Samples per second.
    :param sample_count: Samples per channel.
    :param channels: Number of channels.
    """

    def __init__(self, sample_rate: int, sample_count: int, channels: int):
        self.sample_rate = sample_rate
        self.sample_count = sample_count
        self.channels = channels

    @property
    def duration_seconds(self) -> float:
        return self.sample_count / float(self.sample_rate)

    def __repr__(self) -> str:
        return (
            f"AudioFileInfo(sample_rate={self.sample_rate}, sample_count={self.sample_count}, "
            f"channels={self.channels}, duration={self.duration_seconds:.3f}s)"
        )


def _read_wav(path: str) -> Tuple[np.ndarray, int]:
    with wave.open(path, "rb") as w:
        channels = w.getnchannels()
        width = w.getsampwidth()
        rate = w.getframerate()
        raw = w.readframes(w.getnframes())

    if width == 1:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif width == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 4:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    elif width == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        packed = (b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)).astype(np.int32)
        packed = np.where(packed >= 1 << 23, packed - (1 << 24), packed)
        data = packed.astype(np.float32) / 8388608.0
    else:
        raise ValueError(f"unsupported WAV sample width: {width} bytes")

    if channels > 1:
        data = data.reshape(-1, channels)
    else:
        data = data.reshape(-1, 1)
    return data, rate


def read_audio(path: str) -> Tuple[np.ndarray, int]:
    """Decode an audio file to float32 samples in ``[-1, 1]``.

    :param path: Local path to the audio file.
    :return: ``(samples, sample_rate)`` where ``samples`` has shape
        ``(sample_count, channels)``.

    WAV is decoded with the standard library. Other formats require
    ``soundfile``; a clear :class:`ImportError` is raised if it is missing.
    """
    if path.lower().endswith(".wav"):
        try:
            return _read_wav(path)
        except wave.Error:
            pass  # compressed payload in a .wav container -- fall through

    try:
        import soundfile  # noqa: PLC0415
    except ImportError:
        raise ImportError(
            f"reading {path!r} needs the optional 'soundfile' package "
            "(pip install supervisely[audio]). Only uncompressed WAV is "
            "supported without it."
        )
    data, rate = soundfile.read(path, dtype="float32", always_2d=True)
    return data, rate


def get_audio_info(path: str) -> AudioFileInfo:
    """Read sample rate, sample count and channel count from the file header.

    The samples are never decoded: a one-hour WAV answers this in microseconds
    instead of materialising gigabytes of float32. Only a container that hides
    its layout in the payload -- a compressed stream in a ``.wav`` wrapper --
    falls back to a full decode.
    """
    if path.lower().endswith(".wav"):
        try:
            with wave.open(path, "rb") as w:
                return AudioFileInfo(
                    sample_rate=w.getframerate(),
                    sample_count=w.getnframes(),
                    channels=w.getnchannels(),
                )
        except wave.Error:
            pass  # compressed payload in a .wav container -- fall through

    try:
        import soundfile  # noqa: PLC0415
    except ImportError:
        data, rate = read_audio(path)
        return AudioFileInfo(sample_rate=rate, sample_count=data.shape[0], channels=data.shape[1])

    info = soundfile.info(path)
    return AudioFileInfo(
        sample_rate=info.samplerate, sample_count=info.frames, channels=info.channels
    )


def select_channel(samples: np.ndarray, channel: Optional[int]) -> np.ndarray:
    """Reduce a ``(n, channels)`` array to one mono track.

    :param channel: Zero-based channel index, or ``None`` to average all
        channels into a mixdown -- which is what the labeling tool shows when
        no channel is targeted.
    """
    if samples.ndim == 1:
        return samples
    if channel is None:
        return samples.mean(axis=1)
    if channel < 0 or channel >= samples.shape[1]:
        raise ValueError(
            f"channel {channel} out of range for a {samples.shape[1]}-channel recording"
        )
    return samples[:, channel]

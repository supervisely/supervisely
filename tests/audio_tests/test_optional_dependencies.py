# coding: utf-8
"""Audio support must not add a dependency for anyone who does not use it.

``soundfile`` is the ``supervisely[audio]`` extra and is imported only when a
non-WAV file is decoded. ``torch`` and ``tensorflow`` are never imported by the
SDK; the framework parity tests use them only when installed. Checked in a
fresh interpreter, because this process has already imported whatever the
other tests needed.
"""

import subprocess
import sys

import pytest

OPTIONAL = ("soundfile", "torch", "torchaudio", "tensorflow", "librosa")


def test_importing_audio_support_loads_no_optional_package():
    code = (
        "import sys\n"
        "import supervisely as sly\n"
        "import supervisely.audio, supervisely.api.audio_api\n"
        "import supervisely.project.audio_project\n"
        "import supervisely.convert.audio.audio_converter\n"
        "import supervisely.convert.audio.sly.sly_audio_converter\n"
        "from supervisely.convert.converter import ImportManager\n"
        f"loaded = [m for m in {OPTIONAL!r} if m in sys.modules]\n"
        "print(','.join(loaded))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout.strip()
    assert out == "", f"imported eagerly: {out}"


def test_wav_is_decoded_without_soundfile(tmp_path):
    """The common case, WAV, is read with the standard library, and a missing
    soundfile only matters for other formats."""
    code = (
        "import sys, wave, struct\n"
        "sys.modules['soundfile'] = None  # make any import of it fail\n"
        "sys.modules['av'] = None\n"
        "import supervisely as sly\n"
        f"p = {str(tmp_path / 'a.wav')!r}\n"
        "w = wave.open(p, 'wb'); w.setnchannels(1); w.setsampwidth(2); w.setframerate(8000)\n"
        "w.writeframes(struct.pack('<100h', *range(100))); w.close()\n"
        "samples, rate = sly.audio.read_audio(p)\n"
        "assert samples.shape == (100, 1) and rate == 8000\n"
        "assert sly.audio.get_audio_info(p).sample_count == 100\n"
        "try:\n"
        f"    sly.audio.read_audio({str(tmp_path / 'a.flac')!r})\n"
        "except ImportError as e:\n"
        "    assert 'supervisely[audio]' in str(e)\n"
        "else:\n"
        "    raise AssertionError('expected ImportError')\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def _write_m4a(path, rate=48000, seconds=1.0):
    """Stereo AAC in MP4: 440 Hz on the left, 1000 Hz on the right."""
    import numpy as np

    av = pytest.importorskip("av")
    n = int(rate * seconds)
    t = np.arange(n) / rate
    planar = np.stack(
        [0.3 * np.sin(2 * np.pi * 440 * t), 0.3 * np.sin(2 * np.pi * 1000 * t)]
    ).astype(np.float32)
    with av.open(str(path), "w", format="mp4") as out:
        stream = out.add_stream("aac", rate=rate, layout="stereo")
        step = 1024
        for i in range(0, n, step):
            frame = av.AudioFrame.from_ndarray(
                np.ascontiguousarray(planar[:, i : i + step]), format="fltp", layout="stereo"
            )
            frame.sample_rate = rate
            for packet in stream.encode(frame):
                out.mux(packet)
        for packet in stream.encode(None):
            out.mux(packet)
    return n


def test_m4a_is_decoded_through_av(tmp_path):
    """libsndfile cannot open MP4, so M4A goes through PyAV -- whether or not
    soundfile is installed."""
    import numpy as np

    import supervisely as sly

    path = str(tmp_path / "stereo.m4a")
    n = _write_m4a(path)

    samples, rate = sly.audio.read_audio(path)
    assert rate == 48000
    assert samples.dtype == np.float32 and samples.shape[1] == 2
    # AAC works in 1024-sample frames, plus 1024 priming samples when the file
    # has no edit list -- which a file muxed by PyAV does not.
    assert abs(samples.shape[0] - n) <= 2048

    for channel, tone in ((0, 440), (1, 1000)):
        spectrum = np.abs(np.fft.rfft(samples[:, channel]))
        peak_hz = np.argmax(spectrum) * rate / samples.shape[0]
        assert abs(peak_hz - tone) < 5

    info = sly.audio.get_audio_info(path)
    assert (info.sample_rate, info.channels, info.sample_count) == (48000, 2, samples.shape[0])


def test_m4a_without_av_names_the_extra(tmp_path):
    path = str(tmp_path / "stereo.m4a")
    _write_m4a(path)
    code = (
        "import sys\n"
        "sys.modules['av'] = None\n"
        "import supervisely as sly\n"
        "try:\n"
        f"    sly.audio.read_audio({path!r})\n"
        "except ImportError as e:\n"
        "    assert 'supervisely[audio]' in str(e), e\n"
        "else:\n"
        "    raise AssertionError('expected ImportError')\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)

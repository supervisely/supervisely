"""ffmpeg call of the video converter that re-encodes non-.mp4 videos on import.

Without a thread count ffmpeg and x264 size their threads from the host's CPU count, not the
container's CPU limit, and a 4K source then needed more than a 4 GB pod. The call now takes its
threads from the cgroup CPU quota (capped), writes 8-bit 4:2:0 where the size allows it, and a
failed or killed ffmpeg raises instead of leaving a truncated file to be uploaded.
"""

import os
import shutil
import subprocess

import pytest

import supervisely.convert.video.video_converter as vc
from supervisely.convert.video.video_converter import VideoConverter


@pytest.fixture
def cgroup(tmp_path, monkeypatch):
    """Point the converter at fake cgroup files; returns a writer for them."""
    v2 = tmp_path / "v2" / "cpu.max"
    v1 = tmp_path / "v1"
    monkeypatch.setattr(vc, "_CGROUP_V2_CPU_MAX", str(v2))
    monkeypatch.setattr(vc, "_CGROUP_V1_CPU_DIRS", (str(v1),))

    def write(v2_text=None, v1_quota=None, v1_period=100000):
        if v2_text is not None:
            v2.parent.mkdir(exist_ok=True)
            v2.write_text(v2_text)
        if v1_quota is not None:
            v1.mkdir(exist_ok=True)
            (v1 / "cpu.cfs_quota_us").write_text(f"{v1_quota}\n")
            (v1 / "cpu.cfs_period_us").write_text(f"{v1_period}\n")

    return write


@pytest.mark.parametrize(
    "cpu_max, threads",
    [
        ("200000 100000\n", 2),
        ("50000 100000\n", 1),
        ("150000 100000\n", 2),
        ("1600000 100000\n", 4),
        ("max 100000\n", 2),
        ("garbage\n", 2),
        ("100000 0\n", 2),
    ],
)
def test_threads_from_cgroup_v2(cgroup, cpu_max, threads):
    cgroup(v2_text=cpu_max)
    assert vc._get_ffmpeg_threads() == threads


@pytest.mark.parametrize("quota, threads", [(300000, 3), (-1, 2), (2400000, 4)])
def test_threads_from_cgroup_v1(cgroup, quota, threads):
    cgroup(v1_quota=quota)
    assert vc._get_ffmpeg_threads() == threads


def test_threads_without_cgroup(cgroup):
    assert vc._get_ffmpeg_threads() == 2


def _converter(tmp_path):
    return VideoConverter(str(tmp_path), None, False)


def _meta(width, height):
    return {"streams": [{"codecType": "video", "width": width, "height": height}]}


def test_convert_args(tmp_path, monkeypatch, cgroup):
    cgroup(v2_text="200000 100000")
    calls = []
    monkeypatch.setattr(vc.subprocess, "call", lambda args: calls.append(args) or 0)
    conv = _converter(tmp_path)

    conv._convert("in.mov", "out.mp4", True, False, video_meta=_meta(3840, 2160))
    args = calls[-1]
    assert args.index("-threads") < args.index("-i")
    assert args[args.index("-c:v") + 1 :][:2] == ["libx264", "-threads"]
    assert args[args.index("-pix_fmt") + 1] == "yuv420p"
    assert args[-1] == "out.mp4"

    # 4:2:0 cannot hold an odd size; leave the pixel format to ffmpeg as before
    conv._convert("in.avi", "out.mp4", True, False, video_meta=_meta(641, 481))
    assert "-pix_fmt" not in calls[-1]

    # rotation swaps width and height; parity is checked on the stored frame
    rotated = {
        "streams": [
            {
                "codecType": "video",
                "width": 2160,
                "height": 3840,
                "originalWidth": 3840,
                "originalHeight": 2160,
            }
        ]
    }
    conv._convert("in.mov", "out.mp4", True, False, video_meta=rotated)
    assert "-pix_fmt" in calls[-1]

    # metadata could not be read: the size may be odd, so do not force 4:2:0
    conv._convert("in.mkv", "out.mp4", True, False, video_meta=None)
    assert "-pix_fmt" not in calls[-1]
    assert calls[-1].index("-threads") < calls[-1].index("-i")


def test_failed_ffmpeg_removes_partial_output_and_keeps_source(tmp_path, monkeypatch):
    src = tmp_path / "clip.mov"
    src.write_bytes(b"\0" * 64)
    out = tmp_path / "clip_h264.mp4"

    def killed_partway(args):
        out.write_bytes(b"\0" * 48)  # what a killed ffmpeg leaves: no moov atom
        return -9

    monkeypatch.setattr(vc.subprocess, "call", killed_partway)
    monkeypatch.setattr(vc, "get_info", lambda path: _meta(3840, 2160))
    conv = _converter(tmp_path)

    with pytest.raises(RuntimeError, match="clip.mov.*code -9"):
        conv.convert_to_mp4_if_needed(str(src))
    assert src.exists()
    assert not out.exists()


def test_failed_ffmpeg_keeps_existing_file_it_did_not_write(tmp_path, monkeypatch):
    existing = tmp_path / "clip_h264.mp4"  # e.g. a user's own video next to clip.mov
    existing.write_bytes(b"user video")
    monkeypatch.setattr(vc.subprocess, "call", lambda args: 1)  # fails before opening output
    conv = _converter(tmp_path)

    with pytest.raises(RuntimeError):
        conv._convert(str(tmp_path / "clip.mov"), str(existing), True, False)
    assert existing.read_bytes() == b"user video"


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")
@pytest.mark.parametrize(
    "size, pix_fmt_in, pix_fmt_out",
    [("640x480", "yuv420p10le", "yuv420p"), ("641x481", "yuv444p", "yuv444p")],
)
def test_real_ffmpeg(tmp_path, size, pix_fmt_in, pix_fmt_out):
    src = tmp_path / "clip.mov"
    subprocess.run(
        [
            "ffmpeg", "-nostdin", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", f"testsrc=size={size}:rate=30:duration=1",
            "-c:v", "libx264", "-pix_fmt", pix_fmt_in, str(src),
        ],
        check=True,
    )  # fmt: skip
    conv = _converter(tmp_path)
    name, out = conv.convert_to_mp4_if_needed(str(src))
    assert name == "clip.mp4"
    assert not src.exists()
    probe = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,pix_fmt,width,height,nb_frames",
            "-of", "default=nw=1", out,
        ],
        capture_output=True, text=True, check=True,
    ).stdout  # fmt: skip
    stream = dict(line.split("=", 1) for line in probe.split())
    width, height = size.split("x")
    assert stream == {
        "codec_name": "h264",
        "pix_fmt": pix_fmt_out,
        "width": width,
        "height": height,
        "nb_frames": "30",
    }
    assert os.path.getsize(out) > 0


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")
@pytest.mark.parametrize(
    "size, pix_fmt_in, pix_fmt_out",
    [("640x480", "yuv420p10le", "yuv420p"), ("641x481", "yuv444p", "yuv444p")],
)
def test_real_ffmpeg_without_metadata(tmp_path, monkeypatch, size, pix_fmt_in, pix_fmt_out):
    """get_info can fail (no duration in a piped .mkv, a killed ffprobe); the size is then
    read from the header so that 10-bit still becomes 8-bit and odd sizes still convert."""
    src = tmp_path / "clip.mov"
    subprocess.run(
        [
            "ffmpeg", "-nostdin", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", f"testsrc=size={size}:rate=30:duration=1",
            "-c:v", "libx264", "-pix_fmt", pix_fmt_in, str(src),
        ],
        check=True,
    )  # fmt: skip

    def unreadable(path):
        raise KeyError("duration")

    monkeypatch.setattr(vc, "get_info", unreadable)
    _, out = _converter(tmp_path).convert_to_mp4_if_needed(str(src))
    probe = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=pix_fmt,width,height", "-of", "default=nw=1", out,
        ],
        capture_output=True, text=True, check=True,
    ).stdout  # fmt: skip
    width, height = size.split("x")
    assert dict(line.split("=", 1) for line in probe.split()) == {
        "pix_fmt": pix_fmt_out,
        "width": width,
        "height": height,
    }
    assert not src.exists()

"""Tests for the upload ``progress_cb`` contract.

Uploads report through a ``MultipartEncoderMonitor``. Callbacks in the wild come in two
shapes: delta style ones that take a byte increment (``tqdm.update``,
``Progress.iters_done_report``, plain lambdas) and monitor style ones written back when the
SDK handed the monitor over (they read ``monitor.bytes_read`` / ``monitor.len``). Both are
used inside the SDK itself, so the adapter has to serve both.

The tests drive a real ``MultipartEncoder`` and never touch the network.
"""

import copy
import gc
import io
import pickle
import weakref
from functools import partial
from unittest import mock

import pytest
from requests_toolbelt import MultipartEncoder, MultipartEncoderMonitor
from tqdm import tqdm  # importing supervisely rebinds this name to tqdm_sly
from tqdm.std import tqdm as vanilla_tqdm

from supervisely.api.video.video_api import VideoApi
from supervisely.task.progress import (
    Progress,
    UploadProgressDelta,
    build_multipart_monitor_callback,
    tqdm_sly,
)

PAYLOAD = b"x" * 40000
CHUNK = 8192


def make_monitor(progress_cb, files=1):
    fields = {
        f"{idx}-file": (str(idx), io.BytesIO(PAYLOAD), "application/octet-stream")
        for idx in range(files)
    }
    encoder = MultipartEncoder(fields=fields)
    callback = build_multipart_monitor_callback(progress_cb)
    return MultipartEncoderMonitor(encoder, callback) if callback else encoder


def drain(monitor):
    """Read the whole body, the way the HTTP adapter does."""
    while monitor.read(CHUNK):
        pass
    return monitor.len


# --------------------------------------------------------------------------------------
# delta style callbacks
# --------------------------------------------------------------------------------------
def test_vanilla_tqdm_is_not_callable_but_still_works():
    """A tqdm imported before supervisely stays the original class, which is not callable."""
    bar = vanilla_tqdm(total=None, file=io.StringIO())
    assert not callable(bar)
    assert not hasattr(bar, "get_partial")
    total = drain(make_monitor(bar))
    assert bar.n == total


def test_patched_tqdm_goes_through_its_monitor_hook():
    """`supervisely/__init__.py` rebinds tqdm.tqdm to tqdm_sly, so user code that imports
    tqdm after supervisely gets the monitor aware one."""
    assert tqdm is tqdm_sly
    bar = tqdm(desc="uploading", total=len(PAYLOAD), file=io.StringIO())
    total = drain(make_monitor(bar))
    assert bar.n == total


def test_tqdm_update_bound_method():
    bar = vanilla_tqdm(total=None, file=io.StringIO())
    total = drain(make_monitor(bar.update))
    assert bar.n == total


def test_plain_lambda_gets_increments():
    seen = []
    total = drain(make_monitor(lambda count: seen.append(int(count))))
    assert sum(seen) == total
    assert all(delta >= 0 for delta in seen)


def test_progress_iters_done_report():
    progress = Progress("uploading", total_cnt=len(PAYLOAD) * 4, is_size=True)
    total = drain(make_monitor(progress.iters_done_report))
    assert progress.current == total


def test_bare_progress_object():
    """`Progress` is not callable, but it reports increments through iters_done_report."""
    progress = Progress("uploading", total_cnt=len(PAYLOAD) * 4, is_size=True)
    assert not callable(progress)
    total = drain(make_monitor(progress))
    assert progress.current == total


def test_object_with_update_only():
    class Bar:
        def __init__(self):
            self.n = 0

        def update(self, count):
            self.n += int(count)

    bar = Bar()
    total = drain(make_monitor(bar))
    assert bar.n == total


def test_callable_object_delta_style():
    class Counter:
        def __init__(self):
            self.total = 0

        def __call__(self, count):
            self.total += int(count)

    counter = Counter()
    total = drain(make_monitor(counter))
    assert counter.total == total


# --------------------------------------------------------------------------------------
# monitor style callbacks: what apps and the SDK itself were written against
# --------------------------------------------------------------------------------------
def test_monitor_style_function():
    seen = {}

    def on_progress(monitor):
        seen["bytes_read"] = monitor.bytes_read
        seen["len"] = monitor.len

    total = drain(make_monitor(on_progress))
    assert seen["bytes_read"] == total
    assert seen["len"] == total


def test_monitor_style_partial():
    """The shape used by the yolov8 train app: partial() over a monitor style function."""
    seen = []

    def upload_monitor(monitor, tag):
        seen.append((tag, monitor.bytes_read, monitor.len))

    total = drain(make_monitor(partial(upload_monitor, tag="artifacts")))
    assert seen[-1] == ("artifacts", total, total)
    assert [read for _, read, _ in seen] == sorted(read for _, read, _ in seen)


def test_monitor_style_callable_object():
    class Reporter:
        def __init__(self):
            self.last = None

        def __call__(self, monitor):
            self.last = (monitor.bytes_read, monitor.len)

    reporter = Reporter()
    total = drain(make_monitor(reporter))
    assert reporter.last == (total, total)


def test_sly_output_set_download_shape():
    """`sly.output.set_download` passes a lambda that reads len first, then bytes_read."""
    state = {}

    def _print_progress(monitor, holder):
        if not holder:
            holder.append(Progress("Uploading", total_cnt=monitor.len, is_size=True))
        holder[0].set_current_value(monitor.bytes_read)
        state["progress"] = holder[0]

    holder = []
    total = drain(make_monitor(lambda m: _print_progress(m, holder)))
    assert state["progress"].current == total
    assert state["progress"].total == total


# --------------------------------------------------------------------------------------
# monitor aware progress objects report on their own
# --------------------------------------------------------------------------------------
def test_tqdm_sly_uses_its_own_monitor_hook():
    bar = tqdm_sly(desc="uploading", total=len(PAYLOAD), unit="B", file=io.StringIO())
    assert callable(bar.get_partial())
    total = drain(make_monitor(bar))
    assert bar.n == total


def test_get_partial_wins_over_being_callable():
    calls = {"partial": 0, "call": 0}

    class MonitorAware:
        def __call__(self, count):
            calls["call"] += 1

        def get_partial(self):
            def hook(monitor):
                calls["partial"] += 1

            return hook

    drain(make_monitor(MonitorAware()))
    assert calls["partial"] > 0
    assert calls["call"] == 0


def test_non_callable_get_partial_is_ignored():
    """An attribute that merely happens to be named get_partial must not be called."""
    seen = []

    class Weird:
        get_partial = "not a method"

        def __call__(self, count):
            seen.append(int(count))

    total = drain(make_monitor(Weird()))
    assert sum(seen) == total


# --------------------------------------------------------------------------------------
# the value handed to delta style callbacks
# --------------------------------------------------------------------------------------
def test_value_is_an_int_carrying_the_monitor():
    values = []
    monitor = make_monitor(values.append)
    total = drain(monitor)

    assert all(isinstance(value, int) for value in values)
    assert all(isinstance(value, UploadProgressDelta) for value in values)
    assert sum(values) == total
    assert values[-1].bytes_read == total
    assert values[-1].len == total
    assert values[-1].monitor is monitor
    # arbitrary monitor attributes are forwarded while the upload is running
    assert values[-1].encoder is monitor.encoder


def test_stored_increments_do_not_pin_the_monitor():
    """A callback that keeps what it was handed must not keep the encoder alive with it."""
    values = []
    monitor = make_monitor(values.append)
    drain(monitor)
    monitor_ref = weakref.ref(monitor)

    del monitor
    gc.collect()
    assert monitor_ref() is None, "stored increments pinned the multipart monitor"
    assert values[-1].monitor is None


def test_stored_increments_do_not_pin_the_uploaded_file(tmp_path):
    """The team files upload never closes its streams explicitly, so nothing the callback
    keeps may hold them open."""
    path = tmp_path / "payload.bin"
    path.write_bytes(PAYLOAD)
    handle = open(path, "rb")
    handle_ref = weakref.ref(handle)

    values = []
    encoder = MultipartEncoder(
        fields=[("file", ("payload.bin", handle, "application/octet-stream"))]
    )
    monitor = MultipartEncoderMonitor(encoder, build_multipart_monitor_callback(values.append))
    drain(monitor)

    del encoder, monitor, handle
    gc.collect()
    assert values, "no progress was reported"
    assert handle_ref() is None, "stored increments held the uploaded file open"


def test_numbers_outlive_the_monitor():
    values = []
    monitor = make_monitor(values.append)
    total = drain(monitor)

    del monitor
    gc.collect()
    assert values[-1].bytes_read == total
    assert values[-1].len == total
    with pytest.raises(AttributeError, match="only while the upload is running"):
        values[-1].encoder


@pytest.mark.parametrize("protocol", range(pickle.HIGHEST_PROTOCOL + 1))
def test_value_pickles_as_a_plain_int(protocol):
    """A monitor cannot cross a process boundary, but the increment used to be a plain int
    and code that puts it on a multiprocessing queue must keep working."""
    value = UploadProgressDelta(7, _FakeMonitor(bytes_read=7, len=10))
    restored = pickle.loads(pickle.dumps(value, protocol=protocol))
    assert restored == 7
    assert type(restored) is int


def test_value_copies_as_a_plain_int():
    value = UploadProgressDelta(7, _FakeMonitor(bytes_read=7, len=10))
    assert copy.copy(value) == 7
    assert copy.deepcopy(value) == 7


def test_value_behaves_like_a_number():
    value = UploadProgressDelta(7, _FakeMonitor(bytes_read=7, len=10))
    assert value + 1 == 8
    assert value * 2 == 14
    assert float(value) == 7.0
    assert f"{value}" == "7"
    assert value < 8 and value > 6
    assert sum([value, value]) == 14


def test_unknown_attribute_raises_attribute_error():
    value = UploadProgressDelta(1, _FakeMonitor(bytes_read=1, len=2))
    with pytest.raises(AttributeError):
        value.definitely_not_there


class _FakeMonitor:
    def __init__(self, bytes_read, len):  # noqa: A002 - mirrors the toolbelt attribute
        self.bytes_read = bytes_read
        self.len = len


# --------------------------------------------------------------------------------------
# increments, ordering and multi file uploads
# --------------------------------------------------------------------------------------
def test_increments_sum_up_and_never_go_backwards():
    deltas, cumulative = [], []

    def on_progress(value):
        deltas.append(int(value))
        cumulative.append(value.bytes_read)

    total = drain(make_monitor(on_progress))
    assert sum(deltas) == total
    assert cumulative == sorted(cumulative)
    assert cumulative[-1] == total


def test_bulk_upload_reports_across_all_files():
    values = []
    total = drain(make_monitor(values.append, files=3))
    assert sum(values) == total
    assert total > 3 * len(PAYLOAD)  # payload plus multipart overhead


# --------------------------------------------------------------------------------------
# nothing to report to, and bad input
# --------------------------------------------------------------------------------------
def test_none_means_no_monitor():
    assert build_multipart_monitor_callback(None) is None


@pytest.mark.parametrize("bad", [42, "a string", object(), ["list"]])
def test_non_callable_progress_cb_is_rejected(bad):
    with pytest.raises(TypeError, match="progress_cb must be callable"):
        build_multipart_monitor_callback(bad)


def test_callback_is_independent_per_upload():
    """Two uploads with the same callback must not share the reported offset."""
    values = []
    drain(make_monitor(values.append))
    first = sum(values)

    values.clear()
    drain(make_monitor(values.append))
    assert sum(values) == first


def test_callback_errors_are_not_swallowed():
    """A broken callback must fail loudly instead of corrupting the upload silently."""

    def boom(value):
        raise RuntimeError("callback is broken")

    with pytest.raises(RuntimeError, match="callback is broken"):
        drain(make_monitor(boom))


def test_widget_progress_shape_is_the_get_partial_one():
    """The SlyTqdm widget (CustomTqdm) cannot be built outside an app, but it reports the
    same way tqdm_sly does: through get_partial(), which is covered above."""
    from supervisely.app.widgets.sly_tqdm.sly_tqdm import CustomTqdm

    assert callable(getattr(CustomTqdm, "get_partial", None))
    assert callable(getattr(CustomTqdm, "_progress_monitor", None))


def test_video_converter_progress_follows_the_upload():
    """The converters share one byte progress bar for the whole upload, and the callback is
    handed an increment, so it has to report increments rather than set an absolute value."""
    from supervisely.convert.video import video_converter

    created = []

    class SpyProgress(video_converter.Progress):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            created.append(self)

    original = video_converter.Progress
    video_converter.Progress = SpyProgress
    try:
        # self is unused by the helper, and building a converter needs a real dataset
        callback = video_converter.VideoConverter._get_video_upload_progress(None)
        total = drain(make_monitor(callback))
    finally:
        video_converter.Progress = original

    assert created, "no Progress was created"
    assert created[0].current == total


def test_progress_set_current_value_keeps_getting_the_running_total():
    """`Progress.set_current_value` advances by `value - current`, so increments would leave it
    stuck near the first chunk. Callers hand it to uploads directly, so it keeps the totals."""
    progress = Progress("uploading", total_cnt=len(PAYLOAD) * 4, is_size=True)
    total = drain(make_monitor(progress.set_current_value))
    assert progress.current == total


def test_set_current_value_of_a_subclass_is_recognised():
    """The check is on the function, not on the name, so subclasses are served too."""

    class MyProgress(Progress):
        pass

    progress = MyProgress("uploading", total_cnt=len(PAYLOAD) * 4, is_size=True)
    total = drain(make_monitor(progress.set_current_value))
    assert progress.current == total


def test_a_foreign_set_current_value_still_gets_increments():
    """Only Progress.set_current_value is special cased; a method that merely shares the name
    belongs to the delta contract like every other callable."""
    seen = []

    class Own:
        def set_current_value(self, value):
            seen.append(value)

    total = drain(make_monitor(Own().set_current_value))
    assert sum(seen) == total, "a foreign callback must keep receiving increments"


def test_video_upload_path_bar_follows_the_upload():
    """VideoApi.upload_path(item_progress=True) used to bind Progress.set_current_value, which was
    right while the upload fed it monitor.bytes_read and wrong once it fed increments. Checked
    while the upload runs: the handler tops the bar up to total when it returns, either way."""
    seen = {}

    def fake_upload_paths(self, dataset_id, names, paths, item_progress=None, **kwargs):
        seen["uploaded"] = drain(make_monitor(item_progress))
        seen["on_the_bar"] = item_progress.__self__.current
        return [object()]

    api = VideoApi.__new__(VideoApi)
    with mock.patch.object(VideoApi, "upload_paths", fake_upload_paths), mock.patch(
        "supervisely.api.video.video_api.get_file_size", return_value=len(PAYLOAD) * 4
    ):
        api.upload_path(dataset_id=1, name="a.mp4", path="/tmp/a.mp4", item_progress=True)

    assert seen["on_the_bar"] == seen["uploaded"], (
        f"{seen['on_the_bar']} on the bar while {seen['uploaded']} bytes were uploaded"
    )


def test_video_converter_bar_adds_up_across_videos():
    """A large upload sends one video per request, so every request brings its own monitor whose
    bytes_read starts at zero. Reporting the running total left the bar stuck on the first video."""
    from supervisely.convert.video import video_converter

    created = []

    class SpyProgress(video_converter.Progress):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            created.append(self)

    original = video_converter.Progress
    video_converter.Progress = SpyProgress
    try:
        callback = video_converter.VideoConverter._get_video_upload_progress(None)
        # a fresh adapter per request, the way _upload_uniq_videos_single_req builds it
        uploaded = sum(drain(make_monitor(callback)) for _ in range(2))
    finally:
        video_converter.Progress = original

    assert len(created) == 1, "the bar must be shared by every video of the upload"
    assert created[0].current == uploaded, f"{created[0].current} on the bar, {uploaded} uploaded"

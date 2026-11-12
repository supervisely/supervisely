"""
Security tests for two fixes that both turn attacker-controlled bytes into a
filesystem / code-execution primitive when a model is deployed as root in a Linux
container.

F4 - download filename taken from a remote header
    ``supervisely._utils.get_filename_from_headers`` returned the raw
    ``Content-Disposition`` ``filename`` sent by the remote server, and callers do
    ``os.path.join(model_dir, name)`` (``nn/inference/inference.py`` ~865,
    ``nn/training/train_app.py`` ~1766).  A server answering a model-download request
    with ``filename="../../evil.sh"`` (or an absolute path, or a Windows ``..\\..\\``
    name) therefore chose where the downloaded bytes landed.  The fix keeps only the
    basename and rejects ``""`` / ``"."`` / ``".."``.

F5 - unsafe checkpoint load in the OSNet re-ID tracker
    ``OsnetReIDModel.load_pretrained_weights`` and ``init_pretrained_weights``
    (``nn/tracker/botsort/osnet_reid/osnet_reid_interface.py`` and ``osnet.py``)
    called ``torch.load`` on a caller-supplied ``reid_weights`` path.  A pickled
    checkpoint whose ``__reduce__`` runs arbitrary code is executed during the load.
    The fix passes ``weights_only=True`` at both call sites.

    IMPORTANT torch-version note (see the module-level comment on the F5 tests):
    PyTorch >= 2.6 already defaults ``torch.load`` to ``weights_only=True``.  This
    environment runs torch 2.14, so the *unfixed* BASE code is already protected by
    torch's own default - a plain exploit therefore passes on BOTH trees and proves
    nothing (requirement 4).  To genuinely demonstrate the bug the BASE code carries,
    the distinguishing F5 tests emulate the pre-2.6 default (``weights_only=False``
    when the caller omits the argument) so that the *real* ``load_pretrained_weights``
    / ``init_pretrained_weights`` code is exercised unchanged and the latent flaw the
    fix removes becomes observable again.  This emulates a torch version, it does not
    patch the SDK.  A second pair of tests runs the real attack against the installed
    torch and documents that it is blocked on both trees.  Finally, a pair of
    torch-version-independent spy tests assert that the *real* code explicitly passes
    ``weights_only=True`` to ``torch.load`` - the single observable behaviour the fix
    adds - which fails on BASE (the keyword is absent there) whatever the installed
    torch default is.

No real network is used.  Everything an exploit could write is aimed inside
``tmp_path``.
"""

import os
import sys
import types
from pathlib import Path

import pytest
import torch
import torch.nn as nn

import supervisely._utils as sly_utils
from supervisely._utils import get_filename_from_headers
from supervisely.nn.tracker.botsort.osnet_reid import osnet as osnet_mod
from supervisely.nn.tracker.botsort.osnet_reid import osnet_reid_interface as reid_iface
from supervisely.nn.tracker.botsort.osnet_reid.osnet import osnet_x1_0

# ===========================================================================
# F4 - get_filename_from_headers
# ===========================================================================


class _Resp:
    """Minimal stand-in for a ``requests`` response."""

    def __init__(self, status_code=200, content_disposition=None):
        self.status_code = status_code
        self.headers = {}
        if content_disposition is not None:
            self.headers["Content-Disposition"] = content_disposition


@pytest.fixture()
def mock_requests(monkeypatch):
    """Replace the network calls inside ``get_filename_from_headers``.

    The caller sets ``state["head"]`` / ``state["get"]`` to the responses the fake
    ``requests.head`` / ``requests.get`` should return.
    """
    state = {"head": _Resp(200), "get": _Resp(200)}

    def fake_head(url, *args, **kwargs):
        return state["head"]

    def fake_get(url, *args, **kwargs):
        return state["get"]

    monkeypatch.setattr(sly_utils.requests, "head", fake_head)
    monkeypatch.setattr(sly_utils.requests, "get", fake_get)
    return state


def _assert_plain_and_contained(name: str, model_dir: str) -> None:
    """The returned name must be a bare file name that cannot escape ``model_dir``."""
    assert name is not None
    assert name not in ("", ".", "..")
    assert "/" not in name
    assert "\\" not in name
    assert os.path.basename(name) == name
    base = os.path.abspath(model_dir)
    joined = os.path.abspath(os.path.join(base, name))
    assert os.path.commonpath([base, joined]) == base
    assert joined != base


# --- exploit tests: FAIL on BASE, PASS on FIXED ---------------------------
@pytest.mark.parametrize(
    "disposition",
    [
        'attachment; filename="../../evil.sh"',
        'attachment; filename="../../../etc/cron.d/evil"',
        'attachment; filename="/etc/cron.d/evil"',
        'attachment; filename="..\\\\..\\\\evil"',  # Windows-style traversal
        'attachment; filename=".."',
    ],
    ids=["dotdot-rel", "deep-rel", "absolute", "windows-backslash", "bare-dotdot"],
)
def test_get_filename_from_headers_strips_malicious_path(mock_requests, tmp_path, disposition):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    mock_requests["head"] = _Resp(200, disposition)
    mock_requests["get"] = _Resp(200, disposition)

    # The URL ends with a legit segment; for the ``".."`` case the fix falls back to it.
    name = get_filename_from_headers("https://host.invalid/files/legit_name.bin")

    _assert_plain_and_contained(name, str(model_dir))


# --- regression tests: PASS on BOTH trees ---------------------------------
def test_get_filename_from_headers_keeps_normal_name(mock_requests, tmp_path):
    mock_requests["head"] = _Resp(200, 'attachment; filename="model_final.pth"')

    name = get_filename_from_headers("https://host.invalid/download")

    assert name == "model_final.pth"
    _assert_plain_and_contained(name, str(tmp_path))


def test_get_filename_from_headers_falls_back_to_url_segment(mock_requests):
    # head has no Content-Disposition -> code issues a GET, which also has none
    mock_requests["head"] = _Resp(200)
    mock_requests["get"] = _Resp(200)

    name = get_filename_from_headers("https://host.invalid/dir/weights_v2.pt")

    assert name == "weights_v2.pt"


def test_get_filename_from_headers_bad_head_status_uses_get(mock_requests):
    # head >= 400 forces a GET; the GET carries a normal filename
    mock_requests["head"] = _Resp(404)
    mock_requests["get"] = _Resp(200, 'attachment; filename="from_get.onnx"')

    name = get_filename_from_headers("https://host.invalid/x")

    assert name == "from_get.onnx"


def test_get_filename_from_headers_url_without_segment_uses_default(mock_requests):
    mock_requests["head"] = _Resp(200)
    mock_requests["get"] = _Resp(200)

    name = get_filename_from_headers("https://host.invalid/")

    assert name == "downloaded_file"


# ===========================================================================
# F5 - unsafe checkpoint load (torch.load weights_only)
# ===========================================================================

MARKER_NAME = "pwned_by_pickle.txt"


def _drop_marker(path):
    """Payload: proof that arbitrary code ran during ``torch.load``."""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("arbitrary code executed during torch.load")
    return 0


class _MaliciousCheckpoint:
    """Pickles to a call of :func:`_drop_marker` - the classic torch.load RCE."""

    def __init__(self, marker_path):
        self.marker_path = marker_path

    def __reduce__(self):
        return (_drop_marker, (self.marker_path,))


def _write_malicious_checkpoint(dst: Path, marker_path: Path) -> Path:
    # torch.save pickles the object but does NOT run __reduce__'s callable;
    # the payload only fires when the file is later loaded unsafely.
    torch.save({"state_dict": _MaliciousCheckpoint(str(marker_path))}, str(dst))
    return dst


def _reid_model_stub() -> "reid_iface.OsnetReIDModel":
    """Cheapest possible real OsnetReIDModel: no network, no heavy backbone build.

    The malicious payload fires (or is rejected) on the ``torch.load`` line before
    ``self.model`` is touched, so a tiny real module is enough to exercise the real
    ``load_pretrained_weights`` code path.
    """
    model = reid_iface.OsnetReIDModel.__new__(reid_iface.OsnetReIDModel)
    model.device = torch.device("cpu")
    model.model = nn.Linear(2, 2)
    return model


@pytest.fixture()
def legacy_torch_load(monkeypatch):
    """Emulate pre-2.6 torch: when a caller omits ``weights_only``, default to False.

    This restores the environment in which the unfixed SDK code is actually
    exploitable, without changing a single line of the SDK.
    """
    real_load = torch.load

    def legacy(f, *args, **kwargs):
        if kwargs.get("weights_only", None) is None:
            kwargs["weights_only"] = False
        return real_load(f, *args, **kwargs)

    monkeypatch.setattr(torch, "load", legacy)
    return real_load


# --- exploit tests (legacy-torch emulation): FAIL on BASE, PASS on FIXED ---
def test_reid_load_pretrained_weights_blocks_pickle_rce(tmp_path, legacy_torch_load):
    marker = tmp_path / MARKER_NAME
    ckpt = _write_malicious_checkpoint(tmp_path / "checkpoint.pth", marker)
    assert not marker.exists(), "saving the checkpoint must not execute the payload"

    model = _reid_model_stub()

    with pytest.raises(Exception):
        model.load_pretrained_weights(ckpt)

    assert not marker.exists(), (
        "torch.load executed the pickle payload -> load_pretrained_weights did not "
        "pass weights_only=True"
    )


def test_osnet_init_pretrained_weights_blocks_pickle_rce(
    tmp_path, monkeypatch, legacy_torch_load
):
    # Emulate the pretrained-weights cache so the real function reaches torch.load
    # without any network / gdown download.
    monkeypatch.setenv("TORCH_HOME", str(tmp_path))
    fake_gdown = types.ModuleType("gdown")

    def _no_download(*args, **kwargs):
        raise AssertionError("init_pretrained_weights tried to hit the network")

    fake_gdown.download = _no_download
    monkeypatch.setitem(sys.modules, "gdown", fake_gdown)

    cache_dir = tmp_path / "checkpoints"
    cache_dir.mkdir()
    marker = tmp_path / MARKER_NAME
    _write_malicious_checkpoint(cache_dir / "osnet_x1_0_imagenet.pth", marker)
    assert not marker.exists()

    model = nn.Linear(2, 2)

    # The real function swallows key/size mismatches, so a successful-but-poisoned
    # load would not raise; the marker is therefore the decisive security check.
    raised = False
    try:
        osnet_mod.init_pretrained_weights(model, key="osnet_x1_0")
    except Exception:
        raised = True

    assert not marker.exists(), (
        "torch.load executed the pickle payload -> init_pretrained_weights did not "
        "pass weights_only=True"
    )
    # on the fixed code the unsafe pickle is rejected outright
    assert raised, "fixed code should reject the non-weights pickle"


# --- real attack against the installed torch: PASS on BOTH trees ----------
# torch >= 2.6 defaults weights_only=True, so even the unfixed code is protected
# here; these guard the end-to-end behaviour but do NOT distinguish the trees.
def test_reid_load_pretrained_weights_blocks_pickle_rce_installed_torch(tmp_path):
    marker = tmp_path / MARKER_NAME
    ckpt = _write_malicious_checkpoint(tmp_path / "checkpoint_real.pth", marker)

    model = _reid_model_stub()

    with pytest.raises(Exception):
        model.load_pretrained_weights(ckpt)

    assert not marker.exists()


# --- regression tests: a genuine checkpoint still loads (PASS on BOTH) -----
@pytest.fixture(scope="module")
def real_osnet():
    # pretrained=False -> no download; a real OSNet so the key/size matching runs.
    return osnet_x1_0(num_classes=1000, pretrained=False, loss="softmax")


def _fresh_model_with(real_osnet):
    model = reid_iface.OsnetReIDModel.__new__(reid_iface.OsnetReIDModel)
    model.device = torch.device("cpu")
    model.model = osnet_x1_0(num_classes=1000, pretrained=False, loss="softmax")
    return model


def test_reid_load_pretrained_weights_accepts_plain_state_dict(tmp_path, real_osnet):
    path = tmp_path / "good_plain.pth"
    torch.save(real_osnet.state_dict(), str(path))

    model = _fresh_model_with(real_osnet)
    model.load_pretrained_weights(path)  # must not raise

    # a genuine weight was actually copied in
    key = next(iter(real_osnet.state_dict()))
    assert torch.equal(model.model.state_dict()[key], real_osnet.state_dict()[key])


def test_reid_load_pretrained_weights_accepts_state_dict_wrapper(tmp_path, real_osnet):
    path = tmp_path / "good_wrapper.pth"
    torch.save({"state_dict": real_osnet.state_dict()}, str(path))

    model = _fresh_model_with(real_osnet)
    model.load_pretrained_weights(path)  # must not raise

    key = next(iter(real_osnet.state_dict()))
    assert torch.equal(model.model.state_dict()[key], real_osnet.state_dict()[key])


# --- torch-version-independent distinguishing tests: FAIL on BASE, PASS on FIXED ---
# These do not emulate any torch version.  They assert the one observable thing the
# fix adds - ``weights_only=True`` on the real ``torch.load`` call - by spying on
# ``torch.load`` while the real loading code runs against a genuine checkpoint.  On
# BASE the keyword is absent (the call leans on torch's default, unsafe on torch<2.6);
# on FIXED it is explicitly True.  The spy forwards to the real loader, so a legit
# state_dict still loads and nothing is faked.
def test_reid_load_pretrained_weights_passes_weights_only_true(
    tmp_path, real_osnet, monkeypatch
):
    path = tmp_path / "spy_plain.pth"
    torch.save(real_osnet.state_dict(), str(path))
    model = _fresh_model_with(real_osnet)

    seen = {}
    real_load = torch.load

    def spy(f, *args, **kwargs):
        seen["weights_only"] = kwargs.get("weights_only", "ABSENT")
        return real_load(f, *args, **kwargs)

    monkeypatch.setattr(torch, "load", spy)
    model.load_pretrained_weights(path)  # real code path, genuine checkpoint

    assert seen.get("weights_only") is True, (
        "load_pretrained_weights must pass weights_only=True to torch.load; "
        f"got {seen.get('weights_only')!r}"
    )


def test_osnet_init_pretrained_weights_passes_weights_only_true(
    tmp_path, real_osnet, monkeypatch
):
    monkeypatch.setenv("TORCH_HOME", str(tmp_path))

    fake_gdown = types.ModuleType("gdown")

    def _no_download(*args, **kwargs):
        raise AssertionError("init_pretrained_weights tried to hit the network")

    fake_gdown.download = _no_download
    monkeypatch.setitem(sys.modules, "gdown", fake_gdown)

    cache_dir = tmp_path / "checkpoints"
    cache_dir.mkdir()
    # a genuine (non-malicious) checkpoint in the cache, so the real load succeeds
    torch.save(real_osnet.state_dict(), str(cache_dir / "osnet_x1_0_imagenet.pth"))

    model = osnet_x1_0(num_classes=1000, pretrained=False, loss="softmax")

    seen = {}
    real_load = torch.load

    def spy(f, *args, **kwargs):
        seen["weights_only"] = kwargs.get("weights_only", "ABSENT")
        return real_load(f, *args, **kwargs)

    monkeypatch.setattr(torch, "load", spy)
    osnet_mod.init_pretrained_weights(model, key="osnet_x1_0")  # real code path

    assert seen.get("weights_only") is True, (
        "init_pretrained_weights must pass weights_only=True to torch.load; "
        f"got {seen.get('weights_only')!r}"
    )

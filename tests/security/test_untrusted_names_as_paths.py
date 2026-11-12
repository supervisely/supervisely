"""
Security tests for fix F7: names taken from untrusted content used as filesystem paths.

Two independent sinks, both reachable with attacker-controlled data when an inference
app runs as root inside a Linux container:

(a) ``Inference._extract_model_files_from_checkpoint`` (``nn/inference/inference.py``)
    wrote ``checkpoint["model_files"][key]["name"]`` (or the fallback ``"<key>.txt"``)
    joined to ``self.model_dir`` with no sanitising.  The checkpoint is chosen by the
    caller of ``/deploy_from_api`` (``model_files.checkpoint`` points at a Team Files
    path the server downloads, then extracts embedded aux files from).  A crafted
    checkpoint carrying ``name="../../x"`` / an absolute name / a traversing dict key
    therefore wrote (and overwrote) files anywhere the process could reach.  The fix
    reduces the name to ``os.path.basename`` and rejects ``""`` / ``"."`` / ``".."``.

(b) ``InferenceImageCache.download_video`` (``nn/inference/cache.py``) built
    ``Path("/tmp/smart_cache") / ("_<rand>_" + video_info.name)`` and
    ``api.video.download_path`` creates the missing parent directories, so a video
    whose ``name`` is ``"x/../../../somewhere/evil.mp4"`` (the video info comes from the
    server the request points at) wrote outside ``/tmp/smart_cache``.  The fix keeps
    only ``Path(video_info.name).name`` (the final component).

The exploit tests drive the real functions with the crafted input and assert nothing
was written outside the allowed directory; each FAILS on BASE and PASSES on FIXED.
The regression tests assert legitimate behaviour and PASS on BOTH trees.

Everything an exploit could write is redirected into ``tmp_path``; no real host is
contacted and nothing is aimed at a real system path (the hard-coded ``/tmp/smart_cache``
writes are redirected into a per-test sandbox, and the path the code *asked* to write to
is captured and checked for containment).
"""

import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from supervisely.nn.inference.cache import InferenceImageCache
from supervisely.nn.inference.inference import Inference


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _files_under(root: Path):
    return [p.resolve() for p in root.rglob("*") if p.is_file()]


def _files_outside(allowed: Path, scan_root: Path, ignore=()):
    """Files under ``scan_root`` that are not inside ``allowed`` (and not ignored)."""
    allowed = allowed.resolve()
    ignore = [Path(p).resolve() for p in ignore]
    leaked = []
    for p in _files_under(scan_root):
        if p == allowed or allowed in p.parents:
            continue
        if any(p == i or i in p.parents for i in ignore):
            continue
        leaked.append(p)
    return leaked


def _make_crafted_checkpoint(path: Path, abs_target: Path) -> dict:
    """A harmless checkpoint (plain str content only) whose ``model_files`` entries try
    to escape the model dir in every way the task calls out.  Returns the raw dict."""
    ckpt = {
        "model_files": {
            # legitimate entry: must still be written into model_dir
            "config": {"name": "config.yaml", "content": "legit: 1\n"},
            # relative traversal in the name
            "rel": {"name": "../../escaped_by_name.txt", "content": "REL"},
            # absolute name pointing outside model_dir (still inside the sandbox)
            "abs": {"name": str(abs_target), "content": "ABS"},
            # no "name" -> fallback "<key>.txt"; the dict KEY carries the traversal
            "../../escaped_by_key": {"content": "KEY"},
            # names the fix must reject outright without crashing the extraction
            "dotdot": {"name": "..", "content": "DOTDOT"},
            "empty": {"name": "", "content": "EMPTY"},
            "dot": {"name": ".", "content": "DOT"},
        }
    }
    torch.save(ckpt, str(path))
    return ckpt


def _bare_inference(model_dir: Path) -> Inference:
    """Cheapest possible real ``Inference`` instance: only ``_model_dir`` is needed by
    ``_extract_model_files_from_checkpoint`` (verified against the source)."""
    inst = Inference.__new__(Inference)
    inst._model_dir = str(model_dir)
    return inst


# ===========================================================================
# (a) _extract_model_files_from_checkpoint
# ===========================================================================
# ---- exploit (real function): FAIL on BASE, PASS on FIXED -----------------
def test_extract_checkpoint_does_not_escape_model_dir(tmp_path: Path):
    # model_dir nested so a couple of ".." escape it but stay inside tmp_path
    model_dir = tmp_path / "a" / "b" / "model_dir"
    model_dir.mkdir(parents=True)
    ckpt_store = tmp_path / "remote"
    ckpt_store.mkdir()
    abs_zone = tmp_path / "abs_zone"
    abs_target = abs_zone / "abs_escaped.txt"

    ckpt_path = ckpt_store / "crafted.pt"
    _make_crafted_checkpoint(ckpt_path, abs_target)

    inst = _bare_inference(model_dir)
    # the real extraction routine the /deploy_from_api download path invokes
    result = inst._extract_model_files_from_checkpoint(str(ckpt_path))

    # Nothing may be written outside model_dir (the crafted checkpoint is the only
    # pre-existing file under ckpt_store and is ignored).
    leaked = _files_outside(model_dir, tmp_path, ignore=[ckpt_path])
    assert leaked == [], f"checkpoint extraction wrote outside model_dir: {leaked}"

    # The specific escape targets must not exist anywhere.
    assert not abs_target.exists()
    assert not (tmp_path / "a" / "escaped_by_name.txt").exists()
    assert not (tmp_path / "a" / "escaped_by_key.txt").exists()

    # Every file that *was* produced lives inside model_dir, and the result map only
    # points at paths inside model_dir.
    for p in _files_under(model_dir):
        assert model_dir.resolve() in p.parents
    for key, dst in result.items():
        dst = Path(dst).resolve()
        assert model_dir.resolve() in dst.parents, f"{key} -> {dst} escaped model_dir"


# ---- regression (real function): PASS on BOTH trees -----------------------
def test_extract_checkpoint_writes_legit_files(tmp_path: Path):
    model_dir = tmp_path / "a" / "b" / "model_dir"
    model_dir.mkdir(parents=True)
    ckpt_path = tmp_path / "crafted.pt"
    torch.save(
        {
            "model_files": {
                "config": {"name": "config.yaml", "content": "legit: 1\n"},
                "labels": {"name": "labels.txt", "content": "cat\ndog\n"},
                # fallback name "<key>.txt" for an entry without an explicit name
                "notes": {"content": "hello"},
            }
        },
        str(ckpt_path),
    )

    inst = _bare_inference(model_dir)
    result = inst._extract_model_files_from_checkpoint(str(ckpt_path))

    assert (model_dir / "config.yaml").read_text() == "legit: 1\n"
    assert (model_dir / "labels.txt").read_text() == "cat\ndog\n"
    assert (model_dir / "notes.txt").read_text() == "hello"
    assert result["config"] == str(model_dir / "config.yaml")
    assert result["labels"] == str(model_dir / "labels.txt")
    assert result["notes"] == str(model_dir / "notes.txt")


def test_extract_checkpoint_overwrites_existing_file(tmp_path: Path):
    model_dir = tmp_path / "a" / "b" / "model_dir"
    model_dir.mkdir(parents=True)
    existing = model_dir / "config.yaml"
    existing.write_text("OLD CONTENT")

    ckpt_path = tmp_path / "crafted.pt"
    torch.save(
        {"model_files": {"config": {"name": "config.yaml", "content": "NEW CONTENT"}}},
        str(ckpt_path),
    )

    inst = _bare_inference(model_dir)
    result = inst._extract_model_files_from_checkpoint(str(ckpt_path))

    assert existing.read_text() == "NEW CONTENT"
    assert result["config"] == str(existing)


# ---------------------------------------------------------------------------
# (a) end-to-end through the real POST /deploy_from_api route
# ---------------------------------------------------------------------------
import supervisely.app.fastapi.subapp as subapp  # noqa: E402
import supervisely.nn.inference.gui as GUI  # noqa: E402
from supervisely.nn.utils import ModelSource, RuntimeType  # noqa: E402

REMOTE_CHECKPOINT = "/experiments/1_proj/2_task/checkpoints/crafted.pt"


@pytest.fixture(scope="module")
def deploy_served(tmp_path_factory):
    """A minimal real ``Inference`` served through ``TestClient`` with a
    ``ServingGUITemplate``-shaped GUI, so that ``/deploy_from_api`` takes the branch
    that downloads the (caller-chosen) checkpoint and extracts its embedded files via
    the real ``_download_custom_model`` -> ``_extract_model_files_from_checkpoint``.

    ``api.file`` is mocked to serve the crafted checkpoint; everything from the HTTP
    route down to the file extraction is the real code under test.
    """
    from fastapi.testclient import TestClient

    sandbox = tmp_path_factory.mktemp("f7_deploy")
    model_dir = sandbox / "md_root" / "lvl" / "model_dir"
    model_dir.mkdir(parents=True)
    cache_dir = sandbox / "cache"
    cache_dir.mkdir()
    team_files = sandbox / "team_files"
    team_files.mkdir()
    abs_target = sandbox / "abs_zone" / "abs_escaped.txt"

    crafted = team_files / "crafted.pt"
    _make_crafted_checkpoint(crafted, abs_target)

    class MiniInference(Inference):
        FRAMEWORK_NAME = "f7"

        def load_model(self, model_files, model_source, model_info, device, runtime, **kwargs):
            self._loaded_model_files = dict(model_files)

        def get_classes(self):
            return ["dummy"]

        def _get_obj_class_shape(self):
            from supervisely.geometry.rectangle import Rectangle

            return Rectangle

    import logging

    from supervisely.app.singleton import Singleton

    saved_argv = sys.argv
    saved_env = {k: os.environ.get(k) for k in ("SMART_CACHE_CONTAINER_DIR", "SERVER_ADDRESS", "API_TOKEN", "TEAM_ID", "ENV")}
    # Application / _MainServer are process-wide singletons: if an earlier module already
    # served an Inference, serve() would reuse that server, whose /deploy_from_api route is
    # bound to the earlier instance.  Serve this one on fresh singletons and restore the
    # previous ones on teardown.
    saved_singletons = (Singleton._instances, Singleton._nested_instances)
    uvicorn_access = logging.getLogger("uvicorn.access")
    saved_uvicorn_filters = list(uvicorn_access.filters)
    Singleton._instances, Singleton._nested_instances = {}, {}
    sys.argv = ["pytest-f7-deploy"]
    os.environ["SMART_CACHE_CONTAINER_DIR"] = str(cache_dir)
    os.environ["SERVER_ADDRESS"] = "http://sly-test.invalid"
    os.environ["API_TOKEN"] = "x" * 128
    os.environ["TEAM_ID"] = "1"
    os.environ.pop("ENV", None)  # development mode

    inst = None
    try:
        with mock.patch.object(subapp, "set_autostart_flag_from_state", lambda *a, **k: None):
            inst = MiniInference(model_dir=str(model_dir))
            inst._api = mock.MagicMock(name="app_api")
            inst.serve()

        # ServingGUITemplate-shaped GUI so the template deploy branch is taken.
        gui = mock.MagicMock(spec=GUI.ServingGUITemplate)
        gui._success_label = mock.MagicMock()
        inst._gui = gui
        inst._user_layout_card = mock.MagicMock()
        inst._api_request_model_info = mock.MagicMock()
        inst._api_request_model_layout = mock.MagicMock()

        def get_info_by_path(team_id, path):
            if path == REMOTE_CHECKPOINT:
                return SimpleNamespace(sizeb=os.path.getsize(crafted), id=5, path=path, name="crafted.pt")
            return None

        def download(team_id, remote, local, **kwargs):
            import shutil as _sh

            _sh.copyfile(str(crafted), local)

        inst.api.file.get_info_by_path.side_effect = get_info_by_path
        inst.api.file.download.side_effect = download

        client = TestClient(inst.app.get_server(), raise_server_exceptions=False)
        yield SimpleNamespace(
            client=client, model_dir=model_dir, sandbox=sandbox,
            team_files=team_files, abs_target=abs_target, inst=inst,
        )
    finally:
        if inst is not None and inst._freeze_timer is not None:
            inst._freeze_timer.cancel()
        Singleton._instances, Singleton._nested_instances = saved_singletons
        uvicorn_access.filters[:] = saved_uvicorn_filters
        sys.argv = saved_argv
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_deploy_from_api_crafted_checkpoint_stays_in_model_dir(deploy_served):
    body = {
        "state": {
            "deploy_params": {
                "model_source": ModelSource.CUSTOM,
                "model_files": {"checkpoint": REMOTE_CHECKPOINT},
                "model_info": {},
                "device": "cpu",
                "runtime": RuntimeType.PYTORCH,
            }
        }
    }
    resp = deploy_served.client.post("/deploy_from_api", json=body)
    assert resp.status_code == 200, resp.text

    leaked = _files_outside(
        deploy_served.model_dir,
        deploy_served.sandbox,
        ignore=[deploy_served.team_files],
    )
    assert leaked == [], f"/deploy_from_api extraction wrote outside model_dir: {leaked}"
    assert not deploy_served.abs_target.exists()

    # The legitimate embedded file still landed in model_dir with its content.
    assert (deploy_served.model_dir / "config.yaml").read_text() == "legit: 1\n"


# ===========================================================================
# (b) InferenceImageCache.download_video
# ===========================================================================
# The code hard-codes Path("/tmp/smart_cache").  We never touch the real path: the
# download stand-in redirects every write into a per-test sandbox while recording the
# path the code *asked* to write to, and the containment check is made against that
# recorded path (independent of the redirect).
HARDCODED_SMART_CACHE = Path("/tmp/smart_cache")
EVIL_VIDEO_NAME = "x/../../../somewhere/evil.mp4"
VIDEO_PAYLOAD = b"attacker-controlled-video-bytes"


def _to_sandbox(sandbox_root: Path, p) -> Path:
    """Map an absolute path onto ``sandbox_root`` keeping its structure (incl. any
    ``..`` segments, so the OS resolves them exactly as it would on the real fs)."""
    parts = Path(p).parts
    if Path(p).anchor:
        parts = parts[1:]  # drop the drive/root anchor
    return sandbox_root.joinpath(*parts)


@pytest.fixture()
def video_download(tmp_path: Path):
    """Build an ``InferenceImageCache`` plus a mocked api whose ``video.download_path``
    is a faithful stand-in (ensure parent dirs, write bytes) redirected into a sandbox,
    and whose ``video.get_info_by_id`` returns a video with a configurable name."""
    import supervisely.nn.inference.cache as cache_mod
    from supervisely.io.fs import ensure_base_path

    sandbox_fs = tmp_path / "fs"  # stand-in for the real filesystem root
    sandbox_fs.mkdir()
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    requested_paths = []

    def download_path(video_id, path, progress_cb=None):
        requested_paths.append(str(path))
        dst = _to_sandbox(sandbox_fs, path)
        ensure_base_path(str(dst))  # same helper the real download_path uses
        with open(dst, "wb") as f:
            f.write(VIDEO_PAYLOAD)
        if progress_cb is not None:
            progress_cb(len(VIDEO_PAYLOAD))

    # save_video moves the downloaded temp file into the cache dir; redirect the source
    # of that move into the sandbox too, so the move finds the file we actually wrote.
    real_shutil = cache_mod.shutil

    class _ShutilShim:
        def __getattr__(self, name):
            return getattr(real_shutil, name)

        def move(self, src, dst, *a, **k):
            return real_shutil.move(str(_to_sandbox(sandbox_fs, src)), dst, *a, **k)

    name_holder = {"name": EVIL_VIDEO_NAME}

    api = mock.MagicMock(name="request_api")
    api.video.download_path.side_effect = download_path
    api.video.get_info_by_id.side_effect = lambda vid: SimpleNamespace(
        id=vid,
        name=name_holder["name"],
        frames_count=1,
        file_meta={"size": str(len(VIDEO_PAYLOAD))},
    )

    inf_cache = InferenceImageCache(maxsize=10, ttl=100, base_folder=str(cache_dir), log_progress=False)

    with mock.patch.object(cache_mod, "shutil", _ShutilShim()):
        yield SimpleNamespace(
            inf_cache=inf_cache,
            api=api,
            cache_dir=cache_dir,
            sandbox_fs=sandbox_fs,
            requested_paths=requested_paths,
            name_holder=name_holder,
        )


def _assert_inside_smart_cache(requested: str):
    allowed = HARDCODED_SMART_CACHE.resolve()
    target = Path(requested).resolve()
    assert allowed == target or allowed in target.parents, (
        f"download was asked to write to {requested!r} which resolves to {target} "
        f"- outside the smart cache dir {allowed}"
    )


# ---- exploit (real public function): FAIL on BASE, PASS on FIXED ----------
def test_download_video_malicious_name_stays_within_smart_cache(video_download):
    vd = video_download
    vd.name_holder["name"] = EVIL_VIDEO_NAME

    # the real public entry point; return_images=False matches the /smart_cache task
    vd.inf_cache.download_video(vd.api, 7, return_images=False)

    assert vd.requested_paths, "download_path was never called"
    for requested in vd.requested_paths:
        _assert_inside_smart_cache(requested)

    # Corroborate physically: in the sandbox, nothing was written outside the mapped
    # smart-cache dir (the cache dir where the file is finally moved to is separate).
    mapped_smart_cache = _to_sandbox(vd.sandbox_fs, HARDCODED_SMART_CACHE)
    leaked = _files_outside(mapped_smart_cache, vd.sandbox_fs)
    assert leaked == [], f"download escaped the smart cache dir: {leaked}"


# ---- regression (real public function): PASS on BOTH trees ----------------
def test_download_video_legit_name_downloads_and_caches(video_download):
    vd = video_download
    vd.name_holder["name"] = "clip.mp4"

    vd.inf_cache.download_video(vd.api, 8, return_images=False)

    assert vd.requested_paths
    for requested in vd.requested_paths:
        _assert_inside_smart_cache(requested)

    # the video was added to the cache and the cached file exists inside the cache dir
    assert 8 in vd.inf_cache._cache
    cached = Path(vd.inf_cache.get_video_path(8)).resolve()
    assert vd.cache_dir.resolve() in cached.parents
    assert cached.read_bytes() == VIDEO_PAYLOAD


# ===========================================================================
# Sibling sink confirmation: InferenceVideoInterface (nn/inference/video_inference.py)
# ===========================================================================
# Same F7 pattern: ``self._local_video_path = os.path.join(self._imgs_dir,
# f"{time.time_ns()}_{video_info.name}")`` (now wrapped in os.path.basename).  Confirmed
# exploitable on BASE, fixed on FIXED (see notes).  Exercised through the real class.
def _make_video_interface(api, imgs_dir: Path, name: str):
    from supervisely.nn.inference.video_inference import InferenceVideoInterface

    video_info = SimpleNamespace(
        id=1,
        name=name,
        frames_count=1,
        frames_to_timecodes=[0, 0.04],
        file_meta={"size": str(len(VIDEO_PAYLOAD))},
    )
    return InferenceVideoInterface(
        api=api,
        start_frame_index=0,
        frames_count=1,
        frames_direction="forward",
        video_info=video_info,
        imgs_dir=str(imgs_dir),
        preparing_progress={"current": 0, "total": 1},
    )


@pytest.fixture()
def video_iface_api():
    class _Resp:
        def iter_content(self, chunk_size=None):
            yield VIDEO_PAYLOAD

    api = mock.MagicMock(name="iface_api")
    api.video._download.return_value = _Resp()
    return api


# NOTE: a *leading* "../.." name (e.g. "../../evil.mp4") does NOT escape here, because
# the code prepends "{time_ns}_" to the name: the first segment becomes a literal
# "<ns>_.." directory, not a traversal.  Only an *embedded* "/.." (its own path segment)
# escapes - which is exactly the shape of the attacker-supplied name the task calls out.
@pytest.mark.parametrize(
    "name",
    ["x/../../iface_escaped/evil.mp4", "sub/dir/../../../iface_escaped2/evil.mp4"],
    ids=["embedded-traversal", "embedded-traversal-deep"],
)
def test_video_interface_download_stays_in_imgs_dir(tmp_path, video_iface_api, name):
    imgs_dir = tmp_path / "d0" / "d1" / "imgs"
    imgs_dir.mkdir(parents=True)

    iface = _make_video_interface(video_iface_api, imgs_dir, name)
    iface._download_entire_video()

    local = Path(iface._local_video_path).resolve()
    assert imgs_dir.resolve() in local.parents, f"local video path escaped imgs_dir: {local}"
    leaked = _files_outside(imgs_dir, tmp_path)
    assert leaked == [], f"video interface wrote outside imgs_dir: {leaked}"


def test_video_interface_download_legit_name(tmp_path, video_iface_api):
    imgs_dir = tmp_path / "d0" / "d1" / "imgs"
    imgs_dir.mkdir(parents=True)

    iface = _make_video_interface(video_iface_api, imgs_dir, "clip.mp4")
    iface._download_entire_video()

    local = Path(iface._local_video_path).resolve()
    assert imgs_dir.resolve() in local.parents
    assert local.name.endswith("clip.mp4")
    assert local.read_bytes() == VIDEO_PAYLOAD

"""
Security tests for fix F1: upload-filename path traversal in the inference server.

Vulnerable code paths (BASE tree):
  * ``supervisely/nn/inference/cache.py`` -- ``PersistentImageTTLCache.save_image`` /
    ``save_video`` built the on-disk path as ``base_dir / <key>`` with no containment
    check, so a key containing ``..`` or an absolute path escaped the cache directory.
  * ``supervisely/nn/inference/inference.py`` -- the ``/inference_batch``,
    ``/inference_batch_async`` and ``/inference_video_async`` routes passed the
    client-controlled ``UploadFile.filename`` straight through as that key.

On a Linux container the inference app runs as root, so this is an arbitrary file
write as root driven by the upload filename.

The fix applies ``os.path.basename`` to the filename in the three routes and adds a
``_check_path_in_base_dir`` containment guard (raising ``ValueError``) in
``save_image`` / ``save_video``.

The "exploit" tests perform the real attack (through the real FastAPI routes via
``TestClient`` where feasible, and through the real cache functions otherwise) and
assert the harmful outcome -- a file written outside the cache directory -- did not
happen.  Each of them FAILS on BASE and PASSES on FIXED.  The regression tests assert
legitimate behaviour and PASS on BOTH trees.

Everything is confined to ``tmp_path`` / a pytest-managed temp dir; no real host is
contacted and no file is aimed at a real system path.
"""

import io
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import mock
import numpy as np
import pytest

import supervisely.imaging.image as sly_image
from supervisely.nn.inference.cache import InferenceImageCache, PersistentImageTTLCache


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _img() -> np.ndarray:
    return np.random.randint(0, 255, size=(16, 24, 3), dtype=np.uint8)


def _png_bytes() -> bytes:
    """A real PNG payload, so decoding on the way into the cache succeeds on both trees
    and the only difference between trees is *where* the file is written."""
    return sly_image.write_bytes(_img(), ".png")


def _deep_base(root: Path) -> Path:
    """A cache base dir nested several levels below ``root`` so that a handful of ``..``
    segments escape the cache dir but still land inside ``root`` (never a real path)."""
    base = root / "c0" / "c1" / "c2" / "c3" / "smart_cache"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _files_outside(base: Path, scan_root: Path):
    """All files under ``scan_root`` that are not inside ``base``."""
    base = base.resolve()
    leaked = []
    for p in scan_root.rglob("*"):
        if p.is_file():
            rp = p.resolve()
            if rp != base and base not in rp.parents:
                leaked.append(rp)
    return leaked


# ===========================================================================
# Cache-layer exploit tests (real cache functions).
# FAIL on BASE, PASS on FIXED.
# ===========================================================================
@pytest.mark.parametrize(
    "evil_key",
    [
        "../escape_img",          # one level up
        "../../escape_img",       # nested traversal
        "sub/../../escape_img",   # traversal hidden behind a benign-looking segment
    ],
)
def test_save_image_rejects_relative_traversal(tmp_path: Path, evil_key: str):
    base = _deep_base(tmp_path)
    cache = PersistentImageTTLCache(maxsize=16, ttl=600, filepath=base)

    # Where the unguarded code would write (base_dir / key, mirroring the real logic).
    escaped = (base / Path(evil_key).with_suffix(".png")).resolve()
    assert base.resolve() not in escaped.parents  # sanity: this really is an escape

    try:
        cache.save_image(evil_key, _img())
    except ValueError:
        pass  # FIXED: containment guard rejected the key

    assert not escaped.exists(), f"path traversal wrote outside the cache dir: {escaped}"
    assert _files_outside(base, tmp_path) == []


def test_save_image_rejects_absolute_path(tmp_path: Path):
    base = _deep_base(tmp_path)
    cache = PersistentImageTTLCache(maxsize=16, ttl=600, filepath=base)

    # Absolute key that points outside the cache dir but still inside tmp_path.
    abs_target = tmp_path / "abs_zone" / "abs_evil"
    escaped = Path(str(abs_target)).with_suffix(".png").resolve()
    assert base.resolve() not in escaped.parents

    try:
        cache.save_image(str(abs_target), _img())
    except ValueError:
        pass

    assert not escaped.exists(), f"absolute key wrote outside the cache dir: {escaped}"
    assert _files_outside(base, tmp_path) == []


@pytest.mark.parametrize(
    "evil_key",
    [
        "../../../vid_escape",     # enough `..` to overcome the `video_` prefix
        "../../../../vid_escape",
    ],
)
def test_save_video_rejects_relative_traversal(tmp_path: Path, evil_key: str):
    base = _deep_base(tmp_path)
    cache = PersistentImageTTLCache(maxsize=16, ttl=600, filepath=base)

    # Mirror the real path construction: base_dir / f"video_{key}{ext}" (ext == "" for IO).
    escaped = (base / f"video_{evil_key}").resolve()
    assert base.resolve() not in escaped.parents

    try:
        cache.save_video(evil_key, io.BytesIO(b"\x00\x01\x02video"))
    except ValueError:
        pass

    assert not escaped.exists(), f"path traversal wrote outside the cache dir: {escaped}"
    assert _files_outside(base, tmp_path) == []


def test_add_image_to_cache_rejects_traversal(tmp_path: Path):
    """The SDK helper that the routes call directly."""
    base = _deep_base(tmp_path)
    inf_cache = InferenceImageCache(maxsize=16, ttl=600, base_folder=str(base))

    evil_key = "../../pwned_by_add_image"
    escaped = (base / Path(evil_key).with_suffix(".png")).resolve()
    assert base.resolve() not in escaped.parents

    try:
        inf_cache.add_image_to_cache(evil_key, _png_bytes(), ext=".png")
    except ValueError:
        pass

    assert not escaped.exists(), f"add_image_to_cache escaped the cache dir: {escaped}"
    assert _files_outside(base, tmp_path) == []


def test_add_video_to_cache_rejects_traversal(tmp_path: Path):
    base = _deep_base(tmp_path)
    inf_cache = InferenceImageCache(maxsize=16, ttl=600, base_folder=str(base))

    evil_key = "../../../pwned_by_add_video"
    escaped = (base / f"video_{evil_key}").resolve()
    assert base.resolve() not in escaped.parents

    try:
        inf_cache.add_video_to_cache(evil_key, io.BytesIO(b"\x00\x01\x02video"))
    except ValueError:
        pass

    assert not escaped.exists(), f"add_video_to_cache escaped the cache dir: {escaped}"
    assert _files_outside(base, tmp_path) == []


# ===========================================================================
# Cache-layer regression tests (legitimate keys the SDK itself uses).
# PASS on BOTH trees.
# ===========================================================================
@pytest.mark.parametrize("ext", [".png", "", None])
def test_save_image_legit_names(tmp_path: Path, ext):
    base = _deep_base(tmp_path)
    cache = PersistentImageTTLCache(maxsize=16, ttl=600, filepath=base)

    img = _img()
    cache.save_image("plain_name", img, ext=ext) if ext is not None else cache.save_image(
        "plain_name", img
    )

    assert (base / "plain_name.png").exists()
    assert np.allclose(cache.get_image("plain_name"), img)
    assert _files_outside(base, tmp_path) == []


@pytest.mark.parametrize(
    "name",
    [
        "image_12345",          # SDK's own string key form (via _image_name)
        "frame_7_42",           # SDK's own frame key form (via _frame_name)
        "name with spaces",     # spaces
    ],
)
def test_save_image_legit_string_keys(tmp_path: Path, name: str):
    base = _deep_base(tmp_path)
    cache = PersistentImageTTLCache(maxsize=32, ttl=600, filepath=base)

    img = _img()
    cache.save_image(name, img)

    # save_image normalises the extension with Path(key).with_suffix(ext).
    expected = base / Path(name).with_suffix(".png")
    assert expected.exists()
    assert np.allclose(cache.get_image(name), img)
    assert _files_outside(base, tmp_path) == []


def test_save_video_legit_integer_key(tmp_path: Path):
    base = _deep_base(tmp_path)
    cache = PersistentImageTTLCache(maxsize=16, ttl=600, filepath=base)

    cache.save_video(777, io.BytesIO(b"\x00\x01\x02video"))

    assert (base / "video_777").exists()
    assert Path(cache.get_video_path(777)).read_bytes() == b"\x00\x01\x02video"
    assert _files_outside(base, tmp_path) == []


def test_add_image_to_cache_legit(tmp_path: Path):
    base = _deep_base(tmp_path)
    inf_cache = InferenceImageCache(maxsize=16, ttl=600, base_folder=str(base))

    out = inf_cache.add_image_to_cache("photo.jpg", _png_bytes(), ext=".png")
    assert isinstance(out, np.ndarray)
    assert (base / "photo.png").exists()
    assert _files_outside(base, tmp_path) == []


# ===========================================================================
# End-to-end route tests through the real FastAPI server (TestClient).
# ===========================================================================
@pytest.fixture(scope="module")
def served(tmp_path_factory):
    """Serve a minimal real Inference subclass through TestClient, with no platform.

    Only the pieces that would require a live Supervisely instance or a GPU are
    mocked: ``self.api`` is a MagicMock and the autostart helper (which would build a
    real ``sly.Api``) is turned into a no-op.  The HTTP routes, request parsing,
    filename handling and the cache are all the real code under test.
    """
    from supervisely.nn.inference.inference import Inference

    sandbox = tmp_path_factory.mktemp("f1_routes_sandbox")
    cache_dir = _deep_base(sandbox)
    model_dir = tmp_path_factory.mktemp("f1_models")

    class MiniInference(Inference):
        def load_on_device(self, model_dir, device="cpu"):
            self._model_served = True

        def get_classes(self):
            return ["dummy"]

        def _get_obj_class_shape(self):
            from supervisely.geometry.rectangle import Rectangle

            return Rectangle

        # No-op the heavy inference bodies; the routes still cache files for real first.
        def _inference_images(self, images, state, inference_request):
            return None

        def _inference_video(self, path, state, inference_request):
            return None

    import logging

    from supervisely.app.singleton import Singleton

    saved_argv = sys.argv
    saved_env = {k: os.environ.get(k) for k in ("SMART_CACHE_CONTAINER_DIR", "SERVER_ADDRESS", "API_TOKEN", "ENV")}
    # Application, _MainServer, StateJson, ... are process-wide singletons and the server
    # keeps every route registered on it (bound to *this* module's MiniInference).  Build
    # the app on fresh singletons and put the previous ones back on teardown, so a later
    # module that serves its own Inference gets its own server, not this one's routes.
    saved_singletons = (Singleton._instances, Singleton._nested_instances)
    uvicorn_access = logging.getLogger("uvicorn.access")
    saved_uvicorn_filters = list(uvicorn_access.filters)
    Singleton._instances, Singleton._nested_instances = {}, {}
    sys.argv = ["pytest-f1-security"]
    os.environ["SMART_CACHE_CONTAINER_DIR"] = str(cache_dir)
    os.environ["SERVER_ADDRESS"] = "http://sly-test.invalid"
    os.environ["API_TOKEN"] = "x" * 128
    os.environ.pop("ENV", None)  # development mode

    import supervisely.app.fastapi.subapp as subapp

    try:
        with mock.patch.object(subapp, "set_autostart_flag_from_state", lambda *a, **k: None):
            inst = MiniInference(model_dir=str(model_dir))
            inst._api = mock.MagicMock()
            inst._model_served = True
            inst.serve()

        from fastapi.testclient import TestClient

        client = TestClient(inst.app.get_server(), raise_server_exceptions=False)

        yield SimpleNamespace(
            client=client, cache_dir=cache_dir, sandbox=sandbox, inst=inst
        )
    finally:
        Singleton._instances, Singleton._nested_instances = saved_singletons
        uvicorn_access.filters[:] = saved_uvicorn_filters
        sys.argv = saved_argv
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# ---- route exploit tests: FAIL on BASE, PASS on FIXED ---------------------
def test_route_inference_batch_traversal_blocked(served):
    name = "../../batch_pwned.png"
    resp = served.client.post(
        "/inference_batch",
        files=[("files", (name, _png_bytes(), "image/png"))],
    )
    assert resp.status_code == 200, resp.text
    leaked = _files_outside(served.cache_dir, served.sandbox)
    assert leaked == [], f"/inference_batch upload escaped the cache dir: {leaked}"


def test_route_inference_batch_async_traversal_blocked(served):
    name = "../../batch_async_pwned.png"
    resp = served.client.post(
        "/inference_batch_async",
        files=[("files", (name, _png_bytes(), "image/png"))],
    )
    assert resp.status_code == 200, resp.text
    leaked = _files_outside(served.cache_dir, served.sandbox)
    assert leaked == [], f"/inference_batch_async upload escaped the cache dir: {leaked}"


def test_route_inference_video_async_traversal_blocked(served):
    name = "../../../video_async_pwned.mp4"
    resp = served.client.post(
        "/inference_video_async",
        files=[("files", (name, b"\x00\x01\x02fakevideo", "video/mp4"))],
    )
    assert resp.status_code == 200, resp.text
    leaked = _files_outside(served.cache_dir, served.sandbox)
    assert leaked == [], f"/inference_video_async upload escaped the cache dir: {leaked}"


def test_route_inference_batch_absolute_name_blocked(served):
    # An absolute filename pointing OUTSIDE the cache dir but still inside the test
    # sandbox (never a real system path).  os.path.basename strips it on both
    # Windows and Linux; without the fix the absolute path is used verbatim and the
    # file lands outside the cache dir.
    abs_name = str(served.sandbox / "abs_batch_pwned.png")
    resp = served.client.post(
        "/inference_batch",
        files=[("files", (abs_name, _png_bytes(), "image/png"))],
    )
    assert resp.status_code == 200, resp.text
    leaked = _files_outside(served.cache_dir, served.sandbox)
    assert leaked == [], f"/inference_batch absolute name escaped the cache dir: {leaked}"


# ---- route regression tests: PASS on BOTH trees ---------------------------
# These assert that a legitimate upload is cached at the correct location inside the
# cache dir.  They deliberately do NOT scan the whole (module-shared) sandbox for
# stray files: on BASE the exploit tests above leave escaped files behind, and those
# leftovers are not this test's concern -- legitimate placement is.
def test_route_inference_batch_legit_name(served):
    name = "normal_photo.png"
    resp = served.client.post(
        "/inference_batch",
        files=[("files", (name, _png_bytes(), "image/png"))],
    )
    assert resp.status_code == 200, resp.text
    cached = served.cache_dir / "normal_photo.png"
    assert cached.exists()
    assert np.allclose(sly_image.read(str(cached)), sly_image.read(str(cached)))


def test_route_inference_batch_name_with_dots_and_spaces(served):
    name = "my nice.photo.v2.png"
    resp = served.client.post(
        "/inference_batch",
        files=[("files", (name, _png_bytes(), "image/png"))],
    )
    assert resp.status_code == 200, resp.text
    assert (served.cache_dir / "my nice.photo.v2.png").exists()


def test_route_inference_batch_name_without_extension(served):
    name = "noext_image"
    resp = served.client.post(
        "/inference_batch",
        files=[("files", (name, _png_bytes(), "image/png"))],
    )
    assert resp.status_code == 200, resp.text
    assert (served.cache_dir / "noext_image.png").exists()


def test_route_inference_video_async_legit_name(served):
    name = "clip.mp4"
    resp = served.client.post(
        "/inference_video_async",
        files=[("files", (name, b"\x00\x01\x02fakevideo", "video/mp4"))],
    )
    assert resp.status_code == 200, resp.text
    assert (served.cache_dir / "video_clip.mp4").exists()

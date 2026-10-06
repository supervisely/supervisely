"""Security tests for F9: the remaining ``tarfile`` extraction sites.

Two places in the SDK extracted a tar that comes from somewhere else with a bare
``tar.extractall(path)``:

* ``FileApi.download_directory`` (``supervisely/api/file_api.py``) - the archive is
  whatever the server answers to ``file-storage.download``;
* ``VideoProject.restore_snapshot`` (``supervisely/project/video_project.py``, reached
  through the public ``VideoProject.upload_bin``) - the archive is a ``.tar.zst``
  snapshot file handed to the SDK.

Without an extraction filter a member named ``../x``, an absolute member, or a link
that points outside of the destination is written (as root, in the deployment
containers) wherever the archive says.  The fix routes both sites through the new
helper ``supervisely.io.fs._extractall_safely`` which uses tarfile's ``data`` filter
when the running Python has one and validates every member right before extracting
it otherwise.  An unsafe member is skipped: a warning starting with
``Skipping unsafe archive member`` goes to the SDK logger, nothing is raised for it
and the remaining (safe) members are extracted as usual.

Layout of this file:

1. ``_extractall_safely`` itself, in normal mode and in stream mode (``"r|"`` over a
   forward-only file object), on the ``data`` filter branch and on the manual branch.
2. ``FileApi.download_directory`` through the real ``sly.Api`` with a fake HTTP
   transport that serves the archive.
3. ``VideoProject.upload_bin`` / ``restore_snapshot`` with a zstd-compressed archive,
   for the streaming branch and for the one-shot branch of the function.

Every archive is built in memory with ``tarfile.TarInfo``.  Everything an archive
could write is aimed at a sandbox directory inside ``tmp_path``; "outside" always
means "inside the sandbox, but not inside the directory the code was told to use".
No real network is used.
"""

import io
import json
import logging
import os
import sys
import tarfile
import tempfile
import types
from pathlib import Path
from unittest import mock

import pytest
import requests

import supervisely as sly
from supervisely.project.versioning.common import DEFAULT_VIDEO_SCHEMA_VERSION
from supervisely.sly_logger import create_formatter
from supervisely.video_annotation.key_id_map import KeyIdMap

PLATFORM = "https://platform.invalid"
TOKEN = "t" * 128
SECRET = b"top-secret"
SKIP_WARNING = "Skipping unsafe archive member"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _symlinks_supported() -> bool:
    with tempfile.TemporaryDirectory() as probe:
        try:
            os.symlink("target", os.path.join(probe, "link"))
        except (OSError, NotImplementedError, AttributeError):
            return False
    return True


def _hardlinks_supported() -> bool:
    with tempfile.TemporaryDirectory() as probe:
        source = os.path.join(probe, "source")
        with open(source, "wb"):
            pass
        try:
            os.link(source, os.path.join(probe, "link"))
        except (OSError, NotImplementedError, AttributeError):
            return False
    return True


needs_symlinks = pytest.mark.skipif(
    not _symlinks_supported(),
    reason="creating symlinks needs a privilege this account does not have "
    "(Windows without Developer Mode); the test runs on POSIX",
)
needs_hardlinks = pytest.mark.skipif(
    not _hardlinks_supported(), reason="this filesystem can not create hardlinks"
)


def _reg(name, data=b"owned"):
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.type = tarfile.REGTYPE
    info.mode = 0o644
    return info, io.BytesIO(data)


def _dir(name):
    info = tarfile.TarInfo(name)
    info.type = tarfile.DIRTYPE
    info.mode = 0o755
    return info, None


def _sym(name, linkname):
    info = tarfile.TarInfo(name)
    info.type = tarfile.SYMTYPE
    info.linkname = linkname
    return info, None


def _hardlink(name, linkname):
    info = tarfile.TarInfo(name)
    info.type = tarfile.LNKTYPE
    info.linkname = linkname
    return info, None


def _special(name, member_type):
    """Character / block device or fifo.  1:3 is the null device, harmless if created."""
    info = tarfile.TarInfo(name)
    info.type = member_type
    info.mode = 0o644
    info.devmajor = 1
    info.devminor = 3
    return info, None


def _tar_bytes(members) -> bytes:
    """``members`` is a list of ``(TarInfo, fileobj_or_None)`` tuples."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        for info, fileobj in members:
            if fileobj is not None:
                fileobj.seek(0)
            tar.addfile(info, fileobj)
    return buffer.getvalue()


class _ForwardOnly:
    """A file object that can only be read front to back, like a decompression stream."""

    def __init__(self, data: bytes):
        self._buffer = io.BytesIO(data)

    def read(self, size=-1):
        return self._buffer.read(size)


def _open_normal(data: bytes) -> tarfile.TarFile:
    return tarfile.open(fileobj=io.BytesIO(data), mode="r")


def _open_stream(data: bytes) -> tarfile.TarFile:
    return tarfile.open(fileobj=_ForwardOnly(data), mode="r|")


OPENERS = [
    pytest.param(_open_normal, id="normal"),
    pytest.param(_open_stream, id="stream"),
]


def _snapshot_of(root: Path, allowed: Path) -> dict:
    """Everything under ``root`` that is not under ``allowed``.

    Maps the relative path to the file content (``None`` for a directory), so a new
    file, a new directory and a changed file all show up as a difference.
    """
    allowed = os.path.abspath(str(allowed))
    state = {}
    for dirpath, dirnames, filenames in os.walk(str(root)):
        dirnames[:] = [
            name for name in dirnames if os.path.abspath(os.path.join(dirpath, name)) != allowed
        ]
        for name in dirnames:
            state[os.path.relpath(os.path.join(dirpath, name), str(root))] = None
        for name in filenames:
            path = os.path.join(dirpath, name)
            with open(path, "rb") as file:
                state[os.path.relpath(path, str(root))] = file.read()
    return state


def _files_in(root: Path) -> dict:
    """Relative posix path -> content, for every regular file under ``root``."""
    files = {}
    for dirpath, _, filenames in os.walk(str(root)):
        for name in filenames:
            path = Path(dirpath) / name
            files[path.relative_to(root).as_posix()] = path.read_bytes()
    return files


def _attempt(func, *args, **kwargs):
    """Run ``func``; return the exception it raised, or ``None``."""
    try:
        func(*args, **kwargs)
    except Exception as exc:  # pylint: disable=broad-except
        return exc
    return None


def _drop_extraction_filters(monkeypatch) -> None:
    """Make tarfile behave like a Python that has no extraction filters.

    ``tarfile.data_filter`` disappears (this is what the SDK looks at) and extraction
    without an explicit filter is fully trusted again, whatever the default of the
    running Python is.
    """
    monkeypatch.delattr(tarfile, "data_filter", raising=False)
    if hasattr(tarfile, "fully_trusted_filter"):
        monkeypatch.setattr(
            tarfile.TarFile,
            "extraction_filter",
            staticmethod(tarfile.fully_trusted_filter),
            raising=False,
        )


@pytest.fixture(params=["data_filter", "manual_checks"])
def branch(request, monkeypatch) -> str:
    """Which branch of ``_extractall_safely`` the test goes through."""
    if request.param == "data_filter":
        if not hasattr(tarfile, "data_filter"):
            pytest.skip("this Python has no tarfile extraction filters")
    else:
        _drop_extraction_filters(monkeypatch)
    return request.param


@pytest.fixture()
def sandbox(tmp_path: Path) -> Path:
    """The only place an archive in this file can reach."""
    root = tmp_path / "sandbox"
    root.mkdir()
    (root / "secret.txt").write_bytes(SECRET)
    return root


@pytest.fixture()
def skip_warnings() -> list:
    """Messages of the ``Skipping unsafe archive member`` warnings the SDK logger writes.

    The SDK logger does not propagate to the root logger, so ``caplog`` does not see it;
    a handler is attached to it for the duration of the test instead.
    """

    class _ListHandler(logging.Handler):
        def __init__(self):
            super().__init__(level=logging.WARNING)
            self.messages = []

        def emit(self, record):
            message = record.getMessage()
            if message.startswith(SKIP_WARNING):
                self.messages.append(message)

    handler = _ListHandler()
    handler.setFormatter(create_formatter())
    previous_level = sly.logger.level
    if sly.logger.getEffectiveLevel() > logging.WARNING:
        sly.logger.setLevel(logging.WARNING)
    sly.logger.addHandler(handler)
    yield handler.messages
    sly.logger.removeHandler(handler)
    sly.logger.setLevel(previous_level)


def _assert_skipped(messages: list, fragment: str) -> None:
    """A skip warning that mentions ``fragment`` (part of the member name) was logged."""
    assert any(
        fragment in message for message in messages
    ), f"no {SKIP_WARNING!r} warning mentioning {fragment!r} was logged: {messages!r}"


def _absolute_name(sandbox: Path) -> str:
    """An absolute member name that points at a file directly inside the sandbox."""
    return str(sandbox / "PWNED_abs.txt").replace(os.sep, "/")


def _absolute_member_skipped(name: str, branch: str) -> bool:
    """Whether the fix skips an absolute member (with a warning) or keeps it inside.

    The ``data`` filter strips a leading ``/`` and extracts the rest inside the
    destination (POSIX); a name that is still absolute after that (a Windows drive) is
    skipped.  The manual branch skips every absolute member.
    """
    return branch == "manual_checks" or os.path.isabs(name.lstrip("/" + os.sep))


def _attack_members(kind: str, sandbox: Path, depth: int) -> list:
    """Hostile members for a destination that is ``depth`` levels below ``sandbox``."""
    up = "../" * depth
    if kind == "parent_traversal":
        return [_reg(up + "PWNED_parent.txt")]
    if kind == "absolute_path":
        return [_reg(_absolute_name(sandbox))]
    if kind == "hardlink_overwrite":
        # a hardlink to a file outside, then a regular member written over that name
        return [_hardlink("hl", up + "secret.txt"), _reg("hl", b"overwritten")]
    raise ValueError(kind)


# part of the skip warning that names the unsafe member of each attack
ATTACK_WARNING_FRAGMENT = {
    "parent_traversal": "PWNED_parent.txt",
    "absolute_path": "PWNED_abs.txt",
    "hardlink_overwrite": "'hl'",
}


def _attack_is_skipped(kind: str, sandbox: Path, branch: str) -> bool:
    """Whether the fix skips the attack member with a warning (see above for absolute)."""
    if kind == "absolute_path":
        return _absolute_member_skipped(_absolute_name(sandbox), branch)
    return True


ATTACKS = [
    pytest.param("parent_traversal", id="parent_traversal"),
    pytest.param("absolute_path", id="absolute_path"),
    pytest.param("hardlink_overwrite", id="hardlink_overwrite", marks=needs_hardlinks),
]


# ===========================================================================
# 1. supervisely.io.fs._extractall_safely
# ===========================================================================


def _extract(tar: tarfile.TarFile, target: Path) -> None:
    """Extract through the helper under test.

    The helper does not exist before the fix: the sites it now guards called a bare
    ``tar.extractall(path)``.  On such a tree that call is what gets exercised, so the
    tests below fail there on the write outside of the destination and not on a
    missing attribute.
    """
    helper = getattr(sly.fs, "_extractall_safely", None)
    if helper is None:
        tar.extractall(str(target))
    else:
        helper(tar, str(target))


def _extract_bytes(opener, data: bytes, target: Path):
    def run():
        with opener(data) as tar:
            _extract(tar, target)

    return _attempt(run)


@pytest.fixture()
def target(sandbox: Path) -> Path:
    path = sandbox / "extract"
    path.mkdir()
    return path


# --- vulnerability tests: FAIL on base, PASS on fixed ----------------------


@pytest.mark.parametrize("opener", OPENERS)
@pytest.mark.parametrize(
    "name",
    ["../PWNED_parent.txt", "folder/../../PWNED_parent.txt", "./../PWNED_parent.txt"],
)
def test_extractall_parent_traversal_blocked(sandbox, target, branch, opener, name, skip_warnings):
    """A member that climbs out of the destination with ``..`` is skipped, the rest extracts."""
    data = _tar_bytes([_reg("ok.txt", b"ok"), _reg(name), _reg("after.txt", b"after")])
    before = _snapshot_of(sandbox, target)

    exc = _extract_bytes(opener, data, target)

    assert _snapshot_of(sandbox, target) == before, "a tar member was written outside target_dir"
    assert exc is None, f"extraction was aborted instead of skipping the member: {exc!r}"
    _assert_skipped(skip_warnings, "PWNED_parent.txt")
    assert _files_in(target) == {"ok.txt": b"ok", "after.txt": b"after"}


@pytest.mark.parametrize("opener", OPENERS)
def test_extractall_absolute_member_contained(sandbox, target, branch, opener, skip_warnings):
    """An absolute member name never reaches that absolute location.

    The ``data`` filter skips it where the name stays absolute (Windows drive) and
    strips the leading separator elsewhere, which keeps the file inside the
    destination; the manual branch skips it.
    """
    name = _absolute_name(sandbox)
    data = _tar_bytes([_reg(name), _reg("after.txt", b"after")])
    before = _snapshot_of(sandbox, target)

    exc = _extract_bytes(opener, data, target)

    assert _snapshot_of(sandbox, target) == before, "absolute member was written outside"
    assert exc is None, f"extraction was aborted instead of skipping the member: {exc!r}"
    if _absolute_member_skipped(name, branch):
        _assert_skipped(skip_warnings, "PWNED_abs.txt")
        assert _files_in(target) == {"after.txt": b"after"}
    else:
        # leading "/" stripped by the data filter: the member lands inside target_dir
        assert _files_in(target) == {"after.txt": b"after", name.lstrip("/"): b"owned"}


@needs_symlinks
@pytest.mark.parametrize("opener", OPENERS)
@pytest.mark.parametrize("link_target", ["relative", "absolute"])
def test_extractall_symlink_escape_blocked(
    sandbox, target, branch, opener, link_target, skip_warnings
):
    """A symlink that points outside, then a file written through it.

    The link is skipped; the file member behind it then lands in an ordinary
    directory named like the link inside the destination, which is harmless.
    """
    outside = sandbox / "outside"
    outside.mkdir()
    linkname = "../outside" if link_target == "relative" else str(outside)
    data = _tar_bytes(
        [_sym("link", linkname), _reg("link/PWNED_symlink.txt"), _reg("after.txt", b"after")]
    )
    before = _snapshot_of(sandbox, target)

    exc = _extract_bytes(opener, data, target)

    assert _snapshot_of(sandbox, target) == before, "a write went through a symlink to outside"
    assert not os.path.islink(str(target / "link")), "symlink to outside was created"
    assert exc is None, f"extraction was aborted instead of skipping the member: {exc!r}"
    _assert_skipped(skip_warnings, "'link'")
    assert (target / "after.txt").read_bytes() == b"after"


@needs_hardlinks
@pytest.mark.parametrize("opener", OPENERS)
def test_extractall_hardlink_to_outside_blocked(sandbox, target, branch, opener, skip_warnings):
    """A hardlink to a file outside, then a member written over the link.

    Without the fix the second member is written into the file outside of the
    destination, because both names are the same file on disk.  With the fix the link
    is skipped and the second member is just an ordinary file inside the destination.
    """
    data = _tar_bytes(
        _attack_members("hardlink_overwrite", sandbox, depth=1) + [_reg("after.txt", b"after")]
    )
    before = _snapshot_of(sandbox, target)

    exc = _extract_bytes(opener, data, target)

    assert (sandbox / "secret.txt").read_bytes() == SECRET, "file outside target_dir was overwritten"
    assert _snapshot_of(sandbox, target) == before
    assert exc is None, f"extraction was aborted instead of skipping the member: {exc!r}"
    _assert_skipped(skip_warnings, "'hl'")
    assert _files_in(target) == {"hl": b"overwritten", "after.txt": b"after"}


@pytest.mark.parametrize("opener", OPENERS)
@pytest.mark.parametrize(
    "member_type",
    [
        pytest.param(tarfile.CHRTYPE, id="char_device"),
        pytest.param(tarfile.BLKTYPE, id="block_device"),
        pytest.param(tarfile.FIFOTYPE, id="fifo"),
    ],
)
def test_extractall_device_member_rejected(
    sandbox, target, branch, opener, member_type, skip_warnings
):
    """Device nodes and fifos are not data: the member is skipped with a warning.

    (Unfiltered extraction creates the node when it is allowed to - root in the
    deployment containers - and silently skips it where the OS has no such thing.)
    """
    data = _tar_bytes(
        [_reg("ok.txt", b"ok"), _special("node", member_type), _reg("after.txt", b"after")]
    )
    before = _snapshot_of(sandbox, target)

    exc = _extract_bytes(opener, data, target)

    assert not os.path.lexists(str(target / "node")), "special file was created"
    assert _snapshot_of(sandbox, target) == before
    assert exc is None, f"extraction was aborted instead of skipping the member: {exc!r}"
    _assert_skipped(skip_warnings, "'node'")
    assert _files_in(target) == {"ok.txt": b"ok", "after.txt": b"after"}


# --- regression tests: PASS on both trees ----------------------------------

LEGIT_FILES = {
    "top.txt": b"top-level",
    "folder/a.txt": b"nested-a",
    "folder/sub/b.txt": b"nested-b",
    "folder/sub/deeper/c.bin": bytes(range(256)) * 64,
    "empty.txt": b"",
}


def _legit_members(prefix: str = "") -> list:
    members = []
    if prefix:
        members.append(_dir(prefix.rstrip("/")))
    members += [_dir(prefix + "folder"), _dir(prefix + "folder/sub")]
    # "folder/sub/deeper" has no directory member on purpose: it is created implicitly
    members += [_reg(prefix + name, data) for name, data in LEGIT_FILES.items()]
    return members


@pytest.mark.parametrize("opener", OPENERS)
@pytest.mark.parametrize("prefix", ["", "./"], ids=["plain_names", "dot_prefixed"])
def test_extractall_legit_archive(target, branch, opener, prefix):
    """Nested folders extract to the same tree in every mode and on every branch.

    ``./`` prefixed names with a ``.`` directory member are what
    ``tar.add(directory, arcname=".")`` produces (the snapshot format).
    """
    data = _tar_bytes(_legit_members(prefix))

    with opener(data) as tar:
        _extract(tar, target)

    assert _files_in(target) == LEGIT_FILES
    assert (target / "folder" / "sub" / "deeper").is_dir()


@pytest.mark.parametrize("opener", OPENERS)
def test_extractall_legit_dotdot_inside_destination(target, branch, opener):
    """``..`` that stays inside the destination is not an attack."""
    data = _tar_bytes([_reg("folder/../kept.txt", b"kept")])

    with opener(data) as tar:
        _extract(tar, target)

    assert _files_in(target) == {"kept.txt": b"kept"}


@needs_hardlinks
@pytest.mark.parametrize("opener", OPENERS)
def test_extractall_legit_internal_hardlink(target, branch, opener):
    """A hardlink to another member of the same archive still works."""
    data = _tar_bytes([_reg("folder/a.txt", b"nested-a"), _hardlink("alias.txt", "folder/a.txt")])

    with opener(data) as tar:
        _extract(tar, target)

    assert _files_in(target) == {"folder/a.txt": b"nested-a", "alias.txt": b"nested-a"}


@needs_symlinks
@pytest.mark.parametrize("opener", OPENERS)
def test_extractall_legit_internal_symlink(target, branch, opener):
    """A relative symlink that stays inside the destination still works."""
    data = _tar_bytes([_reg("folder/a.txt", b"nested-a"), _sym("folder/alias", "a.txt")])

    with opener(data) as tar:
        _extract(tar, target)

    assert os.path.islink(str(target / "folder" / "alias"))
    assert (target / "folder" / "alias").read_bytes() == b"nested-a"


# ===========================================================================
# 2. FileApi.download_directory
# ===========================================================================

REMOTE_DIR = "/My_App_Test/ds1"


@pytest.fixture()
def team_files(monkeypatch) -> dict:
    """A fake platform: every request is answered with ``team_files["body"]``."""
    served = {"body": b"", "urls": []}

    def fake_send(adapter, request, **kwargs):
        served["urls"].append(request.url)
        response = requests.Response()
        response.status_code = 200
        response.reason = "OK"
        response.url = request.url
        response.request = request
        response.raw = io.BytesIO(served["body"])
        return response

    monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", fake_send)
    return served


@pytest.fixture()
def api(monkeypatch, team_files) -> sly.Api:
    monkeypatch.setenv("SERVER_ADDRESS", PLATFORM)
    monkeypatch.setenv("API_TOKEN", TOKEN)
    monkeypatch.delenv("SUPERVISELY_API_SERVER_ADDRESS", raising=False)
    monkeypatch.delenv("TASK_ID", raising=False)
    monkeypatch.setattr(sly.Api, "_checked_servers", set())
    return sly.Api(PLATFORM, TOKEN)


def _remote_dir_members() -> list:
    """What the platform sends for ``REMOTE_DIR``: one top-level folder with its name."""
    return [
        _dir("ds1"),
        _reg("ds1/a.txt", b"file-a"),
        _dir("ds1/sub"),
        _reg("ds1/sub/b.txt", b"file-b"),
    ]


# --- vulnerability tests: FAIL on base, PASS on fixed ----------------------


@pytest.mark.parametrize("kind", ATTACKS)
def test_download_directory_archive_cannot_write_outside(
    sandbox, api, team_files, branch, kind, skip_warnings
):
    """The archive served for a directory download stays inside ``local_save_path``.

    The unsafe member is skipped and the download finishes with the legit files.
    """
    local_save_path = sandbox / "download"
    team_files["body"] = _tar_bytes(
        _remote_dir_members() + _attack_members(kind, sandbox, depth=1)
    )
    before = _snapshot_of(sandbox, local_save_path)

    exc = _attempt(api.file.download_directory, 9, REMOTE_DIR, str(local_save_path))

    # the real download path was taken, and it only talked to the fake platform
    assert team_files["urls"] == [PLATFORM + "/public/api/v3/file-storage.download"]
    assert (
        _snapshot_of(sandbox, local_save_path) == before
    ), "download_directory wrote outside of local_save_path"
    assert exc is None, f"download failed instead of skipping the unsafe member: {exc!r}"
    if _attack_is_skipped(kind, sandbox, branch):
        _assert_skipped(skip_warnings, ATTACK_WARNING_FRAGMENT[kind])
    files = _files_in(local_save_path)
    assert files["a.txt"] == b"file-a"
    assert files["sub/b.txt"] == b"file-b"


@needs_symlinks
def test_download_directory_symlink_escape_blocked(sandbox, api, team_files, branch, skip_warnings):
    """A symlink to outside plus a file written through it, served as a directory."""
    local_save_path = sandbox / "download"
    (sandbox / "outside").mkdir()
    team_files["body"] = _tar_bytes(
        _remote_dir_members() + [_sym("link", "../outside"), _reg("link/PWNED_symlink.txt")]
    )
    before = _snapshot_of(sandbox, local_save_path)

    exc = _attempt(api.file.download_directory, 9, REMOTE_DIR, str(local_save_path))

    assert _snapshot_of(sandbox, local_save_path) == before, "a write went through a symlink"
    assert not os.path.islink(str(local_save_path / "link")), "symlink to outside was created"
    assert exc is None, f"download failed instead of skipping the unsafe member: {exc!r}"
    _assert_skipped(skip_warnings, "'link'")
    files = _files_in(local_save_path)
    assert files["a.txt"] == b"file-a"
    assert files["sub/b.txt"] == b"file-b"


# --- regression tests -------------------------------------------------------


@pytest.mark.parametrize("remote_path", [REMOTE_DIR, REMOTE_DIR + "/"])
def test_download_directory_legit_layout(sandbox, api, team_files, branch, remote_path):
    """A normal directory archive ends up flattened into ``local_save_path``.

    Passes before and after the fix on POSIX.  On Windows the pre-fix code can not get
    this far for an unrelated reason: it never closed the archive before deleting
    ``temp.tar`` (``PermissionError``, WinError 32); the fix closes it.
    """
    local_save_path = sandbox / "download"
    team_files["body"] = _tar_bytes(_remote_dir_members())
    before = _snapshot_of(sandbox, local_save_path)
    progress = []

    api.file.download_directory(9, remote_path, str(local_save_path), progress_cb=progress.append)

    assert _files_in(local_save_path) == {"a.txt": b"file-a", "sub/b.txt": b"file-b"}
    # no temp.tar, no "ds1" folder, no leftover temporary folder
    assert sorted(os.listdir(str(local_save_path))) == ["a.txt", "sub"]
    assert _snapshot_of(sandbox, local_save_path) == before
    assert sum(progress) == len(team_files["body"])


# ===========================================================================
# 3. VideoProject.upload_bin -> VideoProject.restore_snapshot
# ===========================================================================

SNAPSHOT_PROJECT_NAME = "restored videos"
# restore_snapshot extracts into <tempdir>/<mkdtemp>/payload: three levels below sandbox
SNAPSHOT_DEPTH = 3


@pytest.fixture()
def fake_pyarrow(monkeypatch) -> None:
    """``restore_snapshot`` refuses to start without pyarrow.

    The snapshots in this file have no parquet tables, so nothing of pyarrow is ever
    called: an empty stand-in module is enough when the real one is not installed.
    """
    try:
        import pyarrow.parquet  # noqa: F401  pylint: disable=import-error,unused-import

        return
    except Exception:  # pylint: disable=broad-except
        pass
    pyarrow = types.ModuleType("pyarrow")
    parquet = types.ModuleType("pyarrow.parquet")
    pyarrow.parquet = parquet
    monkeypatch.setitem(sys.modules, "pyarrow", pyarrow)
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", parquet)


class _StreamReader:
    """Forward-only decompressed stream, usable as a context manager."""

    def __init__(self, data: bytes):
        self._buffer = io.BytesIO(data)

    def read(self, size=-1):
        return self._buffer.read(size)

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


@pytest.fixture()
def real_zstd():
    """The installed ``zstd`` module, imported as late as the SDK itself imports it."""
    return pytest.importorskip("zstd")


@pytest.fixture()
def compress(real_zstd):
    """tar bytes -> ``.tar.zst`` bytes, the snapshot container format."""
    return real_zstd.compress


@pytest.fixture(params=["streaming", "one_shot"])
def zstd_branch(request, monkeypatch, real_zstd) -> str:
    """Which extraction branch of ``restore_snapshot`` runs first.

    The function tries ``zstd.ZstdDecompressor().stream_reader(...)`` with the tar in
    stream mode and falls back to ``zstd.decompress`` with the tar in normal mode.
    The ``zstd`` package the SDK depends on has no ``ZstdDecompressor``, so with it
    only the one-shot branch can run.  The module the function imports is replaced by
    a thin wrapper around the real one that has (``streaming``) or does not have
    (``one_shot``) that class, which makes both branches reachable and the test
    independent of the installed zstd flavour.
    """
    module = types.ModuleType("zstd")
    module.compress = real_zstd.compress
    module.decompress = real_zstd.decompress
    if request.param == "streaming":

        class ZstdDecompressor:
            def stream_reader(self, source):
                return _StreamReader(real_zstd.decompress(source.read()))

        module.ZstdDecompressor = ZstdDecompressor
    monkeypatch.setitem(sys.modules, "zstd", module)
    return request.param


@pytest.fixture()
def tar_read_modes(monkeypatch) -> list:
    """The mode of every tar opened for reading while the test runs."""
    modes = []
    real_open = tarfile.open

    def recording_open(*args, **kwargs):
        mode = kwargs.get("mode", args[1] if len(args) > 1 else "r")
        if mode.startswith("r"):
            modes.append(mode)
        return real_open(*args, **kwargs)

    monkeypatch.setattr(tarfile, "open", recording_open)
    return modes


@pytest.fixture()
def snapshot_env(sandbox, monkeypatch, fake_pyarrow) -> Path:
    """Send the function's ``tempfile.mkdtemp()`` into the sandbox; return that dir."""
    temp_dir = sandbox / "tmp"
    temp_dir.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(temp_dir))
    return temp_dir


@pytest.fixture()
def platform_api():
    """Stand-in for ``sly.Api``: records what the restore asks the platform to do."""
    api = mock.MagicMock(name="api")
    api.project.exists.return_value = False
    api.project.create.return_value = mock.MagicMock(name="created_project", id=77)
    return api


def _snapshot_payload_files() -> dict:
    """The files of a (videoless) snapshot payload, as ``build_snapshot`` writes them."""
    return {
        "project_info.json": json.dumps(
            {"name": SNAPSHOT_PROJECT_NAME, "description": "from snapshot", "readme": ""}
        ).encode(),
        "project_meta.json": json.dumps(sly.ProjectMeta().to_json()).encode(),
        "key_id_map.json": json.dumps(KeyIdMap().to_dict()).encode(),
        "manifest.json": json.dumps(
            {"schema_version": DEFAULT_VIDEO_SCHEMA_VERSION, "tables": []}
        ).encode(),
    }


def _snapshot_members() -> list:
    # same shape as tar.add(payload_dir, arcname="."): a "." directory and "./name" files
    return [_dir(".")] + [_reg("./" + name, data) for name, data in _snapshot_payload_files().items()]


def _snapshot_tar(extra_members=()) -> bytes:
    """The (not yet compressed) tar of a valid snapshot, plus any extra members."""
    return _tar_bytes(_snapshot_members() + list(extra_members))


def _first_mode(zstd_branch: str) -> str:
    return "r|" if zstd_branch == "streaming" else "r"


# --- vulnerability tests: FAIL on base, PASS on fixed ----------------------


@pytest.mark.parametrize("kind", ATTACKS)
def test_restore_snapshot_archive_cannot_write_outside(
    sandbox,
    snapshot_env,
    platform_api,
    tar_read_modes,
    compress,
    zstd_branch,
    branch,
    kind,
    skip_warnings,
):
    """A snapshot file can not write outside of the function's temporary directory.

    The snapshot is complete and valid apart from the extra members, so before the fix
    the restore goes through without any error while the file outside is written.
    With the fix the extra members are skipped and the restore still goes through.
    """
    snapshot = sandbox / "project.tar.zst"
    snapshot.write_bytes(
        compress(_snapshot_tar(_attack_members(kind, sandbox, SNAPSHOT_DEPTH)))
    )
    before = _snapshot_of(sandbox, snapshot_env)

    exc = _attempt(sly.VideoProject.upload_bin, platform_api, str(snapshot), workspace_id=5)

    assert tar_read_modes[0] == _first_mode(zstd_branch), "the expected branch was not reached"
    assert (
        _snapshot_of(sandbox, snapshot_env) == before
    ), "restore_snapshot wrote outside of its temporary directory"
    assert exc is None, f"restore failed instead of skipping the unsafe member: {exc!r}"
    # the first branch extracted everything it could; nothing fell back to the other one
    assert tar_read_modes == [_first_mode(zstd_branch)]
    if _attack_is_skipped(kind, sandbox, branch):
        _assert_skipped(skip_warnings, ATTACK_WARNING_FRAGMENT[kind])
    platform_api.project.create.assert_called_once()
    create_args = platform_api.project.create.call_args
    assert create_args.args[:3] == (5, SNAPSHOT_PROJECT_NAME, sly.ProjectType.VIDEOS)
    assert _files_in(snapshot_env) == {}


@pytest.mark.parametrize("kind", ATTACKS)
def test_restore_snapshot_bytes_cannot_write_outside(
    sandbox,
    snapshot_env,
    platform_api,
    tar_read_modes,
    compress,
    zstd_branch,
    branch,
    kind,
    skip_warnings,
):
    """Same attack, handed to ``restore_snapshot`` directly as bytes.

    The hostile members come first and the rest of the payload is missing, so the
    function fails later anyway; what matters is whether the write happened first.
    The unsafe member itself no longer raises: the extraction finishes and the error
    is about the missing payload.
    """
    data = compress(_tar_bytes(_attack_members(kind, sandbox, SNAPSHOT_DEPTH)))
    before = _snapshot_of(sandbox, snapshot_env)

    exc = _attempt(
        sly.VideoProject.restore_snapshot, platform_api, snapshot_bytes=data, workspace_id=5
    )

    assert tar_read_modes[0] == _first_mode(zstd_branch), "the expected branch was not reached"
    assert (
        _snapshot_of(sandbox, snapshot_env) == before
    ), "restore_snapshot wrote outside of its temporary directory"
    assert tar_read_modes == [_first_mode(zstd_branch)], "the extraction itself failed"
    assert exc is not None
    assert not isinstance(exc, tarfile.TarError), f"unsafe member was not skipped: {exc!r}"
    if _attack_is_skipped(kind, sandbox, branch):
        _assert_skipped(skip_warnings, ATTACK_WARNING_FRAGMENT[kind])
    platform_api.project.create.assert_not_called()


@needs_symlinks
def test_restore_snapshot_symlink_escape_blocked(
    sandbox, snapshot_env, platform_api, compress, zstd_branch, branch, skip_warnings
):
    """A symlink to outside plus a file written through it, inside a snapshot."""
    (sandbox / "outside").mkdir()
    up = "../" * SNAPSHOT_DEPTH
    data = compress(
        _snapshot_tar([_sym("link", up + "outside"), _reg("link/PWNED_symlink.txt")])
    )
    before = _snapshot_of(sandbox, snapshot_env)

    exc = _attempt(
        sly.VideoProject.restore_snapshot, platform_api, snapshot_bytes=data, workspace_id=5
    )

    assert _snapshot_of(sandbox, snapshot_env) == before, "a write went through a symlink"
    assert exc is None, f"restore failed instead of skipping the unsafe member: {exc!r}"
    _assert_skipped(skip_warnings, "'link'")
    platform_api.project.create.assert_called_once()


# --- regression tests: PASS on both trees ----------------------------------


@pytest.mark.parametrize("source", ["path", "bytesio"])
def test_restore_snapshot_legit(
    sandbox, snapshot_env, platform_api, tar_read_modes, compress, zstd_branch, branch, source
):
    """A well-formed snapshot is still restored, on either branch of the function."""
    data = compress(_snapshot_tar())
    if source == "path":
        file = sandbox / "project.tar.zst"
        file.write_bytes(data)
        file = str(file)
    else:
        file = io.BytesIO(data)
    before = _snapshot_of(sandbox, snapshot_env)

    project = sly.VideoProject.upload_bin(platform_api, file, workspace_id=5, log_progress=False)

    assert project is platform_api.project.create.return_value
    assert tar_read_modes == [_first_mode(zstd_branch)], "the first branch should have succeeded"
    create_args = platform_api.project.create.call_args
    assert create_args.args[:3] == (5, SNAPSHOT_PROJECT_NAME, sly.ProjectType.VIDEOS)
    assert create_args.args[3] == "from snapshot"
    platform_api.project.update_meta.assert_called_once_with(77, sly.ProjectMeta().to_json())
    # nothing outside of the temporary directory, and that directory is cleaned up
    assert _snapshot_of(sandbox, snapshot_env) == before
    assert _files_in(snapshot_env) == {}

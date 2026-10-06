"""Security tests for F3: tar path-traversal in ``sly.fs.unpack_archive`` /
``unpack_archive_async``.

The SDK unpacks user-uploaded import archives.  Before the fix both functions
called ``shutil.unpack_archive`` with no extraction filter, so a tar member named
``../evil`` / ``/abs/evil`` / a symlink or hardlink pointing outside the target
directory escaped it and let an attacker write (as root, in the deployment
containers) anywhere on disk.  The fix routes extraction through
``_unpack_archive_safely`` which uses tarfile's ``data`` filter (or a manual
member-validation fallback on Pythons without extraction filters).  An unsafe
member is SKIPPED with a warning ("Skipping unsafe archive member ...") logged
through the supervisely logger, and extraction continues with the remaining
safe members; no exception is raised for unsafe members.

The vulnerability tests below build the malicious archive programmatically with
``tarfile.TarInfo`` (no OS symlink privilege needed to *create* one), put safe
members before and after the unsafe one, run the real public functions, and
assert: no exception, nothing written outside the target dir, the safe members
extracted, and the skip warning logged.  They FAIL on the pre-fix tree (the
escaping member is written) and PASS on the fixed tree.  Everything is
self-contained and lives under ``tmp_path`` (an escaping member lands at a
sibling *inside* ``tmp_path``, never a real system path).

The error-behaviour regression tests check that non-archives (a tar saved under
a ``.zip`` name, missing paths, directories) still raise exactly what
``shutil.unpack_archive`` raises, as on the pre-fix tree.

Style follows tests/inference_cache (plain functions, pytest fixtures, tmp_path).
"""

import asyncio
import io
import logging
import os
import shutil
import tarfile
import zipfile
from pathlib import Path

import pytest

import supervisely as sly

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

skip_on_windows = pytest.mark.skipif(
    os.name == "nt",
    reason="extracting real symlinks/hardlinks needs privileges on Windows; "
    "the malicious archive is still built here, extraction is validated on POSIX",
)

# (extension, tarfile write mode) for the tar archive formats we cover
TAR_FORMATS = [
    pytest.param(".tar", "w", id="tar"),
    pytest.param(".tar.gz", "w:gz", id="targz"),
    pytest.param(".tgz", "w:gz", id="tgz"),
]


def _run_sync(archive, target, **kwargs):
    return sly.fs.unpack_archive(str(archive), str(target), **kwargs)


def _run_async(archive, target, **kwargs):
    return asyncio.run(sly.fs.unpack_archive_async(str(archive), str(target), **kwargs))


# Both real public entry points must enforce the same safety.
UNPACKERS = [
    pytest.param(_run_sync, id="unpack_archive"),
    pytest.param(_run_async, id="unpack_archive_async"),
]


def _reg_member(name, data=b"owned"):
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.type = tarfile.REGTYPE
    return info, io.BytesIO(data)


def _sym_member(name, linkname):
    info = tarfile.TarInfo(name)
    info.type = tarfile.SYMTYPE
    info.linkname = linkname
    return info, None


def _hardlink_member(name, linkname):
    info = tarfile.TarInfo(name)
    info.type = tarfile.LNKTYPE
    info.linkname = linkname
    return info, None


def _write_tar(path, mode, members):
    """``members`` is a list of ``(TarInfo, fileobj_or_None)`` tuples."""
    with tarfile.open(str(path), mode) as tar:
        for info, fileobj in members:
            if fileobj is None:
                tar.addfile(info)
            else:
                tar.addfile(info, fileobj)


def _write_zip(path, entries):
    """``entries`` is a list of ``(arcname, data)`` tuples."""
    with zipfile.ZipFile(str(path), "w") as zf:
        for arcname, data in entries:
            zf.writestr(arcname, data)


def _unpack_collecting_error(unpacker, archive, target, **kwargs):
    """Run an unpacker; return the exception it raised, or None."""
    try:
        unpacker(archive, target, **kwargs)
    except Exception as e:  # noqa: BLE001 - the test asserts there is none
        return e
    return None


# Safe members placed before and after the unsafe one: an unsafe member is
# skipped and the extraction must carry on with the rest of the archive.
_SAFE_BEFORE = ("good.txt", b"good")
_SAFE_AFTER = ("after/ok.txt", b"ok")


def _with_safe_members(*unsafe_members):
    return [_reg_member(*_SAFE_BEFORE), *unsafe_members, _reg_member(*_SAFE_AFTER)]


def _assert_safe_members_extracted(target: Path):
    assert (target / "good.txt").read_bytes() == b"good", "safe member before the unsafe one"
    assert (target / "after" / "ok.txt").read_bytes() == b"ok", "safe member after the unsafe one"


SKIP_WARNING_PREFIX = "Skipping unsafe archive member"


class _RecordList(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.records = []

    def emit(self, record):
        self.records.append(record)


@pytest.fixture
def sly_log():
    """Records of the supervisely logger (it does not propagate, so caplog's
    root handler would not see them)."""
    handler = _RecordList()
    old_level = sly.logger.level
    sly.logger.addHandler(handler)
    if not sly.logger.isEnabledFor(logging.WARNING):
        sly.logger.setLevel(logging.WARNING)
    try:
        yield handler.records
    finally:
        sly.logger.removeHandler(handler)
        sly.logger.setLevel(old_level)


def _skip_warnings(records, member_hint):
    return [
        r
        for r in records
        if r.levelno == logging.WARNING
        and r.getMessage().startswith(SKIP_WARNING_PREFIX)
        and member_hint in r.getMessage()
    ]


# ---------------------------------------------------------------------------
# vulnerability tests  (FAIL on base, PASS on fixed)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("unpacker", UNPACKERS)
@pytest.mark.parametrize("ext, mode", TAR_FORMATS)
def test_parent_traversal_member_blocked(tmp_path: Path, unpacker, ext, mode, sly_log):
    """A ``../evil`` tar member must not escape the extraction directory; it is
    skipped with a warning and the safe members are still extracted."""
    target = tmp_path / "extract"
    target.mkdir()
    escaped = tmp_path / "PWNED_parent.txt"
    archive = tmp_path / f"evil{ext}"
    _write_tar(archive, mode, _with_safe_members(_reg_member("../PWNED_parent.txt")))

    error = _unpack_collecting_error(unpacker, archive, target)

    assert not escaped.exists(), "tar member escaped target_dir via '..'"
    assert error is None, f"unsafe member aborted the extraction: {error!r}"
    _assert_safe_members_extracted(target)
    assert _skip_warnings(sly_log, "PWNED_parent.txt"), "no skip warning was logged"


@pytest.mark.parametrize("unpacker", UNPACKERS)
@pytest.mark.parametrize("ext, mode", TAR_FORMATS)
def test_deep_parent_traversal_member_blocked(tmp_path: Path, unpacker, ext, mode, sly_log):
    """A deeper ``../../evil`` member must not escape either."""
    target = tmp_path / "nested" / "extract"
    target.mkdir(parents=True)
    escaped = tmp_path / "PWNED_deep.txt"  # two levels up from target
    archive = tmp_path / f"evil_deep{ext}"
    _write_tar(archive, mode, _with_safe_members(_reg_member("../../PWNED_deep.txt")))

    error = _unpack_collecting_error(unpacker, archive, target)

    assert not escaped.exists(), "tar member escaped target_dir via '../../'"
    assert error is None, f"unsafe member aborted the extraction: {error!r}"
    _assert_safe_members_extracted(target)
    assert _skip_warnings(sly_log, "PWNED_deep.txt"), "no skip warning was logged"


@pytest.mark.parametrize("unpacker", UNPACKERS)
@pytest.mark.parametrize("ext, mode", TAR_FORMATS)
def test_absolute_path_member_contained(tmp_path: Path, unpacker, ext, mode, sly_log):
    """An absolute-path member must not be written to that absolute location.

    On Windows the drive survives stripping, so the member is skipped with a
    warning; on POSIX the leading separator is stripped and the file is kept
    *inside* the target.  Either way the attacker-chosen absolute path stays
    empty and no exception is raised.  We aim the absolute path at a sibling
    inside ``tmp_path`` so nothing touches a real system location.
    """
    target = tmp_path / "extract"
    target.mkdir()
    escaped = tmp_path / "PWNED_abs.txt"
    abs_name = str(escaped).replace(os.sep, "/")  # tar uses forward slashes
    archive = tmp_path / f"evil_abs{ext}"
    _write_tar(archive, mode, _with_safe_members(_reg_member(abs_name)))

    error = _unpack_collecting_error(unpacker, archive, target)

    assert not escaped.exists(), "absolute-path member was written outside target_dir"
    assert error is None, f"unsafe member aborted the extraction: {error!r}"
    _assert_safe_members_extracted(target)
    if os.name == "nt":
        assert _skip_warnings(sly_log, "PWNED_abs.txt"), "no skip warning was logged"
    else:
        assert (target / abs_name.lstrip("/")).is_file(), "stripped member not kept inside target"


@pytest.mark.parametrize("unpacker", UNPACKERS)
@pytest.mark.parametrize("ext, mode", TAR_FORMATS)
def test_fallback_branch_parent_traversal_blocked(
    tmp_path: Path, unpacker, ext, mode, monkeypatch, sly_log
):
    """Same ``../`` attack with tarfile's extraction filters removed.

    This exercises ``_unpack_archive_safely``'s manual member-validation
    fallback (used on Python builds without ``tarfile.data_filter``) by deleting
    that attribute.  On the pre-fix tree the fallback does not exist, so the
    member still escapes.
    """
    monkeypatch.delattr(tarfile, "data_filter", raising=False)

    target = tmp_path / "extract"
    target.mkdir()
    escaped = tmp_path / "PWNED_fallback.txt"
    archive = tmp_path / f"evil_fb{ext}"
    _write_tar(archive, mode, _with_safe_members(_reg_member("../PWNED_fallback.txt")))

    error = _unpack_collecting_error(unpacker, archive, target)

    assert not escaped.exists(), "fallback branch let a '..' member escape target_dir"
    assert error is None, f"fallback branch aborted the extraction: {error!r}"
    _assert_safe_members_extracted(target)
    assert _skip_warnings(sly_log, "PWNED_fallback.txt"), "no skip warning was logged"


@skip_on_windows
@pytest.mark.parametrize("unpacker", UNPACKERS)
def test_symlink_escape_blocked(tmp_path: Path, unpacker, sly_log):
    """A symlink pointing outside the target, followed by a file written through
    it, must not land outside the target directory (classic symlink escape).
    The link is skipped; the same-named regular file then lands inside target."""
    target = tmp_path / "extract"
    target.mkdir()
    escaped = tmp_path / "PWNED_symlink.txt"
    archive = tmp_path / "evil_symlink.tar"
    # "link" -> tmp_path (outside target); then write through it.
    _write_tar(
        archive,
        "w",
        _with_safe_members(
            _sym_member("link", str(tmp_path)),
            _reg_member("link/PWNED_symlink.txt"),
        ),
    )

    error = _unpack_collecting_error(unpacker, archive, target)

    assert not escaped.exists(), "symlink member allowed a write outside target_dir"
    assert not (target / "link").is_symlink(), "symlink to outside target_dir was created"
    assert error is None, f"unsafe member aborted the extraction: {error!r}"
    _assert_safe_members_extracted(target)
    assert _skip_warnings(sly_log, "link"), "no skip warning was logged"


@skip_on_windows
@pytest.mark.parametrize("unpacker", UNPACKERS)
def test_hardlink_outside_blocked(tmp_path: Path, unpacker, sly_log):
    """A hardlink member pointing at a file outside the target must be skipped;
    the dangerous link must not be created inside the target."""
    outside = tmp_path / "secret.txt"
    outside.write_bytes(b"top-secret")
    target = tmp_path / "extract"
    target.mkdir()
    link_inside = target / "hl"
    archive = tmp_path / "evil_hardlink.tar"
    _write_tar(archive, "w", _with_safe_members(_hardlink_member("hl", "../secret.txt")))

    error = _unpack_collecting_error(unpacker, archive, target)

    assert not link_inside.exists(), "hardlink to a file outside target_dir was created"
    assert error is None, f"unsafe member aborted the extraction: {error!r}"
    _assert_safe_members_extracted(target)
    assert _skip_warnings(sly_log, "hl"), "no skip warning was logged"


# ---------------------------------------------------------------------------
# regression tests  (PASS on both trees)
# ---------------------------------------------------------------------------


def _build_legit_tar(path, mode):
    _write_tar(
        path,
        mode,
        [
            _reg_member("top.txt", b"top-level"),
            _reg_member("folder/a.txt", b"nested-a"),
            _reg_member("folder/sub/b.txt", b"nested-b"),
        ],
    )


def _assert_legit_extracted(target: Path):
    assert (target / "top.txt").read_bytes() == b"top-level"
    assert (target / "folder" / "a.txt").read_bytes() == b"nested-a"
    assert (target / "folder" / "sub" / "b.txt").read_bytes() == b"nested-b"


@pytest.mark.parametrize("unpacker", UNPACKERS)
@pytest.mark.parametrize("ext, mode", TAR_FORMATS)
def test_legit_tar_roundtrip(tmp_path: Path, unpacker, ext, mode):
    """Ordinary tar / tar.gz / tgz archives with nested folders still extract."""
    target = tmp_path / "extract"
    target.mkdir()
    archive = tmp_path / f"legit{ext}"
    _build_legit_tar(archive, mode)

    unpacker(archive, target)

    _assert_legit_extracted(target)


@pytest.mark.parametrize("unpacker", UNPACKERS)
def test_legit_tar_roundtrip_fallback_branch(tmp_path: Path, unpacker, monkeypatch):
    """A legit tar still extracts correctly through the manual-validation fallback."""
    monkeypatch.delattr(tarfile, "data_filter", raising=False)
    target = tmp_path / "extract"
    target.mkdir()
    archive = tmp_path / "legit_fb.tar"
    _build_legit_tar(archive, "w")

    unpacker(archive, target)

    _assert_legit_extracted(target)


@pytest.mark.parametrize("unpacker", UNPACKERS)
def test_legit_zip_roundtrip(tmp_path: Path, unpacker):
    """Ordinary zip archives with nested folders still extract (zip is delegated
    to shutil, whose zip extraction already sanitizes members)."""
    target = tmp_path / "extract"
    target.mkdir()
    archive = tmp_path / "legit.zip"
    _write_zip(
        archive,
        [
            ("top.txt", b"top-level"),
            ("folder/a.txt", b"nested-a"),
            ("folder/sub/b.txt", b"nested-b"),
        ],
    )

    unpacker(archive, target)

    _assert_legit_extracted(target)


@pytest.mark.parametrize("unpacker", UNPACKERS)
def test_zip_traversal_stays_contained(tmp_path: Path, unpacker):
    """A zip traversal attempt must stay contained.

    ``shutil._unpack_zipfile`` skips any member whose name contains ``..`` (or
    is absolute), so the escaping member is written nowhere outside the target.
    This already holds on both trees; it is a regression guard for zip handling.
    """
    target = tmp_path / "extract"
    target.mkdir()
    escaped = tmp_path / "PWNED_zip.txt"
    archive = tmp_path / "evil.zip"
    _write_zip(archive, [("../PWNED_zip.txt", b"owned"), ("good.txt", b"good")])

    unpacker(archive, target)

    assert not escaped.exists(), "zip member escaped target_dir"
    # a legitimate sibling member in the same archive still extracts
    assert (target / "good.txt").read_bytes() == b"good"


@pytest.mark.parametrize("unpacker", UNPACKERS)
def test_is_split_roundtrip(tmp_path: Path, unpacker):
    """The is_split option (reassemble parts, then extract) still works."""
    split_dir = tmp_path / "parts"
    split_dir.mkdir()
    target = tmp_path / "extract"
    target.mkdir()

    whole = tmp_path / "payload.tar"
    _build_legit_tar(whole, "w")
    raw = whole.read_bytes()
    half = len(raw) // 2
    (split_dir / "payload.tar.001").write_bytes(raw[:half])
    (split_dir / "payload.tar.002").write_bytes(raw[half:])

    unpacker(split_dir / "payload.tar.001", target, is_split=True)

    _assert_legit_extracted(target)


@pytest.mark.parametrize("unpacker", UNPACKERS)
def test_remove_junk_option(tmp_path: Path, unpacker):
    """remove_junk removes junk files after extraction, and keeps them when off."""
    # remove_junk=True (default): .DS_Store removed, real file kept
    target_on = tmp_path / "on"
    target_on.mkdir()
    archive_on = tmp_path / "junk_on.tar"
    _write_tar(
        archive_on,
        "w",
        [_reg_member(".DS_Store", b"junk"), _reg_member("keep.txt", b"keep")],
    )
    unpacker(archive_on, target_on, remove_junk=True)
    assert (target_on / "keep.txt").read_bytes() == b"keep"
    assert not (target_on / ".DS_Store").exists()

    # remove_junk=False: junk stays
    target_off = tmp_path / "off"
    target_off.mkdir()
    archive_off = tmp_path / "junk_off.tar"
    _write_tar(
        archive_off,
        "w",
        [_reg_member(".DS_Store", b"junk"), _reg_member("keep.txt", b"keep")],
    )
    unpacker(archive_off, target_off, remove_junk=False)
    assert (target_off / "keep.txt").read_bytes() == b"keep"
    assert (target_off / ".DS_Store").exists()


def _make_bad_archive_path(kind: str, tmp_path: Path) -> Path:
    if kind == "tar_saved_as_zip":
        path = tmp_path / "really_a_tar.zip"
        _build_legit_tar(path, "w")
    elif kind == "missing_zip":
        path = tmp_path / "missing.zip"
    elif kind == "missing_tar":
        path = tmp_path / "missing.tar"
    elif kind == "directory":
        path = tmp_path / "some_dir"
        path.mkdir()
    elif kind == "directory_named_tar":
        path = tmp_path / "some_dir.tar"
        path.mkdir()
    else:
        raise AssertionError(kind)
    return path


# (case, exception type shutil.unpack_archive raises for it on every platform;
# a directory named *.tar raises PermissionError on Windows, IsADirectoryError on
# POSIX, so only the common OSError base is pinned there)
BAD_ARCHIVES = [
    pytest.param("tar_saved_as_zip", shutil.ReadError, id="tar-saved-as-zip"),
    pytest.param("missing_zip", shutil.ReadError, id="missing-zip"),
    pytest.param("missing_tar", FileNotFoundError, id="missing-tar"),
    pytest.param("directory", shutil.ReadError, id="directory"),
    pytest.param("directory_named_tar", OSError, id="directory-named-tar"),
]


@pytest.mark.parametrize("unpacker", UNPACKERS)
@pytest.mark.parametrize("kind, expected", BAD_ARCHIVES)
def test_non_archive_errors_unchanged(tmp_path: Path, unpacker, kind, expected):
    """Non-archives raise exactly what ``shutil.unpack_archive`` raises (the
    pre-fix behaviour), never a TypeError from the safe-extraction path."""
    target = tmp_path / "extract"
    target.mkdir()
    path = _make_bad_archive_path(kind, tmp_path)

    with pytest.raises(Exception) as reference:
        shutil.unpack_archive(str(path), str(target))
    with pytest.raises(expected) as raised:
        unpacker(path, target)

    assert not isinstance(raised.value, TypeError)
    assert type(raised.value) is type(reference.value), (
        f"{type(raised.value).__name__} raised, shutil raises "
        f"{type(reference.value).__name__}"
    )
    assert list(target.iterdir()) == [], "something was extracted from a non-archive"

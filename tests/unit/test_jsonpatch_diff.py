"""Fuzz tests for JSON-patch generation over FastTable-like payloads.

Upstream jsonpatch (<=1.33) folds an add and a value-equal remove into a
single 'move' even across unrelated subtrees and rewrites neighbouring list
indices incorrectly: from_diff(pre, post).apply(pre) raises JsonPatchConflict
or silently yields a document != post. The trigger is compound values
(dicts/lists) shared between sibling lists that change in the same diff,
which is exactly what FastTable payloads look like. The hit rate depends on
PYTHONHASHSEED, but the aggregate over 2000 pairs is stable at any seed.

The SDK must broadcast patches that always replay correctly, since the
frontend applies them as-is. Run as `python -m pytest` from the repo root
(bare `pytest` may import an installed supervisely instead of this checkout).
"""

import copy
import random

import jsonpatch
import pytest

import supervisely.app.content as content
from supervisely.app.singleton import Singleton

# _safe_diff prefers patchdiff and falls back to jsonpatch's from_diff when
# patchdiff is not installed (python 3.8). Every engine available in this
# environment is tested; the fallback is always tested by forcing it.
ENGINES = ["patchdiff", "jsonpatch-fallback"] if content.patchdiff is not None else ["jsonpatch-fallback"]

N_PAIRS = 2000

COLPOOL = ["class", "count", "score", "iou"]
OPTSPOOL = [{"type": "class"}, {"maxValue": 100}, {"postfix": "%"}, {}]
CELLS = ["cat", "dog", 0, 1, 100]


def _random_table_payload(rng):
    n = rng.randint(2, 4)
    cols = rng.sample(COLPOOL, k=n)
    opts = [copy.deepcopy(rng.choice(OPTSPOOL)) for _ in range(n)]
    rows = [[rng.choice(CELLS) for _ in range(n)] for _ in range(rng.randint(2, 5))]
    for row in rows:
        row[0] = rng.choice(["cat", "dog"])
    return {"columns": cols, "columnsOptions": opts, "data": rows}


@pytest.fixture(params=ENGINES)
def diff_engine(request, monkeypatch):
    if request.param == "jsonpatch-fallback":
        monkeypatch.setattr(content, "patchdiff", None)
    return request.param


@pytest.fixture(scope="module")
def doc_pairs():
    """Consecutive DataJson snapshots around FastTable.read_json calls."""
    saved_instances = dict(Singleton._instances)
    saved_nested = dict(Singleton._nested_instances)
    Singleton._instances.clear()
    Singleton._nested_instances.clear()
    try:
        from supervisely.app.content import DataJson
        from supervisely.app.widgets import FastTable

        table = FastTable()
        rng = random.Random(0)
        pairs = []
        prev = copy.deepcopy(dict(DataJson()))
        for _ in range(N_PAIRS):
            table._columns_first_idx = None  # otherwise columns get pinned by the first read
            table.read_json(_random_table_payload(rng))
            cur = copy.deepcopy(dict(DataJson()))
            pairs.append((prev, cur))
            prev = cur
        return pairs
    finally:
        Singleton._instances.clear()
        Singleton._instances.update(saved_instances)
        Singleton._nested_instances.clear()
        Singleton._nested_instances.update(saved_nested)


def _replay_errors(pairs, diff_fn):
    apply_failed = mismatch = 0
    for pre, post in pairs:
        patch = diff_fn(pre, post)
        try:
            result = patch.apply(pre)
        except jsonpatch.JsonPatchException:
            apply_failed += 1
            continue
        if result != post:
            mismatch += 1
    return apply_failed, mismatch


def test_safe_diff_replays_exactly(doc_pairs, diff_engine):
    apply_failed = mismatch = moves = 0
    for pre, post in doc_pairs:
        patch = content._safe_diff(pre, post)
        moves += sum(op["op"] == "move" for op in patch.patch)
        try:
            result = patch.apply(pre)
        except jsonpatch.JsonPatchException:
            apply_failed += 1
            continue
        if result != post:
            mismatch += 1
    assert (apply_failed, mismatch) == (0, 0)
    if diff_engine == "patchdiff":
        assert moves == 0, "patchdiff never folds add+remove into 'move'"


@pytest.mark.xfail(
    strict=False,
    reason="upstream jsonpatch bug: add+remove folds into 'move' across unrelated "
    "subtrees; hit rate is PYTHONHASHSEED-dependent, so occasionally 0/2000",
)
def test_bare_jsonpatch_from_diff_replays_exactly(doc_pairs):
    apply_failed, mismatch = _replay_errors(doc_pairs, jsonpatch.JsonPatch.from_diff)
    assert (apply_failed, mismatch) == (0, 0)


def test_safe_diff_scalar_change_is_single_replace(diff_engine):
    patch = content._safe_diff({"a": 1, "b": 2}, {"a": 3, "b": 2})
    assert patch.patch == [{"op": "replace", "path": "/a", "value": 3}]


def test_safe_diff_falls_back_on_diff_exception(monkeypatch):
    # if the diff step raises for any reason, _safe_diff must not propagate:
    # it falls back to an explicit per-top-level-key replace
    def _raise(src, dst):
        raise ValueError("simulated diff failure")

    monkeypatch.setattr(content, "_diff", _raise)
    patch = content._safe_diff({"a": 1}, {"a": 2})
    assert patch.apply({"a": 1}) == {"a": 2}


def test_safe_diff_falls_back_on_broken_from_diff_patch(monkeypatch):
    # python 3.8 path: from_diff result is verified inside _diff; a broken
    # patch raises there and _safe_diff falls back to _fallback_patch
    broken = jsonpatch.JsonPatch([{"op": "remove", "path": "/nonexistent"}])
    monkeypatch.setattr(content, "patchdiff", None)
    monkeypatch.setattr(
        jsonpatch.JsonPatch, "from_diff", classmethod(lambda cls, src, dst: broken)
    )
    patch = content._safe_diff({"a": 1}, {"a": 2})
    assert patch.apply({"a": 1}) == {"a": 2}


def test_safe_diff_all_changed_rows_replay(diff_engine):
    pre = {"t": {"data": [{"idx": i, "v": f"a{i}"} for i in range(150)]}}
    post = {"t": {"data": [{"idx": i, "v": f"b{i}"} for i in range(150)]}}
    patch = content._safe_diff(pre, post)
    assert patch.apply(pre) == post


def test_safe_diff_prepend(diff_engine):
    rows = [{"idx": i, "items": [i, f"r{i}"]} for i in range(2000)]
    pre = {"t": {"data": copy.deepcopy(rows)}}
    post = {"t": {"data": [{"idx": -1, "items": [-1, "new"]}] + copy.deepcopy(rows)}}
    patch = content._safe_diff(pre, post)
    assert patch.apply(pre) == post
    if diff_engine == "patchdiff":
        assert len(patch.patch) <= 3, f"prepend produced {len(patch.patch)} ops instead of ~1"


def test_fallback_patch_handles_all_key_ops():
    src = {"same": 1, "changed": {"x": [1, 2]}, "gone": "bye", "we/ird~key": 1}
    dst = {"same": 1, "changed": {"x": [2]}, "added": [{"k": "v"}], "we/ird~key": 2}
    patch = content._fallback_patch(src, dst)
    assert patch.apply(src) == dst
    assert "/same" not in {op["path"] for op in patch.patch}

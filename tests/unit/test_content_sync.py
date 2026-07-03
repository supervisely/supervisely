"""Stress tests for DataJson/StateJson concurrent send_changes().

Before the fix, the patch was computed outside the lock, causing
JsonPatchConflict or silent dropped updates under concurrency.
"""

import copy
import threading
from concurrent.futures import ThreadPoolExecutor

import jsonpatch
import pytest

import supervisely.app.content as content
from supervisely.app.content import DataJson, StateJson
from supervisely.app.fastapi.websocket import WebsocketManager
from supervisely.app.singleton import Singleton

N_THREADS = 8
N_ITERS = 50

INITIAL = {"counters": {}, "items": []}


@pytest.fixture(autouse=True)
def isolated_singletons(monkeypatch):
    saved_instances = dict(Singleton._instances)
    saved_nested = dict(Singleton._nested_instances)
    Singleton._instances.clear()
    Singleton._nested_instances.clear()
    monkeypatch.delenv("SUPERVISELY_MULTIUSER_APP_MODE", raising=False)
    yield
    Singleton._instances.clear()
    Singleton._instances.update(saved_instances)
    Singleton._nested_instances.clear()
    Singleton._nested_instances.update(saved_nested)


@pytest.fixture()
def broadcasts(monkeypatch):
    recorded = []
    record_lock = threading.Lock()

    async def recording_broadcast(self, d, user_id=None):
        with record_lock:
            recorded.append(copy.deepcopy(d))

    monkeypatch.setattr(WebsocketManager, "broadcast", recording_broadcast)
    return recorded


def _stress(obj):
    obj.update(copy.deepcopy(INITIAL))
    obj._last = copy.deepcopy(dict(obj))
    errors = []
    err_lock = threading.Lock()
    mutate_lock = threading.Lock()

    def worker(tid):
        for i in range(N_ITERS):
            with mutate_lock:
                obj["counters"][f"t{tid}"] = i
                obj["items"] = list(range((tid * N_ITERS + i) % 7))
            try:
                obj.send_changes()
            except BaseException as e:  # noqa: B036
                with err_lock:
                    errors.append(f"{type(e).__name__}: {e}")

    with ThreadPoolExecutor(max_workers=N_THREADS) as pool:
        futures = [pool.submit(worker, tid) for tid in range(N_THREADS)]
        for f in futures:
            f.result(timeout=120)

    obj.send_changes()  # final flush
    return errors


def _assert_consistent(obj, field, errors, broadcasts):
    assert not errors, f"{len(errors)} exceptions during concurrent send_changes: {errors[:3]}"
    assert obj._last == dict(obj), "silent divergence: _last != dict(self)"

    # replaying patches must reproduce final state (frontend consistency check)
    doc = copy.deepcopy(INITIAL)
    for payload in broadcasts:
        ops = payload.get(field)
        if ops is None:
            continue
        doc = jsonpatch.JsonPatch(ops).apply(doc)
    assert doc == dict(obj), "frontend replay diverges from backend state"


def test_concurrent_send_changes_datajson(broadcasts):
    data = DataJson()
    errors = _stress(data)
    _assert_consistent(data, "data", errors, broadcasts)


def test_concurrent_send_changes_statejson(broadcasts):
    # also covers the nested-lock path (self._lock → ContentOrigin)
    state = StateJson()
    errors = _stress(state)
    _assert_consistent(state, "state", errors, broadcasts)


def test_no_deadlock_on_small_pool(broadcasts, monkeypatch):
    # waiters used to block pool threads indefinitely, deadlocking a small pool
    monkeypatch.setattr(content, "_pool", ThreadPoolExecutor(max_workers=2))
    state = StateJson()
    state.update(copy.deepcopy(INITIAL))
    state._last = copy.deepcopy(dict(state))
    errors = []

    def worker(tid):
        for i in range(20):
            state["counters"][f"t{tid}"] = i
            try:
                state.send_changes()
            except BaseException as e:  # noqa: B036
                errors.append(f"{type(e).__name__}: {e}")

    threads = [threading.Thread(target=worker, args=(tid,)) for tid in range(N_THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    assert not any(t.is_alive() for t in threads), "deadlock: workers did not finish"
    assert not errors

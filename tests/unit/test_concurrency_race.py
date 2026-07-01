"""
Reproduction tests for the DataJson/StateJson `send_changes()` race
condition.

`_PatchableJson._get_patch()` (diff generation) is not protected by any lock
-- only `_apply_patch()` (mutating `self._last`) is:

    async def synchronize_changes(self, user_id=None):
        patch = self._get_patch()          # <-- unprotected read of self._last
        ...
        await self._apply_patch(patch)     # <-- lock-protected mutation

When two calls to `send_changes()` run concurrently, both can read the same
`_last` baseline before either applies its patch. Whichever applies second
ends up applying a patch computed against an already-stale baseline --
`jsonpatch.JsonPatchConflict` ("can't remove/replace ...") or a silently
dropped update. This is a pre-existing issue independent of which diffing
library computes the patch (reproduced identically on `jsonpatch` and on an
unreleased `patchdiff`-based migration branch).

Each test below reproduces one REALISTIC scenario where this fires WITHOUT
any "incorrect" app code:

  1. Two different users, each doing one normal click, in a multi-user app
     (`session_context()` only tags log/WS routing -- it does not isolate
     DataJson/StateJson per user; there is a single shared singleton).
  2. A `Progress`-driven background task (the officially recommended pattern
     for long-running work) racing a click on an unrelated widget.
  3. Two independently live-updating widgets (e.g. two live charts), each
     driven by its own loop with no coordination between them.
  4. Ordinary rapid / double clicking by a single user on one widget.

These tests assert zero errors across N concurrent repetitions -- the
correct invariant once the race is fixed (e.g. by locking the whole
get-patch-then-apply cycle). They are expected to FAIL on the current
codebase; a fix should make them pass without lowering N or adding retries.

See tests/manual/concurrency_race_demo/ for the same four scenarios,
reproducible by hand in an actual browser against a running app.
"""

import random
import threading

import pytest

from supervisely.app.content import DataJson
from supervisely.app.fastapi import multi_user
from supervisely.app.widgets import Progress, Text


@pytest.fixture
def data_json():
    d = DataJson()
    d.clear()
    d._last = {}
    yield d
    d.clear()
    d._last = {}


def _run_concurrently(targets):
    """Run each zero-arg callable in its own thread; return exceptions raised."""
    errors = []
    lock = threading.Lock()

    def _wrap(fn):
        try:
            fn()
        except Exception as e:
            with lock:
                errors.append(f"{type(e).__name__}: {e}")

    threads = [threading.Thread(target=_wrap, args=(fn,)) for fn in targets]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return errors


def _mutate_list(widget, n_items=None):
    """Replace a widget's own list-valued key with a new, differently-sized
    list. This is the shape that turns a stale patch fatal: a scalar replace
    is a harmless no-op even under the exact same race window, because it
    isn't expressed as index-based add/remove operations."""
    n = n_items if n_items is not None else random.randint(5, 15)
    DataJson()[widget.widget_id]["items"] = [random.randint(0, 50) for _ in range(n)]
    DataJson().send_changes()


# ---------------------------------------------------------------------------
# Scenario 1 -- two different users clicking at the same time (multi-user app)
# ---------------------------------------------------------------------------


def test_two_different_users_concurrent_actions(data_json):
    """Two users, each doing one independent, individually-correct action at
    the same time in a multi-user app, must not corrupt each other's state."""
    widget_a = Text("User A", status="text")
    widget_b = Text("User B", status="text")

    def user_a_action():
        with multi_user.session_context("user-a"):
            _mutate_list(widget_a)

    def user_b_action():
        with multi_user.session_context("user-b"):
            _mutate_list(widget_b)

    errors = []
    for _ in range(20):
        errors += _run_concurrently([user_a_action, user_b_action])

    assert errors == [], f"{len(errors)} concurrent multi-user actions raised: {errors[:3]}"


# ---------------------------------------------------------------------------
# Scenario 2 -- Progress-bar background task racing a click on another widget
# ---------------------------------------------------------------------------


def test_progress_background_task_races_other_widget(data_json):
    """A Progress-driven background task (the recommended pattern for
    long-running work) must not corrupt state when the user interacts with a
    different widget while it runs.

    Drives the progress widget's DataJson entry directly (matching what each
    `pbar.update()` call does under the hood) rather than through
    Progress.__call__'s context manager, which pulls in async/websocket
    machinery that assumes a running application -- irrelevant to the race
    itself and not available in a plain unit test process."""
    progress = Progress("Long task", show_percents=True)
    other_widget = Text("Other widget", status="text")

    def run_task():
        for _ in range(15):
            _mutate_list(progress)

    def click_other(i):
        _mutate_list(other_widget)

    errors = []
    lock = threading.Lock()

    def _wrap(fn, *a):
        try:
            fn(*a)
        except Exception as e:
            with lock:
                errors.append(f"{type(e).__name__}: {e}")

    task_thread = threading.Thread(target=_wrap, args=(run_task,))
    click_threads = [threading.Thread(target=_wrap, args=(click_other, i)) for i in range(15)]

    task_thread.start()
    for t in click_threads:
        t.start()
    for t in click_threads:
        t.join()
    task_thread.join()

    assert errors == [], f"{len(errors)} calls raised during background task + concurrent click: {errors[:3]}"


# ---------------------------------------------------------------------------
# Scenario 3 -- two independently live-updating widgets, no coordination
# ---------------------------------------------------------------------------


def test_two_independent_live_widgets(data_json):
    """Two widgets updated by independent loops (e.g. two live charts) with
    no coordination between them must not corrupt each other's state."""
    feed_a = Text("Feed A", status="text")
    feed_b = Text("Feed B", status="text")

    def feed_loop(widget, n_ticks, errors, lock):
        for _ in range(n_ticks):
            try:
                _mutate_list(widget)
            except Exception as e:
                with lock:
                    errors.append(f"{type(e).__name__}: {e}")

    errors = []
    lock = threading.Lock()
    threads = [
        threading.Thread(target=feed_loop, args=(feed_a, 20, errors, lock)),
        threading.Thread(target=feed_loop, args=(feed_b, 20, errors, lock)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == [], f"{len(errors)}/40 ticks raised across two independent live widgets: {errors[:3]}"


# ---------------------------------------------------------------------------
# Scenario 4 -- rapid / double clicking by a single user on one widget
# ---------------------------------------------------------------------------


def test_rapid_clicks_same_widget(data_json):
    """Ordinary rapid clicking (or two near-simultaneous requests) on a
    single widget must not corrupt its own state."""
    widget = Text("Rapid-click target", status="text")

    errors = _run_concurrently([lambda: _mutate_list(widget) for _ in range(25)])

    assert errors == [], f"{len(errors)}/25 rapid clicks raised: {errors[:3]}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

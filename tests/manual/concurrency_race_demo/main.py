"""
End-to-end reproduction of DataJson/StateJson concurrency races, for manual
verification in a real browser.

Root cause: `_PatchableJson._get_patch()` (diff generation) is not protected
by any lock -- only `_apply_patch()` (mutating `self._last`) is. When two
concurrent calls to `send_changes()` both read the shared state before either
one applies its patch, the second one to apply ends up applying a patch
computed against an already-stale baseline: crash or silently dropped update.

This file has one section per REALISTIC scenario where that race fires
WITHOUT any "incorrect" app code -- multi-user apps, Progress-driven
background tasks, independently live-updating widgets, and ordinary rapid
clicking. Each section has:
  - a "single try" button -- click it by hand, timing-dependent, may or may
    not show a failure on any single attempt
  - a "STRESS xN" button -- fires N concurrent repetitions server-side for a
    reliable, guaranteed-to-reproduce demonstration

See tests/unit/test_concurrency_race.py for the automated (headless, no
browser) version of the same four scenarios.

Run (from the repo root):
    PYTHONPATH=. uvicorn tests.manual.concurrency_race_demo.main:app \
        --host 0.0.0.0 --port 8000 --ws websockets --reload
Then open http://localhost:8000/ and click through each scenario's buttons.
"""
import os
import random
import threading
import time

os.environ.setdefault("SUPERVISELY_MULTIUSER_APP_MODE", "true")

import supervisely as sly
from supervisely.app.content import DataJson
from supervisely.app.fastapi import multi_user
from supervisely.app.widgets import Button, Card, Container, Progress, Text

# ---------------------------------------------------------------------------
# Scenario 1 -- two different users, each doing one normal click, at the
# same time (multi-user apps do NOT get isolated DataJson/StateJson: it is
# one shared singleton for the whole process, `session_context` only tags
# log/WS routing).
# ---------------------------------------------------------------------------
s1_widget_a = Text("User A: 0 updates", status="text")
s1_widget_b = Text("User B: 0 updates", status="text")
s1_status = Text("Ready.", status="text")
s1_btn_single = Button("Scenario 1 · fire User A + User B once, simultaneously")
s1_btn_stress = Button("Scenario 1 · STRESS x20 simultaneous pairs")


def _s1_user_action(user_id, widget, counter_key, n_items):
    with multi_user.session_context(user_id):
        wid = widget.widget_id
        DataJson()[wid]["items"] = [random.randint(0, 50) for _ in range(n_items)]
        DataJson()[wid]["text"] = f"User {user_id}: {counter_key} updates"
        DataJson().send_changes()


def _s1_fire_pair(n_items_a, n_items_b, results, idx):
    errors = []
    threads = [
        threading.Thread(target=lambda: _try(errors, "A", lambda: _s1_user_action("A", s1_widget_a, idx, n_items_a))),
        threading.Thread(target=lambda: _try(errors, "B", lambda: _s1_user_action("B", s1_widget_b, idx, n_items_b))),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    results.extend(errors)


def _try(errors_list, label, fn):
    try:
        fn()
    except Exception as e:
        errors_list.append(f"{label}: {type(e).__name__}: {e}")


@s1_btn_single.click
def s1_do_single():
    results = []
    _s1_fire_pair(random.randint(5, 15), random.randint(5, 15), results, 0)
    if results:
        s1_status.set(f"Scenario 1 (single try): CRASHED -- {results[0]}", status="error")
    else:
        s1_status.set("Scenario 1 (single try): clean this time (timing-dependent -- use STRESS for a guaranteed repro).", status="success")


@s1_btn_stress.click
def s1_do_stress():
    # Fire all 40 (20 pairs x User A/B) threads at once instead of pair-by-pair
    # -- sequential pairs barely overlap, so the race window is too narrow to
    # reliably hit. Real concurrent users don't politely wait their turn either.
    results = []
    threads = []
    for i in range(20):
        threads.append(threading.Thread(target=lambda i=i: _try(results, "A", lambda: _s1_user_action("A", s1_widget_a, i, random.randint(5, 15)))))
        threads.append(threading.Thread(target=lambda i=i: _try(results, "B", lambda: _s1_user_action("B", s1_widget_b, i, random.randint(5, 15)))))
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if results:
        s1_status.set(f"Scenario 1 STRESS: {len(results)}/40 calls raised. First: {results[0]}", status="error")
    else:
        s1_status.set("Scenario 1 STRESS: 0/20 raised. Clean.", status="success")


scenario_1 = Card(
    title="Scenario 1 -- two different users clicking at the same time (multi-user mode)",
    content=Container(widgets=[s1_widget_a, s1_widget_b, s1_status, Container(widgets=[s1_btn_single, s1_btn_stress], direction="horizontal")]),
)

# ---------------------------------------------------------------------------
# Scenario 2 -- a Progress-driven background task (the officially recommended
# pattern for long-running work) racing with a click on a completely
# unrelated widget while it runs.
# ---------------------------------------------------------------------------
s2_progress = Progress("Long task", show_percents=True)
s2_other_widget = Text("Other widget: untouched", status="text")
s2_status = Text("Ready.", status="text")
s2_btn_start_task = Button("Scenario 2 · start background task (10 x 0.1s progress ticks)")
s2_btn_click_other = Button("Scenario 2 · click ME while the task above is running")
s2_btn_stress = Button("Scenario 2 · STRESS x15 (task + 15 rapid clicks while it runs)")

_s2_errors = []
_s2_errors_lock = threading.Lock()


def _s2_run_task(n_ticks=10, tick_s=0.1):
    try:
        with s2_progress(message="Processing...", total=n_ticks) as pbar:
            for i in range(n_ticks):
                time.sleep(tick_s)
                # Varying-length list under the progress widget's OWN key --
                # same shape as the "other widget" mutation, so both sides of
                # the race are hitting index-based list ops.
                wid = s2_progress.widget_id
                DataJson()[wid]["log"] = [random.randint(0, 50) for _ in range(random.randint(5, 15))]
                DataJson().send_changes()
                pbar.update(1)
    except Exception as e:
        with _s2_errors_lock:
            _s2_errors.append(f"progress: {type(e).__name__}: {e}")


def _s2_click_other(i):
    try:
        # A varying-length list is what exposes the index-based race (a plain
        # scalar .set() call is benign even under the same race window).
        wid = s2_other_widget.widget_id
        DataJson()[wid]["items"] = [random.randint(0, 50) for _ in range(random.randint(5, 15))]
        s2_other_widget.set(f"Other widget: click #{i}", status="info")
    except Exception as e:
        with _s2_errors_lock:
            _s2_errors.append(f"other_widget: {type(e).__name__}: {e}")


@s2_btn_start_task.click
def s2_do_start_task():
    t = threading.Thread(target=_s2_run_task, args=(10, 0.1))
    t.start()
    s2_status.set("Background task started -- click the other button NOW while it runs.", status="info")


@s2_btn_click_other.click
def s2_do_click_other():
    _s2_click_other(0)
    s2_status.set("Clicked other widget.", status="success")


@s2_btn_stress.click
def s2_do_stress():
    _s2_errors.clear()
    task_thread = threading.Thread(target=_s2_run_task, args=(15, 0.05))
    task_thread.start()
    click_threads = [threading.Thread(target=_s2_click_other, args=(i,)) for i in range(15)]
    for t in click_threads:
        t.start()
    for t in click_threads:
        t.join()
    task_thread.join()
    if _s2_errors:
        s2_status.set(f"Scenario 2 STRESS: {len(_s2_errors)} calls raised. First: {_s2_errors[0]}", status="error")
    else:
        s2_status.set("Scenario 2 STRESS: 0 calls raised. Clean.", status="success")


scenario_2 = Card(
    title="Scenario 2 -- Progress-bar background task racing a click on another widget",
    content=Container(widgets=[s2_progress, s2_other_widget, s2_status, Container(widgets=[s2_btn_start_task, s2_btn_click_other, s2_btn_stress], direction="horizontal")]),
)

# ---------------------------------------------------------------------------
# Scenario 3 -- two independently live-updating widgets on the same page
# (e.g. two live charts), each driven by its own background loop, with no
# explicit coordination between them -- exactly what "live" dashboards do.
# ---------------------------------------------------------------------------
s3_feed_a = Text("Feed A: idle", status="text")
s3_feed_b = Text("Feed B: idle", status="text")
s3_status = Text("Ready.", status="text")
s3_btn_start = Button("Scenario 3 · start both live feeds (independent loops, 20 ticks each)")

_s3_errors = []
_s3_errors_lock = threading.Lock()


def _s3_feed_loop(widget, label, n_ticks):
    wid = widget.widget_id
    for i in range(n_ticks):
        try:
            # Varying-length list, same as the other scenarios -- a plain
            # scalar .set() is benign even under this exact race window.
            DataJson()[wid]["points"] = [random.randint(0, 1000) for _ in range(random.randint(5, 15))]
            widget.set(f"{label}: tick {i}", status="text")
            time.sleep(0.01)
        except Exception as e:
            with _s3_errors_lock:
                _s3_errors.append(f"{label} tick={i} {type(e).__name__}: {e}")


@s3_btn_start.click
def s3_do_start():
    _s3_errors.clear()
    threads = [
        threading.Thread(target=_s3_feed_loop, args=(s3_feed_a, "Feed A", 20)),
        threading.Thread(target=_s3_feed_loop, args=(s3_feed_b, "Feed B", 20)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if _s3_errors:
        s3_status.set(f"Scenario 3: {len(_s3_errors)}/40 ticks raised. First: {_s3_errors[0]}", status="error")
    else:
        s3_status.set("Scenario 3: 0/40 ticks raised. Clean.", status="success")


scenario_3 = Card(
    title="Scenario 3 -- two independently live-updating widgets (e.g. two live charts)",
    content=Container(widgets=[s3_feed_a, s3_feed_b, s3_status, s3_btn_start]),
)

# ---------------------------------------------------------------------------
# Scenario 4 -- ordinary rapid / double clicking on a single button by one
# user, with a list-valued widget (the shape that makes a stale patch fatal
# instead of a harmless no-op).
# ---------------------------------------------------------------------------
s4_widget = Text("Rapid-click target: 0", status="text")
s4_status = Text("Ready.", status="text")
s4_btn_target = Button("Scenario 4 · click me (try double/rapid-clicking by hand!)")
s4_btn_stress = Button("Scenario 4 · STRESS x25 rapid clicks (simulated)")

_s4_click_count = {"n": 0}


def _s4_action():
    _s4_click_count["n"] += 1
    wid = s4_widget.widget_id
    DataJson()[wid]["items"] = [random.randint(0, 50) for _ in range(random.randint(5, 15))]
    DataJson()[wid]["text"] = f"Rapid-click target: {_s4_click_count['n']}"
    DataJson().send_changes()


@s4_btn_target.click
def s4_do_click():
    try:
        _s4_action()
        s4_status.set(f"Click #{_s4_click_count['n']} OK. Try clicking FAST several times in a row.", status="success")
    except Exception as e:
        s4_status.set(f"CRASHED on click: {type(e).__name__}: {e}", status="error")


@s4_btn_stress.click
def s4_do_stress():
    errors = []
    threads = [threading.Thread(target=lambda: _try(errors, "click", _s4_action)) for _ in range(25)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        s4_status.set(f"Scenario 4 STRESS: {len(errors)}/25 raised. First: {errors[0]}", status="error")
    else:
        s4_status.set("Scenario 4 STRESS: 0/25 raised. Clean.", status="success")


scenario_4 = Card(
    title="Scenario 4 -- rapid / double clicking by a single user",
    content=Container(widgets=[s4_widget, s4_status, Container(widgets=[s4_btn_target, s4_btn_stress], direction="horizontal")]),
)

# ---------------------------------------------------------------------------
layout = Container(widgets=[scenario_1, scenario_2, scenario_3, scenario_4])
app = sly.Application(layout=layout)

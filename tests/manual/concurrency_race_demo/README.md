# Concurrency race — manual browser reproduction

Companion to `tests/unit/test_concurrency_race.py` (same four scenarios,
headless). This is the browser-interactive version, useful for visually
confirming the bug and for verifying a future fix end-to-end.

## Bug

`_PatchableJson._get_patch()` (`supervisely/app/content.py`) computes a diff
against `self._last` without holding any lock — only `_apply_patch()`
(mutating `self._last`) is lock-protected. When two `send_changes()` calls
run concurrently, both can read the same stale `_last` before either applies
its patch. Whichever applies second ends up applying a patch computed
against an already-stale baseline: `jsonpatch.JsonPatchConflict` or a
silently dropped update.

Reproduced identically with `jsonpatch` (current diffing library) and with
`patchdiff` (an unreleased migration branch) — the race is in the missing
lock, not in either diffing library.

## Run

```bash
PYTHONPATH=. uvicorn tests.manual.concurrency_race_demo.main:app \
    --host 0.0.0.0 --port 8000 --ws websockets --reload
```

Open `http://localhost:8000/`.

## Scenarios

1. **Two different users clicking at the same time** (multi-user mode) — `session_context()` only tags log/WS routing, it does not isolate `DataJson`/`StateJson` per user.
2. **Progress-bar background task racing a click on another widget** — the officially recommended pattern for long-running work (`Progress` + a background thread) racing an unrelated click.
3. **Two independently live-updating widgets** — e.g. two live charts, each driven by its own loop, with no coordination between them.
4. **Rapid / double clicking by a single user** — ordinary fast clicking, or two widgets refreshing close together.

Each section has:
- a **single-try button** — click it by hand; timing-dependent, may or may not show a failure on any one attempt
- a **STRESS xN button** — fires N concurrent repetitions server-side for a guaranteed, reliable reproduction

Click each `STRESS` button and watch the status line turn red with a
`JsonPatchConflict` error and a raised/total count. The app itself keeps
running throughout (every crash is caught and reported, not left unhandled)
— but the visible widget state is left corrupted or stale (e.g. Scenario 2's
progress bar can freeze mid-count).

## Suggested fix direction (not included in this PR)

Hold a single lock around the whole get-patch-then-apply cycle in
`synchronize_changes()`, not just inside `_apply_patch()`, so `_get_patch()`
always diffs against the `_last` it will actually be applied to.

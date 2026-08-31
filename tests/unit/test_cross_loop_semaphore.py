"""Tests for :class:`supervisely._utils.CrossLoopSemaphore`.

The semaphore is a process wide request throttle shared by every event loop of the
process: apps keep one long lived ``Api`` object but run work in freshly spawned threads,
and every new thread means a new event loop.

Every helper here joins its threads with a timeout, so a deadlock fails the test instead
of hanging the suite.
"""

import asyncio
import threading
import time

import pytest

from supervisely._utils import CrossLoopSemaphore

JOIN_TIMEOUT = 30


class _Run:
    """One coroutine executed in a brand new thread with a brand new event loop.

    Reproduces what ``run_coroutine()`` does for every training / inference run: the loop
    is created per thread and is intentionally not closed afterwards.
    """

    def __init__(self, coro_factory, daemon=True):
        self.result = None
        self.error = None
        self.loop = None
        self._factory = coro_factory
        self.thread = threading.Thread(target=self._target, daemon=daemon)

    def _target(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        try:
            self.result = self.loop.run_until_complete(self._factory())
        except BaseException as exc:  # noqa: BLE001 - reported to the test
            self.error = exc

    def start(self):
        self.thread.start()
        return self

    def join(self, timeout=JOIN_TIMEOUT):
        self.thread.join(timeout)
        assert not self.thread.is_alive(), "run did not finish: deadlock"
        return self

    def check(self):
        self.join()
        assert self.error is None, f"run failed: {self.error!r}"
        return self.result


def run_in_fresh_loop(coro_factory):
    return _Run(coro_factory).start().check()


def make_contended_batch(semaphore, tasks=12, work=0.005):
    async def batch():
        async def one():
            async with semaphore:
                await asyncio.sleep(work)

        await asyncio.gather(*[one() for _ in range(tasks)])
        return True

    return batch


def test_stock_asyncio_semaphore_breaks_across_loops():
    """Regression guard: documents why the custom primitive exists.

    ``asyncio.Semaphore`` binds itself to the first loop that suspends on it (to the loop
    of the creating thread on python 3.8/3.9), so the same object cannot be reused by the
    next run.
    """
    semaphore = asyncio.Semaphore(2)
    errors = []
    for _ in range(3):
        run = _Run(make_contended_batch(semaphore)).start().join()
        if run.error is not None:
            errors.append(run.error)

    assert errors, "expected asyncio.Semaphore to fail when shared between event loops"
    assert all(isinstance(err, RuntimeError) for err in errors), errors


def test_reused_by_fresh_event_loops():
    semaphore = CrossLoopSemaphore(2)
    for _ in range(3):
        assert run_in_fresh_loop(make_contended_batch(semaphore)) is True

    state = semaphore._state()
    assert state["value"] == 2, state
    assert state["consistent"], state
    assert state["reclaimed"] == 0, state


def test_awaited_from_the_same_loop_twice():
    """A loop that is reused (thread pool worker) must not accumulate state."""
    semaphore = CrossLoopSemaphore(3)
    run = _Run(make_contended_batch(semaphore, tasks=20))
    run.start().check()

    loop = run.loop
    asyncio.set_event_loop(loop)
    try:
        assert loop.run_until_complete(make_contended_batch(semaphore, tasks=20)()) is True
    finally:
        asyncio.set_event_loop(None)
    assert semaphore._state()["value"] == 3, semaphore._state()


def test_global_limit_is_shared_by_concurrent_loops():
    """Three loops running at once must not get three independent limits."""
    limit = 3
    semaphore = CrossLoopSemaphore(limit)
    guard = threading.Lock()
    stats = {"inflight": 0, "peak": 0}

    async def batch():
        async def one():
            async with semaphore:
                with guard:
                    stats["inflight"] += 1
                    stats["peak"] = max(stats["peak"], stats["inflight"])
                await asyncio.sleep(0.002)
                with guard:
                    stats["inflight"] -= 1

        await asyncio.gather(*[one() for _ in range(40)])
        return True

    runs = [_Run(batch).start() for _ in range(3)]
    for run in runs:
        assert run.check() is True

    assert stats["peak"] <= limit, f"limit exceeded: {stats['peak']} > {limit}"
    assert stats["peak"] == limit, "the limit was never reached, the test is not meaningful"
    assert semaphore._state()["consistent"], semaphore._state()


def test_fifo_order_across_loops():
    """Waiters are served in order, so no loop can starve behind another one."""
    semaphore = CrossLoopSemaphore(1)
    order = []

    async def grab(tag, delay):
        await asyncio.sleep(delay)
        async with semaphore:
            order.append(tag)
            await asyncio.sleep(0.01)

    async def loop_a():
        await asyncio.gather(grab("a1", 0.00), grab("a2", 0.02), grab("a3", 0.04))

    async def loop_b():
        await asyncio.gather(grab("b1", 0.01), grab("b2", 0.03), grab("b3", 0.05))

    runs = [_Run(loop_a).start(), _Run(loop_b).start()]
    for run in runs:
        run.check()

    assert order == ["a1", "b1", "a2", "b2", "a3", "b3"], order


def test_works_with_mixed_uvloop_and_asyncio_loops():
    """uvicorn picks uvloop when it is installed, worker threads use plain asyncio."""
    uvloop = pytest.importorskip("uvloop")
    limit = 3
    semaphore = CrossLoopSemaphore(limit)
    guard = threading.Lock()
    stats = {"inflight": 0, "peak": 0}
    errors = []

    async def batch():
        async def one():
            async with semaphore:
                with guard:
                    stats["inflight"] += 1
                    stats["peak"] = max(stats["peak"], stats["inflight"])
                await asyncio.sleep(0.002)
                with guard:
                    stats["inflight"] -= 1

        await asyncio.gather(*[one() for _ in range(30)])

    def worker(loop_factory):
        loop = loop_factory()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(batch())
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [
        threading.Thread(target=worker, args=(uvloop.new_event_loop,), daemon=True),
        threading.Thread(target=worker, args=(asyncio.new_event_loop,), daemon=True),
        threading.Thread(target=worker, args=(uvloop.new_event_loop,), daemon=True),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(JOIN_TIMEOUT)
        assert not thread.is_alive(), "deadlock with mixed loop implementations"

    assert not errors, errors
    assert stats["peak"] == limit, stats
    assert semaphore._state()["consistent"], semaphore._state()


def test_cancelled_waiters_do_not_lose_permits():
    semaphore = CrossLoopSemaphore(2)

    async def scenario():
        released = asyncio.Event()

        async def hold():
            async with semaphore:
                await released.wait()

        holders = [asyncio.ensure_future(hold()) for _ in range(2)]
        await asyncio.sleep(0.05)
        waiters = [asyncio.ensure_future(hold()) for _ in range(5)]
        await asyncio.sleep(0.05)
        for waiter in waiters:
            waiter.cancel()
        await asyncio.gather(*waiters, return_exceptions=True)
        released.set()
        await asyncio.gather(*holders)
        return True

    assert run_in_fresh_loop(scenario) is True
    state = semaphore._state()
    assert state["value"] == 2, state
    assert state["waiters"] == 0, state


def test_cancel_racing_the_handoff():
    """Cancelling a waiter exactly while a permit is handed to it must not lose it."""
    semaphore = CrossLoopSemaphore(1)

    async def scenario():
        async def hold(duration):
            async with semaphore:
                await asyncio.sleep(duration)

        for _ in range(300):
            first = asyncio.ensure_future(hold(0.0005))
            second = asyncio.ensure_future(hold(0.0005))
            await asyncio.sleep(0.0004)
            second.cancel()
            await asyncio.gather(first, second, return_exceptions=True)
        return True

    assert run_in_fresh_loop(scenario) is True
    assert semaphore._state()["value"] == 1, semaphore._state()


def test_wait_for_timeout_returns_the_permit():
    semaphore = CrossLoopSemaphore(1)

    async def scenario():
        async with semaphore:
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(semaphore.acquire(), timeout=0.05)
        return True

    assert run_in_fresh_loop(scenario) is True
    assert semaphore._state()["value"] == 1, semaphore._state()


def _abandon_permits(semaphore, holders=3, daemon=True):
    """Emulate ``download_async_or_sync()``: an exception inside gather aborts the run
    while sibling tasks still hold their permits, and those tasks are never resumed."""
    started = threading.Event()

    async def scenario():
        async def hold():
            async with semaphore:
                started.set()
                await asyncio.sleep(3600)

        async def boom():
            await asyncio.sleep(0.05)
            raise ValueError("download failed")

        await asyncio.gather(*[hold() for _ in range(holders)], boom())

    run = _Run(scenario, daemon=daemon)
    run.start().join()
    assert isinstance(run.error, ValueError), run.error
    assert started.is_set()
    return run


def test_permits_of_a_dead_loop_are_reclaimed():
    semaphore = CrossLoopSemaphore(3)
    _abandon_permits(semaphore, holders=3)

    leaked = semaphore._state()
    assert leaked["held"] == 3 and leaked["value"] == 0, leaked

    # the next run must not deadlock behind the leaked permits
    assert run_in_fresh_loop(make_contended_batch(semaphore, tasks=9)) is True
    state = semaphore._state()
    assert state["value"] == 3, state
    assert state["reclaimed"] == 3, state
    assert state["consistent"], state


def test_permits_of_a_stopped_loop_with_a_live_thread_are_reclaimed():
    """The thread survives the failed async batch and keeps running (sync fallback),
    so liveness of the thread cannot be used to detect the leak."""
    semaphore = CrossLoopSemaphore(2)
    keep_alive = threading.Event()
    stopped = threading.Event()

    async def scenario():
        async def hold():
            async with semaphore:
                await asyncio.sleep(3600)

        async def boom():
            await asyncio.sleep(0.05)
            raise ValueError("boom")

        await asyncio.gather(hold(), hold(), boom())

    def worker():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(scenario())
        except ValueError:
            pass
        stopped.set()
        keep_alive.wait(JOIN_TIMEOUT)  # thread stays alive, loop no longer runs

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    assert stopped.wait(JOIN_TIMEOUT)
    try:
        assert semaphore._state()["held"] == 2, semaphore._state()
        assert run_in_fresh_loop(make_contended_batch(semaphore, tasks=4)) is True
    finally:
        keep_alive.set()
        thread.join(JOIN_TIMEOUT)


def test_resurrected_release_is_dropped():
    """A frozen task resumed later must not push the counter above the limit."""
    semaphore = CrossLoopSemaphore(2)
    resumed = threading.Event()

    async def scenario():
        async def hold():
            async with semaphore:
                while not resumed.is_set():
                    await asyncio.sleep(0.005)

        holders = [asyncio.ensure_future(hold()) for _ in range(2)]
        await asyncio.sleep(0.05)
        scenario.holders = holders
        raise ValueError("boom")

    run = _Run(scenario, daemon=False)
    run.start().join()
    assert isinstance(run.error, ValueError), run.error
    assert semaphore._state()["held"] == 2, semaphore._state()

    # force reclamation from another loop
    assert run_in_fresh_loop(make_contended_batch(semaphore, tasks=4)) is True
    assert semaphore._state()["value"] == 2, semaphore._state()

    # now resurrect the abandoned loop and let the frozen tasks release their permits
    resumed.set()
    loop = run.loop
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(asyncio.gather(*scenario.holders))
    finally:
        asyncio.set_event_loop(None)

    state = semaphore._state()
    assert state["value"] == 2, state
    assert state["consistent"], state


def test_resize_grow_wakes_queued_waiters():
    semaphore = CrossLoopSemaphore(2)

    async def scenario():
        started = []

        async def one(idx):
            async with semaphore:
                started.append(idx)
                await asyncio.sleep(0.08)

        tasks = [asyncio.ensure_future(one(idx)) for idx in range(6)]
        await asyncio.sleep(0.02)
        assert len(started) == 2, started
        semaphore.resize(5)
        await asyncio.sleep(0.02)
        grown = len(started)
        await asyncio.gather(*tasks)
        return grown

    assert run_in_fresh_loop(scenario) == 5
    state = semaphore._state()
    assert state["limit"] == 5 and state["value"] == 5, state
    assert state["consistent"], state


def test_resize_shrink_applies_as_permits_are_returned():
    semaphore = CrossLoopSemaphore(4)

    async def scenario():
        release = asyncio.Event()

        async def hold():
            async with semaphore:
                await release.wait()

        holders = [asyncio.ensure_future(hold()) for _ in range(4)]
        await asyncio.sleep(0.03)
        semaphore.resize(2)  # nothing to take back: all permits are in flight
        release.set()
        await asyncio.gather(*holders)
        return True

    assert run_in_fresh_loop(scenario) is True
    state = semaphore._state()
    assert state["limit"] == 2, state
    assert state["value"] == 2, state
    assert state["debt"] == 0, state
    assert state["consistent"], state

    # the shrunk limit is actually enforced afterwards
    guard = threading.Lock()
    stats = {"inflight": 0, "peak": 0}

    async def batch():
        async def one():
            async with semaphore:
                with guard:
                    stats["inflight"] += 1
                    stats["peak"] = max(stats["peak"], stats["inflight"])
                await asyncio.sleep(0.002)
                with guard:
                    stats["inflight"] -= 1

        await asyncio.gather(*[one() for _ in range(20)])
        return True

    assert run_in_fresh_loop(batch) is True
    assert stats["peak"] == 2, stats


def test_release_from_sync_context():
    semaphore = CrossLoopSemaphore(1)

    async def take():
        await semaphore.acquire()
        return True

    assert run_in_fresh_loop(take) is True
    assert semaphore._state()["value"] == 0, semaphore._state()

    semaphore.release()  # no running loop here
    assert semaphore._state()["value"] == 1, semaphore._state()


def test_locked_and_limit():
    semaphore = CrossLoopSemaphore(1)
    assert semaphore.limit == 1
    assert semaphore.locked() is False

    async def scenario():
        async with semaphore:
            assert semaphore.locked() is True
        return True

    assert run_in_fresh_loop(scenario) is True
    assert semaphore.locked() is False
    assert "CrossLoopSemaphore" in repr(semaphore)


def test_value_attribute_is_free_permits():
    """``semaphore._value`` is read by the debug logging in ``project.py``, keep it
    meaning the same thing as in ``asyncio.Semaphore``: the number of free permits."""
    semaphore = CrossLoopSemaphore(2)
    assert semaphore._value == 2

    async def scenario():
        async with semaphore:
            assert semaphore._value == 1
            async with semaphore:
                assert semaphore._value == 0
        return True

    assert run_in_fresh_loop(scenario) is True
    assert semaphore._value == 2


def test_zero_permits_blocks_until_resized():
    semaphore = CrossLoopSemaphore(0)

    async def scenario():
        acquired = asyncio.ensure_future(semaphore.acquire())
        await asyncio.sleep(0.02)
        assert not acquired.done()
        semaphore.resize(1)
        return await asyncio.wait_for(acquired, timeout=5)

    assert run_in_fresh_loop(scenario) is True


def test_negative_values_rejected():
    with pytest.raises(ValueError):
        CrossLoopSemaphore(-1)
    with pytest.raises(ValueError):
        CrossLoopSemaphore(1).resize(-1)


def test_permit_survives_a_cancel_racing_the_handoff_from_another_thread():
    """The handoff and the cancellation happen on different threads.

    ``_wake_one_locked()`` accounts the permit to the waiter and schedules the grant on the
    waiter's loop. If the waiting task is cancelled before that callback runs, nobody used to
    give the permit back: the acquirer sees a cancelled future and the grant sees a finished
    one. A single loop cannot reproduce it, both sides are serialized on one thread there.
    """
    semaphore = CrossLoopSemaphore(1)
    holder_ready = threading.Event()
    release_now = threading.Event()
    released = threading.Event()
    unblock_waiter_loop = threading.Event()
    box = {}

    def holder_thread():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def hold():
            async with semaphore:
                holder_ready.set()
                while not release_now.is_set():
                    await asyncio.sleep(0.005)
            released.set()

        loop.run_until_complete(hold())

    def waiter_thread():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        box["loop"] = loop

        async def scenario():
            task = asyncio.ensure_future(semaphore.acquire())
            box["task"] = task
            await asyncio.sleep(0.05)  # let it queue behind the holder
            # block the loop so the cancel and the grant queue up behind this callback
            loop.call_soon(unblock_waiter_loop.wait)
            await asyncio.gather(task, return_exceptions=True)

        loop.run_until_complete(scenario())

    holder = threading.Thread(target=holder_thread, daemon=True)
    waiter = threading.Thread(target=waiter_thread, daemon=True)
    holder.start()
    assert holder_ready.wait(JOIN_TIMEOUT)
    waiter.start()

    for _ in range(200):  # wait until the acquire is really queued
        if semaphore._state()["waiters"] == 1:
            break
        time.sleep(0.01)
    assert semaphore._state()["waiters"] == 1

    time.sleep(0.2)  # the waiter loop is now sitting in the blocking callback
    box["loop"].call_soon_threadsafe(box["task"].cancel)  # queued first
    release_now.set()  # the holder releases, so the grant queues second
    assert released.wait(JOIN_TIMEOUT)
    unblock_waiter_loop.set()

    holder.join(JOIN_TIMEOUT)
    waiter.join(JOIN_TIMEOUT)
    assert not holder.is_alive() and not waiter.is_alive()

    state = semaphore._state()
    assert state["value"] == 1, state
    assert state["held"] == 0, state
    assert state["consistent"], state


def test_stale_holder_from_a_collected_loop_is_replaced():
    """id(loop) is reused once a loop is collected, so a bucket may hold a dead loop's entry."""
    from supervisely._utils import _SemaphorePermitHolder

    semaphore = CrossLoopSemaphore(2)
    collected = asyncio.new_event_loop()
    collected.close()

    async def scenario():
        live = asyncio.get_running_loop()
        stale = _SemaphorePermitHolder(collected)
        stale.count = 1
        semaphore._holders[id(live)] = stale  # the new loop got the old id
        semaphore._value -= 1  # the stale permit is accounted as held

        async with semaphore:
            state = semaphore._state()
            assert state["held"] == 1, state  # only ours, the stale one was reclaimed
        return semaphore._state()

    state = run_in_fresh_loop(scenario)
    assert state["value"] == 2, state
    assert state["held"] == 0, state
    assert state["consistent"], state
    assert state["reclaimed"] == 1, state


def test_release_inside_a_loop_never_touches_another_loops_holder():
    semaphore = CrossLoopSemaphore(2)
    taken = threading.Event()
    give_back = threading.Event()

    def holder_thread():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def hold():
            async with semaphore:
                taken.set()
                while not give_back.is_set():
                    await asyncio.sleep(0.005)

        loop.run_until_complete(hold())

    holder = threading.Thread(target=holder_thread, daemon=True)
    holder.start()
    assert taken.wait(JOIN_TIMEOUT)

    async def stray_release():
        semaphore.release()  # this loop holds nothing
        return semaphore._state()

    state = run_in_fresh_loop(stray_release)
    assert state["held"] == 1, state  # the other loop still owns its permit
    assert state["value"] == 1, state

    give_back.set()
    holder.join(JOIN_TIMEOUT)
    assert semaphore._state()["value"] == 2, semaphore._state()


def test_ambiguous_sync_release_is_dropped_with_a_warning(caplog):
    semaphore = CrossLoopSemaphore(3)
    taken = threading.Barrier(3)
    give_back = threading.Event()

    def holder_thread():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def hold():
            async with semaphore:
                taken.wait(JOIN_TIMEOUT)
                while not give_back.is_set():
                    await asyncio.sleep(0.005)

        loop.run_until_complete(hold())

    holders = [threading.Thread(target=holder_thread, daemon=True) for _ in range(2)]
    for thread in holders:
        thread.start()
    taken.wait(JOIN_TIMEOUT)

    before = semaphore._state()
    semaphore.release()  # no running loop here and two loops hold permits
    after = semaphore._state()
    assert after == before, (before, after)

    give_back.set()
    for thread in holders:
        thread.join(JOIN_TIMEOUT)
    assert semaphore._state()["value"] == 3, semaphore._state()


def test_queued_waiter_recovers_permits_frozen_after_it_queued():
    """A waiter that is already queued has to recover the semaphore on its own.

    Reaping happens when someone calls acquire(), so if the holder's loop freezes *after*
    the waiter queued and nothing else acquires, there is nobody left to trigger it. The
    waiter re-checks on a timer instead of waiting forever.
    """
    semaphore = CrossLoopSemaphore(1)
    semaphore.stall_check_sec = 0.2
    holder_took = threading.Event()
    freeze_now = threading.Event()
    outcome = {}

    def holder_thread():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def hold():
            async with semaphore:
                holder_took.set()
                while not freeze_now.is_set():
                    await asyncio.sleep(0.01)
                await asyncio.sleep(3600)  # frozen together with the loop

        async def boom():
            while not freeze_now.is_set():
                await asyncio.sleep(0.01)
            await asyncio.sleep(0.05)
            raise ValueError("gather aborted")

        try:
            loop.run_until_complete(asyncio.gather(hold(), boom()))
        except ValueError:
            pass  # the loop stays open but stops running, with the permit still accounted

    def waiter_thread():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def wait_once():
            async with semaphore:
                return "acquired"

        try:
            outcome["result"] = loop.run_until_complete(wait_once())
        except BaseException as exc:  # noqa: BLE001
            outcome["result"] = f"{type(exc).__name__}: {exc}"

    holder = threading.Thread(target=holder_thread, daemon=True)
    holder.start()
    assert holder_took.wait(JOIN_TIMEOUT)

    waiter = threading.Thread(target=waiter_thread, daemon=True)
    waiter.start()
    for _ in range(500):  # the waiter must be queued before the loop freezes
        if semaphore._state()["waiters"] == 1:
            break
        time.sleep(0.01)
    assert semaphore._state()["waiters"] == 1, semaphore._state()

    freeze_now.set()
    holder.join(JOIN_TIMEOUT)
    waiter.join(JOIN_TIMEOUT)
    assert not waiter.is_alive(), (
        f"the queued waiter never recovered the frozen permit: {semaphore._state()}"
    )
    assert outcome["result"] == "acquired", outcome
    assert semaphore._state()["consistent"], semaphore._state()


def test_a_logging_handler_that_reenters_does_not_deadlock():
    """Warnings are emitted with the lock released.

    A logging handler is arbitrary code — the SDK's own handlers report to the platform —
    so emitting the reclaim warning while holding the counter lock deadlocked every thread
    that touched the semaphore.
    """
    import logging

    from supervisely.sly_logger import logger as sly_logger

    semaphore = CrossLoopSemaphore(1)
    semaphore.stall_check_sec = 0.2
    seen = []

    class ReentrantHandler(logging.Handler):
        def emit(self, record):
            seen.append(semaphore._state())  # takes the lock from inside the handler
            semaphore.locked()

    handler = ReentrantHandler()
    sly_logger.addHandler(handler)
    try:
        _abandon_permits(semaphore, holders=1)  # a permit frozen on a stopped loop
        assert semaphore._state()["held"] == 1, semaphore._state()

        # the next acquirer reclaims it, which logs a warning while the handler re-enters
        assert run_in_fresh_loop(make_contended_batch(semaphore, tasks=3)) is True
    finally:
        sly_logger.removeHandler(handler)

    assert seen, "the reclaim warning never reached the handler"
    assert semaphore._state()["value"] == 1, semaphore._state()


def test_a_waiter_on_a_stopped_loop_does_not_stall_the_others():
    """The permit must not be handed to a waiter that cannot take it.

    A waiter whose loop stopped keeps its place in the queue, so handing it the permit
    parks it until the next reclaim and everyone behind waits a full stall_check_sec.
    """
    semaphore = CrossLoopSemaphore(1)
    semaphore.stall_check_sec = 5  # a stall would be plainly visible
    hog_took = threading.Event()
    release_hog = threading.Event()
    orphan_queued = threading.Event()
    timings = {}

    def orphan_thread():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def scenario():
            asyncio.ensure_future(semaphore.acquire())  # queued, then abandoned
            await asyncio.sleep(0.2)
            orphan_queued.set()
            raise ValueError("abandoned")

        try:
            loop.run_until_complete(scenario())
        except ValueError:
            pass  # the loop stays open but stops running, the waiter stays queued

    def hog_thread():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def hold():
            async with semaphore:
                hog_took.set()
                while not release_hog.is_set():
                    await asyncio.sleep(0.01)

        loop.run_until_complete(hold())

    def live_thread():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def take():
            started = time.monotonic()
            async with semaphore:
                timings["waited"] = time.monotonic() - started

        loop.run_until_complete(take())

    hog = threading.Thread(target=hog_thread, daemon=True)
    hog.start()
    assert hog_took.wait(JOIN_TIMEOUT)

    orphan = threading.Thread(target=orphan_thread, daemon=True)
    orphan.start()
    assert orphan_queued.wait(JOIN_TIMEOUT)
    orphan.join(JOIN_TIMEOUT)
    assert semaphore._state()["waiters"] == 1, semaphore._state()

    live = threading.Thread(target=live_thread, daemon=True)
    live.start()
    time.sleep(0.3)  # let it queue behind the orphan
    release_hog.set()
    hog.join(JOIN_TIMEOUT)
    live.join(JOIN_TIMEOUT)

    assert not live.is_alive(), "the live waiter never got the permit"
    assert timings["waited"] < semaphore.stall_check_sec, (
        f"waited {timings['waited']:.1f}s, the permit went to the stopped loop first"
    )
    # the orphan keeps its place: a later run_until_complete() on that loop may resume it
    assert semaphore._state()["waiters"] == 1, semaphore._state()


def test_a_waiter_paused_between_two_runs_is_still_served():
    """Skipping is not dropping: `run_coroutine()` reuses one loop per thread, and a task
    suspended between two run_until_complete() calls has to survive the gap."""
    semaphore = CrossLoopSemaphore(1)
    semaphore.stall_check_sec = 0.2
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(semaphore.acquire())  # this loop holds the only permit
        waiter = asyncio.ensure_future(semaphore.acquire(), loop=loop)
        loop.run_until_complete(asyncio.sleep(0.05))  # queue it, then leave the loop stopped

        assert semaphore._state()["waiters"] == 1, semaphore._state()
        time.sleep(0.5)  # the loop is stopped, nothing must drop the waiter
        assert semaphore._state()["waiters"] == 1, semaphore._state()

        loop.run_until_complete(_release_and_wait(semaphore, waiter))
        assert waiter.result() is True
    finally:
        asyncio.set_event_loop(None)
        loop.close()


async def _release_and_wait(semaphore, waiter):
    semaphore.release()
    await asyncio.wait_for(waiter, timeout=5)

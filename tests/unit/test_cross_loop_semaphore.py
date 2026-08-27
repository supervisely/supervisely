"""Tests for :class:`supervisely._utils.CrossLoopSemaphore`.

The semaphore is a process wide request throttle shared by every event loop of the
process: apps keep one long lived ``Api`` object but run work in freshly spawned threads,
and every new thread means a new event loop.

Every helper here joins its threads with a timeout, so a deadlock fails the test instead
of hanging the suite.
"""

import asyncio
import threading

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

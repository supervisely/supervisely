import asyncio
import concurrent.futures
import contextvars
import threading
from typing import Dict, Hashable, Optional, Tuple

from supervisely.sly_logger import logger

# The event loop the app server runs on; set at startup and when a websocket connects.
_main_loop: Optional[asyncio.AbstractEventLoop] = None

# Seconds to wait for the server loop to pick up a coroutine before leaving it queued there.
_MAIN_LOOP_PICKUP_TIMEOUT = 1.0
# Seconds the server loop's own thread waits for a send it runs on a separate loop.
_LOOP_THREAD_SEND_TIMEOUT = 2.0
# Set while a coroutine handed to the server loop has not been picked up: the loop is blocked.
_main_loop_blocked = threading.Event()

_queue_state = threading.Lock()
# sends handed to the server loop that have not begun yet, by key; later identical ones join them
_queued: Dict[Hashable, "_QueuedRun"] = {}
# runs the caller handed over last: a caller's runs begin in the order it handed them over.
# A context variable, so that it follows one request rather than a reused worker thread.
_caller_runs: contextvars.ContextVar = contextvars.ContextVar("_caller_runs", default=())


def run_sync(coroutine):
    """
    Runs an asynchronous coroutine in a separate thread and waits for its result.
    It is useful for running async functions in a synchronous
    environment.

    This method creates a new thread using ThreadPoolExecutor and executes the coroutine
    inside a new event loop.

    ⚠️ Note: This function creates a new event loop every time it is called,
    which can cause issues when using objects tied to a specific loop (e.g., asyncio.Semaphore).

    :param coroutine: coroutine to run
    :returns: result of coroutine
    """
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            result = executor.submit(
                lambda coroutine_to_exec: asyncio.run(coroutine_to_exec), coroutine
            ).result()
        return result
    except RuntimeError as ex:
        print(repr(ex))
        return None


def _set_main_loop() -> None:
    """Remembers the running event loop as the one the app server runs on."""
    global _main_loop
    loop = asyncio.get_running_loop()
    if loop is not _main_loop:
        with _queue_state:
            _queued.clear()  # runs queued on a previous loop will never begin
    _main_loop = loop
    _main_loop_blocked.clear()


class _QueuedRun:
    """A coroutine handed to the server loop, shared by the callers that asked for it."""

    def __init__(self, coroutine, key: Optional[Hashable], after: Tuple["_QueuedRun", ...], loop):
        self.coroutine = coroutine
        self.key = key
        self.loop = loop
        # the caller's earlier runs on this loop: they begin first
        self.after = _pending_on(after, loop)
        self.picked_up = False  # the server loop has started handling it
        self.began = False  # the coroutine itself is running
        self.waited = True  # a caller is waiting for the result and gets its error
        self.future: Optional[concurrent.futures.Future] = None

    def done(self, future: concurrent.futures.Future):
        self.after = ()
        with _queue_state:
            if self.key is not None and _queued.get(self.key) is self:
                del _queued[self.key]
            if future.cancelled():
                _main_loop_blocked.clear()  # the loop got to it, even if only to cancel it
        if not self.began:
            self.coroutine.close()  # cancelled before it ran
            return
        if self.waited or future.cancelled() or future.exception() is None:
            return
        logger.warning(
            "Failed to send app changes", exc_info=future.exception(), extra={"key": repr(self.key)}
        )


def _pending_on(runs: Tuple[_QueuedRun, ...], loop) -> Tuple[_QueuedRun, ...]:
    return tuple(
        r for r in runs if r.loop is loop and r.future is not None and not r.future.done()
    )


async def _run_queued(run: _QueuedRun):
    with _queue_state:
        run.picked_up = True
        _main_loop_blocked.clear()  # the loop is running callbacks again
    earlier = [asyncio.wrap_future(r.future) for r in _pending_on(run.after, run.loop)]
    if earlier:
        await asyncio.wait(earlier)
    with _queue_state:
        if run.key is not None and _queued.get(run.key) is run:
            del _queued[run.key]
        run.began = True
    return await run.coroutine


def _run_sync_bounded(coroutine, timeout: float):
    """Like run_sync(), but stops waiting after `timeout` seconds and lets it finish alone."""
    outcome = {}

    def target():
        try:
            outcome["result"] = asyncio.run(coroutine)
        except BaseException as exc:  # pylint: disable=broad-except
            outcome["error"] = exc

    thread = threading.Thread(target=target, name="sly-app-send", daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        logger.warning(
            f"Sending app changes takes longer than {timeout} s; it continues in the background"
        )
        return None
    error = outcome.get("error")
    if isinstance(error, RuntimeError):
        print(repr(error))
        return None
    if error is not None:
        raise error
    return outcome.get("result")


def _run_on_main_loop(coroutine, key: Optional[Hashable] = None):
    """
    Runs a coroutine on the app server's event loop and waits for its result.

    Objects such as websocket connections belong to the loop they were created on, and
    awaiting them from another loop can wait forever. Runs handed over by one caller (a
    thread, or a request) begin in the order it handed them over. Returns None without
    waiting when the server loop does not pick the coroutine up within a second (it is
    blocked, possibly waiting for the caller): the coroutine stays queued and runs when the
    loop is free.

    A coroutine with the same ``key`` as one queued and not begun yet is dropped in favour
    of it: a send takes its snapshot when it begins, so the queued one sends both changes.

    The server loop's own thread cannot wait for itself: it runs the coroutine on a separate
    loop as :func:`run_sync` does, and stops waiting after a few seconds so that a send
    needing the server loop cannot block it for good. Before the server loop starts and
    after it stops, :func:`run_sync` is used.

    :param coroutine: coroutine to run
    :param key: identifies interchangeable sends, e.g. one store's changes
    :returns: result of coroutine
    """
    loop = _main_loop
    if loop is None or loop.is_closed() or not loop.is_running():
        return run_sync(coroutine)
    try:
        if asyncio.get_running_loop() is loop:
            return _run_sync_bounded(coroutine, _LOOP_THREAD_SEND_TIMEOUT)
    except RuntimeError:
        pass

    previous = _caller_runs.get()
    scheduled = False
    with _queue_state:
        run = _queued.get(key) if key is not None else None
        if run is not None and run.future.done():
            run = None  # finished or cancelled: it will not send this change
        if run is not None:
            coroutine.close()
        else:
            run = _QueuedRun(coroutine, key, previous, loop)
            runner = _run_queued(run)
            try:
                run.future = asyncio.run_coroutine_threadsafe(runner, loop)
            except RuntimeError:
                runner.close()  # the loop closed after the check above
            else:
                scheduled = True
                if key is not None:
                    _queued[key] = run
    if run.future is None:
        return run_sync(coroutine)
    if scheduled:
        # outside the lock: the callback takes it, and runs here if the run is already done
        run.future.add_done_callback(run.done)
    # a run joined from another caller may begin before this caller's earlier ones
    _caller_runs.set(_pending_on((run,) + (() if scheduled else previous), loop))

    future = run.future
    try:
        if not _main_loop_blocked.is_set():
            concurrent.futures.wait([future], timeout=_MAIN_LOOP_PICKUP_TIMEOUT)
        with _queue_state:
            if not run.picked_up and not future.done():
                _main_loop_blocked.set()
                run.waited = False  # nobody will see its error: log it instead
                return None
            run.waited = True
        while True:
            done, _ = concurrent.futures.wait([future], timeout=1)
            if done:
                return future.result()
            if loop.is_closed() or not loop.is_running():
                # the server stopped before finishing it; nothing will complete the future
                future.cancel()
                return None
    except concurrent.futures.CancelledError:
        # cancelled by the server shutting down
        return None
    except RuntimeError as ex:
        print(repr(ex))
        return None

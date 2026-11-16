import asyncio
import concurrent.futures
import threading
from typing import Optional

# The event loop the app server runs on; set at startup and when a websocket connects.
_main_loop: Optional[asyncio.AbstractEventLoop] = None

# Seconds to wait for the server loop to start a coroutine before leaving it queued there.
_MAIN_LOOP_PICKUP_TIMEOUT = 1.0
# Set while a coroutine handed to the server loop has not been started: the loop is blocked.
_main_loop_blocked = threading.Event()


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
    _main_loop = asyncio.get_running_loop()
    _main_loop_blocked.clear()


def _run_on_main_loop(coroutine):
    """
    Runs a coroutine on the app server's event loop and waits for its result.

    Objects such as websocket connections belong to the loop they were created on, and
    awaiting them from another loop can wait forever. Falls back to :func:`run_sync` when
    the server loop is not running yet (or any more), or when the caller is running inside
    an event loop itself. If the server loop does not start the coroutine in time (it is
    blocked, possibly waiting for the calling thread), the coroutine stays queued there and
    this returns None without waiting for it.

    :param coroutine: coroutine to run
    :returns: result of coroutine
    """
    loop = _main_loop
    try:
        asyncio.get_running_loop()
        in_event_loop = True
    except RuntimeError:
        in_event_loop = False
    if in_event_loop or loop is None or loop.is_closed() or not loop.is_running():
        return run_sync(coroutine)

    start_lock = threading.Lock()
    started = []

    async def _start():
        with start_lock:
            started.append(True)
            _main_loop_blocked.clear()  # the loop is running callbacks again
        return await coroutine

    try:
        future = asyncio.run_coroutine_threadsafe(_start(), loop)
    except RuntimeError:
        # the loop closed after the check above; the coroutine was not scheduled
        return run_sync(coroutine)
    try:
        if not _main_loop_blocked.is_set():
            concurrent.futures.wait([future], timeout=_MAIN_LOOP_PICKUP_TIMEOUT)
        with start_lock:
            if not started:
                if future.done():  # cancelled by the server shutting down before it started
                    coroutine.close()
                else:
                    _main_loop_blocked.set()
                return None
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
        if not started:
            coroutine.close()
        return None
    except RuntimeError as ex:
        print(repr(ex))
        return None

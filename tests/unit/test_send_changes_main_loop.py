"""send_changes() from a worker thread must run on the server's event loop.

Websocket sends wait on loop-bound objects (uvicorn's write-buffer Event). Awaited
from a throwaway loop, the wakeup set by the server loop never arrives and the
callback hangs forever, holding the DataJson/StateJson lock.
"""

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

import supervisely.app.content as content
import supervisely.app.fastapi.utils as utils
from supervisely.app.content import DataJson
from supervisely.app.fastapi.websocket import WebsocketManager
from supervisely.app.singleton import Singleton


@pytest.fixture(autouse=True)
def isolated_state(monkeypatch):
    saved_instances = dict(Singleton._instances)
    saved_nested = dict(Singleton._nested_instances)
    Singleton._instances.clear()
    Singleton._nested_instances.clear()
    monkeypatch.delenv("SUPERVISELY_MULTIUSER_APP_MODE", raising=False)
    monkeypatch.setattr(utils, "_main_loop", None)
    utils._main_loop_blocked.clear()
    yield
    Singleton._instances.clear()
    Singleton._instances.update(saved_instances)
    Singleton._nested_instances.clear()
    Singleton._nested_instances.update(saved_nested)


@pytest.fixture()
def server_loop():
    """A running loop in its own thread, registered the way app startup does it."""
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    asyncio.run_coroutine_threadsafe(_register(), loop).result(5)
    yield loop
    if loop.is_running():
        loop.call_soon_threadsafe(loop.stop)
    thread.join(5)
    loop.close()


async def _register():
    utils._set_main_loop()


async def _new_event():
    return asyncio.Event()


class _BackpressuredSocket:
    """Mimics uvicorn: send() waits on an Event the server loop sets once the client drains."""

    def __init__(self, loop):
        self.loop = loop
        self.writable = asyncio.run_coroutine_threadsafe(_new_event(), loop).result(5)
        self.sent = []

    async def send_json(self, d):
        await self.writable.wait()
        self.sent.append(d)

    def drain_later(self, delay):
        self.loop.call_soon_threadsafe(self.loop.call_later, delay, self.writable.set)


def _run_in_worker(fn, timeout):
    """Runs fn in a daemon thread; a call that hangs fails the test instead of hanging it."""
    outcome = {}

    def target():
        try:
            outcome["result"] = fn()
        except BaseException as exc:  # noqa: B036
            outcome["error"] = exc

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout)
    assert not thread.is_alive(), f"still blocked after {timeout} s"
    if "error" in outcome:
        raise outcome["error"]
    return outcome["result"]


def test_runs_on_server_loop_and_returns_result(server_loop):
    async def where():
        return asyncio.get_running_loop()

    assert _run_in_worker(lambda: utils._run_on_main_loop(where()), 5) is server_loop


def test_exception_reaches_caller(server_loop):
    async def boom():
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        _run_in_worker(lambda: utils._run_on_main_loop(boom()), 5)


def test_falls_back_without_server_loop():
    async def where():
        return asyncio.get_running_loop()

    assert _run_in_worker(lambda: utils._run_on_main_loop(where()), 5) is not None


def test_falls_back_on_server_loop_thread(server_loop):
    # an async route calling the sync API: waiting on its own loop would deadlock
    async def inner():
        return asyncio.get_running_loop()

    async def caller():
        return utils._run_on_main_loop(inner()), asyncio.get_running_loop()

    used, own = asyncio.run_coroutine_threadsafe(caller(), server_loop).result(5)
    assert used is not own


def test_returns_when_server_loop_stops(server_loop):
    started = threading.Event()

    async def forever():
        started.set()
        await asyncio.sleep(3600)

    def call():
        return utils._run_on_main_loop(forever())

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(call)
        assert started.wait(5)
        server_loop.call_soon_threadsafe(server_loop.stop)
        assert future.result(10) is None


def test_send_changes_survives_backpressure(server_loop, monkeypatch):
    socket = _BackpressuredSocket(server_loop)
    manager = WebsocketManager()
    monkeypatch.setattr(manager, "active_connections", [socket])

    def push(value):
        DataJson()["value"] = value
        DataJson().send_changes()

    socket.drain_later(0.3)
    _run_in_worker(lambda: push(1), 10)
    # the lock was released: later pushes go through too
    _run_in_worker(lambda: push(2), 10)
    assert [d["data"][0]["value"] for d in socket.sent] == [1, 2]


def test_broadcast_drops_broken_and_stalled_clients(server_loop, monkeypatch):
    class Healthy:
        def __init__(self):
            self.sent = []

        async def send_json(self, d):
            self.sent.append(d)

        async def close(self):
            pass

    class Broken(Healthy):
        async def send_json(self, d):
            raise RuntimeError('Cannot call "send" once a close message has been sent.')

    class Stalled(Healthy):
        async def send_json(self, d):
            await asyncio.sleep(3600)

    broken, stalled, healthy = Broken(), Stalled(), Healthy()
    manager = WebsocketManager()
    monkeypatch.setattr(manager, "active_connections", [broken, stalled, healthy])
    monkeypatch.setattr(WebsocketManager, "_send_timeout", 0.2)

    started = time.monotonic()
    asyncio.run_coroutine_threadsafe(manager.broadcast({"data": []}), server_loop).result(5)
    assert time.monotonic() - started < 3
    assert healthy.sent == [{"data": []}]
    assert manager.active_connections == [healthy]


def test_blocked_server_loop_keeps_the_send_queued(server_loop):
    # the server loop blocks on a thread that sends changes (e.g. joining a worker on stop)
    ran_on = []

    async def where():
        ran_on.append(asyncio.get_running_loop())

    async def join_worker():
        started = time.monotonic()
        result = _run_in_worker(lambda: utils._run_on_main_loop(where()), 10)
        return result, time.monotonic() - started

    result, waited = asyncio.run_coroutine_threadsafe(join_worker(), server_loop).result(15)
    assert result is None and waited < 5
    deadline = time.monotonic() + 5
    while not ran_on and time.monotonic() < deadline:
        time.sleep(0.05)
    assert ran_on == [server_loop]  # it ran later, on the server loop, not elsewhere
    assert not utils._main_loop_blocked.is_set()
    assert _run_in_worker(lambda: utils._run_on_main_loop(_running_loop()), 5) is server_loop


def test_busy_server_loop_never_sends_from_another_loop(server_loop):
    ran_on = []

    async def where():
        ran_on.append(asyncio.get_running_loop())

    async def busy():
        time.sleep(2.5)  # sync work on the loop thread

    asyncio.run_coroutine_threadsafe(busy(), server_loop)
    time.sleep(0.1)
    for _ in range(3):
        _run_in_worker(lambda: utils._run_on_main_loop(where()), 5)
    deadline = time.monotonic() + 5
    while len(ran_on) < 3 and time.monotonic() < deadline:
        time.sleep(0.05)
    assert ran_on == [server_loop] * 3


async def _running_loop():
    return asyncio.get_running_loop()


def test_returns_none_when_server_cancels_it(server_loop):
    started = threading.Event()

    async def forever():
        started.set()
        await asyncio.sleep(3600)

    def cancel_all():
        for task in asyncio.all_tasks(server_loop):
            task.cancel()

    def call():
        return utils._run_on_main_loop(forever())

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(call)
        assert started.wait(5)
        server_loop.call_soon_threadsafe(cancel_all)
        assert future.result(10) is None


def test_cancelled_lock_wait_does_not_leave_lock_held(server_loop):
    lock = threading.Lock()
    lock.acquire()

    async def take():
        async with content.async_lock(lock):
            pass

    task = asyncio.run_coroutine_threadsafe(take(), server_loop)
    time.sleep(0.05)
    task.cancel()
    lock.release()  # the pending acquire attempt may still win the lock
    time.sleep(0.5)
    assert not lock.locked()


class _Client:
    def __init__(self, stall=False):
        self.stall = stall
        self.sent = []
        self.closed = threading.Event()

    async def send_json(self, d):
        if self.stall:
            await asyncio.sleep(3600)
        self.sent.append(d)

    async def close(self):
        await asyncio.sleep(0.01)  # a close frame waits for the transport too
        self.closed.set()


def test_stalled_clients_are_dropped_together_and_closed(server_loop, monkeypatch):
    stalled = [_Client(stall=True) for _ in range(3)]
    healthy = _Client()
    manager = WebsocketManager()
    monkeypatch.setattr(manager, "active_connections", stalled + [healthy])
    monkeypatch.setattr(WebsocketManager, "_send_timeout", 0.5)

    started = time.monotonic()
    asyncio.run_coroutine_threadsafe(manager.broadcast({"data": []}), server_loop).result(5)
    assert time.monotonic() - started < 1.2  # one timeout, not three
    assert healthy.sent == [{"data": []}]
    assert manager.active_connections == [healthy]
    assert all(client.closed.wait(2) for client in stalled)


def test_client_dropped_off_the_server_loop_is_closed(monkeypatch):
    stalled = _Client(stall=True)
    manager = WebsocketManager()
    monkeypatch.setattr(manager, "active_connections", [stalled])
    monkeypatch.setattr(WebsocketManager, "_send_timeout", 0.2)

    utils.run_sync(manager.broadcast({"data": []}))
    assert stalled.closed.is_set()
    assert manager.active_connections == []


def test_widget_websocket_action_survives_backpressure(server_loop, monkeypatch):
    from supervisely.app.widgets import ReloadableArea, Text

    class ActionBackpressure(_BackpressuredSocket):
        def __init__(self, loop):
            super().__init__(loop)
            self.writable = asyncio.run_coroutine_threadsafe(_new_event(), loop).result(5)
            self.loop.call_soon_threadsafe(self.writable.set)

        async def send_json(self, d):
            if "runAction" in d:  # the large message: the buffer fills until the client drains
                self.loop.call_soon_threadsafe(self.writable.clear)
                self.drain_later(0.3)
                await asyncio.sleep(0.05)
            await super().send_json(d)

    area = ReloadableArea(content=Text("a"))
    socket = ActionBackpressure(server_loop)
    time.sleep(0.05)
    monkeypatch.setattr(WebsocketManager(), "active_connections", [socket])

    # the unwrapped method: the instance wrapper also re-renders the page for offline sessions
    _run_in_worker(lambda: ReloadableArea.reload(area), 10)
    assert any("runAction" in d for d in socket.sent)

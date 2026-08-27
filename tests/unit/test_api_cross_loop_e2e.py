"""End to end tests for the loop bound state cached on ``Api``.

A real ``Api`` object talks to a local keep alive HTTP server while the test reproduces
the pattern that used to break: one long lived ``Api`` and several runs, each one in a
freshly spawned thread with its own event loop (what ``TrainApp`` and the inference
thread pool do), plus several such loops running at the same time.
"""

import asyncio
import http.server
import json
import socketserver
import threading

import httpx
import pytest

from supervisely._utils import CrossLoopSemaphore, run_coroutine
from supervisely.api.api import Api

JOIN_TIMEOUT = 60
SEMAPHORE_SIZE = 3


class _Stats:
    def __init__(self):
        self.lock = threading.Lock()
        self.inflight = 0
        self.peak = 0
        self.requests = 0
        self.connections = 0

    def enter(self):
        with self.lock:
            self.inflight += 1
            self.requests += 1
            self.peak = max(self.peak, self.inflight)

    def exit(self):
        with self.lock:
            self.inflight -= 1


class _Server(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 256
    stats: _Stats
    delay: float


class _Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"  # real keep-alive, like a production instance

    def setup(self):
        super().setup()
        with self.server.stats.lock:
            self.server.stats.connections += 1

    def _respond(self, body: bytes):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        if length:
            self.rfile.read(length)
        stats = self.server.stats
        stats.enter()
        try:
            if self.server.delay:
                threading.Event().wait(self.server.delay)
            self._respond(json.dumps({"ok": True}).encode())
        finally:
            stats.exit()

    do_GET = do_POST

    def log_message(self, *args):  # silence the default stderr logging
        pass


@pytest.fixture
def server():
    stats = _Stats()
    srv = _Server(("127.0.0.1", 0), _Handler)
    srv.stats = stats
    srv.delay = 0.0
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        yield srv
    finally:
        srv.shutdown()
        srv.server_close()
        thread.join(JOIN_TIMEOUT)


def build_api(server, token="t" * 128):
    """A new Api object for the stub server, like app code does per request."""
    address = f"http://127.0.0.1:{server.server_address[1]}"
    instance = Api(address, token, retry_count=1)
    # do not touch the network for anything but the requests under test
    instance._skip_https_redirect_check = True
    instance._require_https_redirect_check = False
    instance._version_check_completed = True
    instance.retry_sleep_sec = 0
    instance.set_semaphore_size(SEMAPHORE_SIZE)
    return instance


@pytest.fixture
def api(server):
    known = set(Api._semaphores)
    try:
        yield build_api(server)
    finally:
        # the registry of shared semaphores is class level state
        for key in set(Api._semaphores) - known:
            del Api._semaphores[key]


def _batch(api, requests=12):
    """One async batch shaped like the SDK download code: the global semaphore guards
    every request."""

    async def run():
        semaphore = api.get_default_semaphore()

        async def one(idx):
            async with semaphore:
                response = await api.post_async("test", json={"idx": idx}, retries=1)
                return response.status_code

        return await asyncio.gather(*[one(idx) for idx in range(requests)])

    return run


class _Run(threading.Thread):
    """A run in its own thread and its own event loop, via ``run_coroutine()``."""

    def __init__(self, api, requests=12, close_loop=False):
        super().__init__(daemon=True)
        self._api = api
        self._requests = requests
        self._close_loop = close_loop
        self.result = None
        self.error = None
        self.loop = None

    def start(self):
        super().start()
        return self

    def run(self):
        try:
            self.result = run_coroutine(_batch(self._api, self._requests)())
        except BaseException as exc:  # noqa: BLE001 - reported to the test
            self.error = exc
        finally:
            self.loop = asyncio.get_event_loop_policy().get_event_loop()
            if self._close_loop and self.loop is not None:
                self.loop.close()

    def check(self):
        self.join(JOIN_TIMEOUT)
        assert not self.is_alive(), "run did not finish: deadlock"
        assert self.error is None, f"run failed: {self.error!r}"
        assert self.result == [200] * self._requests, self.result
        return self


def test_three_sequential_runs_share_one_api(api, server):
    """The regression: run 2 used to die with "bound to a different event loop" and the
    project was silently downloaded synchronously instead."""
    for _ in range(3):
        # start and check one by one to keep the runs strictly sequential
        _Run(api).start().check()

    assert server.stats.requests > 0
    # keep-alive was actually in play: fewer connections than requests
    assert server.stats.connections < server.stats.requests


def test_back_to_back_runs_reuse_warm_connections(api, server):
    """No delay between runs, so the previous loop's keep-alive connections are still
    warm. Sharing one httpx.AsyncClient here is what fails with PoolTimeout / a stale
    anyio Event, so each loop must get its own client."""
    _Run(api, requests=6).start().check()
    _Run(api, requests=6).start().check()
    _Run(api, requests=6).start().check()

    assert len(api._async_clients) == 3
    clients = [entry[1] for entry in api._async_clients.values()]
    assert len({id(client) for client in clients}) == 3


def test_concurrent_loops_share_one_api(api):
    """The uvicorn loop and a worker thread hitting the same Api at the same time."""
    runs = [_Run(api, requests=10) for _ in range(3)]
    for run in runs:
        run.start()
    for run in runs:
        run.check()

    assert api._semaphore._state()["consistent"], api._semaphore._state()


def test_request_concurrency_never_exceeds_the_semaphore(api, server):
    """The whole point of the semaphore: the server must never see more concurrent
    requests than the limit, no matter how many event loops are involved."""
    server.delay = 0.05
    runs = [_Run(api, requests=8) for _ in range(3)]
    for run in runs:
        run.start()
    for run in runs:
        run.check()

    assert server.stats.peak <= SEMAPHORE_SIZE, (
        f"the server saw {server.stats.peak} concurrent requests "
        f"with a limit of {SEMAPHORE_SIZE}"
    )
    assert server.stats.peak == SEMAPHORE_SIZE, "the limit was never reached"


def test_stale_clients_are_dropped_when_their_loop_is_closed(api):
    for _ in range(3):
        _Run(api, close_loop=True).start().check()

    _Run(api).start().check()
    # closed loops must not accumulate: only the live one is kept
    assert len(api._async_clients) == 1, api._async_clients


def test_semaphore_is_cross_loop_and_resized_in_place(api):
    semaphore = api.get_default_semaphore()
    assert isinstance(semaphore, CrossLoopSemaphore)
    assert api.get_default_semaphore_size() == SEMAPHORE_SIZE
    assert semaphore.limit == SEMAPHORE_SIZE

    api.set_semaphore_size(SEMAPHORE_SIZE + 4)
    assert api.get_default_semaphore() is semaphore, "resize must keep the same object"
    assert semaphore.limit == SEMAPHORE_SIZE + 4
    assert api.get_default_semaphore_size() == SEMAPHORE_SIZE + 4

    _Run(api, requests=10).start().check()
    assert semaphore._state()["consistent"], semaphore._state()


def test_client_recreation_lock_is_not_loop_bound(api):
    """It used to be an asyncio.Lock created on first use, so it inherited the same
    cross-loop problem in the error handling path."""
    assert isinstance(api._client_recreation_lock, type(threading.Lock()))

    api._client_recreation_cooldown = 0
    api._last_client_recreation_time = None

    async def recreate():
        api._set_async_client()
        before = api.async_httpx_client
        recreated = await api._recreate_client_if_needed(httpx.RemoteProtocolError("boom"))
        return recreated, before is not api.async_httpx_client

    for _ in range(3):
        recreated, replaced = run_in_own_loop(recreate)
        assert recreated is True
        assert replaced is True


def test_injected_client_is_still_used(api):
    """Backward compatibility: assigning a client outside of a loop (tests, app code)
    keeps working and stays shared."""

    class _Fake:
        def __init__(self):
            self.calls = 0

        async def post(self, *args, **kwargs):
            self.calls += 1
            return httpx.Response(200, json={"ok": True}, request=httpx.Request("POST", "/"))

    fake = _Fake()
    api.async_httpx_client = fake
    assert api.async_httpx_client is fake

    async def call():
        return (await api.post_async("test", json={}, retries=1)).status_code

    assert run_in_own_loop(call) == 200
    assert fake.calls == 1
    assert api._async_clients[None][1] is fake


def run_in_own_loop(coro_factory):
    """Run a coroutine in a fresh thread and a fresh event loop."""
    box = {}

    def target():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            box["result"] = loop.run_until_complete(coro_factory())
        except BaseException as exc:  # noqa: BLE001
            box["error"] = exc

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(JOIN_TIMEOUT)
    assert not thread.is_alive(), "run did not finish: deadlock"
    assert "error" not in box, f"run failed: {box['error']!r}"
    return box["result"]


# --------------------------------------------------------------------------------------
# one limit per (server address, token), not per Api object
# --------------------------------------------------------------------------------------
def test_semaphore_is_shared_by_api_objects_with_the_same_key(api, server):
    """app/fastapi/request.py builds a new Api per HTTP request, so the limit cannot live
    on the object: it belongs to the instance the requests go to."""
    twin = build_api(server)
    assert twin.get_default_semaphore() is api.get_default_semaphore()
    assert twin.get_default_semaphore_size() == SEMAPHORE_SIZE


def test_semaphore_is_not_shared_across_tokens(api, server):
    """Different users are throttled separately."""
    other_user = build_api(server, token="u" * 128)
    assert other_user.get_default_semaphore() is not api.get_default_semaphore()


def test_size_change_is_visible_to_every_api_with_the_same_key(api, server):
    twin = build_api(server)
    api.set_semaphore_size(SEMAPHORE_SIZE + 5)

    assert twin.get_default_semaphore().limit == SEMAPHORE_SIZE + 5
    assert twin.get_default_semaphore_size() == SEMAPHORE_SIZE + 5
    assert api.get_default_semaphore_size() == SEMAPHORE_SIZE + 5


def test_request_concurrency_capped_across_several_api_objects(api, server):
    """Three Api objects (three "requests" to the app), three event loops, one instance:
    the server must still never see more than the limit."""
    instances = [api, build_api(server), build_api(server)]
    assert len({id(instance.get_default_semaphore()) for instance in instances}) == 1

    server.delay = 0.05
    runs = [_Run(instance, requests=8) for instance in instances]
    for run in runs:
        run.start()
    for run in runs:
        run.check()

    assert server.stats.peak <= SEMAPHORE_SIZE, (
        f"the server saw {server.stats.peak} concurrent requests from "
        f"{len(instances)} Api objects with a limit of {SEMAPHORE_SIZE}"
    )
    assert server.stats.peak == SEMAPHORE_SIZE, "the limit was never reached"

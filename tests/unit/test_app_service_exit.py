"""AppService (app v1) lets the process exit after it is stopped.

The agent event stream never ends on its own, so the publisher thread stays blocked in
``AgentAPI.get_endless_stream`` after the app is stopped, and a periodic function can be in the
middle of a long sleep. These threads must not keep the interpreter alive. Before the fix
``_shutdown`` "released" ThreadPoolExecutor threads through ``thread._tstate_lock``, which Python
3.13 removed: on 3.13+ the process logged "Successfully shutdown the APP service." and then hung.

Each test runs a real AppService in a subprocess on the running interpreter. Nothing goes over the
network: the agent stream, the app initialization and the version check are stubbed.
"""

import os
import subprocess
import sys
import textwrap
import threading
from pathlib import Path

import pytest

from supervisely.app.v1.app_service import _DaemonThreadPoolExecutor

REPO_ROOT = str(Path(__file__).resolve().parents[2])

APP = textwrap.dedent(
    """
    import asyncio, os, sys, threading, types

    os.environ.update(
        TASK_ID="1", SERVER_ADDRESS="http://localhost:1", API_TOKEN="t" * 128,
        AGENT_TOKEN="a" * 32, DEBUG_APP_DIR=sys.argv[2], DEBUG_CACHE_DIR=sys.argv[2],
    )

    try:
        from supervisely.worker_proto import worker_api_pb2  # needs protobuf
    except ImportError:
        stub = types.ModuleType("supervisely.worker_proto.worker_api_pb2")
        stub.GeneralEvent = object
        stub.Empty = lambda: None
        sys.modules[stub.__name__] = stub

    import supervisely as sly
    from supervisely.api.app_api import AppApi
    from supervisely.worker_api.agent_api import AgentAPI

    AppApi.get_url = lambda self, task_id: "http://localhost:1/app"
    AppApi.initialize = lambda self, *a, **k: None
    sly.Api._check_version = lambda self, *a, **k: None

    _sleep = asyncio.sleep
    asyncio.sleep = lambda delay, *a, **k: _sleep(min(delay, 0.1), *a, **k)  # _shutdown waits 5 s

    def endless_stream(self, *args, **kwargs):  # the agent stream never ends on its own
        threading.Event().wait()
        yield

    AgentAPI.get_endless_stream = endless_stream
    app = sly.AppService()

    @app.periodic(seconds=3600)
    def hourly(api, task_id):
        pass

    @app.callback("go")
    def go(api, task_id, context, state, app_logger):
        if sys.argv[1] == "error":
            raise RuntimeError("callback failed")
        app.stop(wait=sys.argv[1] != "force")

    app.run(initial_events=[{"command": "go", "api_token": "t" * 128}])
    print("RUN RETURNED", flush=True)
    """
)


def _run_app(mode, tmp_path, timeout=60):
    script = tmp_path / "app.py"
    script.write_text(APP)
    env = dict(os.environ)  # import supervisely from this checkout, as pytest does
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [REPO_ROOT, env.get("PYTHONPATH")]))
    return subprocess.run(
        [sys.executable, str(script), mode, str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )


def test_process_exits_after_app_stop(tmp_path):
    res = _run_app("stop", tmp_path)
    assert res.returncode == 0, res.stderr[-2000:]
    assert "RUN RETURNED" in res.stdout


def test_process_exits_after_forced_stop(tmp_path):
    res = _run_app("force", tmp_path)
    assert res.returncode == 0, res.stderr[-2000:]
    assert "RUN RETURNED" in res.stdout


def test_process_exits_when_a_callback_fails(tmp_path):
    # ignore_errors is False: the error stops the app and run() raises it
    res = _run_app("error", tmp_path)
    assert res.returncode == 1
    assert "callback failed" in res.stderr
    assert "RUN RETURNED" not in res.stdout


def test_executor_runs_work_on_daemon_threads():
    executor = _DaemonThreadPoolExecutor(max_workers=2)
    release = threading.Event()
    blocked = executor.submit(release.wait)  # like the publisher on the agent stream
    assert executor.submit(lambda: threading.current_thread().daemon).result(timeout=5) is True
    assert [executor.submit(pow, 2, i).result(timeout=5) for i in range(5)] == [1, 2, 4, 8, 16]
    with pytest.raises(ZeroDivisionError):
        executor.submit(lambda: 1 / 0).result(timeout=5)
    assert len(executor._threads) == 2

    executor.shutdown(wait=False)
    with pytest.raises(RuntimeError):
        executor.submit(print)
    release.set()
    assert blocked.result(timeout=5) is True
    executor.shutdown(wait=True)
    assert not any(t.is_alive() for t in executor._threads)

"""
Security tests for the ``/sly/session-info`` token leak fix.

Every app with a UI gets a ``/sly`` sub-application from the SDK (``Application`` ->
``_init`` -> ``create`` in ``supervisely/app/fastapi/subapp.py``; older apps mount
``create()`` themselves). Its ``POST /session-info`` route has no authentication and used to
answer every caller with::

    {"TASK_ID": ..., "SERVER_ADDRESS": ..., "API_TOKEN": os.environ["API_TOKEN"]}

so anyone who could reach a running app could read the app's own API token. After the fix
``API_TOKEN`` is the app token only in development (``is_development()``) and in advanced
debug (``is_debug_with_sly_net()``, ``DEBUG_WITH_SLY_NET``): there the local UI has no other
source for it. In every production run (``ENV=production``) it is ``null``, whatever the app
runs on - docker, a kubernetes pod, podman, a container no probe recognises or a bare host:
at the instance the web UI uses the token of the logged-in user. ``TASK_ID`` and
``SERVER_ADDRESS`` keep the values they had before the fix: ``"/"`` for production in docker
(``is_docker()``, the "production at the instance" branch of the route), the normalized
address from the environment otherwise.

Layout of this file:

* exploit tests - build a real app with the SDK factory in production mode, send the
  unauthenticated request through FastAPI ``TestClient`` and assert that the app token is
  nowhere in the response. They fail on the unfixed code.
* invariant / regression tests - behaviour that has to hold before and after the fix
  (development and advanced debug still get the token, ``TASK_ID`` and ``SERVER_ADDRESS``
  are returned in every mode with the values they had before, headless apps have no such
  route).

The running mode is switched the way the SDK reads it: the ``ENV`` and ``DEBUG_WITH_SLY_NET``
environment variables. The host the app runs on is switched with the container markers:
``/.dockerenv`` and ``/proc/self/cgroup`` (what ``is_docker()`` reads, it decides
``SERVER_ADDRESS`` only), ``KUBERNETES_SERVICE_HOST`` and ``/run/.containerenv``. The token
does not depend on any of them, they only add more production hosts that all have to get
``null``. The marker files are faked, so the real ``is_docker()`` answers the same on any OS
and does not look at the machine that runs the tests. Every test gets its own set of SDK
singletons (``Application``, the main server, ``StateJson`` ...). No real network is used:
every host name is under the reserved ``.invalid`` TLD and outgoing requests are
intercepted.
"""

import io
import logging
import os
import threading
from pathlib import Path
from typing import Callable, Dict, List, NamedTuple, Optional

import httpx
import pytest
import requests
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

import supervisely as sly
import supervisely._utils as sly_utils
import supervisely.app.fastapi.offline as offline
import supervisely.app.fastapi.subapp as subapp
from supervisely.app.content import ContentOrigin
from supervisely.app.singleton import Singleton
from supervisely.app.widgets import Text

# the app's own credentials (environment of the running app)
PLATFORM = "https://platform.invalid"
APP_TOKEN = "app-own-secret-token-" + "a" * 107
TASK_ID = "4242"

# what a caller can put into a request
ATTACKER = "https://attacker.invalid"
USER_TOKEN = "logged-in-user-token-" + "u" * 107

SESSION_INFO_KEYS = {"TASK_ID", "SERVER_ADDRESS", "API_TOKEN"}
JOIN_TIMEOUT = 30

# /proc/self/cgroup
CGROUP_HOST = "0::/user.slice/user-1000.slice/session-3.scope\n"
CGROUP_V2 = "0::/\n"  # cgroup v2: the same line in a container of any runtime
CGROUP_V1_DOCKER = (
    "12:memory:/docker/3f4e9a1c0b7d\n"
    "11:cpu,cpuacct:/docker/3f4e9a1c0b7d\n"
    "1:name=systemd:/docker/3f4e9a1c0b7d\n"
)
CGROUP_V1_KUBERNETES = (
    "12:memory:/kubepods/burstable/pod6f0c2a9e/3f4e9a1c0b7d\n"
    "11:cpu,cpuacct:/kubepods/burstable/pod6f0c2a9e/3f4e9a1c0b7d\n"
    "1:name=systemd:/kubepods/burstable/pod6f0c2a9e/3f4e9a1c0b7d\n"
)


class Host(NamedTuple):
    """Container markers of the machine the app runs on, as the SDK probes find them."""

    dockerenv: bool = False  # /.dockerenv exists: docker
    cgroup: Optional[str] = None  # content of /proc/self/cgroup; None - no such file
    containerenv: bool = False  # /run/.containerenv exists: podman
    kubernetes: bool = False  # KUBERNETES_SERVICE_HOST is set: every kubernetes pod


class Mode(NamedTuple):
    """Running mode of the app, as the SDK helpers see it."""

    env: Optional[str]  # ENV variable; None - not set, the SDK default is development
    host: Host
    sly_net: bool = False  # DEBUG_WITH_SLY_NET is set (advanced debug)


LINUX_HOST = Host(cgroup=CGROUP_HOST)
NOT_LINUX_HOST = Host()  # a developer's windows or macOS machine
DOCKER = Host(dockerenv=True, cgroup=CGROUP_V2)
KUBERNETES = Host(cgroup=CGROUP_V2, kubernetes=True)
PODMAN = Host(cgroup=CGROUP_V2, containerenv=True)
# a container of a runtime that leaves none of the markers above
UNMARKED_CONTAINER = Host(cgroup=CGROUP_V2)

# production app in a docker container, the way the agent starts it at the instance:
# SERVER_ADDRESS is "/", the token must not be handed out
DOCKER_MODES: Dict[str, Mode] = {
    "production-in-docker": Mode("production", DOCKER),
    "production-in-docker-dockerenv-only": Mode("production", Host(dockerenv=True)),
    "production-in-docker-cgroup-v1": Mode("production", Host(cgroup=CGROUP_V1_DOCKER)),
    "production-in-docker-in-kubernetes": Mode(
        "production", Host(dockerenv=True, cgroup=CGROUP_V2, kubernetes=True)
    ),
}
# production app in a container that is_docker() does not recognise:
# SERVER_ADDRESS is the one of the environment, the token must not be handed out
OTHER_CONTAINER_MODES: Dict[str, Mode] = {
    "production-in-kubernetes": Mode("production", KUBERNETES),
    "production-in-kubernetes-cgroup-v1": Mode(
        "production", Host(cgroup=CGROUP_V1_KUBERNETES, kubernetes=True)
    ),
    "production-in-podman": Mode("production", PODMAN),
}
# production env without a container: SERVER_ADDRESS is the one of the environment (the
# route takes its development branch for it), the token must not be handed out
BARE_HOST_MODES: Dict[str, Mode] = {
    "production-on-bare-host": Mode("production", LINUX_HOST),
    "production-on-bare-host-not-linux": Mode("production", NOT_LINUX_HOST),
}
NO_TOKEN_MODES: Dict[str, Mode] = {**DOCKER_MODES, **OTHER_CONTAINER_MODES, **BARE_HOST_MODES}

# modes in which the UI has no logged-in user to take a token from
TOKEN_MODES: Dict[str, Mode] = {
    "development": Mode(None, LINUX_HOST),
    "development-env": Mode("development", LINUX_HOST),
    "development-not-linux": Mode(None, NOT_LINUX_HOST),
    "development-in-docker": Mode("development", DOCKER),
    "development-in-kubernetes": Mode("development", KUBERNETES),
    "development-in-podman": Mode(None, PODMAN),
    "advanced-debug": Mode(None, LINUX_HOST, sly_net=True),
    "advanced-debug-production": Mode("production", LINUX_HOST, sly_net=True),
    "advanced-debug-production-in-docker": Mode("production", DOCKER, sly_net=True),
    "advanced-debug-production-in-kubernetes": Mode("production", KUBERNETES, sly_net=True),
    "advanced-debug-production-in-podman": Mode("production", PODMAN, sly_net=True),
}

# production app in a container that leaves none of the markers: the token must not be
# handed out either
UNMARKED_CONTAINER_MODE = Mode("production", UNMARKED_CONTAINER)

ALL_MODES: Dict[str, Mode] = {
    **NO_TOKEN_MODES,
    **TOKEN_MODES,
    "production-in-unmarked-container": UNMARKED_CONTAINER_MODE,
}

# one production container of every runtime that leaves a marker
CONTAINERS: Dict[str, Mode] = {
    "docker": DOCKER_MODES["production-in-docker"],
    "kubernetes": OTHER_CONTAINER_MODES["production-in-kubernetes"],
    "podman": OTHER_CONTAINER_MODES["production-in-podman"],
}
PRODUCTION = CONTAINERS["docker"]

# ways to build an app that has the route
UI_APPS = ["layout", "templates", "create-mount"]


class SdkRuntime:
    """Background work of the apps built in a test and what it tried to send to the platform."""

    def __init__(self):
        # started by Application.__init__ in production: renders the page for offline usage
        self.threads: List[threading.Thread] = []
        # UI content the app wanted to store on the platform
        self.platform_updates: List[dict] = []
        # apps whose page was about to be dumped to disk and uploaded to the platform
        self.offline_dumps: List[FastAPI] = []

    def wait(self):
        """Block until everything the apps started in background has finished."""
        for thread in list(self.threads):
            thread.join(JOIN_TIMEOUT)
            assert not thread.is_alive()
        uploader = offline._offline_session_uploader
        if uploader is not None:
            uploader.join(JOIN_TIMEOUT)
            assert not uploader.is_alive()


def token_leaks(response: httpx.Response) -> List[str]:
    """Every part of the HTTP response that carries the app's own token."""
    leaks = []
    if APP_TOKEN in response.text:
        leaks.append("body")
    for name, value in response.headers.items():
        if APP_TOKEN in value:
            leaks.append(f"header {name}")
    return leaks


def post_session_info(client: TestClient, **request_kwargs) -> httpx.Response:
    response = client.post("/sly/session-info", **request_kwargs)
    assert response.status_code == 200, response.text
    assert set(response.json()) == SESSION_INFO_KEYS
    return response


def expected_server_address(mode: Mode) -> str:
    """SERVER_ADDRESS the route returns; the fix does not change it in any mode."""
    # at the instance the UI is served by the platform itself
    return "/" if mode in DOCKER_MODES.values() else PLATFORM


@pytest.fixture(autouse=True)
def app_env(monkeypatch):
    """Environment of a running app: its own server address, its own secret token, its task."""
    monkeypatch.setenv("SERVER_ADDRESS", PLATFORM)
    monkeypatch.setenv("API_TOKEN", APP_TOKEN)
    monkeypatch.setenv("TASK_ID", TASK_ID)
    monkeypatch.setenv("CONTENT_ORIGIN_UPDATE_INTERVAL", "0.05")
    for name in [
        "ENV",
        "DEBUG_WITH_SLY_NET",
        "SUPERVISELY_MULTIUSER_APP_MODE",
        "SUPERVISELY_API_SERVER_ADDRESS",
        "_SUPERVISELY_OFFLINE_FILES_UPLOADED",
    ]:
        monkeypatch.delenv(name, raising=False)
    # per-process memory of the "http -> https" probe, keep the tests independent
    monkeypatch.setattr(sly.Api, "_checked_servers", set())


@pytest.fixture(autouse=True)
def run_on(monkeypatch) -> Callable[[Host], None]:
    """Fake the container markers the SDK looks for: a bare host until a test moves the app.

    The real helpers look at the machine that runs the tests, which may be a container.
    """
    current = [LINUX_HOST]
    real_exists = os.path.exists
    real_isfile = os.path.isfile

    def fake_exists(path):
        if path == "/.dockerenv":
            return current[0].dockerenv
        if path == "/run/.containerenv":
            return current[0].containerenv
        if path == "/proc/self/cgroup":
            return current[0].cgroup is not None
        return real_exists(path)

    def fake_isfile(path):
        if path == "/proc/self/cgroup":
            return current[0].cgroup is not None
        return real_isfile(path)

    def fake_open(path, *args, **kwargs):
        if path == "/proc/self/cgroup":
            return io.StringIO(current[0].cgroup)
        return open(path, *args, **kwargs)

    def move(host: Host) -> None:
        current[0] = host
        if host.kubernetes:
            monkeypatch.setenv("KUBERNETES_SERVICE_HOST", "10.96.0.1")
        else:
            monkeypatch.delenv("KUBERNETES_SERVICE_HOST", raising=False)

    monkeypatch.setattr(os.path, "exists", fake_exists)
    monkeypatch.setattr(os.path, "isfile", fake_isfile)
    # shadows the builtin for supervisely._utils only
    monkeypatch.setattr(sly_utils, "open", fake_open, raising=False)
    move(LINUX_HOST)
    return move


@pytest.fixture(autouse=True)
def http_calls(monkeypatch) -> List[str]:
    """Record every outgoing HTTP request instead of sending it."""
    calls: List[str] = []

    def fake_send(adapter, request, **kwargs):
        calls.append(f"{request.method} {request.url}")
        response = requests.Response()
        response.status_code = 200
        response.reason = "OK"
        response.headers["Content-Type"] = "application/json"
        response._content = b"{}"
        response.url = request.url
        response.request = request
        return response

    def refuse(request):
        calls.append(f"{request.method} {request.url}")
        raise httpx.ConnectError("real network is disabled in tests")

    def fake_handle_request(transport, request):
        refuse(request)

    async def fake_handle_async_request(transport, request):
        refuse(request)

    monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", fake_send)
    # TestClient uses its own in-process transport
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", fake_handle_request)
    monkeypatch.setattr(
        httpx.AsyncHTTPTransport, "handle_async_request", fake_handle_async_request
    )
    return calls


@pytest.fixture(autouse=True)
def sdk_runtime(monkeypatch, app_env, run_on, http_calls) -> SdkRuntime:
    """Fresh SDK singletons for every test and no leftovers of the apps built in it."""
    runtime = SdkRuntime()

    # Application, the main server, StateJson, DataJson, templates ... are one per process
    monkeypatch.setattr(Singleton, "_instances", {})
    monkeypatch.setattr(Singleton, "_nested_instances", {})
    monkeypatch.setattr(offline, "_offline_session_uploader", None)
    monkeypatch.setattr(offline, "_pending_offline_session", None)
    # read from the environment once, when the SDK is imported
    monkeypatch.setattr(subapp, "SUPERVISELY_SERVER_PATH_PREFIX", "")
    uvicorn_logger = logging.getLogger("uvicorn.access")
    monkeypatch.setattr(uvicorn_logger, "filters", list(uvicorn_logger.filters))

    # a production app pushes its UI content to the platform and uploads an offline copy of
    # the page from background threads: keep both away from the network and the file system
    def fake_send_content(origin, data_patch, state):
        runtime.platform_updates.append({"data_patch": list(data_patch), "state": state})

    def fake_dump_files(app, template_response):
        runtime.offline_dumps.append(app)

    class RecordedThread(threading.Thread):
        def start(self):
            runtime.threads.append(self)
            super().start()

    monkeypatch.setattr(ContentOrigin, "_send", fake_send_content)
    monkeypatch.setattr(offline, "dump_files_to_supervisely", fake_dump_files)
    monkeypatch.setattr(subapp, "Thread", RecordedThread)

    yield runtime

    runtime.wait()
    origin = Singleton._instances.get(ContentOrigin)
    if origin is not None:
        origin.stop()
        if origin._loop_thread.is_alive():
            origin._loop_thread.join(JOIN_TIMEOUT)
            assert not origin._loop_thread.is_alive()


@pytest.fixture()
def run_as(monkeypatch, run_on) -> Callable[[Mode], None]:
    """Put the process into one of the SDK running modes."""

    def apply(mode: Mode) -> None:
        if mode.env is None:
            monkeypatch.delenv("ENV", raising=False)
        else:
            monkeypatch.setenv("ENV", mode.env)
        if mode.sly_net:
            monkeypatch.setenv("DEBUG_WITH_SLY_NET", "1")
        else:
            monkeypatch.delenv("DEBUG_WITH_SLY_NET", raising=False)
        run_on(mode.host)

    return apply


@pytest.fixture()
def make_app(sdk_runtime: SdkRuntime, tmp_path: Path) -> Callable[..., object]:
    """Build an app with the SDK factory, in the mode the process is in right now."""

    def make(kind: str = "layout"):
        if kind == "layout":
            # an app made of widgets; the object itself is what "uvicorn main:app" serves
            app = sly.Application(layout=Text("session info test app"))
        elif kind == "templates":
            # an app with its own html templates
            templates_dir = tmp_path / "templates"
            templates_dir.mkdir()
            (templates_dir / "main.html").write_text("<div>session info test app</div>")
            app = sly.Application(templates_dir=str(templates_dir)).get_server()
        elif kind == "create-mount":
            # an app that mounts the sub-application on its own server
            app = FastAPI()
            app.mount("/sly", sly.app.fastapi.create())
        elif kind == "headless":
            # an app without UI, e.g. a serving app
            app = sly.Application().get_server()
        else:
            raise ValueError(kind)
        sdk_runtime.wait()
        return app

    return make


@pytest.fixture()
def make_client(make_app) -> Callable[..., TestClient]:
    def make(kind: str = "layout") -> TestClient:
        return TestClient(make_app(kind))

    return make


# Exploit tests: these fail on the unfixed code


@pytest.mark.parametrize("kind", UI_APPS)
@pytest.mark.parametrize("mode", NO_TOKEN_MODES.values(), ids=list(NO_TOKEN_MODES))
def test_production_app_does_not_return_app_token(run_as, make_client, http_calls, mode, kind):
    run_as(mode)
    client = make_client(kind)

    # the whole attack: one request, no credentials of any kind
    response = post_session_info(client)

    # Should not hand out the app's own token
    assert token_leaks(response) == []
    assert response.json()["API_TOKEN"] is None
    assert http_calls == []


@pytest.mark.parametrize(
    "request_kwargs",
    [
        {"json": {}},
        {"json": {"state": {}, "context": {}}},
        {"json": {"state": {}, "context": {"apiToken": USER_TOKEN, "userId": 7}}},
        {"json": {"server_address": ATTACKER, "api_token": USER_TOKEN}},
        {"json": {"context": {"outside_request": True}}},
        {
            "headers": {
                "Host": "localhost:8000",
                "Origin": "http://localhost:8000",
                "Referer": "http://localhost:8000/?userId=7",
                "X-Forwarded-For": "127.0.0.1",
                "Cookie": "session=not-a-real-session",
                "x-debug-mode": "1",
            }
        },
        {"params": {"ENV": "development", "DEBUG_WITH_SLY_NET": "1", "userId": "7"}},
    ],
    ids=[
        "empty-json",
        "empty-state-and-context",
        "other-token-in-context",
        "caller-chosen-server",
        "outside-request",
        "localhost-headers",
        "mode-in-query",
    ],
)
@pytest.mark.parametrize("mode", CONTAINERS.values(), ids=list(CONTAINERS))
def test_production_app_does_not_return_app_token_for_any_request(
    run_as, make_client, http_calls, mode, request_kwargs
):
    run_as(mode)
    client = make_client()

    response = post_session_info(client, **request_kwargs)

    # Should not let anything in the request talk the route into development behaviour
    assert token_leaks(response) == []
    assert response.json() == {
        "TASK_ID": TASK_ID,
        "SERVER_ADDRESS": expected_server_address(mode),
        "API_TOKEN": None,
    }
    assert http_calls == []


@pytest.mark.parametrize("mode", CONTAINERS.values(), ids=list(CONTAINERS))
def test_production_multiuser_app_does_not_return_app_token(
    run_as, make_client, monkeypatch, mode
):
    monkeypatch.setenv("SUPERVISELY_MULTIUSER_APP_MODE", "true")
    run_as(mode)
    client = make_client()

    anonymous = post_session_info(client)
    as_user = post_session_info(client, params={"userId": "7"}, json={"context": {"userId": 7}})

    assert token_leaks(anonymous) == []
    assert token_leaks(as_user) == []
    assert anonymous.json()["API_TOKEN"] is None
    assert as_user.json()["API_TOKEN"] is None


@pytest.mark.parametrize("mode", CONTAINERS.values(), ids=list(CONTAINERS))
def test_production_app_does_not_return_app_token_after_being_used(run_as, make_app, mode):
    # the app has been opened by a logged-in user before the anonymous caller comes
    run_as(mode)
    app = make_app()
    user = TestClient(app)
    user_request = {"state": {}, "context": {"apiToken": USER_TOKEN, "userId": 7}}
    assert user.get("/").status_code == 200
    assert user.post("/sly/state", json=user_request).status_code == 200
    as_user = post_session_info(user, json=user_request)

    # a client of its own: nothing of the user's session comes with the requests
    anonymous = [post_session_info(TestClient(app)) for _ in range(3)]

    # Should not start to hand out the token once the app is in use
    for response in [as_user, *anonymous]:
        assert token_leaks(response) == []
        assert response.json()["API_TOKEN"] is None


def test_production_app_in_unmarked_container_does_not_return_app_token(run_as, make_client):
    # e.g. plain containerd or LXC: the token does not depend on finding a container marker
    run_as(UNMARKED_CONTAINER_MODE)
    client = make_client()

    response = post_session_info(client)

    assert token_leaks(response) == []
    assert response.json()["API_TOKEN"] is None


# Invariant and regression tests: these pass before and after the fix


@pytest.mark.parametrize(
    "host, docker",
    [
        (LINUX_HOST, False),
        (NOT_LINUX_HOST, False),
        (DOCKER, True),
        (Host(dockerenv=True), True),
        (Host(cgroup=CGROUP_V1_DOCKER), True),
        (KUBERNETES, False),
        (Host(cgroup=CGROUP_V1_KUBERNETES, kubernetes=True), False),
        (PODMAN, False),
        (UNMARKED_CONTAINER, False),
    ],
    ids=[
        "linux-host",
        "not-linux-host",
        "docker",
        "docker-dockerenv-only",
        "docker-cgroup-v1",
        "kubernetes",
        "kubernetes-cgroup-v1",
        "podman",
        "unmarked-container",
    ],
)
def test_run_on_drives_the_real_container_probes(run_on, host, docker):
    run_on(host)

    # the route asks the real helper
    assert subapp.is_docker is sly_utils.is_docker
    assert bool(subapp.is_docker()) == docker
    assert os.path.exists("/.dockerenv") == host.dockerenv
    assert os.path.exists("/run/.containerenv") == host.containerenv
    assert bool(os.environ.get("KUBERNETES_SERVICE_HOST")) == host.kubernetes


@pytest.mark.parametrize("mode", ALL_MODES.values(), ids=list(ALL_MODES))
def test_run_as_switches_the_sdk_mode_helpers(run_as, mode):
    run_as(mode)

    assert sly.is_production() == (mode.env == "production")
    assert sly.is_development() == (mode.env != "production")
    assert sly.is_debug_with_sly_net() == mode.sly_net
    if mode in DOCKER_MODES.values():
        assert subapp.is_docker()
    if mode in OTHER_CONTAINER_MODES.values() or mode == UNMARKED_CONTAINER_MODE:
        assert not subapp.is_docker()


def test_container_probes_do_not_hide_real_files(tmp_path: Path):
    # the fakes answer for the marker paths only
    existing = tmp_path / "file.txt"
    existing.write_text("content")

    assert os.path.exists(str(existing))
    assert os.path.isfile(str(existing))
    assert os.path.exists(str(tmp_path)) and not os.path.isfile(str(tmp_path))
    assert not os.path.exists(str(tmp_path / "missing"))
    assert not os.path.isfile(str(tmp_path / "missing"))


@pytest.mark.parametrize("kind", UI_APPS)
@pytest.mark.parametrize("mode", TOKEN_MODES.values(), ids=list(TOKEN_MODES))
def test_development_and_advanced_debug_still_return_app_token(
    run_as, make_client, mode, kind
):
    run_as(mode)
    client = make_client(kind)

    response = post_session_info(client)

    # Should give the local UI everything it needs to talk to the platform
    assert response.json() == {
        "TASK_ID": TASK_ID,
        "SERVER_ADDRESS": PLATFORM,
        "API_TOKEN": APP_TOKEN,
    }
    # and a check that the helper of the exploit tests is able to see the token
    assert token_leaks(response) == ["body"]


@pytest.mark.parametrize("kind", UI_APPS)
@pytest.mark.parametrize("mode", ALL_MODES.values(), ids=list(ALL_MODES))
def test_task_id_and_server_address_are_returned_in_every_mode(run_as, make_client, mode, kind):
    run_as(mode)
    client = make_client(kind)

    body = post_session_info(client).json()

    assert body["TASK_ID"] == TASK_ID
    assert body["SERVER_ADDRESS"] == expected_server_address(mode)


@pytest.mark.parametrize(
    "address, expected",
    [
        (PLATFORM + "/", PLATFORM),
        ("platform.invalid", "http://platform.invalid"),
        ("http://platform.invalid/", "http://platform.invalid"),
    ],
    ids=["trailing-slash", "no-scheme", "http"],
)
@pytest.mark.parametrize(
    "mode_name",
    [
        "development",
        "advanced-debug-production-in-docker",
        "production-on-bare-host",
        "production-in-kubernetes",
        "production-in-podman",
    ],
)
def test_server_address_from_environment_is_normalized(
    run_as, make_client, monkeypatch, mode_name, address, expected
):
    monkeypatch.setenv("SERVER_ADDRESS", address)
    run_as(ALL_MODES[mode_name])
    client = make_client()

    assert post_session_info(client).json()["SERVER_ADDRESS"] == expected


@pytest.mark.parametrize("mode", ALL_MODES.values(), ids=list(ALL_MODES))
def test_missing_environment_values_are_returned_as_null(run_as, make_client, monkeypatch, mode):
    for name in ["TASK_ID", "SERVER_ADDRESS", "API_TOKEN"]:
        monkeypatch.delenv(name)
    run_as(mode)
    client = make_client("create-mount")

    body = post_session_info(client).json()

    assert body == {
        "TASK_ID": None,
        "SERVER_ADDRESS": "/" if mode in DOCKER_MODES.values() else None,
        "API_TOKEN": None,
    }


@pytest.mark.parametrize("mode", CONTAINERS.values(), ids=list(CONTAINERS))
def test_session_info_only_answers_post(run_as, make_client, mode):
    run_as(mode)
    client = make_client()

    response = client.get("/sly/session-info")

    assert response.status_code == 405
    assert token_leaks(response) == []


@pytest.mark.parametrize("mode", ALL_MODES.values(), ids=list(ALL_MODES))
def test_headless_app_has_no_session_info_route(run_as, make_client, mode):
    run_as(mode)
    client = make_client("headless")

    response = client.post("/sly/session-info")

    assert response.status_code == 404
    assert token_leaks(response) == []


@pytest.mark.parametrize(
    "method, path",
    [("POST", "/sly/data"), ("POST", "/sly/state"), ("GET", "/"), ("POST", "/is_running")],
    ids=["data", "state", "index-page", "is-running"],
)
@pytest.mark.parametrize(
    "mode",
    [*CONTAINERS.values(), TOKEN_MODES["development"]],
    ids=[*CONTAINERS, "development"],
)
def test_other_ui_routes_do_not_contain_app_token(run_as, make_client, mode, method, path):
    run_as(mode)
    client = make_client()

    response = client.request(method, path)

    assert response.status_code == 200, response.text
    assert token_leaks(response) == []


@pytest.mark.parametrize("kind", ["layout", "templates"])
@pytest.mark.parametrize("mode", CONTAINERS.values(), ids=list(CONTAINERS))
def test_production_app_is_really_built_in_production_mode(
    run_as, make_client, sdk_runtime, http_calls, mode, kind
):
    # a check of the exploit tests: the app they attack went through the production branch
    # of Application.__init__, and its platform-facing background work was intercepted
    run_as(mode)
    client = make_client(kind)

    assert client.post("/is_running").json() == {"running": True, "mode": "production"}
    assert Singleton._instances[ContentOrigin]._loop_thread.is_alive()
    assert len(sdk_runtime.threads) == 1
    assert len(sdk_runtime.offline_dumps) == 1
    assert http_calls == []


@pytest.mark.parametrize("mode", CONTAINERS.values(), ids=list(CONTAINERS))
def test_production_request_with_user_token_is_served_with_that_token(
    run_as, make_app, http_calls, mode
):
    # at the instance the UI sends the token of the logged-in user with every request, so
    # the app works on behalf of that user without its own token being known to the browser
    run_as(mode)
    server = make_app().get_server()
    seen = []

    @server.post("/whoami")
    def whoami(request: Request):
        seen.append(request.state.api)
        return {"server_address": request.state.api.server_address}

    client = TestClient(server)
    session_info = post_session_info(client).json()
    response = client.post("/whoami", json={"state": {}, "context": {"apiToken": USER_TOKEN}})

    assert session_info["SERVER_ADDRESS"] == expected_server_address(mode)
    assert response.status_code == 200, response.text
    assert response.json() == {"server_address": PLATFORM}
    assert [(api.server_address, api.token) for api in seen] == [(PLATFORM, USER_TOKEN)]
    assert seen[0].headers["x-api-key"] == USER_TOKEN
    assert token_leaks(response) == []
    assert http_calls == []

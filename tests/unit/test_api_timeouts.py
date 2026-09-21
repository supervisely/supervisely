# coding: utf-8

import asyncio
import os
import unittest
from unittest.mock import MagicMock, patch

import httpx

from supervisely.api.api import DEFAULT_HTTPX_TIMEOUT, Api

SERVER_ADDRESS = "https://app.supervisely.com"
TOKEN = "a" * 128
TIMEOUTS_ENV = "SUPERVISELY_API_TIMEOUTS"


def _make_api(**kwargs) -> Api:
    return Api(SERVER_ADDRESS, TOKEN, **kwargs)


def _ok_response() -> MagicMock:
    response = MagicMock()
    response.status_code = 200
    return response


class _StreamResponse:
    """Minimal stand-in for the context manager returned by httpx stream methods."""

    def __init__(self, chunks=(b"data",)):
        self.status_code = 200
        self.headers = {"content-length": str(sum(len(chunk) for chunk in chunks))}
        self._chunks = chunks

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    def iter_raw(self, chunk_size):
        yield from self._chunks

    async def aiter_raw(self, chunk_size):
        for chunk in self._chunks:
            yield chunk


class EnvTestCase(unittest.TestCase):
    def setUp(self):
        self._env_backup = os.environ.pop(TIMEOUTS_ENV, None)

    def tearDown(self):
        os.environ.pop(TIMEOUTS_ENV, None)
        if self._env_backup is not None:
            os.environ[TIMEOUTS_ENV] = self._env_backup


class TestApiTimeoutsConfig(EnvTestCase):
    def test_not_configured_by_default(self):
        api = _make_api()
        self.assertIsNone(api.timeouts)
        self.assertEqual(api._httpx_timeout, httpx.Timeout(DEFAULT_HTTPX_TIMEOUT))

    def test_from_env(self):
        os.environ[TIMEOUTS_ENV] = "10,45"
        self.assertEqual(_make_api().timeouts, (10.0, 45.0))

    def test_single_value_from_env_applies_to_both_phases(self):
        os.environ[TIMEOUTS_ENV] = "7"
        self.assertEqual(_make_api().timeouts, (7.0, 7.0))

    def test_invalid_env_leaves_timeouts_as_they_are(self):
        for value in ["abc", "1,2,3", "-5,10", "0,10", "", "  "]:
            with self.subTest(value=value):
                os.environ[TIMEOUTS_ENV] = value
                self.assertIsNone(_make_api().timeouts)

    def test_argument_takes_precedence_over_env(self):
        os.environ[TIMEOUTS_ENV] = "10,45"
        self.assertEqual(_make_api(timeouts=(3, 9)).timeouts, (3.0, 9.0))

    def test_invalid_argument_is_an_error(self):
        for value in ["30,300", (30,), (0, -1), (1, "x"), 0, -1, True]:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    _make_api(timeouts=value)

    def test_httpx_timeout_shape(self):
        timeout = _make_api(timeouts=(3, 9))._httpx_timeout
        self.assertEqual(timeout.connect, 3.0)
        self.assertEqual(timeout.read, 9.0)
        self.assertEqual(timeout.write, 9.0)
        self.assertEqual(timeout.pool, 9.0)

    def test_setter_updates_both_transports(self):
        api = _make_api()
        api.timeouts = 5
        self.assertEqual(api.timeouts, (5.0, 5.0))
        self.assertEqual(api._httpx_timeout.read, 5.0)


class TestTimeoutsArePassedToTransport(EnvTestCase):
    def setUp(self):
        super().setUp()
        self.api = _make_api(timeouts=(3, 9))
        self.api._skip_https_redirect_check = True

    def test_requests_post(self):
        with patch("supervisely.api.api.requests.post", return_value=_ok_response()) as mock:
            self.api.post("some.method", {"id": 1})
        self.assertEqual(mock.call_args.kwargs["timeout"], (3.0, 9.0))

    def test_requests_post_per_call_timeout(self):
        with patch("supervisely.api.api.requests.post", return_value=_ok_response()) as mock:
            self.api.post("some.method", {"id": 1}, timeout=(1, 2))
        self.assertEqual(mock.call_args.kwargs["timeout"], (1, 2))

    def test_requests_get(self):
        with patch("supervisely.api.api.requests.get", return_value=_ok_response()) as mock:
            self.api.get("some.method", {"id": 1})
        self.assertEqual(mock.call_args.kwargs["timeout"], (3.0, 9.0))

    def test_requests_without_configured_timeouts(self):
        api = _make_api()
        api._skip_https_redirect_check = True
        with patch("supervisely.api.api.requests.post", return_value=_ok_response()) as mock:
            api.post("some.method", {"id": 1})
        self.assertIsNone(mock.call_args.kwargs["timeout"])

    def test_httpx_post(self):
        self.api.httpx_client = MagicMock()
        self.api.httpx_client.post.return_value = _ok_response()
        self.api.post_httpx("some.method", json={"id": 1})
        timeout = self.api.httpx_client.post.call_args.kwargs["timeout"]
        self.assertEqual((timeout.connect, timeout.read), (3.0, 9.0))

    def test_httpx_get(self):
        self.api.httpx_client = MagicMock()
        self.api.httpx_client.get.return_value = _ok_response()
        self.api.get_httpx("some.method", params={"id": 1})
        timeout = self.api.httpx_client.get.call_args.kwargs["timeout"]
        self.assertEqual((timeout.connect, timeout.read), (3.0, 9.0))

    def test_httpx_per_call_timeout_is_not_overridden(self):
        self.api.httpx_client = MagicMock()
        self.api.httpx_client.post.return_value = _ok_response()
        self.api.post_httpx("some.method", json={"id": 1}, timeout=httpx.Timeout(11))
        timeout = self.api.httpx_client.post.call_args.kwargs["timeout"]
        self.assertEqual(timeout.read, 11.0)

    def test_stream(self):
        self.api.httpx_client = MagicMock()
        self.api.httpx_client.stream.return_value = _StreamResponse()
        list(self.api.stream("some.method", "GET", {"id": 1}))
        timeout = self.api.httpx_client.stream.call_args.kwargs["timeout"]
        self.assertEqual((timeout.connect, timeout.read), (3.0, 9.0))

    def test_stream_async(self):
        self.api.async_httpx_client = MagicMock()
        self.api.async_httpx_client.stream.return_value = _StreamResponse()

        async def consume():
            return [chunk async for chunk in self.api.stream_async("some.method", "GET", {"id": 1})]

        asyncio.run(consume())
        timeout = self.api.async_httpx_client.stream.call_args.kwargs["timeout"]
        self.assertEqual((timeout.connect, timeout.read), (3.0, 9.0))

    def test_httpx_without_configured_timeouts(self):
        api = _make_api()
        api.httpx_client = MagicMock()
        api.httpx_client.post.return_value = _ok_response()
        api.post_httpx("some.method", json={"id": 1})
        timeout = api.httpx_client.post.call_args.kwargs["timeout"]
        self.assertEqual(timeout, httpx.Timeout(DEFAULT_HTTPX_TIMEOUT))


if __name__ == "__main__":
    unittest.main()

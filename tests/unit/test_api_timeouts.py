# coding: utf-8

import os
import unittest
from unittest.mock import MagicMock, patch

import httpx

from supervisely.api.api import (
    DEFAULT_API_CONNECT_TIMEOUT,
    DEFAULT_API_READ_TIMEOUT,
    DEFAULT_HTTPX_CONNECT_TIMEOUT,
    Api,
)

SERVER_ADDRESS = "https://app.supervisely.com"
TOKEN = "a" * 128
TIMEOUTS_ENV = "SUPERVISELY_API_TIMEOUTS"


def _make_api(**kwargs) -> Api:
    return Api(SERVER_ADDRESS, TOKEN, **kwargs)


def _ok_response() -> MagicMock:
    response = MagicMock()
    response.status_code = 200
    return response


class TestApiTimeoutsConfig(unittest.TestCase):
    def setUp(self):
        self._env_backup = os.environ.pop(TIMEOUTS_ENV, None)

    def tearDown(self):
        os.environ.pop(TIMEOUTS_ENV, None)
        if self._env_backup is not None:
            os.environ[TIMEOUTS_ENV] = self._env_backup

    def test_defaults(self):
        self.assertEqual(
            _make_api().timeouts,
            (float(DEFAULT_API_CONNECT_TIMEOUT), float(DEFAULT_API_READ_TIMEOUT)),
        )

    def test_from_env(self):
        os.environ[TIMEOUTS_ENV] = "10,45"
        self.assertEqual(_make_api().timeouts, (10.0, 45.0))

    def test_single_value_from_env_applies_to_both_phases(self):
        os.environ[TIMEOUTS_ENV] = "7"
        self.assertEqual(_make_api().timeouts, (7.0, 7.0))

    def test_invalid_env_falls_back_to_defaults(self):
        defaults = (float(DEFAULT_API_CONNECT_TIMEOUT), float(DEFAULT_API_READ_TIMEOUT))
        for value in ["abc", "1,2,3", "-5,10", "0,10", "", "  "]:
            with self.subTest(value=value):
                os.environ[TIMEOUTS_ENV] = value
                self.assertEqual(_make_api().timeouts, defaults)

    def test_argument_takes_precedence_over_env(self):
        os.environ[TIMEOUTS_ENV] = "10,45"
        self.assertEqual(_make_api(timeouts=(3, 9)).timeouts, (3.0, 9.0))

    def test_httpx_timeout(self):
        timeout = _make_api(timeouts=(3, 9))._httpx_timeout
        self.assertEqual(timeout.connect, 3.0)
        self.assertEqual(timeout.read, 9.0)
        self.assertEqual(timeout.write, 9.0)
        self.assertEqual(timeout.pool, 3.0)

    def test_httpx_keeps_its_own_connect_timeout_until_configured(self):
        timeout = _make_api()._httpx_timeout
        self.assertEqual(timeout.connect, float(DEFAULT_HTTPX_CONNECT_TIMEOUT))
        self.assertEqual(timeout.read, float(DEFAULT_API_READ_TIMEOUT))

    def test_env_applies_to_both_transports(self):
        os.environ[TIMEOUTS_ENV] = "10,45"
        api = _make_api()
        self.assertEqual(api.timeouts, (10.0, 45.0))
        self.assertEqual(api._httpx_timeout.connect, 10.0)


class TestTimeoutsArePassedToTransport(unittest.TestCase):
    def setUp(self):
        self._env_backup = os.environ.pop(TIMEOUTS_ENV, None)
        self.api = _make_api(timeouts=(3, 9))
        self.api._skip_https_redirect_check = True

    def tearDown(self):
        if self._env_backup is not None:
            os.environ[TIMEOUTS_ENV] = self._env_backup

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


if __name__ == "__main__":
    unittest.main()

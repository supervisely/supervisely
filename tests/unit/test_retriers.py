# coding: utf-8
"""Tests for worker_api retrier policies, focused on RetrierConnTO retry behavior."""

import requests

from supervisely.worker_api.retriers import RetrierConnTO
from supervisely.io.network_exceptions import RETRY_STATUS_CODES


def _http_error(status_code):
    resp = requests.Response()
    resp.status_code = status_code
    return requests.exceptions.HTTPError(f"status {status_code}", response=resp)


def _make_retrier(retry_cnt=3, swallow_exc=False):
    # wait_sec 0 keeps the test fast (no real backoff sleeps)
    return RetrierConnTO(
        retry_cnt=retry_cnt,
        wait_sec_first=0,
        wait_sec_max=0,
        timeout=[1, 1],
        swallow_exc=swallow_exc,
    )


class _Counter:
    def __init__(self, exc):
        self.exc = exc
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        raise self.exc


def _run(retrier, cb):
    raised = None
    try:
        retrier.request(cb)
    except Exception as e:  # noqa: BLE001
        raised = e
    return raised


def test_retries_http_500():
    assert 500 in RETRY_STATUS_CODES
    cb = _Counter(_http_error(500))
    raised = _run(_make_retrier(retry_cnt=3), cb)
    assert cb.calls == 3  # all attempts used
    assert isinstance(raised, requests.exceptions.HTTPError)


def test_retries_read_timeout():
    cb = _Counter(requests.exceptions.ReadTimeout("read timed out"))
    raised = _run(_make_retrier(retry_cnt=3), cb)
    assert cb.calls == 3
    assert isinstance(raised, requests.exceptions.ReadTimeout)


def test_retries_connection_error():
    cb = _Counter(requests.exceptions.ConnectionError("conn refused"))
    raised = _run(_make_retrier(retry_cnt=3), cb)
    assert cb.calls == 3
    assert isinstance(raised, requests.exceptions.ConnectionError)


def test_does_not_retry_client_error_400():
    assert 400 not in RETRY_STATUS_CODES
    cb = _Counter(_http_error(400))
    raised = _run(_make_retrier(retry_cnt=3), cb)
    assert cb.calls == 1  # raised immediately, no retries
    assert isinstance(raised, requests.exceptions.HTTPError)


def test_swallow_exc_returns_none_after_retries():
    cb = _Counter(_http_error(500))
    result = _make_retrier(retry_cnt=3, swallow_exc=True).request(cb)
    assert cb.calls == 3
    assert result is None

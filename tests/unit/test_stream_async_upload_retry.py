import asyncio
import json
import time

import httpx
import pytest

from supervisely.api.api import Api
from supervisely.io.fs import get_file_hash


class _FakeResponse:
    def __init__(self, request, body=b"", partial_body=None, mid_stream_error=None):
        self.request = request
        self.status_code = 200
        self.headers = {"content-length": str(len(body))}
        self._body = body
        self._partial_body = partial_body
        self._mid_stream_error = mid_stream_error

    async def aiter_raw(self, chunk_size):
        if self._mid_stream_error is not None:
            yield self._partial_body
            raise self._mid_stream_error
        yield self._body


class _FakeStream:
    """Mimics httpx.AsyncClient.stream() context manager: consumes the request
    body on __aenter__ (like a real send) and either fails or returns a response."""

    def __init__(self, client, call):
        self._client = client
        self._call = call

    async def __aenter__(self):
        content = self._call["content"]
        if content is None:
            body = b""
        elif isinstance(content, bytes):
            body = content
        else:
            chunks = [chunk async for chunk in content]
            body = b"".join(chunks)
        self._call["body"] = body
        self._client.calls.append(self._call)
        attempt = len(self._client.calls) - 1
        behavior = self._client.behaviors[min(attempt, len(self._client.behaviors) - 1)]
        request = httpx.Request(self._call["method"], self._call["url"])
        if behavior.get("send_error") is not None:
            raise behavior["send_error"]
        return _FakeResponse(
            request,
            body=behavior.get("body", b""),
            partial_body=behavior.get("partial_body"),
            mid_stream_error=behavior.get("mid_stream_error"),
        )

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _FakeAsyncClient:
    def __init__(self, behaviors):
        self.behaviors = behaviors
        self.calls = []

    def stream(self, method, url, content=None, json=None, headers=None, timeout=None, params=None):
        call = {
            "method": method,
            "url": url,
            "content": content,
            "json": json,
            "headers": dict(headers or {}),
            "params": params,
        }
        return _FakeStream(self, call)


@pytest.fixture
def api():
    api = Api("https://example.com", "fake-token")
    api.retry_sleep_sec = 0
    api._version_check_completed = True
    # keep the fake client in place: cooldown prevents _recreate_client_if_needed
    # from swapping it for a real httpx.AsyncClient
    api._last_client_recreation_time = time.time()
    api._client_recreation_cooldown = 10**9
    return api


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def _make_file(tmp_path, size=1024 * 1024):
    path = tmp_path / "checkpoint.bin"
    data = bytes(range(256)) * (size // 256)
    path.write_bytes(data)
    return str(path), data


def test_upload_retry_resends_full_body_without_range(api, tmp_path):
    src, data = _make_file(tmp_path)
    upload_response = json.dumps({"hash": get_file_hash(src), "size": len(data)}).encode()
    fake = _FakeAsyncClient(
        behaviors=[
            {"send_error": httpx.WriteError("")},  # socket dies after body was consumed
            {"body": upload_response},
        ]
    )
    api.async_httpx_client = fake

    _run(api.file.upload_async(1, src, "/files/checkpoint.bin"))

    assert len(fake.calls) == 2
    assert fake.calls[0]["body"] == data
    # retry must resend the whole file, not the exhausted generator's tail
    assert fake.calls[1]["body"] == data
    # upload retries must not inherit download resume semantics
    assert "Range" not in fake.calls[1]["headers"]


def test_upload_hash_mismatch_raises(api, tmp_path):
    src, data = _make_file(tmp_path)
    upload_response = json.dumps({"hash": "bogus", "size": len(data)}).encode()
    fake = _FakeAsyncClient(behaviors=[{"body": upload_response}])
    api.async_httpx_client = fake

    with pytest.raises(IOError, match="hash does not match"):
        _run(api.file.upload_async(1, src, "/files/checkpoint.bin"))


def test_upload_size_mismatch_raises(api, tmp_path):
    src, data = _make_file(tmp_path)
    upload_response = json.dumps({"size": len(data) - 1}).encode()
    fake = _FakeAsyncClient(behaviors=[{"body": upload_response}])
    api.async_httpx_client = fake

    with pytest.raises(IOError, match="size does not match"):
        _run(api.file.upload_async(1, src, "/files/checkpoint.bin"))


def test_upload_progress_not_double_counted_on_retry(api, tmp_path):
    src, data = _make_file(tmp_path)
    upload_response = json.dumps({"hash": get_file_hash(src)}).encode()
    fake = _FakeAsyncClient(
        behaviors=[
            {"send_error": httpx.WriteError("")},
            {"body": upload_response},
        ]
    )
    api.async_httpx_client = fake
    reported = []

    _run(api.file.upload_async(1, src, "/files/checkpoint.bin", progress_cb=reported.append))

    assert sum(reported) == len(data)


def test_download_retry_still_resumes_with_range(api):
    full = b"x" * 100
    fake = _FakeAsyncClient(
        behaviors=[
            {
                "body": full,
                "partial_body": full[:40],
                "mid_stream_error": httpx.ReadError(""),
            },
            {"body": full[40:]},
        ]
    )
    api.async_httpx_client = fake

    async def consume():
        chunks = []
        async for chunk, _ in api.stream_async("some-method.download", "GET", data={}):
            chunks.append(chunk)
        return chunks

    _run(consume())

    assert len(fake.calls) == 2
    assert fake.calls[1]["headers"].get("Range", "").startswith("bytes=")

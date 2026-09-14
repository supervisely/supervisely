from types import SimpleNamespace

import pytest
from requests import Timeout

from supervisely.nn.inference import session as session_module
from supervisely.nn.inference.session import Session, SessionJSON


def _session():
    session = object.__new__(SessionJSON)
    session._async_inference_uuid = "inference-request"
    session.async_start_timeout = 600
    session.pending_results_timeout = 600
    return session


def _error_response(traceback=None):
    exception = {
        "type": "RuntimeError",
        "message": "Item pexels-photo-744487.png not found in the project.",
    }
    if traceback is not None:
        exception["traceback"] = traceback
    return {
        "stage": "Error",
        "finished": True,
        "is_inferring": False,
        "progress": {"current": 0, "total": 1},
        "result": False,
        "exception": exception,
    }


def test_wait_for_async_inference_start_raises_terminal_error_without_polling_again(
    monkeypatch,
):
    session = _session()
    responses = [_error_response()]
    cleanup_calls = []
    session._get_inference_progress = lambda: responses.pop(0)
    session._on_async_inference_end = lambda: cleanup_calls.append(True)
    monkeypatch.setattr(
        session_module.time,
        "sleep",
        lambda delay: pytest.fail("terminal responses must not be polled again"),
    )

    with pytest.raises(RuntimeError) as exc_info:
        session._wait_for_async_inference_start()

    message = str(exc_info.value)
    assert "RuntimeError" in message
    assert "Item pexels-photo-744487.png not found in the project." in message
    assert responses == []
    assert cleanup_calls == [True]


def test_wait_for_async_inference_start_includes_app_traceback():
    session = _session()
    session._get_inference_progress = lambda: _error_response("app traceback details")
    session._on_async_inference_end = lambda: None

    with pytest.raises(RuntimeError, match="app traceback details"):
        session._wait_for_async_inference_start(delay=0)


def test_wait_for_new_pending_results_raises_terminal_error(monkeypatch):
    session = _session()
    responses = [_error_response()]
    session._pop_pending_results = lambda: responses.pop(0)
    monkeypatch.setattr(
        session_module.time,
        "sleep",
        lambda delay: pytest.fail("terminal responses must not be polled again"),
    )

    with pytest.raises(RuntimeError, match="RuntimeError.*not found in the project"):
        session._wait_for_new_pending_results()

    assert responses == []


def test_wait_for_new_pending_results_keeps_successful_completion():
    session = _session()
    session._pop_pending_results = lambda: {
        "stage": "Finished",
        "finished": True,
        "is_inferring": False,
        "pending_results": [],
        "result": False,
        "exception": None,
    }

    assert session._wait_for_new_pending_results(delay=0) == []


def test_finished_response_without_results_raises():
    session = _session()
    session._get_inference_progress = lambda: {
        "stage": "Finished",
        "finished": True,
        "is_inferring": False,
        "progress": {"current": 0, "total": 1},
        "pending_results": [],
        "result": False,
        "exception": None,
    }
    session._on_async_inference_end = lambda: None

    with pytest.raises(RuntimeError, match="finished without returning results"):
        session._wait_for_async_inference_start(delay=0)


def test_wait_for_async_inference_start_has_finite_default_timeout(monkeypatch):
    session = _session()
    session._get_inference_progress = lambda: {
        "progress": {"current": 0, "total": 1},
        "pending_results": [],
    }
    cleanup_calls = []
    session.stop_async_inference = lambda: cleanup_calls.append("stop")
    session._on_async_inference_end = lambda: cleanup_calls.append("clear")
    times = iter([0, 601])
    monkeypatch.setattr(
        session_module,
        "time",
        SimpleNamespace(time=lambda: next(times), sleep=lambda delay: None),
    )

    with pytest.raises(Timeout, match="didn't start"):
        session._wait_for_async_inference_start(delay=0)

    assert cleanup_calls == ["stop", "clear"]


def test_wait_for_async_inference_start_keeps_successful_response():
    session = _session()
    response = {
        "stage": "Inference",
        "finished": False,
        "is_inferring": True,
        "progress": {"current": 0, "total": 2},
        "pending_results": [],
        "result": False,
        "exception": None,
    }
    session._get_inference_progress = lambda: response

    actual_response, has_started = session._wait_for_async_inference_start(delay=0)

    assert actual_response is response
    assert has_started is True


@pytest.mark.parametrize("session_cls", [SessionJSON, Session])
def test_session_timeout_defaults_are_forwarded(monkeypatch, session_cls):
    monkeypatch.setattr(SessionJSON, "get_session_info", lambda self: {})

    session = session_cls(api=object(), session_url="http://model")

    assert session.async_start_timeout == 600
    assert session.pending_results_timeout == 600


def test_configured_and_explicit_timeout_values(monkeypatch):
    session = _session()
    session.async_start_timeout = 5
    session._get_inference_progress = lambda: {
        "progress": {"current": 0, "total": 1},
        "pending_results": [],
    }
    cleanup_calls = []
    session.stop_async_inference = lambda: cleanup_calls.append("stop")
    session._on_async_inference_end = lambda: cleanup_calls.append("clear")
    times = iter([0, 6])
    monkeypatch.setattr(
        session_module,
        "time",
        SimpleNamespace(time=lambda: next(times), sleep=lambda delay: None),
    )

    with pytest.raises(Timeout):
        session._wait_for_async_inference_start(delay=0)

    assert cleanup_calls == ["stop", "clear"]

    responses = iter(
        [
            {
                "progress": {"current": 0, "total": 1},
                "pending_results": [],
            },
            {
                "progress": {"current": 0, "total": 2},
                "pending_results": [],
            },
        ]
    )
    session._get_inference_progress = lambda: next(responses)
    times = iter([0, 6, 7])
    monkeypatch.setattr(
        session_module,
        "time",
        SimpleNamespace(time=lambda: next(times), sleep=lambda delay: None),
    )

    _, has_started = session._wait_for_async_inference_start(delay=0, timeout=10)

    assert has_started is True


def test_session_forwards_custom_timeout_values(monkeypatch):
    monkeypatch.setattr(SessionJSON, "get_session_info", lambda self: {})

    session = Session(
        api=object(),
        session_url="http://model",
        async_start_timeout=12,
        pending_results_timeout=34,
    )

    assert session.async_start_timeout == 12
    assert session.pending_results_timeout == 34


def test_configured_pending_results_timeout(monkeypatch):
    session = _session()
    session.pending_results_timeout = 5
    session._pop_pending_results = lambda: {
        "pending_results": [],
        "is_inferring": True,
    }
    cleanup_calls = []
    session.stop_async_inference = lambda: cleanup_calls.append("stop")
    session._on_async_inference_end = lambda: cleanup_calls.append("clear")
    times = iter([0, 6])
    monkeypatch.setattr(
        session_module,
        "time",
        SimpleNamespace(time=lambda: next(times), sleep=lambda delay: None),
    )

    with pytest.raises(Timeout, match="Pending results"):
        session._wait_for_new_pending_results(delay=0)

    assert cleanup_calls == ["stop", "clear"]


def test_none_disables_configured_timeouts(monkeypatch):
    monkeypatch.setattr(SessionJSON, "get_session_info", lambda self: {})
    session = Session(
        api=object(),
        session_url="http://model",
        async_start_timeout=None,
        pending_results_timeout=None,
    )
    progress_responses = iter(
        [
            {
                "progress": {"current": 0, "total": 1},
                "pending_results": [],
            },
            {
                "progress": {"current": 0, "total": 2},
                "pending_results": [],
            },
        ]
    )
    pending_responses = iter(
        [
            {"pending_results": [], "is_inferring": True},
            {"pending_results": [{"result": 1}], "is_inferring": True},
        ]
    )
    session._get_inference_progress = lambda: next(progress_responses)
    session._pop_pending_results = lambda: next(pending_responses)
    times = iter([0, 1000, 0, 1000])
    monkeypatch.setattr(
        session_module,
        "time",
        SimpleNamespace(time=lambda: next(times), sleep=lambda delay: None),
    )

    _, has_started = session._wait_for_async_inference_start(delay=0)
    pending_results = session._wait_for_new_pending_results(delay=0)

    assert has_started is True
    assert pending_results == [{"result": 1}]

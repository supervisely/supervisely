import queue
import concurrent.futures
from typing import Optional, List
from enum import Enum


class RequestType(Enum):
    """Types of requests handled by the live training request queue."""

    START = "start"
    PREDICT = "predict"
    ADD_SAMPLE = "add-sample"
    STATUS = "status"
    ADD_SAMPLE_VIDEO = "add-sample-video"
    PREDICT_BATCH = "predict-batch"
    ADD_SAMPLES_VIDEO = "add-samples-video"
    KEY_FRAMES = "key-frames"


class Request:
    """A simple representation of an API request."""

    def __init__(
        self,
        request_type: RequestType,
        data: Optional[dict] = None,
        future: Optional[concurrent.futures.Future] = None,
    ):
        self.type = request_type
        self.data = data
        self.future = future

    def to_tuple(self):
        return (self.type, self.data, self.future)


class RequestQueue:
    """Thread-safe queue for API requests.

    Futures are ``concurrent.futures.Future`` so they can be awaited from the
    FastAPI event loop (via ``asyncio.wrap_future``) and also blocked on by
    plain background threads (via ``future.result(timeout=...)``).
    """

    def __init__(self):
        self._queue = queue.Queue()

    def put(
        self,
        request_type: RequestType,
        data: Optional[dict] = None,
    ) -> concurrent.futures.Future:
        """Add request and return a future for its result."""
        future = concurrent.futures.Future()
        self._queue.put(Request(request_type, data, future))
        return future

    def get_all(self) -> List[Request]:
        """Get all pending requests (non-blocking)."""
        requests = []
        while not self._queue.empty():
            try:
                requests.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return requests

    def is_empty(self) -> bool:
        return self._queue.empty()

    def get(self, timeout: float = None) -> Request:
        """Get a single request from the queue."""
        return self._queue.get(timeout=timeout)

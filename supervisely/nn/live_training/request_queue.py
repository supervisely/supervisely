import queue
import asyncio
from typing import Any, Optional, List
from enum import Enum


class RequestType(Enum):
    START = "start"
    PREDICT = "predict"
    ADD_SAMPLE = "add-sample"
    STATUS = "status"


class Request:
    """A simple representation of an API request."""
    def __init__(self, request_type: RequestType, data: Optional[dict] = None, future: Optional[asyncio.Future] = None):
        self.type = request_type
        self.data = data
        self.future = future
    
    def to_tuple(self):
        return (self.type, self.data, self.future)
    

class RequestQueue:
    """Thread-safe queue for API requests."""
    
    def __init__(self):
        self._queue = queue.Queue()
    
    def put(self, request_type: RequestType, data: Optional[dict] = None) -> asyncio.Future:
        """Add request and return future for result."""
        future = asyncio.Future()
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
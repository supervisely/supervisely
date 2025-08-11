import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, List, Union

from supervisely.app.singleton import Singleton
from supervisely.sly_logger import logger


# to subscribe to an event, use:
def on_event(topic: Union[str, List[str]]):
    if isinstance(topic, str):
        topic = [topic]

    def decorator(method):
        if hasattr(method, "_event_topic"):
            raise ValueError(
                f"Method {method.__name__} already has an event topic defined: {method._event_topic}. "
            )
        method._event_topic = topic.copy()
        return method

    return decorator


# to publish an event, use:
def publish_event(topic: Union[str, List[str]]):
    if isinstance(topic, str):
        topic = [topic]

    def decorator(method):
        def wrapper(*args, **kwargs):
            message = method(*args, **kwargs)
            broker = PubSubAsync()
            for t in topic:
                broker.publish(t, message)
            return message

        return wrapper

    return decorator


class PubSub(metaclass=Singleton):
    def __init__(self):
        self.subscribers = {}  # Dictionary to store subscribers for each topic

    def _safe_callback_wrapper(self, callback: Callable, message: str, topic: str):
        """Wrapper for safe callback execution"""
        try:
            callback(message)
        except Exception as e:
            logger.error(f"Error calling subscriber for topic '{topic}': {e}", exc_info=True)

    def _callback_wrapper(self, callback: Callable, message: str, topic: str):
        """Wrapper for safe callback execution"""
        return self._safe_callback_wrapper(callback, message, topic)

    def subscribe(self, topic, callback: Callable):
        """Subscribes a callback function to a given topic."""
        if topic not in self.subscribers:
            self.subscribers[topic] = []
        self.subscribers[topic].append(callback)
        logger.info(f"[EVENT]: {callback.__qualname__}() subscribed to topic: '{topic}'")

    def unsubscribe(self, topic, callback: Callable):
        """Unsubscribes a callback function from a given topic."""
        if topic in self.subscribers and callback in self.subscribers[topic]:
            self.subscribers[topic].remove(callback)
            logger.info(f"[EVENT]: {callback.__qualname__}() unsubscribed from topic: '{topic}'")

    def publish(self, topic, message):
        """Publishes a message to a given topic, notifying all subscribers."""
        if topic in self.subscribers:
            for callback in self.subscribers[topic]:
                logger.info(
                    f"[EVENT]: {callback.__qualname__}() called for topic: '{topic}'"
                )
                self._callback_wrapper(callback, message, topic)
        else:
            logger.info(f"No subscribers for topic: '{topic}'")


class PubSubAsync(PubSub):
    """Asynchronous version of PubSub using ThreadPoolExecutor for callback execution."""

    _initialized = False
    _init_lock = threading.Lock()

    def __init__(self):
        with PubSubAsync._init_lock:
            if PubSubAsync._initialized:
                return
            super().__init__()
            self._executor = ThreadPoolExecutor(max_workers=10, thread_name_prefix="sly-pub-sub")
            PubSubAsync._initialized = True

    def _callback_wrapper(self, callback: Callable, message: str, topic: str):
        """Wrapper for safe callback execution in async context"""
        try:
            logger.info(f"[ASYNC EVENT]: {callback.__qualname__}() called for topic: '{topic}'")
            future = self._executor.submit(self._safe_callback_wrapper, callback, message, topic)
            future.add_done_callback(self._log_future_exceptions)
        except Exception as e:
            logger.error(f"Error in callback for event '{topic}': {repr(e)}", exc_info=True)

    def _log_future_exceptions(self, future):
        """Log exceptions from futures"""
        try:
            future.result()  # This will raise if the future encountered an exception
        except Exception as e:
            logger.error(f"Future raised an exception: {repr(e)}", exc_info=True)

    def shutdown(self):
        """Gracefully shutdown the executor"""
        if hasattr(self, "_executor"):
            self._executor.shutdown(wait=True)
            logger.info("PubSubAsync executor shutdown complete.")
        else:
            logger.warning("PubSubAsync executor was not initialized.")


# ! TODO: check for race conditions
# ! TODO: check for circular events
# ! TODO: check for API rate limits

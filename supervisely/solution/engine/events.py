import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

from supervisely.app.singleton import Singleton
from supervisely.sly_logger import logger


# to subscribe to an event, use:
def on_event(topic: str):
    def decorator(method):
        method._event_topic = topic
        return method

    return decorator


# to publish an event, use:
def publish_event(topic: str):
    def decorator(method):
        def wrapper(*args, **kwargs):
            broker = PubSubAsync()
            message = method(*args, **kwargs)
            broker.publish(topic, message)
            return message

        return wrapper

    return decorator


class PubSub(metaclass=Singleton):
    def __init__(self):
        self.subscribers = {}  # Dictionary to store subscribers for each topic
        self.lock = threading.Lock()  # Lock for thread-safe access to subscribers

    def _safe_callback_wrapper(self, callback: Callable, message: str, topic: str):
        """Wrapper for safe callback execution"""
        try:
            callback(message)
        except Exception as e:
            logger.error(f"Error calling subscriber for topic '{topic}': {e}", exc_info=True)

    def _callback_wrapper(self, callback: Callable, message: str, topic: str):
        """Wrapper for safe callback execution"""
        return self._safe_callback_wrapper(callback, message, topic)

    def subscribe(self, topic, callback):
        """Subscribes a callback function to a given topic."""
        with self.lock:
            if topic not in self.subscribers:
                self.subscribers[topic] = []
            self.subscribers[topic].append(callback)
            logger.info(f"Subscribed to topic: '{topic}': {callback.__name__}")

    def unsubscribe(self, topic, callback):
        """Unsubscribes a callback function from a given topic."""
        with self.lock:
            if topic in self.subscribers and callback in self.subscribers[topic]:
                self.subscribers[topic].remove(callback)
                logger.info(f"Unsubscribed from topic: '{topic}': {callback.__name__}")

    def publish(self, topic, message):
        """Publishes a message to a given topic, notifying all subscribers."""
        with self.lock:
            if topic in self.subscribers:
                for callback in self.subscribers[topic]:
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

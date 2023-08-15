from requests.exceptions import HTTPError, RetryError
from typing import List, Union, Callable
from rich.console import Console

from supervisely.sly_logger import logger, EventType
from supervisely.app import DialogWindowError, show_dialog

import traceback
import re

# TODO: Add correct doc link.
# DOC_URL = "https://docs.supervisely.com/errors/"


class HandleException:
    def __init__(
        self,
        exception: Exception,
        stack: List[traceback.FrameSummary] = None,
        **kwargs,
    ):
        self.exception = exception
        self.stack = stack or read_stack_from_exception(self.exception)
        self.code = kwargs.get("code")
        self.title = kwargs.get("title")

        # TODO: Add error code to the title.
        # self.title += f" (SLYE{self.code})"

        self.message = kwargs.get("message")

        # TODO: Add doc link to the message.
        # error_url = DOC_URL + str(self.code)
        # self.message += (
        #     f"<br>Check out the <a href='{error_url}'>documentation</a> "
        #     "for solutions and troubleshooting."
        # )

        self.dev_logging()

    def dev_logging(self):
        # * Option for logging the exception with sly logger.
        # * Due to JSON formatting it's not convinient to read the log.
        # tracelog = " | ".join(traceback.format_list(self.stack))
        # logger.error(
        #     "Detailed info about exception."
        #     f"Exception type: {type(self.exception)}. "
        #     f"Exception value: {self.exception}. "
        #     f"Error code: {self.code}. "
        #     f"Error title: {self.title}. "
        #     f"Error message: {self.message} "
        #     f"Traceback: {tracelog}. "
        # )

        # * Printing the exception to the console line by line.
        console = Console()

        console.print("❗️ Beginning of the error report.", style="bold red")
        console.print(f"{self.exception.__class__.__name__}: {self.exception}")
        # TODO: Uncomment code line when error codes will be added.
        # console.print(f"Error code: {self.code}.", style="bold orange")
        console.print(f"Error title: {self.title}")
        console.print(f"Error message: {self.message}")

        console.print("Traceback (most recent call last):", style="bold red")

        for i, trace in enumerate(traceback.format_list(self.stack)):
            console.print(f"{i + 1}. {trace}")
        console.print("❗️ End of the error report.", style="bold red")

    def raise_error(self):
        raise DialogWindowError(self.title, self.message)

    def show_app_dialog(self):
        show_dialog(
            self.title,
            self.message,
            status="error",
        )

    def log_error_for_agent(self, main_name: str):
        logger.critical(
            self.title,
            exc_info=True,
            extra={
                "main_name": main_name,
                "event_type": EventType.TASK_CRASHED,
                "exc_str": self.message,
            },
        )


class ErrorHandler:
    class SDK:
        pass

    class API:
        class TeamFilesFileNotFound(HandleException):
            def __init__(
                self,
                exception: Exception,
                stack: List[traceback.FrameSummary] = None,
            ):
                self.code = 2001
                self.title = "File on Team Files not found"
                self.message = "The given path to the file on Team Files is incorrect."

                super().__init__(
                    exception,
                    stack,
                    code=self.code,
                    title=self.title,
                    message=self.message,
                )

        class TaskSendRequestError(HandleException):
            def __init__(self, exception: Exception, stack: List[traceback.FrameSummary]):
                self.code = 2010
                self.title = "Task send request error"
                self.message = (
                    "The application has encountered an error while sending a request to the task. "
                    "Please, check that the task is running."
                )

                super().__init__(
                    exception,
                    stack,
                    code=self.code,
                    title=self.title,
                    message=self.message,
                )

        class ConversionNotImplementedError(HandleException):
            def __init__(self, exception: Exception, stack: List[traceback.FrameSummary]):
                self.code = 2009
                self.title = "Not implemented error"
                self.message = (
                    "Conversion is not implemented for this annotations. "
                    "Please, check the geometry of the objects."
                )

                super().__init__(
                    exception,
                    stack,
                    code=self.code,
                    title=self.title,
                    message=self.message,
                )

        class FilesSizeTooLarge(HandleException):
            def __init__(self, exception: Exception, stack: List[traceback.FrameSummary]):
                self.code = 2002
                self.title = "File too large"
                self.message = "The given file size is too large for free community edition."

                super().__init__(
                    exception,
                    stack,
                    code=self.code,
                    title=self.title,
                    message=self.message,
                )

        class ImageFilesSizeTooLarge(HandleException):
            def __init__(self, exception: Exception, stack: List[traceback.FrameSummary]):
                self.code = 2003
                self.title = "Image file too large"
                self.message = "The given image file size is too large for free community edition."

                super().__init__(
                    exception,
                    stack,
                    code=self.code,
                    title=self.title,
                    message=self.message,
                )

        class VideoFilesSizeTooLarge(HandleException):
            def __init__(self, exception: Exception, stack: List[traceback.FrameSummary]):
                self.code = 2005
                self.title = "Video file too large"
                self.message = "The given video file size is too large for free community edition."

                super().__init__(
                    exception,
                    stack,
                    code=self.code,
                    title=self.title,
                    message=self.message,
                )

        class VolumeFilesSizeTooLarge(HandleException):
            def __init__(self, exception: Exception, stack: List[traceback.FrameSummary]):
                self.code = 2006
                self.title = "Volume file too large"
                self.message = "The given volume file size is too large for free community edition."

                super().__init__(
                    exception,
                    stack,
                    code=self.code,
                    title=self.title,
                    message=self.message,
                )

        class OutOfMemory(HandleException):
            def __init__(self, exception: Exception, stack: List[traceback.FrameSummary]):
                self.code = 2007
                self.title = "Out of memory"
                self.message = (
                    "The agent ran out of memory. "
                    "Please, check your agent's memory usage, reduce batch size or use a device with more memory capacity."
                )

                super().__init__(
                    exception,
                    stack,
                    code=self.code,
                    title=self.title,
                    message=self.message,
                )

        class DockerRuntimeError(HandleException):
            def __init__(self, exception: Exception, stack: List[traceback.FrameSummary]):
                self.code = 2008
                self.title = "Docker runtime error"
                self.message = (
                    "The agent has encountered a Docker runtime error. "
                    "Please, check that docker is installed and running."
                )

                super().__init__(
                    exception,
                    stack,
                    code=self.code,
                    title=self.title,
                    message=self.message,
                )


ERROR_PATTERNS = {
    AttributeError: {r".*api\.file\.get_info_by_path.*": ErrorHandler.API.TeamFilesFileNotFound},
    HTTPError: {
        r".*api\.task\.send_request.*": ErrorHandler.API.TaskSendRequestError,
        r".*file-storage\.bulk\.upload.*FileSize.*sizeLimit.*": ErrorHandler.API.FilesSizeTooLarge,
        r".*images\.bulk\.upload.*FileSize.*\"sizeLimit\":1073741824.*": ErrorHandler.API.ImageFilesSizeTooLarge,
        r".*videos\.bulk\.upload.*FileSize.*sizeLimit.*": ErrorHandler.API.VideoFilesSizeTooLarge,
        r".*images\.bulk\.upload.*FileSize.*\"sizeLimit\":157286400.*": ErrorHandler.API.VolumeFilesSizeTooLarge,
    },
    NotImplementedError: {
        r".*from 'graph' to 'polygon'.*": ErrorHandler.API.ConversionNotImplementedError
    },
    # RuntimeError: {r".*CUDA out of memory.*Tried to allocate.*": ErrorHandler.API.OutOfMemory},
    # Exception: {r".*unable to start container process.*": ErrorHandler.API.DockerRuntimeError},
}


def handle_exception(exception: Exception) -> Union[ErrorHandler, None]:
    """Function for handling exceptions, using the stack trace and patterns for known errors.
    Returns an instance of the ErrorHandler class if the pattern is found, otherwise returns None.

    :param exception: Exception to be handled.
    :type exception: Exception
    :return: Instance of the ErrorHandler class or None.
    :rtype: Union[ErrorHandler, None]
    :Usage example:

     .. code-block:: python

        import supervisely as sly

        try:
            # Some code that may raise an exception.
        except Exception as e:
            exception_handler = sly.handle_exception(e)
            if exception_handler:
                # You may raise the exception using the raise_error() method.
                # Or use other approaches for handling the exception (e.g. logging, printing to console, etc.)
                exception_handler.raise_error()
            else:
                # If the pattern is not found, the exception is raised as usual.
                raise
    """
    # Extracting the stack trace.
    stack = read_stack_from_exception(exception)

    # Retrieving the patterns for the given exception type.
    patterns = ERROR_PATTERNS.get(type(exception))
    if not patterns:
        return

    # Looping through the stack trace from the bottom up to find matching pattern with specified Exception type.
    for pattern, handler in patterns.items():
        for frame in stack[::-1]:
            if re.match(pattern, frame.line):
                return handler(exception, stack)
        if re.match(pattern, exception.args[0]):
            return handler(exception, stack)


def handle_exceptions(func: Callable) -> Callable:
    """Decorator for handling exceptions, which tries to find a matching pattern for known errors.
    If the pattern is found, the exception is handled according to the specified handler.
    Otherwise, the exception is raised as usual.

    :param func: Function to be decorated.
    :type func: Callable
    :return: Decorated function.
    :rtype: Callable
    :Usage example:

     .. code-block:: python

        import supervisely as sly

        @sly.handle_exceptions
        def my_func():
            # Some code that may raise an exception.
    """

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            exception_handler = handle_exception(e)
            if exception_handler:
                exception_handler.raise_error()
            else:
                raise

    return wrapper


def read_stack_from_exception(exception):
    stack = traceback.extract_stack()
    stack.extend(traceback.extract_tb(exception.__traceback__))
    return stack

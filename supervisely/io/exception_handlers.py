from requests.exceptions import HTTPError, RetryError
from typing import List, Union, Callable
from rich.console import Console

# TODO: Remove commented line if logger will not be used.
# from supervisely.sly_logger import logger
from supervisely.app import DialogWindowError

import traceback
import re

# TODO: Add correct doc link.
# DOC_URL = "https://docs.supervisely.com/errors/"


class HandleException:
    def __init__(self, exception: Exception, stack: List[traceback.FrameSummary] = None, **kwargs):
        self.exception = exception
        self.stack = stack or self.read_stack()
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

    def read_stack(self):
        stack = traceback.extract_stack()
        stack.append(traceback.extract_tb(self.exception.__traceback__)[-1])
        return stack

    def raise_error(self):
        raise DialogWindowError(self.title, self.message)


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

        class FilesSizeTooLarge(HandleException):
            def __init__(
                self, exception: Exception, stack: List[traceback.FrameSummary], headless: bool
            ):
                self.code = 2002
                self.title = "File too large."
                self.description = (
                    "The given file size is too large for free community edition. "
                    "To use bigger files - get enterprise edition."
                )

                super().__init__(
                    exception,
                    stack,
                    headless,
                    code=self.code,
                    title=self.title,
                    description=self.description,
                )

        class ImageFilesSizeTooLarge(HandleException):
            def __init__(
                self, exception: Exception, stack: List[traceback.FrameSummary], headless: bool
            ):
                self.code = 2003
                self.title = "Image file too large."
                self.description = (
                    "The given image file size is too large for free community edition. "
                    "To use bigger files - get enterprise edition."
                )

                super().__init__(
                    exception,
                    stack,
                    headless,
                    code=self.code,
                    title=self.title,
                    description=self.description,
                )

        class RetryLimitExceeded(HandleException):
            def __init__(
                self, exception: Exception, stack: List[traceback.FrameSummary], headless: bool
            ):
                self.code = 2004
                self.title = "Retry limit exceeded."
                self.description = (
                    "The number of retries for the request has been exceeded. "
                    "Please, check your internet connection, agent status, try again later or contact support."
                )

                super().__init__(
                    exception,
                    stack,
                    headless,
                    code=self.code,
                    title=self.title,
                    description=self.description,
                )

        class VideoFilesSizeTooLarge(HandleException):
            def __init__(
                self, exception: Exception, stack: List[traceback.FrameSummary], headless: bool
            ):
                self.code = 2005
                self.title = "Video file too large."
                self.description = (
                    "The given video file size is too large for free community edition. "
                    "To use bigger files - get enterprise edition."
                )

                super().__init__(
                    exception,
                    stack,
                    headless,
                    code=self.code,
                    title=self.title,
                    description=self.description,
                )

        class VolumeFilesSizeTooLarge(HandleException):
            def __init__(
                self, exception: Exception, stack: List[traceback.FrameSummary], headless: bool
            ):
                self.code = 2006
                self.title = "Volume file too large."
                self.description = (
                    "The given volume file size is too large for free community edition. "
                    "To use bigger files - get enterprise edition."
                )

                super().__init__(
                    exception,
                    stack,
                    headless,
                    code=self.code,
                    title=self.title,
                    description=self.description,
                )

        class ConversionNotImplementedError(HandleException):
            def __init__(
                self, exception: Exception, stack: List[traceback.FrameSummary], headless: bool
            ):
                self.code = 2009
                self.title = "Not implemented error."
                self.description = (
                    "Conversion is not implemented for this annotations. "
                    "Please, check the geometry of the objects."
                )

                super().__init__(
                    exception,
                    stack,
                    headless,
                    code=self.code,
                    title=self.title,
                    description=self.description,
                )

        class OutOfMemory(HandleException):
            def __init__(
                self, exception: Exception, stack: List[traceback.FrameSummary], headless: bool
            ):
                self.code = 2007
                self.title = "Out of memory."
                self.description = (
                    "The agent ran out of memory. "
                    "Please, check your agent's memory usage, reduce batch size or use a device with more memory capacity."
                )

                super().__init__(
                    exception,
                    stack,
                    headless,
                    code=self.code,
                    title=self.title,
                    description=self.description,
                )

        class DockerRuntimeError(HandleException):
            def __init__(
                self, exception: Exception, stack: List[traceback.FrameSummary], headless: bool
            ):
                self.code = 2008
                self.title = "Docker runtime error."
                self.description = (
                    "The agent has encountered a Docker runtime error. "
                    "Please, check that docker is installed and running."
                )

                super().__init__(
                    exception,
                    stack,
                    headless,
                    code=self.code,
                    title=self.title,
                    description=self.description,
                )


ERROR_PATTERNS = {
    AttributeError: {r".*api\.file\.get_info_by_path.*": ErrorHandler.API.TeamFilesFileNotFound},
    HTTPError: {
        r".*file-storage\.bulk\.upload.*FileSize.*sizeLimit.*": ErrorHandler.API.FilesSizeTooLarge,
        r".*images\.bulk\.upload.*FileSize.*\"sizeLimit\":1073741824.*": ErrorHandler.API.ImageFilesSizeTooLarge,
        r".*videos\.bulk\.upload.*FileSize.*sizeLimit.*": ErrorHandler.API.VideoFilesSizeTooLarge,
        r".*images\.bulk\.upload.*FileSize.*\"sizeLimit\":157286400.*": ErrorHandler.API.VolumeFilesSizeTooLarge,
    },
    NotImplementedError: {
        r".*from 'graph' to 'polygon'.*": ErrorHandler.API.ConversionNotImplementedError
    },
    RetryError: {r".*Retry limit exceeded.*": ErrorHandler.API.RetryLimitExceeded},
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
    stack = traceback.extract_stack()
    # Adding the exception's last frame to the stack trace.
    stack.append(traceback.extract_tb(exception.__traceback__)[-1])

    # Retrieving the patterns for the given exception type.
    patterns = ERROR_PATTERNS.get(type(exception))
    if not patterns:
        return

    exception_handler = None

    # Looping through the stack trace from the bottom up to find matching pattern with specified Exception type.
    for frame in stack[::-1]:
        for pattern, handler in patterns.items():
            if re.match(pattern, frame.line):
                exception_handler = handler
                break

    if not exception_handler:
        return

    return exception_handler(exception, stack)


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

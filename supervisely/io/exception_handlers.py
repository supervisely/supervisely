from typing import List, Union
from rich.console import Console

# from supervisely.sly_logger import logger

import traceback
import re

# TODO: Add correct doc link.
# DOC_URL = "https://docs.supervisely.com/errors/"


class HandleException:
    def __init__(self, exception: Exception, stack: List[traceback.FrameSummary], **kwargs):
        self.exception = exception
        self.stack = stack
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
        #     f"Error message: {self.message}. "
        #     f"Traceback: {tracelog}. "
        # )

        # * Printing the exception to the console line by line.
        console = Console()

        console.print("❗️ Beginning of the error report.", style="bold red")
        console.print(f"{self.exception.__class__.__name__}: {self.exception}")
        # TODO: Uncomment code line when error codes will be added.
        # console.print(f"Error code: {self.code}.", style="bold orange")
        console.print(f"Error title: {self.title}.")
        console.print(f"Error message: {self.message}.")

        console.print("Traceback (most recent call last):", style="bold red")

        for i, trace in enumerate(traceback.format_list(self.stack)):
            console.print(f"{i + 1}. {trace}")
        console.print("❗️ End of the error report.", style="bold red")


class ErrorHandler:
    class SDK:
        pass

    class API:
        class TeamFilesFileNotFound(HandleException):
            def __init__(self, exception: Exception, stack: List[traceback.FrameSummary]):
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


ERROR_PATTERNS = {
    AttributeError: {r".*api\.file\.get_info_by_path.*": ErrorHandler.API.TeamFilesFileNotFound}
}


def handle_exception(exception: Exception) -> Union[ErrorHandler, None]:
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

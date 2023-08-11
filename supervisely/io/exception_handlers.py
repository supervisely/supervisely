from typing import List

from supervisely.sly_logger import logger
from supervisely.app import show_dialog

import traceback
import re

BASE_URL = "https://errors.supervisely.com/{code}"


class HandleException:
    def __init__(
        self, exception: Exception, stack: List[traceback.FrameSummary], headless: bool, **kwargs
    ):
        self.exception = exception
        self.stack = stack
        self.headless = headless
        self.code = kwargs.get("code")
        self.title = kwargs.get("title")
        self.description = kwargs.get("description")

        self.dev_logging()

        if not self.headless:
            self.gui_exception()
        else:
            self.headless_exception()

    def dev_logging(self):
        tracelog = " | ".join(traceback.format_list(self.stack))
        logger.error(
            "Detailed info about exception."
            f"Exception type: {type(self.exception)}. "
            f"Exception value: {self.exception}. "
            f"Error code: {self.code}. "
            f"Error title: {self.title}. "
            f"Error description: {self.description}. "
            f"Traceback: {tracelog}. "
        )

    def gui_exception(self):
        url = BASE_URL.format(code=self.code)
        description = (
            f"{self.description}<br><br>"
            f"Please, visit <a href='{url}'>SLY ERROR {self.code}</a> for more information and possible solutions."
        )

        show_dialog(
            title=f"{self.title} (SLY ERROR {self.code})",
            description=description,
            status="error",
        )

    def headless_exception(self):
        pass


class ErrorHandler:
    class SDK:
        pass

    class API:
        class TeamFilesFileNotFound(HandleException):
            def __init__(
                self, exception: Exception, stack: List[traceback.FrameSummary], headless: bool
            ):
                self.code = 2001
                self.title = "File on Team Files not found"
                self.description = "The given path to the file on Team Files is incorrect."

                super().__init__(
                    exception,
                    stack,
                    headless,
                    code=self.code,
                    title=self.title,
                    description=self.description,
                )


ERROR_PATTERNS = {
    AttributeError: {r".*api\.file\.get_info_by_path.*": ErrorHandler.API.TeamFilesFileNotFound}
}


def handle_exception(exception: Exception, headless: bool = False):
    # Extracting the stack trace.
    stack = traceback.extract_stack()
    # Adding the exception's last frame to the stack trace.
    stack.append(traceback.extract_tb(exception.__traceback__)[-1])

    error_type = type(exception)
    patterns = ERROR_PATTERNS.get(error_type)
    if not patterns:
        return

    handlers = []

    # Looping through the stack trace from the bottom up.
    for frame in stack[::-1]:
        for pattern, handler in patterns.items():
            if re.match(pattern, frame.line):
                handlers.append(handler)

    if not handlers:
        return

    for handler in handlers:
        handler(exception, stack, headless)

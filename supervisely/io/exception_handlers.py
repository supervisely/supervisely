import traceback
import re


class HandleException:
    def __init__(self, exception: Exception, headless: bool, **kwargs):
        self.exception = exception
        self.code = kwargs.get("code")
        self.title = kwargs.get("title")
        self.description = kwargs.get("description")


class ErrorHandler:
    class SDK:
        pass

    class API:
        class TeamFilesFileNotFound(HandleException):
            def __init__(self, exception: Exception, headless: bool):
                self.exception = exception
                self.headless = headless
                self.code = 2001
                self.title = "File on Team Files not found"
                self.description = "The given path to the file on Team Files is incorrect."

                super().__init__(
                    exception,
                    headless,
                    code=self.code,
                    title=self.title,
                    description=self.description,
                )


ERROR_PATTERNS = {
    AttributeError: {r".*api\.file\.download.*": ErrorHandler.API.TeamFilesFileNotFound}
}


def handle_exception(exception: Exception, headless: bool = False):
    error_type = type(exception)
    error_value = str(exception)
    error_tb = exception.__traceback__
    traces = traceback.extract_tb(error_tb)[::-1]

    print("----------DEBUG SECTION PRINTING-----------")
    print("error_type: ", error_type)
    print("error_value: ", error_value)
    print("--------------TRACEBACK--------------------")
    print("traceback: ", error_tb)
    print("--------------TRACES------------------------")
    for trace in traces:
        print(trace)
    print("-------END OF DEBUG SECTION PRINTING-------")

    patterns = ERROR_PATTERNS.get(error_type)

    if not patterns:
        return

    handlers = []

    for trace in traces:
        for pattern, handler in patterns.items():
            if re.match(pattern, trace.line):
                handlers.append(handler)

    if not handlers:
        return

    for handler in handlers:
        handler(exception, headless)

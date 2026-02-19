try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


class DialogWindowBase(Exception):
    """Base exception that can be displayed to the user as an in-app dialog window."""

    def __init__(self, title, description, status):
        """Initialize DialogWindowBase.

        :param title: Dialog title.
        :param description: Dialog description/message.
        :param status: Status: "info", "success", "warning", or "error".
        """
        self.title = title
        self.description = description
        self.status = status
        super().__init__(self.get_message())

    def get_message(self):
        return f"{self.title}: {self.description}"

    def __str__(self):
        return self.get_message()


class DialogWindowError(DialogWindowBase):
    """Dialog window exception with `error` status."""

    def __init__(self, title, description):
        """Initialize DialogWindowError.

        :param title: Dialog title.
        :param description: Error message.
        """
        super().__init__(title, description, "error")


class DialogWindowWarning(DialogWindowBase):
    """Dialog window exception with `warning` status."""

    def __init__(self, title, description):
        """Initialize DialogWindowWarning.

        :param title: Dialog title.
        :param description: Warning message.
        """
        super().__init__(title, description, "warning")


# for compatibility
class DialogWindowMessage(DialogWindowError):
    """Backwards-compatible alias for informational dialog exceptions."""

    def __init__(self, title, description):
        """Initialize DialogWindowMessage.

        :param title: Dialog title.
        :param description: Info message.
        """
        super().__init__(title, description, "info")  # pylint: disable=too-many-function-args


def show_dialog(
    title, description, status: Literal["info", "success", "warning", "error"] = "info"
):
    from supervisely.app import StateJson, DataJson

    StateJson()["slyAppShowDialog"] = True
    DataJson()["slyAppDialogTitle"] = title
    DataJson()["slyAppDialogMessage"] = description
    DataJson()["slyAppDialogStatus"] = status
    DataJson().send_changes()
    StateJson().send_changes()

from typing import Callable, Dict, List, Literal, Union

from supervisely import logger
from supervisely._utils import abs_url, is_development
from supervisely.solution.base_node import BaseCardNode
from supervisely.solution.engine.models import ComparisonFinishedMessage
from supervisely.solution.nodes.email_notification.gui import SendEmail
from supervisely.solution.nodes.email_notification.history import SendEmailHistory


class EmailNotificationNode(BaseCardNode):
    TITLE = "Send Email"
    DESCRIPTION = "Automatically send email notifications after each model comparison. You can configure the recipients and subject of the email."
    ICON = "mdi mdi-email-fast"
    ICON_COLOR = "#1976D2"
    ICON_BG_COLOR = "#E3F2FD"

    def __init__(
        self,
        credentials: Union[SendEmail.Credentials, Dict, None] = None,
        tooltip_position: Literal["left", "right"] = "right",
        *args,
        **kwargs,
    ):

        # --- components ---------------------------------------------------
        if isinstance(credentials, dict):
            credentials = SendEmail.Credentials(**credentials)
        self.credentials = credentials or SendEmail.Credentials.from_env()
        self.history = SendEmailHistory()
        self.gui = SendEmail(
            default_subject="Supervisely Solution Notification",
            default_body="""
<html>
<head>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
        }}
        .highlight {{
            color: #1976D2;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <p>Hello,</p>
    <p>This is a notification from the Supervisely Solution.</p><br><br>
    {message}<br><br>
    <p>Supervisely Solution</p>
</body>
</html>
""",
            show_body=False,
        )
        self.modals = [
            self.history.modal,
            self.gui.modal,
            self.history.logs_modal,
        ]

        if is_development():
            self._debug_add_dummy_notification()  # For debugging purposes, delete in production

        # --- init node ------------------------------------------------------
        title = kwargs.pop("title", self.TITLE)
        description = kwargs.pop("description", self.DESCRIPTION)
        icon = kwargs.pop("icon", self.ICON)
        icon_color = kwargs.pop("icon_color", self.ICON_COLOR)
        icon_bg_color = kwargs.pop("icon_bg_color", self.ICON_BG_COLOR)
        self.modal_content = self.gui.content
        super().__init__(
            title=title,
            description=description,
            icon=icon,
            icon_color=icon_color,
            icon_bg_color=icon_bg_color,
            *args,
            **kwargs,
        )

        @self.gui.apply_button.click
        def apply_settings_cb():
            self.gui.modal.hide()
            self._update_properties()

        @self.click
        def on_card_click():
            self.gui.open_modal()

    # ------------------------------------------------------------------
    # Handles ----------------------------------------------------------
    # ------------------------------------------------------------------
    def _get_handles(self):
        return [
            {
                "id": "comparison_finished",
                "type": "target",
                "position": "left",
                "connectable": True,
            },
        ]

    # ------------------------------------------------------------------
    # Events -----------------------------------------------------------
    # ------------------------------------------------------------------
    def _available_subscribe_methods(self) -> Dict[str, Callable]:
        return {
            "comparison_finished": self._process_incomming_message,
        }

    def _process_incomming_message(self, message: ComparisonFinishedMessage):
        url = abs_url(message.report_link) if message.report_link else "#"
        message = f"Comparison report is ready: <a href='{url}' target='_blank'>View Report</a>"
        self.run(text=message)

    # ------------------------------------------------------------------
    # Methods ----------------------------------------------------------
    # ------------------------------------------------------------------
    def _debug_add_dummy_notification(self):
        """
        Adds a dummy notification to the history for debugging purposes.
        """
        dummy_notification = SendEmailHistory.Item(
            ["someuser123@example.com", "dummy228@yahoo.com"], SendEmailHistory.Item.Status.SENT
        )
        self.history.add_task(dummy_notification.to_json())

    def run(self, text: str) -> None:
        """
        Runs the SendEmailNode, sending an email with the configured settings.
        """
        notification = SendEmailHistory.Item(self.gui.get_target_addresses())
        n_idx = self.history.add_task(notification)
        self._update_properties()
        try:
            message = self.gui.get_body()
            if text:
                message = message.format(message=text)
            self.gui.send_email(self.credentials, message=message)
            self.history.update_task(n_idx, {"status": SendEmailHistory.Item.Status.SENT})
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            self.history.update_task(n_idx, {"status": SendEmailHistory.Item.Status.FAILED})

    def _get_email_widget_values(self) -> Dict[str, Union[str, List[str]]]:
        return {
            "subject": self.gui.get_subject(),
            "body": self.gui.get_body(),
            "target_addresses": self.gui.get_target_addresses(),
        }

    def _update_properties(self):
        if self.gui.is_email_sending_enabled:
            self.show_automation_badge()
        else:
            self.hide_automation_badge()

        value = "enabled" if self.gui.is_email_sending_enabled else "disabled"
        self.update_property(key="Send Notification", value=value, highlight=True)
        self.update_property(key="Total", value=f"{(len(self.history.get_tasks()))} notifications")

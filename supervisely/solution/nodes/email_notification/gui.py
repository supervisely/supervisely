import mimetypes
import os
import smtplib
import ssl
from typing import Optional

from supervisely.app.widgets import (
    Button,
    CheckboxField,
    Container,
    Dialog,
    Field,
    Input,
    TextArea,
    Widget,
)
from supervisely.sly_logger import logger
from supervisely.solution.engine.modal_registry import ModalRegistry

SMTP_PROVIDERS = {
    "gmail.com": ("smtp.gmail.com", 587),
    "outlook.com": ("smtp.office365.com", 587),
    "hotmail.com": ("smtp.office365.com", 587),
    "live.com": ("smtp.office365.com", 587),
    "yahoo.com": ("smtp.mail.yahoo.com", 587),
    "icloud.com": ("smtp.mail.me.com", 587),
}


class SendEmail(Widget):
    class Credentials:
        def __init__(
            self,
            username: str,
            password: str,
            host: Optional[str] = None,
            port: Optional[int] = None,
        ):
            if (not username or not password) or (username.strip() == "" or password.strip() == ""):
                pass
                # raise ValueError("Username and password must be provided.")
            self.username = username
            self.password = password

            domain = self.get_domain()
            if not domain:
                _host, _port = None, None
            else:
                _host, _port = SMTP_PROVIDERS.get(domain, (None, None))
            self.host = host or _host
            self.port = port or _port
            if not self.host or not self.port:
                pass
                # raise ValueError(
                #     f"No SMTP settings found for domain '{domain}'. "
                #     "Please pass smtp_host and smtp_port explicitly."
                # )

        def get_domain(self) -> str:
            """
            Extracts the email domain from the username.
            """
            if not self.username:
                return
                # raise ValueError("Username must be provided to extract the domain.")
            return self.username.split("@")[-1].lower()

        @classmethod
        def from_env(cls) -> Optional["SendEmail.Credentials"]:
            """
            Creates an instance of Credentials from environment variables.
            Environment variables should be set as:
            - EMAIL_USERNAME
            - EMAIL_PASSWORD
            - EMAIL_HOST (optional)
            - EMAIL_PORT (optional)
            """
            import os

            username = os.getenv("EMAIL_USERNAME")
            password = os.getenv("EMAIL_PASSWORD")
            host = os.getenv("EMAIL_HOST")
            port = os.getenv("EMAIL_PORT")

            if not username or not password:
                return
                # raise ValueError(
                #     "Environment variables EMAIL_USERNAME and EMAIL_PASSWORD must be set."
                # )

            return cls(username, password, host, port)

    def __init__(
        self,
        default_subject: str = None,
        default_body: str = None,
        widget_id: str = None,
        show_body: bool = True,
    ):
        self._default_subject = default_subject
        self._default_body = default_body

        super().__init__(widget_id=widget_id, file_path=__file__)
        self.content = self._init_ui(show_body=show_body)

        ModalRegistry().attach_settings_widget(
            owner_id=self.widget_id, widget=self.content, size="tiny"
        )

    @property
    def apply_button(self) -> Button:
        return self._apply_button

    def _init_ui(self, show_body: bool) -> Container:
        self.enable_checkbox = CheckboxField(
            title="Enable Email Notifications",
            description="If checked, email notifications will be sent automatically.",
            checked=False,
        )

        @self.enable_checkbox.value_changed
        def on_checkbox_change(checked: bool):
            if checked:
                self.enable_settings()
            else:
                self.disable_settings()

        self._target_addresses_input = Input(
            minlength=1,
            maxlength=100,
            placeholder="user1@example.com, user2@example.com",
            size="small",
            type="textarea",
        )

        target_addresses_field = Field(
            self._target_addresses_input,
            "Target email addresses",
            "Enter email addresses to separated by commas",
        )

        self._subject_input = Input(
            value=self._default_subject or "Supervisely Notification",
            minlength=0,
            maxlength=300,
            placeholder="Enter email subject here...",
            type="textarea",
        )
        subject_input_field = Field(
            self._subject_input, "Email Subject", "Configure the subject of the email notification."
        )

        self._body_input = TextArea(placeholder="Enter email body here...", rows=10, autosize=False)
        if self._default_body is not None:
            self._body_input.set_value(self._default_body)
        body_input_field = Field(
            self._body_input,
            "Email Body",
            "Configure the body of the email notification.",
        )
        if not show_body:
            body_input_field.hide()

        self._apply_button = Button("Apply")
        return Container(
            [
                self.enable_checkbox,
                target_addresses_field,
                subject_input_field,
                body_input_field,
                self._apply_button,
            ]
        )

    @property
    def is_email_sending_enabled(self) -> bool:
        """
        Returns True if email sending is enabled, False otherwise.
        """
        return self.enable_checkbox.is_checked()

    def enable_settings(self):
        """
        Enables the email notification settings.
        """
        self._target_addresses_input.enable()
        self._subject_input.enable()
        self._body_input.enable()

    def disable_settings(self):
        """
        Disables the email notification settings.
        """
        self._target_addresses_input.disable()
        self._subject_input.disable()
        self._body_input.disable()

    def get_target_addresses(self):
        """
        Returns a list of email addresses to send the notification to.
        If no addresses are provided, returns None.
        """
        value = self._target_addresses_input.get_value()
        if not value:
            return None
        return self._target_addresses_input.get_value().split(",")

    def get_subject(self):
        """
        Returns the subject of the email notification.
        If no subject is provided, returns an empty string.
        """
        return self._subject_input.get_value()

    def get_body(self):
        """
        Returns the body of the email notification.
        If no body is provided, returns an empty string.
        """
        return self._body_input.get_value()

    def set_body(self, body: str):
        """
        Sets the body of the email notification.
        :param body: The body text to set.
        """
        if not isinstance(body, str):
            raise ValueError("Body must be a string.")
        self._body_input.set_value(body)

    def get_json_data(self):
        return {}

    def get_json_state(self):
        return {}

    def send_email(
        self,
        credentials: Credentials,
        message: str = None,
        attachments: Optional[list] = None,
    ):
        """
        Send an email via SMTP. If smtp_host/port are not provided,
        they will be inferred from the username's email domain using SMTP_PROVIDERS.
        """
        if not credentials.username or not credentials.password:
            logger.warning("Email notifications are enabled, but no credentials are provided.")
            return
            # raise ValueError("Username and password must be provided in credentials.")

        from email.message import EmailMessage

        msg = EmailMessage()
        msg["Subject"] = self.get_subject() or self._default_subject
        msg["From"] = credentials.username
        msg["To"] = self.get_target_addresses() or [credentials.username]

        message = message or self.get_body() or self._default_body
        if "<html>" in message:
            msg.add_alternative(
                message,
                subtype="html",
            )
        else:
            msg.set_content(message)

        for path in attachments or []:
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Attachment not found: {path}")
            ctype, encoding = mimetypes.guess_type(path)
            maintype, subtype = (ctype or "application/octet-stream").split("/", 1)
            with open(path, "rb") as fp:
                msg.add_attachment(
                    fp.read(), maintype=maintype, subtype=subtype, filename=os.path.basename(path)
                )

        try:
            context = ssl.create_default_context()
            with smtplib.SMTP(credentials.host, credentials.port) as smtp:
                smtp.ehlo()
                smtp.starttls(context=context)
                smtp.ehlo()
                smtp.login(credentials.username, credentials.password)
                smtp.send_message(msg)
                logger.info(f"Email sent to {self.get_target_addresses()}")
        except smtplib.SMTPAuthenticationError as e:
            logger.error("Failed to auxthenticate with the provided email credentials.")
            raise e
        except (smtplib.SMTPException, smtplib.SMTPServerDisconnected) as e:
            logger.error(f"Failed to login to SMTP: {e}", exc_info=False)
            raise e

    def to_html(self):
        return self.content.to_html()

    @property
    def modal(self) -> Dialog:
        return ModalRegistry().settings_dialog_tiny

    def open_modal(self):
        ModalRegistry().open_settings(owner_id=self.widget_id, size="tiny")

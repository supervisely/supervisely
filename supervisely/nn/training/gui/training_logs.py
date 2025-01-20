from typing import Any, Dict

import supervisely.io.env as sly_env
from supervisely import Api
from supervisely._utils import is_production
from supervisely.app.widgets import (
    Button,
    Card,
    Container,
    Progress,
    RunAppButton,
    TaskLogs,
    Text,
)


class TrainingLogs:
    title = "Training Logs"
    description = "Track training progress"
    lock_message = "Start training to unlock"

    def __init__(self, app_options: Dict[str, Any]):
        self.display_widgets = []
        api = Api.from_env()
        self.app_options = app_options

        # GUI Components
        self.validator_text = Text("")
        self.validator_text.hide()

        if is_production():
            task_id = sly_env.task_id(raise_not_found=False)
        else:
            task_id = None

        # Tensorboard button
        if is_production():
            task_info = api.task.get_info_by_id(task_id)
            session_token = task_info["meta"]["sessionToken"]
            sly_url_prefix = f"/net/{session_token}"
            self.tensorboard_link = f"{api.server_address}{sly_url_prefix}/tensorboard/"
        else:
            self.tensorboard_link = "http://localhost:8000/tensorboard"
        self.tensorboard_button = Button(
            "Open Tensorboard",
            button_type="info",
            plain=True,
            icon="zmdi zmdi-chart",
            link=self.tensorboard_link,
            visible_by_vue_field="!isStaticVersion",
        )
        self.tensorboard_button.disable()

        # Offline session Tensorboard button
        if is_production():
            workspace_id = sly_env.workspace_id()
            app_id = task_info["meta"]["app"]["id"]
            app_info = api.app.get_info_by_id(app_id)
            app_module_id = app_info.module_id

            self.tensorboard_offline_button = RunAppButton(
                workspace_id=workspace_id,
                module_id=app_module_id,
                payload={},
                text="Open Tensorboard",
                button_type="text",
                plain=True,
                icon="zmdi zmdi-chart",
                available_in_offline=True,
                visible_by_vue_field="isStaticVersion",
            )

        self.display_widgets.extend(
            [self.validator_text, self.tensorboard_button, self.tensorboard_offline_button]
        )

        # Optional Show logs button
        if app_options.get("show_logs_in_gui", False):
            self.logs_button = Button(
                text="Show logs",
                plain=True,
                button_size="mini",
                icon="zmdi zmdi-caret-down-circle",
            )
            self.task_logs = TaskLogs(task_id)
            self.task_logs.hide()
            logs_container = Container([self.logs_button, self.task_logs])
            self.display_widgets.extend([logs_container])
        # -------------------------------- #

        # Progress bars
        self.progress_bar_main = Progress(hide_on_finish=False)
        self.progress_bar_main.hide()
        self.progress_bar_secondary = Progress(hide_on_finish=False)
        self.progress_bar_secondary.hide()
        self.display_widgets.extend([self.progress_bar_main, self.progress_bar_secondary])
        # -------------------------------- #

        self.container = Container(self.display_widgets)
        self.card = Card(
            title=self.title,
            description=self.description,
            content=self.container,
            lock_message=self.lock_message,
        )
        self.card.lock()

    @property
    def widgets_to_disable(self) -> list:
        return []

    def validate_step(self) -> bool:
        return True

    def toggle_logs(self):
        if self.task_logs.is_hidden():
            self.task_logs.show()
            self.logs_button.text = "Hide logs"
            self.logs_button.icon = "zmdi zmdi-caret-up-circle"
        else:
            self.task_logs.hide()
            self.logs_button.text = "Show logs"
            self.logs_button.icon = "zmdi zmdi-caret-down-circle"

from typing import Any, Dict

import supervisely.io.env as sly_env
import supervisely.nn.training.gui.utils as gui_utils
from supervisely import Api, logger
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
        # Init widgets
        self.tensorboard_button = None
        self.tensorboard_offline_button = None
        self.logs_button = None
        self.task_logs = None
        self.progress_bar_main = None
        self.progress_bar_secondary = None
        self.validator_text = None
        self.container = None
        self.card = None
        # -------------------------------- #

        self.display_widgets = []
        self.app_options = app_options
        api = Api.from_env()
        team_id = sly_env.team_id()

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
        self.display_widgets.extend([self.validator_text, self.tensorboard_button])

        # Offline session Tensorboard button
        if is_production():
            workspace_id = sly_env.workspace_id()
            app_name = "Tensorboard Experiments Viewer"
            module_info = gui_utils.get_module_info_by_name(api, app_name)
            if module_info is not None:
                self.tensorboard_offline_button = RunAppButton(
                    team_id=team_id,
                    workspace_id=workspace_id,
                    module_id=module_info["id"],
                    payload={},
                    text="Open Tensorboard",
                    button_type="info",
                    plain=True,
                    icon="zmdi zmdi-chart",
                    available_in_offline=True,
                    visible_by_vue_field=None,
                )
                self.tensorboard_offline_button.disable()
                self.tensorboard_offline_button.hide()
                self.display_widgets.extend([self.tensorboard_offline_button])
            else:
                logger.warning(
                    f"App '{app_name}' not found. Tensorboard button will not be displayed in offline mode."
                )

        # Optional Show logs button
        if self.app_options.get("show_logs_in_gui", False):
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

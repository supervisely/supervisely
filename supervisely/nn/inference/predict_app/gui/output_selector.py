from typing import Any, Dict, List

from supervisely import logger
from supervisely.api.api import Api
from supervisely.app.widgets import (
    Button,
    Card,
    Checkbox,
    Container,
    Field,
    Input,
    Progress,
    ProjectThumbnail,
    Text,
)


class OutputSelector:
    title = "Select Output"
    description = "Select the output mode"
    lock_message = "Select previous step to unlock"

    def __init__(self, api: Api):
        # Init Step
        self.api = api
        self.display_widgets: List[Any] = []
        # -------------------------------- #

        # Init Base Widgets
        self.validator_text = None
        self.start_button = None
        self.stop_button = None
        self.container = None
        self.card = None
        # -------------------------------- #

        # Init Step Widgets
        self.stop_serving_on_finish = None
        self.stop_self_on_finish = None
        self.project_name_input = None
        self.project_name_field = None
        self.progress = None
        self.project_thumbnail = None
        # -------------------------------- #

        # TODO: Implement option later
        # Stop Apps on Finish
        # self.stop_serving_on_finish = Checkbox("Stop Serving App on prediction finish", False)
        # self.stop_self_on_finish = Checkbox("Stop Predict App on prediction finish", True)
        # Add widgets to display ------------ #
        # self.display_widgets.extend([self.stop_serving_on_finish, self.stop_self_on_finish])
        # ----------------------------------- #

        # Project Name
        self.project_name_input = Input(minlength=1, maxlength=255, placeholder="New Project Name")
        self.project_name_field = Field(
            content=self.project_name_input,
            title="New Project Name",
            description="Name of the new project to create for the results. The created project will have the same dataset structure as the input project.",
        )
        # Add widgets to display ------------ #
        self.display_widgets.extend([self.project_name_field])
        # ----------------------------------- #

        # Base Widgets
        self.validator_text = Text("", status="text")
        self.validator_text.hide()
        self.start_button = Button("Run", icon="zmdi zmdi-play")
        self.stop_button = Button("Stop", icon="zmdi zmdi-stop")
        # Add widgets to display ------------ #
        self.display_widgets.extend([self.start_button, self.validator_text])
        # ----------------------------------- #

        # Progress
        self.progress = Progress(hide_on_finish=False)
        self.progress.hide()
        # Add widgets to display ------------ #
        self.display_widgets.extend([self.progress])
        # ----------------------------------- #

        # Result
        self.project_thumbnail = ProjectThumbnail()
        self.project_thumbnail.hide()
        # Add widgets to display ------------ #
        self.display_widgets.extend([self.project_thumbnail])
        # ----------------------------------- #

        # Card Layout
        self.container = Container(self.display_widgets)
        self.card = Card(
            title=self.title,
            description=self.description,
            content=self.container,
            lock_message=self.lock_message,
        )
        self.card.lock()
        # ----------------------------------- #

    @property
    def widgets_to_disable(self) -> list:
        return [self.project_name_input]

    def set_result_thumbnail(self, project_id: int):
        try:
            project_info = self.api.project.get_info_by_id(project_id)
            self.project_thumbnail.set(project_info)
            self.project_thumbnail.show()
        except Exception as e:
            logger.error(f"Failed to set result thumbnail: {str(e)}")
            self.project_thumbnail.hide()

    def get_settings(self) -> Dict[str, Any]:
        settings = {}
        settings["project_name"] = self.project_name_input.get_value()
        return settings

    def should_stop_serving_on_finish(self) -> bool:
        if self.stop_serving_on_finish is not None:
            return self.stop_serving_on_finish.is_checked()
        return False

    def should_stop_self_on_finish(self) -> bool:
        if self.stop_self_on_finish is not None:
            return self.stop_self_on_finish.is_checked()
        return True

    def load_from_json(self, data):
        project_name = data.get("project_name", None)
        if project_name:
            self.project_name_input.set_value(project_name)

    def validate_step(self) -> bool:
        self.validator_text.hide()
        if self.project_name_input.get_value() == "":
            self.validator_text.set(text="Project name is required", status="error")
            self.validator_text.show()
            return False

        return True

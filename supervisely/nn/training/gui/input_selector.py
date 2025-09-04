from supervisely.api.project_api import ProjectInfo
from supervisely.app.widgets import (
    Button,
    Card,
    Checkbox,
    Container,
    ProjectThumbnail,
    Text,
)
from supervisely.project.download import is_cached


class InputSelector:
    title = "Input project"
    description = "Selected project from which items and annotations will be downloaded"
    lock_message = None

    def __init__(self, project_info: ProjectInfo, app_options: dict = {}):
        # Init widgets
        self.project_thumbnail = None
        self.use_cache_text = None
        self.use_cache_checkbox = None
        self.validator_text = None
        self.button = None
        self.container = None
        self.card = None
        # -------------------------------- #

        self.display_widgets = []
        self.app_options = app_options

        self.project_id = project_info.id
        self.project_info = project_info

        # GUI Components
        self.project_thumbnail = ProjectThumbnail(self.project_info)

        if is_cached(self.project_id):
            _text = "Use cached data stored on the agent to optimize project download"
        else:
            _text = "Cache data on the agent to optimize project download for future trainings"
        self.use_cache_text = Text(_text)
        self.use_cache_checkbox = Checkbox(self.use_cache_text, checked=True)

        self.validator_text = Text("")
        self.validator_text.hide()
        self.button = Button("Select")
        self.display_widgets.extend(
            [
                self.project_thumbnail,
                self.use_cache_checkbox,
                self.validator_text,
                self.button,
            ]
        )
        # -------------------------------- #

        self.container = Container(self.display_widgets)
        self.card = Card(
            title=self.title,
            description=self.description,
            content=self.container,
            collapsable=self.app_options.get("collapsable", False),
        )

    @property
    def widgets_to_disable(self) -> list:
        return [self.use_cache_checkbox]

    def get_project_id(self) -> int:
        return self.project_id

    def set_cache(self, value: bool) -> None:
        if value:
            self.use_cache_checkbox.check()
        else:
            self.use_cache_checkbox.uncheck()

    def get_cache_value(self) -> bool:
        return self.use_cache_checkbox.is_checked()

    def validate_step(self) -> bool:
        return True

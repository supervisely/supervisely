from supervisely.api.project_api import ProjectInfo
from supervisely.app.widgets import (
    Button,
    Card,
    Checkbox,
    Container,
    Field,
    ProjectThumbnail,
    Text,
)
from supervisely.project.download import is_cached


class InputSelector:
    title = "Input Selector"

    def __init__(self, project_info: ProjectInfo):
        self.project_id = project_info.id
        self.project_info = project_info

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
        container = Container(
            widgets=[
                self.project_thumbnail,
                self.use_cache_checkbox,
                self.validator_text,
                self.button,
            ]
        )
        self.card = Card(
            title="Input project",
            description="Selected project from which images and annotations will be downloaded",
            content=container,
        )

    @property
    def widgets_to_disable(self):
        return [
            self.use_cache_checkbox,
        ]

    def get_project_id(self):
        return self.project_id

    def set_cache(self, value: bool):
        if value:
            self.use_cache_checkbox.check()
        else:
            self.use_cache_checkbox.uncheck()

    def get_cache_value(self):
        return self.use_cache_checkbox.is_checked()

    def validate_step(self):
        return True

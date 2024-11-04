from supervisely.api.project_api import ProjectInfo
from supervisely.app.widgets import (
    Button,
    Card,
    Checkbox,
    Container,
    Field,
    ProjectThumbnail,
    SelectDatasetTree,
    Text,
)
from supervisely.project.download import is_cached


class InputSelector:
    title = "Input Selector"

    def __init__(self, project_info: ProjectInfo):
        self.project_id = project_info.id
        self.project_info = project_info

        self.project_thumbnail = ProjectThumbnail(self.project_info)

        self.train_dataset_selector = SelectDatasetTree(
            project_id=self.project_id,
            multiselect=False,
            compact=True,
            team_is_selectable=False,
            workspace_is_selectable=False,
            show_select_all_datasets_checkbox=False,
        )
        train_dataset_selector_field = Field(
            title="Train dataset",
            description="Select dataset for training",
            content=self.train_dataset_selector,
        )
        self.val_dataset_selector = SelectDatasetTree(
            project_id=self.project_id,
            multiselect=False,
            compact=True,
            team_is_selectable=False,
            workspace_is_selectable=False,
            show_select_all_datasets_checkbox=False,
        )
        val_dataset_selector_field = Field(
            title="Val dataset",
            description="Select dataset for validation",
            content=self.val_dataset_selector,
        )

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
                train_dataset_selector_field,
                val_dataset_selector_field,
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
            self.train_dataset_selector,
            self.val_dataset_selector,
            self.use_cache_checkbox,
        ]

    def get_project_id(self):
        return self.project_id

    def get_train_dataset_id(self):
        return self.train_dataset_selector.get_selected_id()

    def set_train_dataset_id(self, dataset_id: int):
        self.train_dataset_selector.set_dataset_id(dataset_id)

    def get_val_dataset_id(self):
        return self.val_dataset_selector.get_selected_id()

    def set_val_dataset_id(self, dataset_id: int):
        self.val_dataset_selector.set_dataset_id(dataset_id)

    def set_cache(self, value: bool):
        if value:
            self.use_cache_checkbox.check(value)
        else:
            self.use_cache_checkbox.uncheck(value)

    def get_cache_value(self):
        return self.use_cache_checkbox.is_checked()

    def validate_step(self):
        self.validator_text.hide()
        train_dataset_id = self.get_train_dataset_id()
        val_dataset_id = self.get_val_dataset_id()

        error_messages = []
        if train_dataset_id is None:
            error_messages.append("Train dataset is not selected")
        if val_dataset_id is None:
            error_messages.append("Val dataset is not selected")

        if error_messages:
            self.validator_text.set(text="; ".join(error_messages), status="error")
            self.validator_text.show()
            return False

        if train_dataset_id == val_dataset_id:
            self.validator_text.set(
                text=(
                    "Train and val datasets are the same. "
                    "Using the same dataset for training and validation leads to overfitting, "
                    "poor generalization, biased model selection, "
                    "misleading metrics, and reduced robustness on unseen data."
                ),
                status="warning",
            )
            self.validator_text.show()
            return True
        else:
            self.validator_text.set(text="Train and val datasets are selected", status="success")
            self.validator_text.show()
            return True

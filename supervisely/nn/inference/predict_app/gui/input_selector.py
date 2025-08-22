from typing import Any, Dict, List

from supervisely.app.widgets import (
    Button,
    Card,
    Container,
    OneOf,
    RadioGroup,
    RadioTable,
    SelectDatasetTree,
    Text,
)
from supervisely.project.project import ProjectType


class InputSelector:
    title = "Select Input"
    description = "Select input data on which to run model for prediction"
    lock_message = None

    def __init__(self, workspace_id: int):
        # Init Step
        self.workspace_id = workspace_id
        self.display_widgets: List[Any] = []
        # -------------------------------- #

        # Init Base Widgets
        self.validator_text = None
        self.button = None
        self.container = None
        self.card = None
        # -------------------------------- #

        # Init Step Widgets
        # Images
        self.select_dataset_for_images = None
        self.select_image_container = None
        # Videos
        self.select_dataset_for_video = None
        self.select_video = None
        self.select_video_container = None
        # Selector
        self.radio = None
        self.one_of = None
        # -------------------------------- #

        # Images
        self.select_dataset_for_images = SelectDatasetTree(
            multiselect=True,
            flat=True,
            select_all_datasets=False,
            allowed_project_types=[ProjectType.IMAGES],
            always_open=False,
            compact=False,
            team_is_selectable=False,
            workspace_is_selectable=False,
            show_select_all_datasets_checkbox=True,
        )
        self.select_image_container = Container(widgets=[self.select_dataset_for_images])
        self._radio_item_images = RadioGroup.Item(
            ProjectType.IMAGES.value, "Images", content=self.select_image_container
        )
        # -------------------------------- #

        # Videos
        self.select_dataset_for_video = SelectDatasetTree(
            flat=True,
            select_all_datasets=False,
            allowed_project_types=[ProjectType.VIDEOS],
            always_open=False,
            compact=False,
            team_is_selectable=False,
            workspace_is_selectable=False,
            show_select_all_datasets_checkbox=False,
        )
        self.select_video = RadioTable(columns=["id", "name", "dataset"], rows=[])
        self.select_video_container = Container(
            widgets=[self.select_dataset_for_video, self.select_video]
        )
        self._radio_item_videos = RadioGroup.Item(
            ProjectType.VIDEOS.value, "Videos", content=self.select_video_container
        )
        # -------------------------------- #

        # Data type Radio Selector
        # self.radio = RadioGroup(items=[self._radio_item_images, self._radio_item_videos])
        self.radio = RadioGroup(items=[self._radio_item_images])
        self.radio.hide()
        self.one_of = OneOf(conditional_widget=self.radio)
        # Add widgets to display ------------ #
        self.display_widgets.extend([self.radio, self.one_of])
        # ----------------------------------- #

        # Base Widgets
        self.validator_text = Text("")
        self.validator_text.hide()
        self.button = Button("Select")
        # Add widgets to display ------------ #
        self.display_widgets.extend([self.validator_text, self.button])
        # ----------------------------------- #

        # Card Layout
        self.container = Container(self.display_widgets)
        self.card = Card(
            title=self.title,
            description=self.description,
            content=self.container,
            lock_message=self.lock_message,
        )
        # ----------------------------------- #

    @property
    def widgets_to_disable(self) -> list:
        return [
            self.select_dataset_for_images,
            self.select_dataset_for_video,
            self.select_video,
            self.radio,
            self.one_of,
        ]

    def get_settings(self) -> Dict[str, Any]:
        if self.radio.get_value() == ProjectType.IMAGES.value:
            return {
                "project_id": self.select_dataset_for_images.get_selected_project_id(),
                "dataset_ids": self.select_dataset_for_images.get_selected_ids(),
            }
        if self.radio.get_value() == ProjectType.VIDEOS.value:
            return {"video_id": self.select_video.get_selected_row()}

    def load_from_json(self, data):
        if "project_id" in data:
            self.select_dataset_for_images.set_project_id(data["project_id"])
            self.select_dataset_for_images.select_all()
            self.radio.set_value(ProjectType.IMAGES.value)
        if "dataset_ids" in data:
            self.select_dataset_for_images.set_dataset_ids(data["dataset_ids"])
            self.radio.set_value(ProjectType.IMAGES.value)
        if "video_id" in data:
            self.select_video.select_row_by_value("id", data["video_id"])
            self.radio.set_value(ProjectType.VIDEOS.value)

    def validate_step(self) -> bool:
        self.validator_text.hide()
        if self.radio.get_value() == ProjectType.IMAGES.value:
            if len(self.select_dataset_for_images.get_selected_ids()) == 0:
                self.validator_text.set(text="Select at least one dataset", status="error")
                self.validator_text.show()
                return False
        if self.radio.get_value() == ProjectType.VIDEOS.value:
            if self.select_dataset_for_video.get_selected_id() is None:
                self.validator_text.set(text="Select a dataset", status="error")
                self.validator_text.show()
                return False
            if len(self.select_video.rows) == 0:
                self.validator_text.set(
                    text="No videos found in the selected dataset", status="error"
                )
                self.validator_text.show()
                return False
            if self.select_video.get_selected_row() == []:
                self.validator_text.set(text="Select a video", status="error")
                self.validator_text.show()
                return False
        return True

from typing import Any, Dict, List

from supervisely.api.api import Api
from supervisely.app.widgets import (
    Button,
    Container,
    Card,
    OneOf,
    RadioGroup,
    RadioTable,
    SelectDataset,
    Text,
)
from supervisely.project.project import ProjectType


class InputSelector:
    title = "Select Items"
    description = "Select the data modality on which to run model"
    lock_message = None

    def __init__(self, api: Api, workspace_id: int):
        # Init basic state
        self.api = api
        self.workspace_id = workspace_id
        self.display_widgets: List[Any] = []
        # -------------------------------- #

        # Init widgets
        self.select_dataset_for_images = None
        self.select_image_container = None
        self.select_dataset_for_video = None
        self.select_video = None
        self.select_video_container = None
        self.radio = None
        self.one_of = None
        self.validator_text = None
        self.button = None
        self.container = None
        self.card = None
        # -------------------------------- #

        # Images
        self.select_dataset_for_images = SelectDataset(
            multiselect=True, allowed_project_types=[ProjectType.IMAGES]
        )
        self.select_dataset_for_images._project_selector._ws_id = workspace_id
        self.select_dataset_for_images._project_selector._compact = True
        self.select_dataset_for_images._project_selector.update_data()
        self.select_image_container = Container(widgets=[self.select_dataset_for_images])
        self._radio_item_images = RadioGroup.Item(
            ProjectType.IMAGES.value, "Images", content=self.select_image_container
        )
        # -------------------------------- #

        # Videos
        self.select_dataset_for_video = SelectDataset(allowed_project_types=[ProjectType.VIDEOS])
        self.select_dataset_for_video._project_selector._ws_id = workspace_id
        self.select_dataset_for_video._project_selector._compact = True
        self.select_dataset_for_video._project_selector.update_data()
        self.select_video = RadioTable(columns=["id", "name", "dataset"], rows=[])
        self.select_video_container = Container(
            widgets=[self.select_dataset_for_video, self.select_video]
        )
        self._radio_item_videos = RadioGroup.Item(
            ProjectType.VIDEOS.value, "Videos", content=self.select_video_container
        )

        @self.select_dataset_for_video.value_changed
        def dataset_for_video_changed(dataset_id: int):
            self.select_video.loading = True
            if dataset_id is None:
                rows = []
            else:
                dataset_info = self.api.dataset.get_info_by_id(dataset_id)
                videos = self.api.video.get_list(dataset_id)
                rows = [[video.id, video.name, dataset_info.name] for video in videos]
            self.select_video.rows = rows
            self.select_video.loading = False

        # -------------------------------- #

        # Data type Radio Selector
        self.radio = RadioGroup(items=[self._radio_item_images, self._radio_item_videos])
        self.one_of = OneOf(conditional_widget=self.radio)
        # -------------------------------- #

        self.validator_text = Text("")
        self.validator_text.hide()
        self.button = Button("Select")
        self.display_widgets.extend([self.radio, self.one_of])

        self.container = Container(self.display_widgets, gap=20)
        self.card = Card(
            title=self.title,
            description=self.description,
            content=self.container,
        )

    @property
    def widgets_to_disable(self) -> list:
        return [
            self.radio,
            self.select_image_container,
            self.select_video_container,
        ]

    def get_item_settings(self) -> Dict[str, Any]:
        if self.radio.get_value() == ProjectType.IMAGES.value:
            return {"dataset_ids": self.select_dataset_for_images.get_selected_ids()}
        if self.radio.get_value() == ProjectType.VIDEOS.value:
            return {"video_id": self.select_video.get_selected_row()}

    def validate_step(self) -> bool:
        return True

    def load_from_json(self, data):
        # @TODO: add images or videos if
        if "project_id" in data:
            self.select_dataset_for_images.set_project_id(data["project_id"])
            self.select_dataset_for_images.set_select_all_datasets(True)
            self.radio.set_value("dataset")
        if "dataset_ids" in data:
            self.select_dataset_for_images.set_dataset_ids(data["dataset_ids"])
            self.radio.set_value("dataset")
        if "video_id" in data:
            self.select_video.select_row_by_value("id", data["video_id"])
            self.radio.set_value("video")

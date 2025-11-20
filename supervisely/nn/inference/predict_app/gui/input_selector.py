import threading
from typing import Any, Dict, List

from supervisely.api.api import Api
from supervisely.api.dataset_api import DatasetInfo
from supervisely.api.project_api import ProjectInfo
from supervisely.api.video.video_api import VideoInfo
from supervisely.app.widgets import (
    Button,
    Card,
    Container,
    FastTable,
    OneOf,
    RadioGroup,
    SelectDatasetTree,
    Text,
)
from supervisely.app.widgets.widget import Widget
from supervisely.project.project import ProjectType


class InputSelector:
    title = "Input data"
    description = "Select input data on which to run model for prediction"
    lock_message = None

    def __init__(self, workspace_id: int, api: Api):
        # Init Step
        self.workspace_id = workspace_id
        self.api = api
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
            select_all_datasets=True,
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
            multiselect=True,
            flat=True,
            select_all_datasets=True,
            allowed_project_types=[ProjectType.VIDEOS],
            always_open=False,
            compact=False,
            team_is_selectable=False,
            workspace_is_selectable=False,
            show_select_all_datasets_checkbox=True,
        )
        self._video_table_columns = [
            "Video id",
            "Video name",
            "Size",
            "Duration",
            "FPS",
            "Frames count",
            "Dataset name",
            "Dataset id",
        ]
        self.select_video = FastTable(
            columns=self._video_table_columns,
            is_selectable=True,
        )
        self.select_video.hide()
        self.select_video_container = Container(
            widgets=[self.select_dataset_for_video, self.select_video]
        )
        self._radio_item_videos = RadioGroup.Item(
            ProjectType.VIDEOS.value, "Videos", content=self.select_video_container
        )
        # -------------------------------- #

        # Data type Radio Selector
        self.radio = RadioGroup(items=[self._radio_item_images, self._radio_item_videos])
        # self.radio = RadioGroup(items=[self._radio_item_images])
        # self.radio.hide()
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

        self._refresh_table_lock = threading.Lock()
        self._refresh_table_thread: threading.Thread = None
        self._refresh_called = False

        @self.radio.value_changed
        def input_selector_type_changed(value: str):
            self.validator_text.hide()

        @self.select_dataset_for_images.project_changed
        def _images_project_changed(project_id):
            self.validator_text.hide()

        @self.select_dataset_for_images.value_changed
        def _images_dataset_changed(dataset_ids):
            self.validator_text.hide()

        @self.select_dataset_for_video.project_changed
        def _videos_project_changed(project_id: int):
            self._refresh_video_table_called()

        @self.select_dataset_for_video.value_changed
        def _videos_dataset_changed(datasets_ids):
            self._refresh_video_table_called()

    def _refresh_video_table_called(self):
        with self._refresh_table_lock:
            self._refresh_called = True
            if self._refresh_table_thread is None or not self._refresh_table_thread.is_alive():
                self._refresh_table_thread = threading.Thread(target=self._refresh_video_table_loop)
        if self._refresh_table_thread is not None and not self._refresh_table_thread.is_alive():
            self._refresh_table_thread.start()

    def _refresh_video_table_loop(self):
        while self._refresh_called:
            with self._refresh_table_lock:
                self._refresh_called = False
            self.select_video.loading = True
            self._refresh_video_table()
            if not self._refresh_called:
                self.select_video.loading = False

    def _refresh_video_table(self):
        self.validator_text.hide()
        self.select_video.clear()
        selected_datasets = self.select_dataset_for_video.get_selected_ids()
        if not selected_datasets:
            self.select_video.hide()
        else:
            rows = []
            self.select_video.show()
            for dataset_id in selected_datasets:
                dataset_info = self.api.dataset.get_info_by_id(dataset_id)
                videos = self.api.video.get_list(dataset_id)
                for video in videos:
                    size = f"{video.frame_height}x{video.frame_width}"
                    try:
                        frame_rate = int(video.frames_count / video.duration)
                    except:
                        frame_rate = "N/A"
                    rows.append(
                        [
                            video.id,
                            video.name,
                            size,
                            video.duration,
                            frame_rate,
                            video.frames_count,
                            dataset_info.name,
                            dataset_info.id,
                        ]
                    )

            self.select_video.add_rows(rows)

    def select_project(self, project_id: int, project_info: ProjectInfo = None):
        if project_info is None:
            project_info = self.api.project.get_info_by_id(project_id)
        if project_info.type == ProjectType.IMAGES.value:
            self.select_dataset_for_images.set_project_id(project_id)
            self.select_dataset_for_images.select_all()
            self.radio.set_value(ProjectType.IMAGES.value)
        elif project_info.type == ProjectType.VIDEOS.value:
            self.select_dataset_for_video.set_project_id(project_id)
            self.select_dataset_for_video.select_all()
            self._refresh_video_table()
            self.select_video.select_rows(list(range(len(self.select_video._rows_total))))
            self.radio.set_value(ProjectType.VIDEOS.value)
        else:
            raise ValueError(f"Project of type {project_info.type} is not supported.")

    def select_datasets(self, dataset_ids: List[int], dataset_infos: List[DatasetInfo] = None):
        if dataset_infos is None:
            dataset_infos = [self.api.dataset.get_info_by_id(ds_id) for ds_id in dataset_ids]
        project_ids = set(ds.project_id for ds in dataset_infos)
        if len(project_ids) > 1:
            raise ValueError("Cannot select datasets from different projects")
        project_id = project_ids.pop()
        project_info = self.api.project.get_info_by_id(project_id)
        if project_info.type == ProjectType.IMAGES.value:
            self.select_dataset_for_images.set_project_id(project_id)
            self.select_dataset_for_images.set_dataset_ids(dataset_ids)
            self.radio.set_value(ProjectType.IMAGES.value)
        elif project_info.type == ProjectType.VIDEOS.value:
            self.select_dataset_for_video.set_project_id(project_id)
            self.select_dataset_for_video.set_dataset_ids(dataset_ids)
            self._refresh_video_table()
            self.select_video.select_rows(list(range(self.select_video._rows_total)))
            self.radio.set_value(ProjectType.VIDEOS.value)
        else:
            raise ValueError(f"Project of type {project_info.type} is not supported.")

    def select_videos(self, video_ids: List[int], video_infos: List[VideoInfo] = None):
        if video_infos is None:
            video_infos = self.api.video.get_info_by_id_batch(video_ids)
        project_id = video_infos[0].project_id
        self.select_dataset_for_video.set_project_id(project_id)
        self.select_dataset_for_video.select_all()
        self._refresh_video_table()
        self.select_video.select_row_by_value("id", video_ids)
        self.radio.set_value(ProjectType.VIDEOS.value)

    def disable(self):
        for widget in self.widgets_to_disable:
            widget.disable()

    def enable(self):
        for widget in self.widgets_to_disable:
            widget.enable()

    @property
    def widgets_to_disable(self) -> List[Widget]:
        return [
            # Images Selector
            self.select_dataset_for_images,
            self.select_dataset_for_images._select_project,
            self.select_dataset_for_images._select_dataset,
            # Videos Selector
            self.select_dataset_for_video,
            self.select_dataset_for_video._select_project,
            self.select_dataset_for_video._select_dataset,
            self.select_video,
            # Controls
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
            rows = self.select_video.get_selected_rows()
            if rows:
                video_ids = [row.row[0] for row in rows]
            else:
                video_ids = None
            return {"video_ids": video_ids}

    def load_from_json(self, data):
        if "video_ids" in data:
            video_ids = data["video_ids"]
            if not video_ids:
                raise ValueError("Video ids cannot be empty")
            video_infos = self.api.video.get_info_by_id_batch(video_ids)
            if not video_infos:
                raise ValueError(f"Videos with video ids {video_ids} are not found")
            self.select_videos(video_ids, video_infos)
        elif "dataset_ids" in data:
            dataset_ids = data["dataset_ids"]
            self.select_datasets(dataset_ids)
        elif "project_id" in data:
            project_id = data["project_id"]
            self.select_project(project_id)

    def get_project_id(self) -> int:
        if self.radio.get_value() == ProjectType.IMAGES.value:
            return self.select_dataset_for_images.project_id
        if self.radio.get_value() == ProjectType.VIDEOS.value:
            return self.select_dataset_for_video.project_id
        return None

    def validate_step(self) -> bool:
        self.validator_text.hide()
        if self.radio.get_value() == ProjectType.IMAGES.value:
            selected_ids = self.select_dataset_for_images.get_selected_ids()
            if selected_ids is None:
                self.validator_text.set(text="Select a project", status="error")
                self.validator_text.show()
                return False
            if len(selected_ids) == 0:
                self.validator_text.set(text="Select at least one dataset", status="error")
                self.validator_text.show()
                return False
        if self.radio.get_value() == ProjectType.VIDEOS.value:
            if not self.select_dataset_for_video.get_selected_ids():
                self.validator_text.set(text="Select a dataset", status="error")
                self.validator_text.show()
                return False
            if self.select_video._rows_total == 0:
                self.validator_text.set(
                    text="No videos found in the selected dataset", status="error"
                )
                self.validator_text.show()
                return False
            if self.select_video.get_selected_rows() == []:
                self.validator_text.set(text="Select a video", status="error")
                self.validator_text.show()
                return False
        return True

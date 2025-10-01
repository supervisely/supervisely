import os
import random
import shutil
import threading
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Callable, Dict, List

import cv2
import yaml

from supervisely._utils import logger
from supervisely.api.api import Api
from supervisely.app.widgets import (
    Button,
    Card,
    Container,
    Editor,
    Field,
    GridGallery,
    Input,
    OneOf,
    Progress,
    Select,
    Text,
    VideoPlayer,
)
from supervisely.app.widgets.checkbox.checkbox import Checkbox
from supervisely.nn.inference.predict_app.gui.input_selector import InputSelector
from supervisely.nn.inference.predict_app.gui.model_selector import ModelSelector
from supervisely.nn.model.model_api import ModelAPI, Prediction
from supervisely.nn.tracker import TrackingVisualizer
from supervisely.project import ProjectMeta
from supervisely.project.project_meta import ProjectType
from supervisely.video.video import VideoFrameReader, create_from_frames
from supervisely.video_annotation.video_annotation import VideoAnnotation


class InferenceMode:
    FULL_IMAGE = "Full Image"
    SLIDING_WINDOW = "Sliding Window"


class AddPredictionsMode:
    MERGE_WITH_EXISTING_LABELS = "Merge with existing labels"
    REPLACE_EXISTING_LABELS = "Replace existing labels"
    REPLACE_EXISTING_LABELS_AND_SAVE_IMAGE_TAGS = "Replace existing labels and save image tags"


class Preview:
    lock_message = "Select previous step to unlock"

    def __init__(
        self,
        api: Api,
        preview_dir: str,
        get_model_api_fn: Callable[[], ModelAPI],
        get_input_settings_fn: Callable[[], Dict[str, Any]],
        get_settings_fn: Callable[[], Dict[str, Any]],
    ):
        self.api = api
        self.preview_dir = preview_dir
        self.get_model_api_fn = get_model_api_fn
        self.get_input_settings_fn = get_input_settings_fn
        self.get_settings_fn = get_settings_fn
        os.makedirs(self.preview_dir, exist_ok=True)
        self.image_preview_path = None
        self.image_peview_url = None
        self.video_preview_path = None
        self.video_peview_url = None

        self.progress_widget = Progress(show_percents=True, hide_on_finish=True)
        self.image_gallery = GridGallery(
            2,
            sync_views=True,
            enable_zoom=True,
            resize_on_zoom=True,
            empty_message="",
        )
        self.image_preview_container = Container(widgets=[self.image_gallery])

        self.video_player = VideoPlayer()
        self.video_preview_container = Container(widgets=[self.video_player])

        self.empty_text = Text("Click preview to visualize predictions")
        self.error_text = Text("Failed to generate preview", status="error")

        self.select = Select(
            items=[
                Select.Item("empty", content=self.empty_text),
                Select.Item(ProjectType.IMAGES.value, content=self.image_preview_container),
                Select.Item(ProjectType.VIDEOS.value, content=self.video_preview_container),
                Select.Item("error", content=self.error_text),
                Select.Item("loading", content=Container(widgets=[])),
                Select.Item("progress", content=self.progress_widget),
            ]
        )
        self.select.set_value("empty")
        self.oneof = OneOf(self.select)

        self.run_button = Button("Preview", icon="zmdi zmdi-slideshow")
        self.card = Card(
            title="Preview",
            description="Preview model predictions on a random image or video from the selected input source.",
            content=self.oneof,
            content_top_right=self.run_button,
            lock_message=self.lock_message,
        )
        self.card.lock()

        @self.run_button.click
        def _run_preview():
            self.run_preview()

    @contextmanager
    def progress(self, message: str, total: int, **kwargs):
        current_item = self.select.get_value()
        try:
            with self.progress_widget(message=message, total=total, **kwargs) as pbar:
                self.select_item("progress")
                yield pbar
        finally:
            self.select_item(current_item)

    def select_item(self, item: str):
        self.select.set_value(item)

    def _download_preview_item(self, with_progress: bool = False):
        input_settings = self.get_input_settings_fn()
        video_ids = input_settings.get("video_ids", None)
        if video_ids is None:
            project_id = input_settings.get("project_id", None)
            dataset_ids = input_settings.get("dataset_ids", None)
            if dataset_ids:
                dataset_id = random.choice(dataset_ids)
            else:
                datasets = self.api.dataset.get_list(project_id)
            total_items = sum(ds.items_count for ds in datasets)
            if total_items == 0:
                raise RuntimeError("No images found in the selected datasets")
            images = []
            while not images:
                dataset_id = random.choice(datasets).id
                images = self.api.image.get_list(dataset_id)
            image_id = random.choice(images).id
            image_info = self.api.image.get_info_by_id(image_id)
            self.image_preview_path = Path(self.preview_dir, image_info.name)
            self.api.image.download_path(image_id, self.image_preview_path)
            self._current_item_id = image_id
            self.image_peview_url = f"/static/preview/{image_info.name}"
        elif len(video_ids) == 0:
            self._current_item_id = None
            self.video_preview_path = None
            self.video_peview_url = None
        else:
            video_id = random.choice(video_ids)
            video_id = video_ids[0]
            video_info = self.api.video.get_info_by_id(video_id)
            frame_start = 0
            seconds = 5
            video_info = self.api.video.get_info_by_id(video_id)
            preview_path = Path(self.preview_dir, video_info.name)
            self.video_preview_path = preview_path
            fps = int(video_info.frames_count / video_info.duration)
            frames_number = min(video_info.frames_count, int(fps * seconds))
            if video_info.frames_count > 30 * 60 * 5:
                with (
                    self.progress(message="Downloading video frames:", total=frames_number)
                    if with_progress
                    else nullcontext()
                ) as pbar:
                    paths = [
                        f"/tmp/{i}.jpg" for i in range(frame_start, frame_start + frames_number)
                    ]
                    self.api.video.download_frames(
                        video_id,
                        frames=list(range(frame_start, frame_start + frames_number)),
                        paths=paths,
                        progress_cb=pbar.update if pbar else None,
                    )
                    frames_gen = (cv2.imread(p) for p in paths)
                    create_from_frames(frames_gen, self.video_preview_path, fps=fps)
            else:
                try:
                    size = int(video_info.file_meta["size"])
                except:
                    size = None
                with (
                    self.progress(message="Downloading video:", total=size)
                    if with_progress and size
                    else nullcontext()
                ) as pbar:
                    self.api.video.download_path(
                        video_id,
                        self.video_preview_path,
                        progress_cb=pbar.update if pbar else None,
                    )
            self._current_item_id = video_id
            self.video_peview_url = f"/static/preview/{video_info.name}"

    def set_image_preview(
        self,
    ):
        self.image_gallery.clean_up()
        if not self._current_item_id:
            self._download_preview_item(with_progress=True)
        image_id = self._current_item_id
        model_api = self.get_model_api_fn()
        settings = self.get_settings_fn()
        inference_settings = settings.get("inference_settings", {})
        with self.progress("Running Model:", total=1) as pbar:
            prediction = model_api.predict(
                image_id=image_id, inference_settings=inference_settings, tqdm=pbar
            )[0]
        self.image_gallery.append(
            self.image_peview_url, title="Source", annotation=prediction.annotation
        )
        self.image_gallery.append(
            self.image_peview_url, title="Prediction", annotation=prediction.annotation
        )
        self.select_item(ProjectType.IMAGES.value)

    def set_video_preview(
        self,
    ):
        self.video_player.set_video(None)
        input_settings = self.get_input_settings_fn()
        video_ids = input_settings.get("video_ids", None)
        if not video_ids:
            raise RuntimeError("No videos selected")
        if not self._current_item_id:
            self._download_preview_item(with_progress=True)
        video_id = self._current_item_id

        frame_start = 0
        seconds = 5
        video_info = self.api.video.get_info_by_id(video_id)
        fps = int(video_info.frames_count / video_info.duration)
        frames_number = min(video_info.frames_count, int(fps * seconds))
        project_meta = ProjectMeta.from_json(self.api.project.get_meta(video_info.project_id))
        model_api = self.get_model_api_fn()
        settings = self.get_settings_fn()
        inference_settings = settings.get("inference_settings", {})
        tracking = settings.get("tracking", False)
        with self.progress("Running model:", total=frames_number) as pbar:
            with model_api.predict(
                video_id=video_id,
                inference_settings=inference_settings,
                tracking=tracking,
                start_frame=frame_start,
                frames_num=frames_number,
                tqdm=pbar,
            ) as session:
                predictions: List[Prediction] = list(session)

        if tracking:
            video_annotation = session.final_result.get("video_ann", {})
            if video_annotation is None:
                raise RuntimeError("Model did not return video annotation")
            video_annotation = VideoAnnotation.from_json(
                video_annotation, project_meta=project_meta
            )
            visualizer = TrackingVisualizer(
                output_fps=fps,
                box_thickness=video_info.frame_height // 110,
                text_scale=video_info.frame_height / 900,
                trajectory_thickness=video_info.frame_width // 110,
            )
            visualizer.visualize_video_annotation(
                video_annotation,
                source=self.video_preview_path,
                output_path=self.video_preview_path,
            )
        else:
            video_writer = cv2.VideoWriter(
                str(self.video_preview_path / "tmp"),
                cv2.VideoWriter.fourcc(*"mp4v"),
                fps,
                (video_info.frame_width, video_info.frame_height),
            )
            with VideoFrameReader(
                self.video_preview_path,
                frame_indexes=list(range(frame_start, frame_start + frames_number)),
            ) as video_reader:
                for pred, frame in zip(predictions, video_reader):
                    if pred.annotation is not None and len(pred.annotation.labels) > 0:
                        img = pred.annotation.draw(frame, draw_class_names=True)
                    else:
                        img = frame
                    video_writer.write(img)
            video_writer.release()
            shutil.move(str(self.video_preview_path / "tmp"), self.video_preview_path)
        self.video_player.set_video(self.video_peview_url)
        self.select_item(ProjectType.VIDEOS.value)

    def set_error(self, text: str):
        self.error_text.text = text
        self.select_item("error")

    def run_preview(self):
        self.select_item("loading")
        try:
            input_settings = self.get_input_settings_fn()
            video_ids = input_settings.get("video_ids", None)
            if video_ids is None:
                self.set_image_preview()
            elif len(video_ids) == 0:
                self.set_error("No videos selected")
            else:
                self.set_video_preview()
        except Exception as e:
            logger.error(f"Failed to generate preview: {str(e)}", exc_info=True)
            self.set_error("Failed to generate preview: " + str(e))

    def _preload_item(self):
        threading.Thread(
            target=self._download_preview_item, kwargs={"with_progress": False}, daemon=True
        ).start()

    def update_item_type(self, item_type: str):
        self.select_item("empty")
        self._current_item_id = None
        # self._preload_item() # need to handle race condition with run_preview and multiple clicks


class SettingsSelector:
    title = "Inference (settings + preview)"
    description = "Select additional settings for model inference"
    lock_message = "Select previous step to unlock"

    def __init__(
        self,
        api: Api,
        static_dir: str,
        input_selector: InputSelector,
        model_selector: ModelSelector,
    ):
        # Init Step
        self.api = api
        self.static_dir = static_dir
        self.input_selector = input_selector
        self.model_selector = model_selector
        self.display_widgets: List[Any] = []
        # -------------------------------- #

        # Init Base Widgets
        self.validator_text = None
        self.button = None
        self.run_button = None
        self.container = None
        self.cards = None
        # -------------------------------- #

        # Init Step Widgets
        self.inference_mode_selector = None
        self.inference_mode_field = None
        self.model_prediction_suffix_input = None
        self.model_prediction_suffix_field = None
        # self.model_prediction_suffix_checkbox = None
        self.predictions_mode_selector = None
        self.predictions_mode_field = None
        self.inference_settings = None
        # -------------------------------- #

        self.settings_widgets = []
        self.image_settings_widgets = []
        self.video_settings_widgets = []

        # Prediction Mode
        self.prediction_modes = [
            AddPredictionsMode.MERGE_WITH_EXISTING_LABELS,
            AddPredictionsMode.REPLACE_EXISTING_LABELS,
            # AddPredictionsMode.REPLACE_EXISTING_LABELS_AND_SAVE_IMAGE_TAGS, # @TODO: Implement later
        ]
        self.predictions_mode_selector = Select(
            items=[Select.Item(mode) for mode in self.prediction_modes]
        )
        self.predictions_mode_selector.set_value(self.prediction_modes[0])
        self.predictions_mode_field = Field(
            content=self.predictions_mode_selector,
            title="Add predictions mode",
            description="Select how to add predictions to the project: by merging with existing labels or by replacing them.",
        )
        # Add widgets to display ------------ #
        self.image_settings_widgets.extend([self.predictions_mode_field])
        # ----------------------------------- #

        # Tracking
        self.tracking_checkbox = Checkbox(content="Enable tracking", checked=True)
        self.tracking_checkbox_field = Field(
            content=self.tracking_checkbox,
            title="Tracking",
            description="Enable tracking for video predictions. The tracking algorithm is BoT-SORT version improved by Supervisely team.",
        )
        # Add widgets to display ------------ #
        self.video_settings_widgets.extend([self.tracking_checkbox_field])
        self.image_settings_container = Container(widgets=self.image_settings_widgets, gap=15)
        self.video_settings_container = Container(widgets=self.video_settings_widgets, gap=15)
        self.image_or_video_container = Container(
            widgets=[self.image_settings_container, self.video_settings_container], gap=0
        )
        self.video_settings_container.hide()
        self.settings_widgets.extend([self.image_or_video_container])
        # ----------------------------------- #

        # Class / Tag Suffix
        self.model_prediction_suffix_input = Input(
            value="_model", minlength=1, placeholder="Enter suffix e.g: _model"
        )
        self.model_prediction_suffix_field = Field(
            content=self.model_prediction_suffix_input,
            title="Class and tag suffix",
            description=(
                "Suffix that will be added to conflicting class and tag names. "
                "E.g. your project has a class 'person' with shape 'bitmap' and model has class 'person' with shape 'rectangle', "
                "then suffix will be added to the model predictions to avoid conflicts. E.g. 'person_model'."
            ),
        )
        # Add widgets to display ------------ #
        self.settings_widgets.extend([self.model_prediction_suffix_field])
        # ----------------------------------- #

        # Inference Settings
        self.inference_settings = Editor("", language_mode="yaml", height_px=300)
        # Add widgets to display ------------ #
        self.settings_widgets.extend([self.inference_settings])
        # ----------------------------------- #

        # Preview
        self.preview_dir = os.path.join(self.static_dir, "preview")
        self.preview = Preview(
            api=self.api,
            preview_dir=self.preview_dir,
            get_model_api_fn=lambda: self.model_selector.model.model_api,
            get_input_settings_fn=self.input_selector.get_settings,
            get_settings_fn=self.get_settings,
        )

        self.settings_container = Container(widgets=self.settings_widgets, gap=15)
        self.display_widgets.extend([self.settings_container])
        # Base Widgets
        self.validator_text = Text("")
        self.validator_text.hide()
        self.button = Button("Select")
        # Add widgets to display ------------ #
        self.display_widgets.extend([self.validator_text, self.button])
        # ----------------------------------- #

        # Card Layout
        self.container = Container(self.display_widgets)
        self.settings_card = Card(
            title=self.title,
            description=self.description,
            content=self.container,
            lock_message=self.lock_message,
        )
        self.settings_card.lock()
        self.cards = [self.settings_card, self.preview.card]
        self.cards_container = Container(
            widgets=self.cards,
            gap=15,
            direction="horizontal",
            fractions=[3, 7],
        )
        # ----------------------------------- #

    @property
    def widgets_to_disable(self) -> list:
        return [
            # self.inference_mode_selector,
            self.model_prediction_suffix_input,
            # self.model_prediction_suffix_checkbox,
            self.predictions_mode_selector,
            self.inference_settings,
        ]

    def set_inference_settings(self, settings: Dict[str, Any]):
        if isinstance(settings, str):
            self.inference_settings.set_text(settings)
        else:
            self.inference_settings.set_text(yaml.safe_dump(settings))

    def get_inference_settings(self) -> Dict:
        settings = yaml.safe_load(self.inference_settings.get_text())
        if settings:
            return settings
        return {}

    def get_settings(self) -> Dict[str, Any]:
        settings = {
            # "inference_mode": self.inference_mode_selector.get_value(),
            "inference_mode": InferenceMode.FULL_IMAGE,
            "model_prediction_suffix": self.model_prediction_suffix_input.get_value(),
            "predictions_mode": self.predictions_mode_selector.get_value(),
            "inference_settings": self.get_inference_settings(),
        }
        if self.input_selector.get_settings().get("video_ids", None) is not None:
            settings["tracking"] = self.tracking_checkbox.is_checked()
        return settings

    def load_from_json(self, data):
        # inference_mode = data.get("inference_mode", None)
        # if inference_mode:
        #     self.inference_mode_selector.set_value(inference_mode)

        model_prediction_suffix = data.get("model_prediction_suffix", None)
        if model_prediction_suffix is not None:
            self.model_prediction_suffix_input.set_value(model_prediction_suffix)

        predictions_mode = data.get("predictions_mode", None)
        if predictions_mode:
            self.predictions_mode_selector.set_value(predictions_mode)

        inference_settings = data.get("inference_settings", None)
        if inference_settings is not None:
            self.set_inference_settings(inference_settings)

    def update_item_type(self, item_type: str):
        if item_type == ProjectType.IMAGES.value:
            self.video_settings_container.hide()
            self.image_settings_container.show()
        elif item_type == ProjectType.VIDEOS.value:
            self.image_settings_container.hide()
            self.video_settings_container.show()
        else:
            raise ValueError(f"Unsupported item type: {item_type}")
        self.preview.update_item_type(item_type)

    def validate_step(self) -> bool:
        return True

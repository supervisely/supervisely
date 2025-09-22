import os
import random
from typing import Any, Dict, List

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
    Select,
    Text,
    VideoPlayer,
)
from supervisely.app.widgets.checkbox.checkbox import Checkbox
from supervisely.nn.inference.predict_app.gui.input_selector import InputSelector
from supervisely.nn.inference.predict_app.gui.model_selector import ModelSelector
from supervisely.nn.model.model_api import ModelAPI
from supervisely.nn.tracker import TrackingVisualizer
from supervisely.project import ProjectMeta
from supervisely.project.project_meta import ProjectType
from supervisely.video.video import create_from_frames
from supervisely.video_annotation.video_annotation import VideoAnnotation


class InferenceMode:
    FULL_IMAGE = "Full Image"
    SLIDING_WINDOW = "Sliding Window"


class AddPredictionsMode:
    MERGE_WITH_EXISTING_LABELS = "Merge with existing labels"
    REPLACE_EXISTING_LABELS = "Replace existing labels"
    REPLACE_EXISTING_LABELS_AND_SAVE_IMAGE_TAGS = "Replace existing labels and save image tags"


class Preview:
    def __init__(self, api: Any, static_dir: str):
        self.static_dir = static_dir
        self.display_widgets = []
        # Preview
        self.gallery = None
        self.video_player = None
        # -------------------------------- #

        # Preview Directory
        self.preview_dir = os.path.join(self.static_dir, "preview")
        os.makedirs(self.preview_dir, exist_ok=True)
        self.image_preview_path = os.path.join(self.preview_dir, "preview.jpg")
        self.image_peview_url = f"/static/preview/preview.jpg"
        self.video_preview_path = os.path.join(self.preview_dir, "preview.mp4")
        self.video_peview_url = f"/static/preview/preview.mp4"
        # ----------------------------------- #

        # Preview Widget
        self.gallery = GridGallery(
            2,
            sync_views=True,
            enable_zoom=True,
            resize_on_zoom=True,
            empty_message="Click 'Preview' to see the model output.",
        )
        self.video_player = VideoPlayer()
        # Add widgets to display ------------ #
        self.display_widgets.extend([self.gallery])
        # ----------------------------------- #


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
            description="Enable tracking for video predictions. The tracking algorith is BoT-SORT",
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
        os.makedirs(self.preview_dir, exist_ok=True)
        self.image_preview_path = os.path.join(self.preview_dir, "preview.jpg")
        self.image_peview_url = f"/static/preview/preview.jpg"
        self.video_preview_path = os.path.join(self.preview_dir, "preview.mp4")
        self.video_peview_url = f"/static/preview/preview.mp4"

        self.run_button = Button("Preview", icon="zmdi zmdi-slideshow")
        self.image_preview_gallery = GridGallery(
            2,
            sync_views=True,
            enable_zoom=True,
            resize_on_zoom=True,
            empty_message="Click 'Preview' to see the model output.",
        )
        self.last_video_id = None
        self.video_player = VideoPlayer()
        self.video_player.hide()
        self.preview_error = Text("Failed to generate preview", status="error")
        self.preview_error.hide()
        self.preview_container = Container(
            widgets=[self.image_preview_gallery, self.video_player, self.preview_error], gap=0
        )
        self.preview_card = Card(
            title="Preview",
            description="Preview model predictions on a random image or video from the selected input source.",
            content=self.preview_container,
            content_top_right=self.run_button,
            lock_message=self.lock_message,
        )
        self.preview_card.lock()

        @self.run_button.click
        def run_preview():
            self.run_preview()

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
        self.cards = [self.settings_card, self.preview_card]
        self.cards_container = Container(
            widgets=[self.settings_card, self.preview_card],
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
        if self.input_selector.get_settings().get("video_id", None) is not None:
            settings["tracking"] = self.tracking_checkbox.is_checked()

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
            self.image_settings_container.show()
            self.video_settings_container.hide()
            self.video_player.hide()
            self.image_preview_gallery.show()
        elif item_type == ProjectType.VIDEOS.value:
            self.image_settings_container.hide()
            self.video_settings_container.show()
            self.image_preview_gallery.hide()
            self.video_player.show()
        else:
            raise ValueError(f"Unsupported item type: {item_type}")

    def set_image_preview(self, image_id: int):
        with self.model_selector.model.model_api.predict_detached(
            image_id=image_id, inference_settings=self.get_inference_settings()
        ) as session:
            self.api.image.download_path(image_id, self.image_preview_path)
            prediction = list(session)[0]
        self.image_preview_gallery.clean_up()
        self.image_preview_gallery.append(
            self.image_peview_url, title="Source", annotation=prediction.annotation
        )
        self.image_preview_gallery.append(
            self.image_peview_url, title="Prediction", annotation=prediction.annotation
        )

    def set_video_preview(self, video_id: int):
        self.video_player.set_video(None)
        frame_start = 0
        seconds = 5
        video_info = self.api.video.get_info_by_id(video_id)
        fps = int(video_info.frames_count / video_info.duration)
        frames_number = min(video_info.frames_count, int(fps * seconds))
        project_meta = ProjectMeta.from_json(self.api.project.get_meta(video_info.project_id))
        with self.model_selector.model.model_api.predict_detached(
            video_id=video_id,
            inference_settings=self.get_inference_settings(),
            tracking=True,
            start_frame=frame_start,
            frames_num=frames_number,
        ) as session:
            if self.last_video_id != video_id:
                paths = [f"/tmp/{i}.jpg" for i in range(frame_start, frame_start + frames_number)]
                self.api.video.download_frames(
                    video_id,
                    frames=list(range(frame_start, frame_start + frames_number)),
                    paths=paths,
                )
                frames_gen = (cv2.imread(p) for p in paths)
                create_from_frames(frames_gen, self.video_preview_path, fps=fps)
                self.last_video_id = video_id

            list(session)

            video_annotation = session.final_result.get("video_ann", {})
        video_annotation = VideoAnnotation.from_json(video_annotation, project_meta=project_meta)
        visualizer = TrackingVisualizer(
            output_fps=fps,
            box_thickness=video_info.frame_width // 200,
            text_scale=0.6,
            text_thickness=video_info.frame_width // 200,
            trajectory_thickness=video_info.frame_width // 200,
        )
        visualizer.visualize_video_annotation(
            video_annotation,
            source=self.video_preview_path,
            output_path=self.video_preview_path,
        )
        self.video_player.set_video(self.video_peview_url)

    def run_preview(self):
        try:
            self.preview_error.hide()
            video_id = self.input_selector.get_settings().get("video_id", None)
            if video_id is None:
                project_id = self.input_selector.get_settings().get("project_id", None)
                dataset_ids = self.input_selector.get_settings().get("dataset_ids", None)
                if dataset_ids:
                    dataset_id = random.choice(dataset_ids)
                else:
                    datasets = self.api.dataset.get_list(project_id)
                    dataset_id = random.choice(datasets).id
                images = self.api.image.get_list(dataset_id)
                image_id = random.choice(images).id
                self.set_image_preview(image_id)
                self.image_preview_gallery.show()
            else:
                self.set_video_preview(video_id)
                self.video_player.show()
        except Exception as e:
            logger.error(f"Failed to generate preview: {str(e)}", exc_info=True)
            self.video_player.hide()
            self.image_preview_gallery.hide()
            self.image_preview_gallery.clean_up()
            self.preview_error.text = f"Failed to generate preview: {str(e)}"
            self.preview_error.show()

    def validate_step(self) -> bool:
        return True

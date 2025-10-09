import os
import random
import shutil
import subprocess
import threading
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Callable, Dict, List

import cv2
import yaml

from supervisely._utils import logger
from supervisely.api.api import Api
from supervisely.api.video.video_api import VideoInfo
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
from supervisely.app.widgets.widget import Widget
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
        os.makedirs(Path(self.preview_dir, "annotated"), exist_ok=True)
        self.image_preview_path = None
        self.image_peview_url = None
        self.video_preview_path = None
        self.video_preview_annotated_path = None
        self.video_peview_url = None

        self.progress_widget = Progress(show_percents=True, hide_on_finish=True)
        self.download_error = Text("", status="warning")
        self.download_error.hide()
        self.progress_container = Container(widgets=[self.download_error, self.progress_widget])
        self.loading_container = Container(widgets=[self.download_error, Text("Loading...")])

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

        self.locked_text = Text("Select input and model to unlock", status="info")
        self.empty_text = Text("Click preview to visualize predictions")
        self.error_text = Text("Failed to generate preview", status="error")

        self.select = Select(
            items=[
                Select.Item("locked", content=self.locked_text),
                Select.Item("empty", content=self.empty_text),
                Select.Item(ProjectType.IMAGES.value, content=self.image_preview_container),
                Select.Item(ProjectType.VIDEOS.value, content=self.video_preview_container),
                Select.Item("error", content=self.error_text),
                Select.Item("loading", content=self.loading_container),
                Select.Item("progress", content=self.progress_container),
            ]
        )
        self.select.set_value("empty")
        self.oneof = OneOf(self.select)

        self.run_button = Button("Preview", icon="zmdi zmdi-slideshow")
        self.run_button.disable()
        self.card = Card(
            title="Preview",
            description="Preview model predictions on a random image or video from the selected input source.",
            content=self.oneof,
            content_top_right=self.run_button,
            lock_message=self.lock_message,
        )

        @self.run_button.click
        def _run_preview():
            self.run_preview()

    def lock(self):
        self.run_button.disable()
        self.card.lock(self.lock_message)

    def unlock(self):
        self.run_button.enable()
        self.card.unlock()

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

    def _download_video_by_frames(
        self, video_info: VideoInfo, save_path: str, frames_number=150, progress_cb=None
    ):
        if Path(save_path).exists():
            Path(save_path).unlink()
        tmp_dir = Path(self.preview_dir, "tmp_frames")
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        os.makedirs(tmp_dir, exist_ok=True)
        self.api.video.download_frames(
            video_info.id,
            frames=list(range(frames_number)),
            paths=[str(tmp_dir / f"frame_{i}.jpg") for i in range(frames_number)],
            progress_cb=progress_cb,
        )
        fps = int(video_info.frames_count / video_info.duration)
        fourcc = cv2.VideoWriter.fourcc(*"mp4v")  # or 'avc1', 'XVID', 'H264'
        out = cv2.VideoWriter(
            save_path, fourcc, fps, (video_info.frame_width, video_info.frame_height)
        )
        for i in range(frames_number):
            frame_path = tmp_dir / f"frame_{i}.jpg"
            if not frame_path.exists():
                continue
            img = cv2.imread(str(frame_path))
            out.write(img)
        out.release()
        shutil.rmtree(tmp_dir)

    def _download_full_video(
        self, video_id: int, save_path: str, duration: int = 5, progress_cb=None
    ):
        if Path(save_path).exists():
            Path(save_path).unlink()
        temp = Path(self.preview_dir) / f"temp_{video_id}.mp4"
        if temp.exists():
            temp.unlink()
        self.api.video.download_path(video_id, temp, progress_cb=progress_cb)
        minutes = duration // 60
        hours = minutes // 60
        minutes = minutes % 60
        seconds = duration % 60
        duration_str = f"{hours:02}:{minutes:02}:{seconds:02}"
        try:
            process = subprocess.Popen(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(temp),
                    "-c",
                    "copy",
                    "-t",
                    duration_str,
                    save_path,
                ],
                stderr=subprocess.PIPE,
            )
            process.wait()
            logger.debug("FFmpeg exited with code: " + str(process.returncode))
            logger.debug(f"FFmpeg stderr: {process.stderr.read().decode()}")
            if len(VideoFrameReader(save_path).read_frames()) == 0:
                raise RuntimeError("No frames read from the video")
            temp.unlink()
        except Exception as e:
            if Path(save_path).exists():
                Path(save_path).unlink()
            shutil.copy(temp, save_path)
            temp.unlink()
            logger.warning(f"FFmpeg trimming failed: {str(e)}", exc_info=True)

    def _download_video_preview(self, video_info: VideoInfo, with_progress=True):
        video_id = video_info.id
        duration = 5
        video_path = Path(self.preview_dir, video_info.name)
        self.video_preview_path = video_path
        self.video_preview_annotated_path = Path(self.preview_dir, "annotated") / Path(
            self.video_preview_path
        ).relative_to(self.preview_dir)
        success = False
        try:
            try:
                size = int(video_info.file_meta["size"])
                size = int(size / video_info.duration * duration)
            except:
                size = None
            with (
                self.progress("Downloading video part:", total=size, unit="B", unit_scale=True)
                if with_progress and size
                else nullcontext()
            ) as pbar:
                success = self._partial_download(
                    video_id, duration, str(self.video_preview_path), progress_cb=pbar.update
                )
        except Exception as e:
            logger.warning(f"Partial download failed: {str(e)}", exc_info=True)
            success = False
        if success:
            return

        video_length_threshold = 120  # seconds
        if video_info.duration > video_length_threshold:
            self.download_error.text = (
                f"Partial download failed. Will Download separate video frames"
            )
            self.download_error.show()

            fps = int(video_info.frames_count / video_info.duration)
            frames_number = min(video_info.frames_count, int(fps * duration))
            with (
                self.progress(
                    "Downloading video frames:", total=frames_number, unit="it", unit_scale=False
                )
                if with_progress
                else nullcontext()
            ) as pbar:
                self._download_video_by_frames(
                    video_info,
                    str(self.video_preview_path),
                    frames_number=frames_number,
                    progress_cb=pbar.update,
                )
        else:
            self.download_error.text = f"Partial download failed. Will Download full video"
            self.download_error.show()
            size = int(video_info.file_meta["size"])
            with (
                self.progress("Downloading video:", total=size, unit="B", unit_scale=True)
                if with_progress
                else nullcontext()
            ) as pbar:
                self._download_full_video(
                    video_info.id,
                    str(self.video_preview_path),
                    duration=duration,
                    progress_cb=pbar.update,
                )

    def _partial_download(self, video_id: int, duration: int, save_path: str, progress_cb=None):
        if Path(save_path).exists():
            Path(save_path).unlink()
        duration_minutes = duration // 60
        duration_hours = duration_minutes // 60
        duration_minutes = duration_minutes % 60
        duration_seconds = duration % 60
        duration_str = f"{duration_hours:02}:{duration_minutes:02}:{duration_seconds:02}"
        response = self.api.video._download(video_id, is_stream=True)
        process = subprocess.Popen(
            [
                "ffmpeg",
                "-y",
                "-t",
                duration_str,
                "-probesize",
                "50M",
                "-analyzeduration",
                "50M",
                "-i",
                "pipe:0",
                "-movflags",
                "frag_keyframe+empty_moov+default_base_moof",
                "-c",
                "copy",
                save_path,
            ],
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        bytes_written = 0
        try:
            for chunk in response.iter_content(chunk_size=8192):
                process.stdin.write(chunk)
                bytes_written += len(chunk)
                if progress_cb:
                    progress_cb(len(chunk))
        except (BrokenPipeError, IOError):
            logger.debug("FFmpeg process closed the pipe, stopping download.", exc_info=True)
            pass
        finally:
            process.stdin.close()
            process.wait()
            response.close()
            logger.debug("FFmpeg exited with code: " + str(process.returncode))
            logger.debug(f"FFmpeg stderr: {process.stderr.read().decode()}")
            logger.debug(f"Total bytes written: {bytes_written}")
        try:
            with VideoFrameReader(save_path) as reader:
                if len(reader.read_frames()) == 0:
                    return False
            return True
        except Exception as e:
            return False

    def _download_preview_item(self, with_progress: bool = False):
        input_settings = self.get_input_settings_fn()
        video_ids = input_settings.get("video_ids", None)
        if video_ids is None:
            project_id = input_settings.get("project_id", None)
            dataset_ids = input_settings.get("dataset_ids", None)
            if dataset_ids:
                images = []
                candidate_ids = list(dataset_ids)
                random.shuffle(candidate_ids)
                dataset_id = None
                for ds_id in candidate_ids:
                    images = self.api.image.get_list(ds_id)
                    if images:
                        dataset_id = ds_id
                        break
                if not images:
                    raise RuntimeError("No images found in the selected datasets")
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
            self.video_preview_annotated_path = None
        else:
            video_id = random.choice(video_ids)
            video_id = video_ids[0]
            video_info = self.api.video.get_info_by_id(video_id)
            self._download_video_preview(video_info, with_progress)
            self._current_item_id = video_id
            self.video_peview_url = f"/static/preview/annotated/{video_info.name}"

    def set_image_preview(self):
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
        model_api = self.get_model_api_fn()
        model_meta = model_api.get_model_meta()
        src_project_meta = ProjectMeta.from_json(self.api.project.get_meta(video_info.project_id))
        project_meta = src_project_meta.merge(model_meta)

        settings = self.get_settings_fn()
        inference_settings = settings.get("inference_settings", {})
        tracking = settings.get("tracking", False)
        with self.progress("Running model:", total=frames_number) as pbar:
            with model_api.predict_detached(
                video_id=video_id,
                inference_settings=inference_settings,
                tracking=tracking,
                start_frame=frame_start,
                num_frames=frames_number,
                tqdm=pbar,
            ) as session:
                predictions: List[Prediction] = list(session)

        if os.path.exists(self.video_preview_annotated_path):
            os.remove(self.video_preview_annotated_path)
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
                output_path=self.video_preview_annotated_path,
            )
        else:
            tmp_path = str(self.video_preview_path / "tmp")
            video_writer = cv2.VideoWriter(
                tmp_path,
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
            shutil.move(tmp_path, self.video_preview_annotated_path)
        self.video_player.set_video(self.video_peview_url)
        self.select_item(ProjectType.VIDEOS.value)

    def set_error(self, text: str):
        self.error_text.text = text
        self.select_item("error")

    def run_preview(self):
        self.download_error.hide()
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
        self.download_error.hide()
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
        self.inference_settings_editor = None
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
        self.inference_settings_editor = Editor("", language_mode="yaml", height_px=300)
        # Add widgets to display ------------ #
        self.settings_widgets.extend([self.inference_settings_editor])
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
        self.cards = [self.settings_card, self.preview.card]
        self.cards_container = Container(
            widgets=self.cards,
            gap=15,
            direction="horizontal",
            fractions=[3, 7],
        )
        # ----------------------------------- #

    def lock(self):
        self.settings_card.lock(self.lock_message)
        self.preview.lock()

    def unlock(self):
        self.settings_card.unlock()
        self.preview.unlock()

    def disable(self):
        for widget in self.widgets_to_disable:
            widget.disable()

    def enable(self):
        for widget in self.widgets_to_disable:
            widget.enable()

    @property
    def widgets_to_disable(self) -> List[Widget]:
        return [
            # self.inference_mode_selector,
            self.model_prediction_suffix_input,
            # self.model_prediction_suffix_checkbox,
            self.predictions_mode_selector,
            self.inference_settings_editor,
        ]

    def set_inference_settings(self, settings: Dict[str, Any]):
        settings = "# Inference settings\n" + settings
        if isinstance(settings, str):
            self.inference_settings_editor.set_text(settings)
        else:
            self.inference_settings_editor.set_text(yaml.safe_dump(settings))

    def set_tracking_settings(self, settings: Dict[str, Any]):
        if self.input_selector.radio.get_value() != ProjectType.VIDEOS.value:
            return

        current_settings = self.inference_settings_editor.get_text()
        if isinstance(settings, str):
            all_settings = current_settings + "\n\n# Tracking settings\n" + settings
            self.inference_settings_editor.set_text(all_settings)
        else:
            all_settings = current_settings + "\n\n# Tracking settings\n" + yaml.safe_dump(settings)
            self.inference_settings_editor.set_text(all_settings)

    def get_inference_settings(self) -> Dict:
        text = self.inference_settings_editor.get_text()
        inference_settings_text = text.split("# Tracking settings")[0]
        settings = yaml.safe_load(inference_settings_text)
        if settings:
            return settings
        return {}

    def get_tracking_settings(self) -> Dict:
        if self.input_selector.radio.get_value() != ProjectType.VIDEOS.value:
            return {}

        text = self.inference_settings_editor.get_text()
        text_parts = text.split("# Tracking settings")
        if len(text_parts) > 1:
            tracking_settings_text = text_parts[1]
            settings = yaml.safe_load(tracking_settings_text)
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
        if self.input_selector.radio.get_value() == ProjectType.VIDEOS.value:
            settings["tracking_settings"] = self.get_tracking_settings()
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

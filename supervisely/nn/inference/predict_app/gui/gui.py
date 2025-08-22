import random
import time
from typing import Any, Callable, Dict, List, Optional

import yaml

from supervisely._utils import is_development, logger
from supervisely.annotation.annotation import Annotation
from supervisely.annotation.label import Label
from supervisely.api.api import Api
from supervisely.api.video.video_api import VideoInfo
from supervisely.app.widgets import Button, Card, Container, Stepper, Widget
from supervisely.io import env
from supervisely.nn.inference.predict_app.gui.classes_selector import ClassesSelector
from supervisely.nn.inference.predict_app.gui.input_selector import InputSelector
from supervisely.nn.inference.predict_app.gui.model_selector import ModelSelector
from supervisely.nn.inference.predict_app.gui.output_selector import OutputSelector
from supervisely.nn.inference.predict_app.gui.preview import Preview
from supervisely.nn.inference.predict_app.gui.settings_selector import (
    AddPredictionsMode,
    SettingsSelector,
)
from supervisely.nn.inference.predict_app.gui.tags_selector import TagsSelector
from supervisely.nn.inference.predict_app.gui.utils import (
    copy_project,
    disable_enable,
    set_stepper_step,
    wrap_button_click,
)
from supervisely.nn.model.model_api import ModelAPI
from supervisely.nn.model.prediction import Prediction
from supervisely.project.project_meta import ProjectMeta
from supervisely.video_annotation.key_id_map import KeyIdMap
from supervisely.video_annotation.video_annotation import VideoAnnotation


class StepFlow:

    def __init__(self, stepper: Stepper):
        self.stepper = stepper
        self.steps = {}
        self.step_sequence = []

    def register_step(
        self,
        name: str,
        card: Card,
        button: Optional[Button] = None,
        widgets_to_disable: Optional[List[Widget]] = None,
        validation_text: Optional[Widget] = None,
        validation_func: Optional[Callable] = None,
        position: Optional[int] = None,
    ) -> "StepFlow":
        self.steps[name] = {
            "card": card,
            "button": button,
            "widgets_to_disable": widgets_to_disable or [],
            "validation_text": validation_text,
            "validation_func": validation_func,
            "position": position,
            "next_steps": [],
            "on_select_click": [],
            "on_reselect_click": [],
            "wrapper": None,
            "has_button": button is not None,
        }

        if position is not None:
            while len(self.step_sequence) <= position:
                self.step_sequence.append(None)
            self.step_sequence[position] = name

        return self

    def set_next_steps(self, step_name: str, next_steps: List[str]) -> "StepFlow":
        if step_name in self.steps:
            self.steps[step_name]["next_steps"] = next_steps
        return self

    def add_on_select_actions(
        self, step_name: str, actions: List[Callable], is_reselect: bool = False
    ) -> "StepFlow":
        if step_name in self.steps:
            key = "on_reselect_click" if is_reselect else "on_select_click"
            self.steps[step_name][key].extend(actions)
        return self

    def build_wrappers(self) -> Dict[str, Callable]:
        valid_sequence = [s for s in self.step_sequence if s is not None and s in self.steps]

        for step_name in reversed(valid_sequence):
            step = self.steps[step_name]

            cards_to_unlock = []
            for next_step_name in step["next_steps"]:
                if next_step_name in self.steps:
                    cards_to_unlock.append(self.steps[next_step_name]["card"])

            callback = None
            if step["next_steps"] and step["has_button"]:
                for next_step_name in step["next_steps"]:
                    if (
                        next_step_name in self.steps
                        and self.steps[next_step_name].get("wrapper")
                        and self.steps[next_step_name]["has_button"]
                    ):
                        callback = self.steps[next_step_name]["wrapper"]
                        break

            if step["has_button"]:
                wrapper = wrap_button_click(
                    button=step["button"],
                    cards_to_unlock=cards_to_unlock,
                    widgets_to_disable=step["widgets_to_disable"],
                    callback=callback,
                    validation_text=step["validation_text"],
                    validation_func=step["validation_func"],
                    on_select_click=step["on_select_click"],
                    on_reselect_click=step["on_reselect_click"],
                    collapse_card=None,
                )

                step["wrapper"] = wrapper

        return {
            name: self.steps[name]["wrapper"]
            for name in self.steps
            if self.steps[name].get("wrapper") and self.steps[name]["has_button"]
        }

    def setup_button_handlers(self) -> None:
        positions = {}
        pos = 1

        for i, step_name in enumerate(self.step_sequence):
            if step_name is not None and step_name in self.steps:
                positions[step_name] = pos
                pos += 1

        for step_name, step in self.steps.items():
            if step_name in positions and step.get("wrapper") and step["has_button"]:

                button = step["button"]
                wrapper = step["wrapper"]
                position = positions[step_name]
                next_position = position + 1

                def create_handler(btn, cb, next_pos):
                    def handler():
                        cb()
                        set_stepper_step(self.stepper, btn, next_pos=next_pos)

                    return handler

                button.click(create_handler(button, wrapper, next_position))

    def build(self) -> Dict[str, Callable]:
        wrappers = self.build_wrappers()
        self.setup_button_handlers()
        return wrappers


class PredictAppGui:
    def __init__(self, api: Api, static_dir: str = "static"):
        self.api = api
        self.static_dir = static_dir

        # Environment variables
        self.team_id = env.team_id()
        self.workspace_id = env.workspace_id()
        self.project_id = env.project_id(raise_not_found=False)
        # -------------------------------- #

        # Flags
        self._stop_flag = False
        self._is_running = False
        # -------------------------------- #

        # GUI
        # Steps
        self.steps = []

        # 1. Input selector
        self.input_selector = InputSelector(self.workspace_id)
        self.steps.append(self.input_selector.card)

        # 2. Model selector
        self.model_selector = ModelSelector(self.api, self.team_id)
        self.steps.append(self.model_selector.card)

        # 3. Classes selector
        self.classes_selector = None
        if True:
            self.classes_selector = ClassesSelector()
            self.steps.append(self.classes_selector.card)

        # 4. Tags selector
        self.tags_selector = None
        if False:
            self.tags_selector = TagsSelector()
            self.steps.append(self.tags_selector.card)

        # 5. Settings selector
        self.settings_selector = SettingsSelector()
        self.steps.append(self.settings_selector.card)

        # 6. Preview
        self.preview = None
        if False:
            self.preview = Preview(api, static_dir)
            self.steps.append(self.preview.card)

        # 7. Output selector
        self.output_selector = OutputSelector(self.api)
        self.steps.append(self.output_selector.card)
        # -------------------------------- #

        # Stepper
        self.stepper = Stepper(widgets=self.steps)
        # ---------------------------- #

        # Layout
        self.layout = Container([self.stepper])
        # ---------------------------- #

        # Button Utils
        def deploy_model() -> ModelAPI:
            self.model_selector.validator_text.hide()
            model_api = None
            try:
                model_api = type(self.model_selector.model).deploy(self.model_selector.model)
            except:
                self.output_selector.start_button.disable()
                raise
            else:
                self.output_selector.start_button.enable()
            return model_api

        # Reimplement deploy method for DeployModel widget
        self.model_selector.model.deploy = deploy_model

        def set_entity_meta():
            model_api = self.model_selector.model.model_api

            model_meta = model_api.get_model_meta()
            if self.classes_selector is not None:
                self.classes_selector.classes_table.set_project_meta(model_meta)
                self.classes_selector.classes_table.show()
            if self.tags_selector is not None:
                self.tags_selector.tags_table.set_project_meta(model_meta)
                self.tags_selector.tags_table.show()

            inference_settings = model_api.get_settings()
            self.settings_selector.set_inference_settings(inference_settings)

            if self.preview is not None:
                self.preview.inference_settings = inference_settings

        def reset_entity_meta():
            empty_meta = ProjectMeta()
            if self.classes_selector is not None:
                self.classes_selector.classes_table.set_project_meta(empty_meta)
                self.classes_selector.classes_table.hide()
            if self.tags_selector is not None:
                self.tags_selector.tags_table.set_project_meta(empty_meta)
                self.tags_selector.tags_table.hide()

            self.settings_selector.set_inference_settings("")

            if self.preview is not None:
                self.preview.inference_settings = None

        def disable_settings_editor():
            if self.settings_selector.inference_settings.readonly:
                self.settings_selector.inference_settings.readonly = False
            else:
                self.settings_selector.inference_settings.readonly = True

        def generate_preview():
            def _get_frame_annotation(
                video_info: VideoInfo, frame_index: int, project_meta: ProjectMeta
            ) -> Annotation:
                video_annotation = VideoAnnotation.from_json(
                    self.api.video.annotation.download(video_info.id, frame_index),
                    project_meta=project_meta,
                    key_id_map=KeyIdMap(),
                )
                frame = video_annotation.frames.get(frame_index)
                img_size = (video_info.frame_height, video_info.frame_width)
                if frame is None:
                    return Annotation(img_size)
                labels = []
                for figure in frame.figures:
                    labels.append(Label(figure.geometry, figure.video_object.obj_class))
                ann = Annotation(img_size, labels=labels)
                return ann

            if self.preview is None:
                return

            self.preview.validator_text.hide()
            self.preview.gallery.clean_up()
            self.preview.gallery.show()
            self.preview.gallery.loading = True
            try:
                items_settings = self.input_selector.get_settings()
                if "video_id" in items_settings:
                    video_id = items_settings["video_id"]
                    video_info = self.api.video.get_info_by_id(video_id)
                    video_frame = random.randint(0, video_info.frames_count - 1)
                    self.api.video.frame.download_path(
                        video_info.id, video_frame, self.preview.preview_path
                    )
                    img_url = self.preview.peview_url
                    project_meta = ProjectMeta.from_json(
                        self.api.project.get_meta(video_info.project_id)
                    )
                    input_ann = _get_frame_annotation(video_info, video_frame, project_meta)
                    prediction = self.model_selector.model.model_api.predict(
                        input=self.preview.preview_path, **self.settings_selector.get_settings()
                    )[0]
                    output_ann = prediction.annotation
                else:
                    if "project_id" in items_settings:
                        project_id = items_settings["project_id"]
                        dataset_infos = self.api.dataset.get_list(project_id, recursive=True)
                        dataset_infos = [ds for ds in dataset_infos if ds.items_count > 0]
                        if not dataset_infos:
                            raise ValueError("No datasets with items found in the project.")
                        dataset_info = random.choice(dataset_infos)
                    elif "dataset_ids" in items_settings:
                        dataset_ids = items_settings["dataset_ids"]
                        dataset_infos = [
                            self.api.dataset.get_info_by_id(dataset_id)
                            for dataset_id in dataset_ids
                        ]
                        dataset_infos = [ds for ds in dataset_infos if ds.items_count > 0]
                        if not dataset_infos:
                            raise ValueError("No items in selected datasets.")
                        dataset_info = random.choice(dataset_infos)
                    else:
                        raise ValueError("No valid item settings found for preview.")
                    images = self.api.image.get_list(dataset_info.id)
                    image_info = random.choice(images)
                    img_url = image_info.preview_url

                    project_meta = ProjectMeta.from_json(
                        self.api.project.get_meta(dataset_info.project_id)
                    )
                    input_ann = Annotation.from_json(
                        self.api.annotation.download(image_info.id).annotation,
                        project_meta=project_meta,
                    )
                    prediction = self.model_selector.model.model_api.predict(
                        image_id=image_info.id, **self.settings_selector.get_settings()
                    )[0]
                    output_ann = prediction.annotation

                self.preview.gallery.append(img_url, input_ann, "Input")
                self.preview.gallery.append(img_url, output_ann, "Output")
                self.preview.validator_text.hide()
                self.preview.gallery.show()
                return prediction
            except Exception as e:
                self.preview.gallery.hide()
                self.preview.validator_text.set(
                    text=f"Error during preview: {str(e)}", status="error"
                )
                self.preview.validator_text.show()
                self.preview.gallery.clean_up()
            finally:
                self.preview.gallery.loading = False

        # ---------------------------- #

        # StepFlow callbacks and wiring
        self.step_flow = StepFlow(self.stepper)
        position = 0

        # 1. Input selector
        self.step_flow.register_step(
            "input_selector",
            self.input_selector.card,
            self.input_selector.button,
            self.input_selector.widgets_to_disable,
            self.input_selector.validator_text,
            self.input_selector.validate_step,
            position=position,
        )
        position += 1

        # 2. Model selector
        self.step_flow.register_step(
            "model_selector",
            self.model_selector.card,
            self.model_selector.button,
            self.model_selector.widgets_to_disable,
            self.model_selector.validator_text,
            self.model_selector.validate_step,
            position=position,
        )
        self.step_flow.add_on_select_actions("model_selector", [set_entity_meta])
        self.step_flow.add_on_select_actions(
            "model_selector", [reset_entity_meta], is_reselect=True
        )
        position += 1

        # 3. Classes selector
        if self.classes_selector is not None:
            self.step_flow.register_step(
                "classes_selector",
                self.classes_selector.card,
                self.classes_selector.button,
                self.classes_selector.widgets_to_disable,
                self.classes_selector.validator_text,
                self.classes_selector.validate_step,
                position=position,
            )
            position += 1

        # 4. Tags selector
        if self.tags_selector is not None:
            self.step_flow.register_step(
                "tags_selector",
                self.tags_selector.card,
                self.tags_selector.button,
                self.tags_selector.widgets_to_disable,
                self.tags_selector.validator_text,
                self.tags_selector.validate_step,
                position=position,
            )
            position += 1

        # 5. Settings selector
        self.step_flow.register_step(
            "settings_selector",
            self.settings_selector.card,
            self.settings_selector.button,
            self.settings_selector.widgets_to_disable,
            self.settings_selector.validator_text,
            self.settings_selector.validate_step,
            position=position,
        )
        self.step_flow.add_on_select_actions("settings_selector", [disable_settings_editor])
        self.step_flow.add_on_select_actions("settings_selector", [disable_settings_editor], True)
        position += 1

        # 6. Preview
        if self.preview is not None:
            self.step_flow.register_step(
                "preview",
                self.preview.card,
                self.preview.button,
                self.preview.widgets_to_disable,
                self.preview.validator_text,
                self.preview.validate_step,
                position=position,
            ).add_on_select_actions("preview", [generate_preview])
            position += 1

        # 7. Output selector
        self.step_flow.register_step(
            "output_selector",
            self.output_selector.card,
            None,
            self.output_selector.widgets_to_disable,
            self.output_selector.validator_text,
            self.output_selector.validate_step,
            position=position,
        )

        # Dependencies Chain
        has_model_selector = self.model_selector is not None
        has_classes_selector = self.classes_selector is not None
        has_tags_selector = self.tags_selector is not None
        has_preview = self.preview is not None

        # Step 1 -> Step 2
        prev_step = "input_selector"
        if has_model_selector:
            self.step_flow.set_next_steps(prev_step, ["model_selector"])
            prev_step = "model_selector"
        # Step 2 -> Step 3
        if has_classes_selector:
            self.step_flow.set_next_steps(prev_step, ["classes_selector"])
            prev_step = "classes_selector"
        # Step 3 -> Step 4
        if has_tags_selector:
            self.step_flow.set_next_steps(prev_step, ["tags_selector"])
            prev_step = "tags_selector"
        # Step 4 -> Step 5
        self.step_flow.set_next_steps(prev_step, ["settings_selector"])
        prev_step = "settings_selector"
        # Step 5 -> Step 6
        if has_preview:
            self.step_flow.set_next_steps(prev_step, ["preview"])
            prev_step = "preview"
        # Step 6 -> Step 7
        self.step_flow.set_next_steps(prev_step, ["output_selector"])

        # Create all wrappers and set button handlers
        wrappers = self.step_flow.build()

        self.input_selector_cb = wrappers.get("input_selector")
        self.classes_selector_cb = wrappers.get("classes_selector")
        self.tags_selector_cb = wrappers.get("tags_selector")
        self.model_selector_cb = wrappers.get("model_selector")
        self.settings_selector_cb = wrappers.get("settings_selector")
        self.preview_cb = wrappers.get("preview")
        self.output_selector_cb = wrappers.get("output_selector")
        # ------------------------------------------------- #

        # Other Handlers
        @self.input_selector.radio.value_changed
        def input_selector_type_changed(value: str):
            self.input_selector.validator_text.hide()

        @self.input_selector.select_dataset_for_video.value_changed
        def dataset_for_video_changed(dataset_id: int):
            self.input_selector.select_video.loading = True
            if dataset_id is None:
                rows = []
            else:
                dataset_info = self.api.dataset.get_info_by_id(dataset_id)
                videos = self.api.video.get_list(dataset_id)
                rows = [[video.id, video.name, dataset_info.name] for video in videos]
            self.input_selector.select_video.rows = rows
            self.input_selector.select_video.loading = False

        # ------------------------------------------------- #

    def run(self, run_parameters: Dict[str, Any] = None) -> List[Prediction]:
        self.show_validator_text()
        self.set_validator_text("Preparing settings for prediction...", "info")
        if run_parameters is None:
            run_parameters = self.get_run_parameters()

        if self.model_selector.model.model_api is None:
            self.model_selector.model._deploy()

        model_api = self.model_selector.model.model_api
        if model_api is None:
            logger.error("Model Deployed with an error")
            self.set_validator_text("Model Deployed with an error", "error")
            return

        kwargs = {}

        # Input
        # Input would be newely created project
        input_parameters = run_parameters["input"]
        input_project_id = input_parameters.get("project_id", None)
        if input_project_id is None:
            raise ValueError("Input project ID is required for prediction.")
        input_dataset_ids = input_parameters.get("dataset_ids", [])
        if not input_dataset_ids:
            raise ValueError("At least one dataset must be selected for prediction.")

        # Settings
        settings = run_parameters["settings"]
        prediction_mode = settings.pop("predictions_mode")
        upload_mode = None
        with_annotations = None
        if prediction_mode == AddPredictionsMode.REPLACE_EXISTING_LABELS:
            upload_mode = "replace"
            with_annotations = False
        elif prediction_mode == AddPredictionsMode.MERGE_WITH_EXISTING_LABELS:
            upload_mode = "append"
            with_annotations = True
        elif prediction_mode == AddPredictionsMode.REPLACE_EXISTING_LABELS_AND_SAVE_IMAGE_TAGS:
            upload_mode = "replace"
            with_annotations = True
        kwargs.update(settings)
        kwargs["upload_mode"] = upload_mode

        # Classes
        classes = run_parameters["classes"]
        if classes:
            kwargs["classes"] = classes

        # Output
        # Always create new project
        # But the actual inference will happen inplace
        output_parameters = run_parameters["output"]
        project_name = output_parameters["project_name"]
        if not project_name:
            input_project_info = self.api.project.get_info_by_id(input_project_id)
            project_name = input_project_info.name + " [Predictions]"
            logger.warning("Project name is empty, using auto-generated name: " + project_name)

        # Copy project
        self.set_validator_text("Copying project...", "info")
        created_project = copy_project(
            self.api,
            project_name,
            self.workspace_id,
            input_project_id,
            input_dataset_ids,
            with_annotations,
            self.output_selector.progress,
        )
        # ------------------------ #

        # Run prediction
        self.set_validator_text("Running prediction...", "info")
        predictions = []
        self._is_running = True
        try:
            with model_api.predict_detached(
                project_id=created_project.id,
                tqdm=self.output_selector.progress(),
                **kwargs,
            ) as session:
                self.output_selector.progress.show()
                i = 0
                for prediction in session:
                    predictions.append(prediction)
                    i += 1
                    if self._stop_flag:
                        logger.info("Prediction stopped by user.")
                        break
            self.output_selector.progress.hide()
        except Exception as e:
            self.output_selector.progress.hide()
            logger.error(f"Error during prediction: {str(e)}")
            self.set_validator_text(f"Error during prediction: {str(e)}", "error")
            disable_enable(self.output_selector.widgets_to_disable, False)
            self._is_running = False
            self._stop_flag = False
            raise e
        finally:
            self._is_running = False
            self._stop_flag = False
        # ------------------------ #

        # Set result thumbnail
        self.set_validator_text("Project successfully processed", "success")
        self.output_selector.set_result_thumbnail(created_project.id)
        # ------------------------ #
        return predictions

    def stop(self):
        logger.info("Stopping prediction...")
        self._stop_flag = True

    def wait_for_stop(self, timeout: int = None):
        logger.info(
            "Waiting " + ""
            if timeout is None
            else f"{timeout} seconds " + "for prediction to stop..."
        )
        t = time.monotonic()
        while self._is_running:
            if timeout is not None and time.monotonic() - t > timeout:
                raise TimeoutError("Timeout while waiting for stop.")
            time.sleep(0.1)
        logger.info("Prediction stopped.")

    def shutdown_model(self):
        self.stop()
        self.wait_for_stop(10)
        self.model_selector.model.stop()

    def get_run_parameters(self) -> Dict[str, Any]:
        settings = {
            "model": self.model_selector.model.get_deploy_parameters(),
            "settings": self.settings_selector.get_settings(),
            "input": self.input_selector.get_settings(),
            "output": self.output_selector.get_settings(),
        }
        if self.classes_selector is not None:
            settings["classes"] = self.classes_selector.get_selected_classes()
        if self.tags_selector is not None:
            settings["tags"] = self.tags_selector.get_selected_tags()
        return settings

    def load_from_json(self, data):
        # 1. Input selector
        self.input_selector.load_from_json(data.get("input", {}))
        # self.input_selector_cb()

        # 2. Model selector
        self.model_selector.model.load_from_json(data.get("model", {}))

        # 3. Classes selector
        if self.classes_selector is not None:
            self.classes_selector.load_from_json(data.get("classes", {}))

        # 4. Tags selector
        if self.tags_selector is not None:
            self.tags_selector.load_from_json(data.get("tags", {}))

        # 5. Settings selector
        self.settings_selector.load_from_json(data.get("settings", {}))

        # 6. Preview (No need?)
        if self.preview is not None:
            self.preview.load_from_json(data.get("preview", {}))

        # 7. Output selector
        self.output_selector.load_from_json(data.get("output", {}))

    def set_validator_text(self, text: str, status: str = "text"):
        self.output_selector.validator_text.set(text=text, status=status)

    def show_validator_text(self):
        self.output_selector.validator_text.show()

    def hide_validator_text(self):
        self.output_selector.validator_text.hide()

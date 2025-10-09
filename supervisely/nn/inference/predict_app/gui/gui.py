import json
import random
import time
from typing import Any, Callable, Dict, List, Optional, Union

from supervisely._utils import is_development, logger
from supervisely.annotation.annotation import Annotation
from supervisely.annotation.label import Label
from supervisely.api.annotation_api import AnnotationInfo
from supervisely.api.api import Api
from supervisely.api.image_api import ImageInfo
from supervisely.api.video.video_api import VideoInfo
from supervisely.app.widgets import Button, Card, Container, Stepper, Widget
from supervisely.io import env
from supervisely.nn.inference.predict_app.gui.classes_selector import ClassesSelector
from supervisely.nn.inference.predict_app.gui.input_selector import InputSelector
from supervisely.nn.inference.predict_app.gui.model_selector import ModelSelector
from supervisely.nn.inference.predict_app.gui.output_selector import OutputSelector
from supervisely.nn.inference.predict_app.gui.settings_selector import (
    AddPredictionsMode,
    SettingsSelector,
)
from supervisely.nn.inference.predict_app.gui.tags_selector import TagsSelector
from supervisely.nn.inference.predict_app.gui.utils import (
    copy_items_to_project,
    copy_project,
    create_project,
    disable_enable,
    set_stepper_step,
    wrap_button_click,
)
from supervisely.nn.model.model_api import ModelAPI
from supervisely.nn.model.prediction import Prediction
from supervisely.project.project_meta import ProjectMeta
from supervisely.project.project_type import ProjectType
from supervisely.video_annotation.frame_collection import Frame, FrameCollection
from supervisely.video_annotation.key_id_map import KeyIdMap
from supervisely.video_annotation.video_annotation import (
    VideoAnnotation,
    VideoFigure,
    VideoObjectCollection,
)
from supervisely.video_annotation.video_figure import VideoObject


class StepFlow:

    def __init__(self, stepper: Stepper):
        self.stepper = stepper
        self.steps = {}
        self.step_sequence = []

    def register_step(
        self,
        name: str,
        card: Union[Card, List[Card]],
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
            visited = set()
            queue = list(step["next_steps"])
            while queue:
                nxt = queue.pop(0)
                if nxt in visited:
                    continue
                visited.add(nxt)
                if nxt in self.steps:
                    nxt_step = self.steps[nxt]
                    if isinstance(nxt_step["card"], list):
                        cards_to_unlock.extend(nxt_step["card"])
                    else:
                        cards_to_unlock.append(nxt_step["card"])
                    queue.extend(nxt_step.get("next_steps", []))

            callback = None
            if step["next_steps"] and step["has_button"]:
                visited = set()
                queue = list(step["next_steps"])
                while queue:
                    next_step_name = queue.pop(0)
                    if next_step_name in visited:
                        continue
                    visited.add(next_step_name)
                    if next_step_name in self.steps:
                        next_step = self.steps[next_step_name]
                        if next_step.get("wrapper") and next_step["has_button"]:
                            callback = next_step["wrapper"]
                            break
                        queue.extend(next_step.get("next_steps", []))

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

        # 5. Settings selector & Preview
        self.settings_selector = SettingsSelector(
            api=self.api,
            static_dir=self.static_dir,
            model_selector=self.model_selector,
            input_selector=self.input_selector,
        )
        self.steps.append(self.settings_selector.cards_container)

        # 6. Output selector
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
                self.classes_selector.set_project_meta(model_meta)
                self.classes_selector.classes_table.show()
            if self.tags_selector is not None:
                self.tags_selector.tags_table.set_project_meta(model_meta)
                self.tags_selector.tags_table.show()

            inference_settings = model_api.get_settings()
            self.settings_selector.set_inference_settings(inference_settings)

            if self.input_selector.radio.get_value() == ProjectType.VIDEOS.value:
                tracking_settings = model_api.get_tracking_settings()
                self.settings_selector.set_tracking_settings(tracking_settings)

        def reset_entity_meta():
            empty_meta = ProjectMeta()
            if self.classes_selector is not None:
                self.classes_selector.set_project_meta(empty_meta)
                self.classes_selector.classes_table.hide()
            if self.tags_selector is not None:
                self.tags_selector.tags_table.set_project_meta(empty_meta)
                self.tags_selector.tags_table.hide()

            self.settings_selector.set_inference_settings("")

        def disable_settings_editor():
            if self.settings_selector.inference_settings.readonly:
                self.settings_selector.inference_settings.readonly = False
            else:
                self.settings_selector.inference_settings.readonly = True

        def enable_preview_button():
            self.settings_selector.preview.run_button.enable()

        def disable_preview_button():
            self.settings_selector.preview.run_button.disable()
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
        self.step_flow.add_on_select_actions("input_selector", [self.update_item_type])
        self.step_flow.add_on_select_actions("input_selector", [disable_preview_button])
        self.step_flow.add_on_select_actions("input_selector", [disable_preview_button], True)

        # 2. Model selector
        self.step_flow.register_step(
            "model_selector",
            self.model_selector.card,
            None,
            self.model_selector.widgets_to_disable,
            self.model_selector.validator_text,
            self.model_selector.validate_step,
            position=position,
        )
        self.step_flow.add_on_select_actions("model_selector", [disable_preview_button])
        self.step_flow.add_on_select_actions("model_selector", [disable_preview_button], True)

        current_position = position + 1

        def deploy_and_set_step():
            model_api = type(self.model_selector.model).deploy(self.model_selector.model)
            if model_api is not None:
                self.step_flow.stepper.set_active_step(current_position + 1)
                set_entity_meta()
                # @TODO: move to def connect and def deploy
                # So card unlocks only after stop and disconnect buttons appear
                self.classes_selector.card.unlock()
            else:
                self.step_flow.stepper.set_active_step(current_position)
                reset_entity_meta()
                # @TODO: move to def connect and def deploy
                # So card locks only after stop and disconnect buttons appear
                self.classes_selector.card.lock() 
            return model_api

        def stop_and_reset_step():
            type(self.model_selector.model).stop(self.model_selector.model)
            self.step_flow.stepper.set_active_step(current_position)
            reset_entity_meta()
            self.classes_selector.card.lock()

        def disconnect_and_reset_step():
            type(self.model_selector.model).disconnect(self.model_selector.model)
            self.step_flow.stepper.set_active_step(current_position)
            reset_entity_meta()
            self.classes_selector.card.lock()

        self.model_selector.model.deploy = deploy_and_set_step
        self.model_selector.model.stop = stop_and_reset_step
        self.model_selector.model.disconnect = disconnect_and_reset_step
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

        # Enable preview button after selecting tags if tags selector is present
        if self.tags_selector is not None:
            self.step_flow.add_on_select_actions("tags_selector", [enable_preview_button])
            self.step_flow.add_on_select_actions("tags_selector", [disable_preview_button], True)
            if self.classes_selector is not None:
                self.step_flow.add_on_select_actions("classes_selector", [disable_preview_button])
                self.step_flow.add_on_select_actions("classes_selector", [disable_preview_button], True)
        # Enable preview button after selecting classes
        else:
            self.step_flow.add_on_select_actions("classes_selector", [enable_preview_button])
            self.step_flow.add_on_select_actions("classes_selector", [disable_preview_button], True)

        # 5. Settings selector & Preview
        self.step_flow.register_step(
            "settings_selector",
            self.settings_selector.cards,
            self.settings_selector.button,
            self.settings_selector.widgets_to_disable,
            self.settings_selector.validator_text,
            self.settings_selector.validate_step,
            position=position,
        )
        self.step_flow.add_on_select_actions("settings_selector", [disable_settings_editor])
        self.step_flow.add_on_select_actions("settings_selector", [disable_settings_editor], True)
        self.step_flow.add_on_select_actions("settings_selector", [enable_preview_button], True)
        position += 1

        # 6. Output selector
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

        @self.input_selector.select_dataset_for_video.project_changed
        def project_for_videos_changed(project_id: int):
            self.input_selector.select_video.clear()
            self.input_selector.select_video.hide()

        @self.input_selector.select_dataset_for_video.value_changed
        def dataset_for_video_changed(dataset_id: int):
            self.input_selector.select_video.loading = True
            self.input_selector.select_video.clear()
            if dataset_id is None:
                self.input_selector.select_video.hide()
            else:
                self.input_selector.select_video.show()
                dataset_info = self.api.dataset.get_info_by_id(dataset_id)
                videos = self.api.video.get_list(dataset_id)
                for video in videos:
                    size = f"{video.frame_height}x{video.frame_width}"
                    try:
                        frame_rate = int(video.frames_count / video.duration)
                    except:
                        frame_rate = "N/A"
                    self.input_selector.select_video.insert_row(
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
            self.input_selector.select_video.loading = False

        # ------------------------------------------------- #

    @property
    def model_api(self) -> Optional[ModelAPI]:
        return self.model_selector.model.model_api

    def update_item_type(self):
        item_type = self.input_selector.radio.get_value()
        self.settings_selector.update_item_type(item_type)
        self.output_selector.update_item_type(item_type)

    def _run_videos(self, run_parameters: Dict[str, Any]) -> List[Prediction]:
        if self.model_api is None:
            self.set_validator_text("Deploying model...", "info")
            self.model_selector.model._deploy()
        if self.model_api is None:
            logger.error("Model Deployed with an error")
            raise RuntimeError("Model Deployed with an error")

        self.set_validator_text("Preparing settings for prediction...", "info")
        if run_parameters is None:
            run_parameters = self.get_run_parameters()

        input_parameters = run_parameters["input"]
        input_video_ids = input_parameters["video_ids"]
        if not input_video_ids:
            raise ValueError("No video IDs provided for video prediction.")

        predict_kwargs = {}
        # Settings
        settings = run_parameters["settings"]
        prediction_mode = settings.pop("predictions_mode")
        tracking = settings.pop("tracking", False)
        predict_kwargs.update(settings)

        # Classes
        classes = run_parameters["classes"]
        if classes:
            predict_kwargs["classes"] = classes

        output_parameters = run_parameters["output"]
        project_name = output_parameters.get("project_name", "")
        upload_to_source_project = output_parameters.get("upload_to_source_project", False)
        skip_project_versioning = output_parameters.get("skip_project_versioning", False)
        skip_annotated = output_parameters.get("skip_annotated", False)

        video_infos_by_project_id: Dict[int, List[VideoInfo]] = {}
        video_infos_by_dataset_id: Dict[int, List[VideoInfo]] = {}
        for info in self.api.video.get_info_by_id_batch(input_video_ids):
            video_infos_by_project_id.setdefault(info.project_id, []).append(info)
            video_infos_by_dataset_id.setdefault(info.dataset_id, []).append(info)
        src_project_metas: Dict[int, ProjectMeta] = {}
        for project_id in video_infos_by_project_id.keys():
            src_project_metas[project_id] = ProjectMeta.from_json(
                self.api.project.get_meta(project_id)
            )

        video_ids_to_skip = set()
        if skip_annotated:
            self.set_validator_text("Checking for already annotated videos...", "info")
            secondary_pbar = self.output_selector.secondary_progress(
                message="Checking for already annotated videos...", total=len(input_video_ids)
            )
            self.output_selector.secondary_progress.show()
            for dataset_id, video_infos in video_infos_by_dataset_id.items():
                annotations = self.api.video.annotation.download_bulk(
                    dataset_id, [info.id for info in video_infos]
                )
                for ann_json, video_info in zip(annotations, video_infos):
                    if ann_json:
                        project_meta = src_project_metas[video_info.project_id]
                        ann = VideoAnnotation.from_json(ann_json, project_meta=project_meta)
                        if len(ann.figures) > 0:
                            video_ids_to_skip.add(video_info.id)
                    secondary_pbar.update()
            self.output_selector.secondary_progress.hide()
            if video_ids_to_skip:
                video_infos_by_project_id = {
                    pid: [info for info in infos if info.id not in video_ids_to_skip]
                    for pid, infos in video_infos_by_project_id.items()
                }

        main_pbar_str = "Processing videos..."
        if video_ids_to_skip:
            main_pbar_str += f" (Skipped {len(video_ids_to_skip)} already annotated videos)"
        total_videos = sum(len(v) for v in video_infos_by_project_id.values())
        if total_videos == 0:
            self.set_validator_text(
                f"No videos to process. Skipped {len(video_ids_to_skip)} already annotated videos",
                "warning",
            )
            return []
        main_pbar = self.output_selector.progress(message=main_pbar_str, total=total_videos)
        self.output_selector.progress.show()
        all_predictictions: List[Prediction] = []
        for src_project_id, src_video_infos in video_infos_by_project_id.items():
            if len(src_video_infos) == 0:
                continue
            project_info = self.api.project.get_info_by_id(src_project_id)
            project_validator_text_str = (
                f"Processing project: {project_info.name} [id: {src_project_id}]"
            )
            if upload_to_source_project:
                if not skip_project_versioning and not is_development():
                    logger.info("Creating new project version...")
                    self.set_validator_text(
                        project_validator_text_str + ": Creating project version",
                        "info",
                    )
                    version_id = self.api.project.version.create(
                        project_info,
                        "Created by Predict App. Task Id: " + str(env.task_id()),
                    )
                    logger.info("New project version created: " + str(version_id))
                output_project_id = src_project_id
                output_videos: List[VideoInfo] = src_video_infos
            else:
                self.set_validator_text(
                    project_validator_text_str + ": Creating project...", "info"
                )
                if not project_name:
                    project_name = project_info.name + " [Predictions]"
                    logger.warning(
                        "Project name is empty, using auto-generated name: " + project_name
                    )
                with_annotations = prediction_mode == AddPredictionsMode.MERGE_WITH_EXISTING_LABELS
                created_project = create_project(
                    api=self.api,
                    project_id=src_project_id,
                    project_name=project_name,
                    workspace_id=self.workspace_id,
                    copy_meta=with_annotations,
                    project_type=ProjectType.VIDEOS,
                )
                output_project_id = created_project.id
                output_videos: List[VideoInfo] = copy_items_to_project(
                    api=self.api,
                    src_project_id=src_project_id,
                    items=src_video_infos,
                    dst_project_id=created_project.id,
                    with_annotations=with_annotations,
                    progress=self.output_selector.secondary_progress,
                    project_type=ProjectType.VIDEOS,
                )

            self.set_validator_text(
                project_validator_text_str + ": Merging project meta",
                "info",
            )
            model_meta = self.model_api.get_model_meta()
            project_meta = src_project_metas[src_project_id]
            project_meta = project_meta.merge(model_meta)
            project_meta = self.api.project.update_meta(output_project_id, project_meta)
            for src_video_info, output_video_info in zip(src_video_infos, output_videos):
                video_validator_text_str = (
                    project_validator_text_str
                    + f", video: {src_video_info.name} [id: {src_video_info.id}]"
                )
                self.set_validator_text(
                    video_validator_text_str + ": Predicting",
                    "info",
                )
                frames_predictions: List[Prediction] = []
                with self.model_api.predict_detached(
                    video_id=src_video_info.id,
                    tqdm=self.output_selector.secondary_progress(),
                    tracking=tracking,
                    **predict_kwargs,
                ) as session:
                    self.output_selector.secondary_progress.show()
                    for prediction in session:
                        if self._stop_flag:
                            logger.info("Prediction stopped by user.")
                            raise StopIteration("Stopped by user.")
                        frames_predictions.append(prediction)
                    all_predictictions.extend(frames_predictions)
                    if tracking:
                        prediction_video_annotation: VideoAnnotation = VideoAnnotation.from_json(
                            session.final_result["video_ann"],
                            project_meta=project_meta,
                        )
                    else:
                        objects = {}
                        frames = []
                        for i, prediction in enumerate(frames_predictions):
                            figures = []
                            for label in prediction.annotation.labels:
                                obj_name = label.obj_class.name
                                if not obj_name in objects:
                                    obj_class = project_meta.get_obj_class(obj_name)
                                    if obj_class is None:
                                        continue
                                    objects[obj_name] = VideoObject(obj_class)

                                vid_object = objects[obj_name]
                                if vid_object:
                                    figures.append(
                                        VideoFigure(vid_object, label.geometry, frame_index=i)
                                    )
                            frame = Frame(i, figures=figures)
                            frames.append(frame)
                        prediction_video_annotation = VideoAnnotation(
                            img_size=(src_video_info.frame_height, src_video_info.frame_width),
                            frames_count=src_video_info.frames_count,
                            objects=VideoObjectCollection(list(objects.values())),
                            frames=FrameCollection(frames),
                        )
                if prediction_video_annotation is None:
                    logger.warning(
                        f"No predictions were made for video {src_video_info.name} [id: {src_video_info.id}]"
                    )
                    main_pbar.update()
                    continue
                self.set_validator_text(
                    video_validator_text_str + ": Uploading predictions",
                    "info",
                )
                if upload_to_source_project:
                    if prediction_mode in [
                        AddPredictionsMode.REPLACE_EXISTING_LABELS,
                        AddPredictionsMode.REPLACE_EXISTING_LABELS_AND_SAVE_IMAGE_TAGS,
                    ]:
                        self.output_selector.secondary_progress.hide()
                        with open("/tmp/prediction_video_annotation.json", "w") as f:
                            json.dump(prediction_video_annotation.to_json(), f)
                        self.api.video.annotation.upload_paths(
                            video_ids=[src_video_info.id],
                            paths=["/tmp/prediction_video_annotation.json"],
                            project_meta=project_meta,
                        )
                    else:
                        secondary_pbar = self.output_selector.secondary_progress(
                            message="Uploading annotations...",
                            total=len(prediction_video_annotation.figures),
                        )
                        self.output_selector.secondary_progress.show()
                        self.api.video.annotation.append(
                            video_id=src_video_info.id,
                            ann=prediction_video_annotation,
                            key_id_map=KeyIdMap(),
                            progress_cb=secondary_pbar.update,
                        )
                else:
                    secondary_pbar = self.output_selector.secondary_progress(
                        message="Uploading annotations...",
                        total=len(prediction_video_annotation.figures),
                    )
                    self.output_selector.secondary_progress.show()
                    self.api.video.annotation.append(
                        video_id=output_video_info.id,
                        ann=prediction_video_annotation,
                        key_id_map=KeyIdMap(),
                        progress_cb=secondary_pbar.update,
                    )
                main_pbar.update()
        return all_predictictions

    def _run_images(self, run_parameters: Dict[str, Any] = None) -> List[Prediction]:
        if self.model_api is None:
            self.set_validator_text("Deploying model...", "info")
            self.model_selector.model._deploy()
        if self.model_api is None:
            logger.error("Model Deployed with an error")
            raise RuntimeError("Model Deployed with an error")

        self.set_validator_text("Preparing settings for prediction...", "info")
        if run_parameters is None:
            run_parameters = self.get_run_parameters()

        predict_kwargs = {}
        # Input
        input_args = {}
        input_parameters = run_parameters["input"]
        input_project_id = input_parameters.get("project_id", None)
        input_dataset_ids = input_parameters.get("dataset_ids", [])
        input_image_ids = input_parameters.get("image_ids", [])
        if input_image_ids:
            input_args["image_ids"] = input_image_ids
        elif input_dataset_ids:
            input_args["dataset_ids"] = input_dataset_ids
        elif input_project_id:
            input_args["project_id"] = input_project_id
        else:
            raise ValueError("No valid input parameters found for prediction.")

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
        predict_kwargs.update(settings)
        predict_kwargs["upload_mode"] = upload_mode

        # Classes
        classes = run_parameters["classes"]
        if classes:
            predict_kwargs["classes"] = classes

        # Output
        output_parameters = run_parameters["output"]
        upload_to_source_project = output_parameters.get("upload_to_source_project", False)
        skip_project_versioning = output_parameters.get("skip_project_versioning", False)
        skip_annotated = output_parameters.get("skip_annotated", False)

        image_infos = []
        if input_image_ids:
            image_infos = self.api.image.get_info_by_id_batch(input_image_ids)
        elif input_dataset_ids:
            for dataset_id in input_dataset_ids:
                image_infos.extend(self.api.image.get_list(dataset_id))
        elif input_project_id:
            datasets = self.api.dataset.get_list(input_project_id, recursive=True)
            for dataset in datasets:
                image_infos.extend(self.api.image.get_list(dataset.id))
        if len(image_infos) == 0:
            raise ValueError("No images found for the given input parameters.")

        to_skip = []
        if skip_annotated:
            to_skip = [image_info.id for image_info in image_infos if image_info.labels_count == 0]
        if to_skip:
            image_infos = [info for info in image_infos if info.id not in to_skip]
        if len(image_infos) == 0:
            self.set_validator_text(
                f"All images are already annotated. Nothing to predict.", "warning"
            )
            return []

        image_infos_by_project_id: Dict[int, List[ImageInfo]] = {}
        image_infos_by_dataset_id: Dict[int, List[ImageInfo]] = {}
        for info in image_infos:
            image_infos_by_project_id.setdefault(info.project_id, []).append(info)
            image_infos_by_dataset_id.setdefault(info.dataset_id, []).append(info)
        src_project_metas: Dict[int, ProjectMeta] = {}
        for project_id in image_infos_by_project_id.keys():
            src_project_metas[project_id] = ProjectMeta.from_json(
                self.api.project.get_meta(project_id)
            )

        self.output_selector.progress.show()
        for src_project_id, infos in image_infos_by_project_id.items():
            if len(infos) == 0:
                continue
            project_info = self.api.project.get_info_by_id(src_project_id)
            project_validator_text_str = (
                f"Processing project: {project_info.name} [id: {src_project_id}]"
            )
            if upload_to_source_project:
                if not skip_project_versioning and not is_development():
                    logger.info("Creating new project version...")
                    self.set_validator_text(
                        project_validator_text_str + ": Creating project version", "info"
                    )
                    version_id = self.api.project.version.create(
                        project_info,
                        "Created by Predict App. Task Id: " + str(env.task_id()),
                    )
                    logger.info("New project version created: " + str(version_id))
                output_project_id = src_project_id
                output_image_infos: List[ImageInfo] = infos
            else:
                self.set_validator_text(
                    project_validator_text_str + ": Creating project...", "info"
                )
                if not project_name:
                    project_name = project_info.name + " [Predictions]"
                    logger.warning(
                        "Project name is empty, using auto-generated name: " + project_name
                    )
                created_project = create_project(
                    api=self.api,
                    project_id=src_project_id,
                    project_name=project_name,
                    workspace_id=self.workspace_id,
                    copy_meta=with_annotations,
                    project_type=ProjectType.IMAGES,
                )
                output_project_id = created_project.id
                output_image_infos: List[ImageInfo] = copy_items_to_project(
                    api=self.api,
                    src_project_id=src_project_id,
                    items=infos,
                    dst_project_id=created_project.id,
                    with_annotations=with_annotations,
                    progress=self.output_selector.progress,
                    project_type=ProjectType.IMAGES,
                )
            self.set_validator_text(
                project_validator_text_str + ": Merging project meta",
                "info",
            )
            model_meta = self.model_api.get_model_meta()
            project_meta = src_project_metas[src_project_id]
            project_meta = project_meta.merge(model_meta)
            project_meta = self.api.project.update_meta(output_project_id, project_meta)

        # Run prediction
        self.set_validator_text("Running prediction...", "info")
        predictions: List[Prediction] = []
        self._is_running = True
        with self.model_api.predict_detached(
            image_ids=[info.id for info in output_image_infos],
            **predict_kwargs,
            tqdm=self.output_selector.progress(),
        ) as session:
            for prediction in session:
                if self._stop_flag:
                    logger.info("Prediction stopped by user.")
                    raise StopIteration("Stopped by user.")
                predictions.append(prediction)
        return predictions

    def run(self, run_parameters: Dict[str, Any] = None) -> List[Prediction]:
        self.show_validator_text()
        if run_parameters is None:
            run_parameters = self.get_run_parameters()
        input_parameters = run_parameters["input"]
        video_ids = input_parameters.get("video_ids", None)
        try:
            if video_ids:
                run_f = self._run_videos
            else:
                run_f = self._run_images
            predictions = run_f(run_parameters)
            if predictions:
                output_project_id = predictions[
                    0
                ].project_id  # normally all predictions belong to the same project
                self.set_validator_text("Project successfully processed", "success")
                self.output_selector.set_result_thumbnail(output_project_id)
            else:
                # items were skipped
                pass
        except StopIteration:
            logger.info("Prediction stopped by user.")
            self.set_validator_text("Prediction stopped by user.", "warning")
            raise
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            self.set_validator_text(f"Error during prediction: {str(e)}", "error")
            disable_enable(self.output_selector.widgets_to_disable, False)
            raise
        finally:
            self.output_selector.secondary_progress.hide()
            self.output_selector.progress.hide()
            self._is_running = False
            self._stop_flag = False

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

        # 5. Settings selector & Preview
        self.settings_selector.load_from_json(data.get("settings", {}))

        # 6. Output selector
        self.output_selector.load_from_json(data.get("output", {}))

    def set_validator_text(self, text: str, status: str = "text"):
        self.output_selector.validator_text.set(text=text, status=status)

    def show_validator_text(self):
        self.output_selector.validator_text.show()

    def hide_validator_text(self):
        self.output_selector.validator_text.hide()

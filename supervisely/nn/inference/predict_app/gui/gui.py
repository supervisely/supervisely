import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import yaml

from supervisely._utils import is_development, logger
from supervisely.api.api import Api
from supervisely.api.image_api import ImageInfo
from supervisely.api.video.video_api import VideoInfo
from supervisely.app.widgets import Button, Card, Container, Stepper, Widget
from supervisely.geometry.any_geometry import AnyGeometry
from supervisely.io import env
from supervisely.nn.inference.inference import update_meta_and_ann_for_video_annotation
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
    create_project,
    disable_enable,
    update_custom_button_params,
    video_annotation_from_predictions,
)
from supervisely.nn.model.model_api import ModelAPI
from supervisely.nn.model.prediction import Prediction
from supervisely.project.project_meta import ProjectMeta
from supervisely.project.project_type import ProjectType
from supervisely.video_annotation.key_id_map import KeyIdMap
from supervisely.video_annotation.video_annotation import VideoAnnotation


class StepFlow:
    def __init__(self):
        self._stepper = None
        self.steps = {}
        self.steps_sequence = []

    def add_step(
        self,
        name: str,
        widget: Widget,
        on_select: Optional[Callable] = None,
        on_reactivate: Optional[Callable] = None,
        depends_on: Optional[List[Widget]] = None,
        on_lock: Optional[Callable] = None,
        on_unlock: Optional[Callable] = None,
        button: Optional[Button] = None,
        position: Optional[int] = None,
    ):
        if depends_on is None:
            depends_on = []
        self.steps[name] = {
            "widget": widget,
            "on_select": on_select,
            "on_reactivate": on_reactivate,
            "depends_on": depends_on,
            "on_lock": on_lock,
            "on_unlock": on_unlock,
            "button": button,
            "is_selected": False,
            "is_locked": False,
        }
        if button is not None:
            self._wrap_button(button, name)
        if position is not None:
            self.steps_sequence.insert(position, name)
        else:
            self.steps_sequence.append(name)
        self.update_locks()

    def _create_stepper(self):
        widgets = []
        for step_name in self.steps_sequence:
            step = self.steps[step_name]
            widgets.append(step["widget"])
        self._stepper = Stepper(widgets=widgets)

    @property
    def stepper(self):
        if self._stepper is None:
            self._create_stepper()
        return self._stepper

    def update_stepper(self):
        for i, step_name in enumerate(self.steps_sequence):
            step = self.steps[step_name]
            if not step["is_selected"]:
                self.stepper.set_active_step(i + 1)
                return

    def update_locks(self):
        for step in self.steps.values():
            should_lock = False
            for dep_name in step["depends_on"]:
                dep = self.steps[dep_name]
                if not dep["is_selected"]:
                    should_lock = True
                    break
            if should_lock and not step["is_locked"]:
                if step["on_lock"] is not None:
                    step["on_lock"]()
                step["is_locked"] = True
            if not should_lock and step["is_locked"]:
                if step["on_unlock"]:
                    step["on_unlock"]()
                step["is_locked"] = False

    def _reactivate_dependents(self, step_name: str, visited=None):
        if visited is None:
            visited = set()
        for dep_name, step in self.steps.items():
            if step_name in step["depends_on"] and not dep_name in visited:
                self._reactivate_step(dep_name, visited)

    def _reactivate_step(self, step_name: str, visited=None):
        step = self.steps[step_name]
        if step["on_reactivate"] is not None:
            step["on_reactivate"]()
        step["is_selected"] = False
        if visited is None:
            visited = set()
        self._reactivate_dependents(step_name, visited)

    def reactivate_step(self, step_name: str):
        self._reactivate_step(step_name)
        self.update_stepper()
        self.update_locks()

    def select_step(self, step_name: str):
        step = self.steps[step_name]
        if step["on_select"] is not None:
            step["on_select"]()
        step["is_selected"] = True
        self.update_stepper()
        self.update_locks()

    def select_or_reactivate(self, step_name: str):
        step = self.steps[step_name]
        if step["is_selected"]:
            self.reactivate_step(step_name)
        else:
            self.select_step(step_name)

    def _wrap_button(self, button: Button, step_name: str):
        button.click(lambda: self.select_or_reactivate(step_name))


class PredictAppGui:
    def __init__(self, api: Api, static_dir: str = "static"):
        self.api = api
        self.static_dir = static_dir

        # Environment variables
        self.team_id = env.team_id()
        self.workspace_id = env.workspace_id()
        self.project_id = env.project_id(raise_not_found=False)
        self.project_meta = None
        if self.project_id:
            self.project_meta = ProjectMeta.from_json(self.api.project.get_meta(self.project_id))
        # -------------------------------- #

        # Flags
        self._stop_flag = False
        self._is_running = False
        # -------------------------------- #

        # GUI
        # Steps
        self.step_flow = StepFlow()
        select_params = {"icon": None, "plain": False, "text": "Select"}
        reselect_params = {"icon": "zmdi zmdi-refresh", "plain": True, "text": "Reselect"}

        # 1. Input selector
        self.input_selector = InputSelector(self.workspace_id, self.api)

        def _on_input_select():
            valid = self.input_selector.validate_step()
            if not valid:
                return
            current_item_type = self.input_selector.radio.get_value()
            self.update_item_type()
            if self.model_api:
                if current_item_type == self.input_selector.radio.get_value():
                    inference_settings = self.model_api.get_settings()
                    self.settings_selector.set_inference_settings(inference_settings)

                    if self.input_selector.radio.get_value() == ProjectType.VIDEOS.value:
                        try:
                            tracking_settings = self.model_api.get_tracking_settings()
                            self.settings_selector.set_tracking_settings(tracking_settings)
                        except Exception as e:
                            logger.warning(
                                "Unable to get tracking settings from the model. Settings defaults"
                            )
                            self.settings_selector.set_default_tracking_settings()
            self.input_selector.disable()

            self.project_id = self.input_selector.get_project_id()
            if self.project_id:
                self.project_meta = ProjectMeta.from_json(self.api.project.get_meta(self.project_id))
            update_custom_button_params(self.input_selector.button, reselect_params)

        def _on_input_reactivate():
            self.input_selector.enable()
            update_custom_button_params(self.input_selector.button, select_params)

        self.step_flow.add_step(
            name="input_selector",
            widget=self.input_selector.card,
            on_select=_on_input_select,
            on_reactivate=_on_input_reactivate,
            button=self.input_selector.button,
        )

        # 2. Model selector
        self.model_selector = ModelSelector(self.api, self.team_id)

        self.step_flow.add_step(
            name="model_selector",
            widget=self.model_selector.card,
        )

        # 3. Classes selector
        self.classes_selector = ClassesSelector()

        def _on_classes_select():
            valid = self.classes_selector.validate_step()
            if not valid:
                return
            self.classes_selector.classes_table.disable()

            # Find conflict between project meta and model meta
            selected_classes_names = self.classes_selector.get_selected_classes()
            project_meta = self.project_meta
            model_meta = self.model_api.get_model_meta()

            has_conflict = False
            for class_name in selected_classes_names:
                project_obj_class = project_meta.get_obj_class(class_name)
                if project_obj_class is None:
                    continue

                model_obj_class = model_meta.get_obj_class(class_name)
                if model_obj_class.geometry_type.name() == AnyGeometry.name():
                    continue

                if project_obj_class.geometry_type.name() == model_obj_class.geometry_type.name():
                    continue

                has_conflict = True
                break

            if has_conflict:
                self.settings_selector.model_prediction_suffix_container.show()
            else:
                self.settings_selector.model_prediction_suffix_container.hide()
            # ------------------------------------------------ #

            update_custom_button_params(self.classes_selector.button, reselect_params)

        def _on_classes_reactivate():
            self.classes_selector.classes_table.enable()
            update_custom_button_params(self.classes_selector.button, select_params)

        self.step_flow.add_step(
            name="classes_selector",
            widget=self.classes_selector.card,
            on_select=_on_classes_select,
            on_reactivate=_on_classes_reactivate,
            depends_on=["input_selector", "model_selector"],
            on_lock=self.classes_selector.lock,
            on_unlock=self.classes_selector.unlock,
            button=self.classes_selector.button,
        )

        # 4. Tags selector
        self.tags_selector = None
        if False:
            self.tags_selector = TagsSelector()
            self.step_flow.add_step("tags_selector", self.tags_selector.card)

        # 5. Settings selector & Preview
        self.settings_selector = SettingsSelector(
            api=self.api,
            static_dir=self.static_dir,
            model_selector=self.model_selector,
            input_selector=self.input_selector,
        )

        def _on_settings_select():
            valid = self.settings_selector.validate_step()
            if not valid:
                return
            self.settings_selector.disable()
            update_custom_button_params(self.settings_selector.button, reselect_params)

        def _on_settings_reactivate():
            self.settings_selector.enable()
            update_custom_button_params(self.settings_selector.button, select_params)

        self.step_flow.add_step(
            name="settings_selector",
            widget=self.settings_selector.cards_container,
            on_select=_on_settings_select,
            on_reactivate=_on_settings_reactivate,
            depends_on=["input_selector", "model_selector", "classes_selector"],
            on_lock=self.settings_selector.lock,
            on_unlock=self.settings_selector.unlock,
            button=self.settings_selector.button,
        )
        self.settings_selector.preview.run_button.disable()

        # 6. Output selector
        self.output_selector = OutputSelector(self.api)

        self.step_flow.add_step(
            "output_selector",
            self.output_selector.card,
            depends_on=[
                "input_selector",
                "model_selector",
                "classes_selector",
                # "tags_selector",
                "settings_selector",
            ],
            on_lock=self.output_selector.lock,
            on_unlock=self.output_selector.unlock,
        )
        # -------------------------------- #

        # Layout
        self.layout = Container([self.step_flow.stepper])
        # ---------------------------- #

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
                try:
                    tracking_settings = model_api.get_tracking_settings()
                    self.settings_selector.set_tracking_settings(tracking_settings)
                except Exception as e:
                    logger.warning(
                        "Unable to get tracking settings from the model. Settings defaults"
                    )
                    self.settings_selector.set_default_tracking_settings()

        def reset_entity_meta():
            empty_meta = ProjectMeta()
            if self.classes_selector is not None:
                self.classes_selector.set_project_meta(empty_meta)
                self.classes_selector.classes_table.hide()
            if self.tags_selector is not None:
                self.tags_selector.tags_table.set_project_meta(empty_meta)
                self.tags_selector.tags_table.hide()

            self.settings_selector.set_inference_settings("")

        def deploy_and_set_step():
            self.model_selector.validator_text.hide()
            model_api = type(self.model_selector.model).deploy(self.model_selector.model)
            if model_api is not None:
                set_entity_meta()
                self.step_flow.select_step("model_selector")
            else:
                reset_entity_meta()
                self.step_flow.reactivate_step("model_selector")
            return model_api

        def stop_and_reset_step():
            type(self.model_selector.model).stop(self.model_selector.model)
            self.step_flow.reactivate_step("model_selector")
            reset_entity_meta()

        def disconnect_and_reset_step():
            type(self.model_selector.model).disconnect(self.model_selector.model)
            self.step_flow.reactivate_step("model_selector")
            reset_entity_meta()

        # Replace deploy methods for DeployModel widget
        self.model_selector.model.deploy = deploy_and_set_step
        self.model_selector.model.stop = stop_and_reset_step
        self.model_selector.model.disconnect = disconnect_and_reset_step

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
        model_prediction_suffix = settings.pop("model_prediction_suffix", "")
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
                with_annotations = prediction_mode in [
                    AddPredictionsMode.APPEND,
                    AddPredictionsMode.IOU_MERGE,
                ]
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
                    ds_progress=self.output_selector.secondary_progress,
                    project_type=ProjectType.VIDEOS,
                )

            self.set_validator_text(
                project_validator_text_str + ": Merging project meta",
                "info",
            )
            project_meta = src_project_metas[src_project_id]
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
                            project_meta=self.model_api.get_model_meta(),
                        )
                    else:
                        prediction_video_annotation = video_annotation_from_predictions(
                            frames_predictions,
                            project_meta,
                            frame_size=(src_video_info.frame_height, src_video_info.frame_width),
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
                project_meta, prediction_video_annotation, meta_changed = (
                    update_meta_and_ann_for_video_annotation(
                        meta=project_meta,
                        ann=prediction_video_annotation,
                        model_prediction_suffix=model_prediction_suffix,
                    )
                )
                if meta_changed:
                    self.api.project.update_meta(output_project_id, project_meta)
                if upload_to_source_project:
                    if prediction_mode in [
                        AddPredictionsMode.REPLACE,
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
        self.set_validator_text("Project successfully processed", "success")
        self.output_selector.set_result_thumbnail(output_project_id)
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
        if prediction_mode == AddPredictionsMode.REPLACE:
            upload_mode = "replace"
            with_annotations = False
        elif prediction_mode == AddPredictionsMode.APPEND:
            upload_mode = "append"
            with_annotations = True
        elif prediction_mode == AddPredictionsMode.IOU_MERGE:
            upload_mode = "iou_merge"
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
        project_name = output_parameters.get("project_name", None)
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
        ds_project_mapping: Dict[int, int] = {}
        for info in image_infos:
            image_infos_by_dataset_id.setdefault(info.dataset_id, []).append(info)
            if info.dataset_id not in ds_project_mapping:
                ds_info = self.api.dataset.get_info_by_id(info.dataset_id)
                ds_project_mapping[info.dataset_id] = ds_info.project_id
            project_id = ds_project_mapping[info.dataset_id]
            image_infos_by_project_id.setdefault(project_id, []).append(info)

        src_project_metas: Dict[int, ProjectMeta] = {}
        for project_id in image_infos_by_project_id.keys():
            src_project_metas[project_id] = ProjectMeta.from_json(
                self.api.project.get_meta(project_id)
            )

        self.output_selector.progress.show()
        total_items = sum(len(v) for v in image_infos_by_project_id.values())
        main_pbar = self.output_selector.progress(message=f"Copying images...", total=total_items)
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
                    ds_progress=self.output_selector.secondary_progress,
                    progress_cb=main_pbar.update,
                    project_type=ProjectType.IMAGES,
                )

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
        self.set_validator_text("Project successfully processed", "success")
        self.output_selector.set_result_thumbnail(output_project_id)
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
            return run_f(run_parameters)
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

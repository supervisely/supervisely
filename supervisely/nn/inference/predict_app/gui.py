import os
import random
import tempfile
from typing import Any, Dict, List

import yaml

from supervisely.annotation.annotation import Annotation, Label, ObjClass
from supervisely.api.api import Api
from supervisely.api.image_api import ImageInfo
from supervisely.api.video.video_api import VideoInfo
from supervisely.app.widgets import (
    Card,
    Container,
    DeployModel,
    Editor,
    FastTable,
    Field,
    GridGallery,
    Input,
    InputNumber,
    OneOf,
    Progress,
    RadioGroup,
    RadioTable,
    SelectDataset,
    SelectProject,
    Text,
)
from supervisely.app.widgets.button.button import Button
from supervisely.io import env
from supervisely.project.project import ProjectType
from supervisely.project.project_meta import ProjectMeta
from supervisely.video_annotation.key_id_map import KeyIdMap
from supervisely.video_annotation.video_annotation import VideoAnnotation


def fetch_frameworks(api: Api) -> List[str]:
    serve_modules = api.app.get_list_ecosystem_modules(
        categories=["serve", "images"], categories_operation="and"
    )
    frameworks = [cat for module in serve_modules for cat in module["config"].get("categories", [])]
    frameworks = [cat[10:] for cat in frameworks if cat.startswith("framework:")]
    return frameworks


class SelectItem:
    def __init__(self, gui: "PredictAppGui"):
        self.gui = gui
        workspace_id = env.workspace_id()

        # project
        self.select_project = SelectProject(
            workspace_id=workspace_id, compact=True, allowed_types=[ProjectType.IMAGES]
        )

        # dataset
        self.select_dataset = SelectDataset(
            multiselect=True,
            allowed_project_types=[ProjectType.IMAGES],
        )
        self.select_dataset._project_selector._ws_id = workspace_id
        self.select_dataset._project_selector._compact = True
        self.select_dataset._project_selector.update_data()

        # video
        self.select_dataset_for_video = SelectDataset(allowed_project_types=[ProjectType.VIDEOS])
        self.select_dataset_for_video._project_selector._ws_id = workspace_id
        self.select_dataset_for_video._project_selector._compact = True
        self.select_dataset_for_video._project_selector.update_data()
        self.select_video = RadioTable(columns=["id", "name", "dataset"], rows=[])
        self.select_video_container = Container(
            widgets=[self.select_dataset_for_video, self.select_video]
        )

        @self.select_dataset_for_video.value_changed
        def dataset_for_video_changed(dataset_id: int):
            self.select_video.loading = True
            if dataset_id is None:
                rows = []
            else:
                dataset_info = self.gui.api.dataset.get_info_by_id(dataset_id)
                videos = self.gui.api.video.get_list(dataset_id)
                rows = [[video.id, video.name, dataset_info.name] for video in videos]
            self.select_video.rows = rows
            self.select_video.loading = False

        # images
        self.select_datasets_for_images = SelectDataset(
            multiselect=True, allowed_project_types=[ProjectType.IMAGES]
        )
        self.select_datasets_for_images._project_selector._ws_id = workspace_id
        self.select_datasets_for_images._project_selector._compact = True
        self.select_datasets_for_images._project_selector.update_data()
        self.select_images = FastTable(columns=["id", "name", "dataset"])
        self.select_images_container = Container(
            widgets=[self.select_datasets_for_images, self.select_images]
        )

        @self.select_datasets_for_images.value_changed
        def datasets_for_images_changed(dataset_ids: list):
            self.select_images.loading = True
            if dataset_ids is None or len(dataset_ids) == 0 or dataset_ids[0] is None:
                self.select_images.clear()
                return
            img_id_to_ds_name: Dict[int, str] = {}
            all_images: List[ImageInfo] = []  # type: ignore
            for dataset_id in dataset_ids:
                dataset_info = self.gui.api.dataset.get_info_by_id(dataset_id)
                images = self.gui.api.image.get_list(dataset_id)
                all_images.extend(images)
                for image in images:
                    img_id_to_ds_name[image.id] = dataset_info.name

            self.select_images.read_json(
                {
                    "data": {
                        "columns": ["id", "name", "dataset"],
                        "rows": [
                            {"idx": 1, "items": [image.id, image.name, img_id_to_ds_name[image.id]]}
                            for image in all_images
                        ],
                    }
                }
            )
            self.select_images.loading = False

        # radio group
        self.radio = RadioGroup(
            items=[
                RadioGroup.Item("project", "Project", content=self.select_project),
                RadioGroup.Item("dataset", "Dataset", content=self.select_dataset),
                RadioGroup.Item("video", "Video", content=self.select_video_container),
                # RadioGroup.Item("images", "Images", content=self.select_images_container),
            ],
        )
        self.one_of = OneOf(conditional_widget=self.radio)
        self.container = Container(
            widgets=[
                self.radio,
                self.one_of,
            ],
            direction="vertical",
            gap=20,
        )
        self.card = Card(
            title="Select Items",
            description="Select the data modality on which to run model",
            content=self.container,
        )

    def get_item_settings(self) -> Dict[str, Any]:
        if self.radio.get_value() == "project":
            return {"project_id": self.select_project.get_selected_id()}
        elif self.radio.get_value() == "dataset":
            return {"dataset_ids": self.select_dataset.get_selected_ids()}
        elif self.radio.get_value() == "video":
            return {
                "video_id": self.select_video.get_selected_row(),
            }
        else:
            return {
                "image_ids": [row["id"] for row in self.select_images.get_selected_rows()],
            }

    def load_from_json(self, data):
        if "project_id" in data:
            self.select_project.set_project_id(data["project_id"])
        if "dataset_ids" in data:
            self.select_dataset.set_dataset_ids(data["dataset_ids"])
        if "video_id" in data:
            self.select_video.select_row_by_value("id", data["video_id"])
        if "image_ids" in data:
            self.select_images.select_rows_by_value("id", data["image_ids"])


class SelectOutput:
    def __init__(self, gui: "PredictAppGui"):
        self.gui = gui

        # new_project
        self.new_project_name = Input()
        self.new_project_description = Text(
            "New project will be created. The created project will have the same dataset structure as the input project."
        )
        self.new_project_name_field = Field(
            content=self.new_project_name,
            title="New Project Name",
            description="Name of the new project to create for the results.",
        )

        self.appned_description = Text("The results will be appended to the existing annotations.")
        self.replace_description = Text(
            "The existing annotations will be replaced with the predictions."
        )

        # iou_merge
        self.iou_merge_threshold = InputNumber(
            value=0, min=0, max=1, step=0.01, controls=False, width=200
        )
        self.iou_merge_description = Text(
            "If the prediction has IOU with any object greater than this value, it will be skipped."
        )
        self.iou_merge_threshold_field = Field(
            content=self.iou_merge_threshold,
            title="IOU Merge Threshold",
            description="Threshold for IOU merge. Float value between 0 and 1.",
        )

        self.radio = RadioGroup(
            items=[
                RadioGroup.Item(
                    "create",
                    "Create",
                    content=Container(
                        widgets=[self.new_project_description, self.new_project_name_field]
                    ),
                ),
                RadioGroup.Item(
                    "append",
                    "Append",
                    content=self.appned_description,
                ),
                RadioGroup.Item(
                    "replace",
                    "Replace",
                    content=self.replace_description,
                ),
                RadioGroup.Item(
                    "iou_merge",
                    "IOU Merge",
                    content=Container(
                        widgets=[self.iou_merge_description, self.iou_merge_threshold_field]
                    ),
                ),
            ],
            direction="horizontal",
        )
        self.one_of = OneOf(self.radio)
        self.container = Container(widgets=[self.radio, self.one_of], direction="vertical", gap=20)
        self.card = Card(title="Output", content=self.container)

    def get_output_settings(self):
        settings = {}
        mode = self.radio.get_value()
        if mode == "create":
            settings["mode"] = "create"
            settings["project_name"] = self.new_project_name.get_value()
        elif mode == "append":
            settings["mode"] = "append"
        elif mode == "replace":
            settings["mode"] = "replace"
        elif mode == "iou_merge":
            settings["mode"] = "iou_merge"
            settings["iou_merge_threshold"] = self.iou_merge_threshold.get_value()

        settings["mode"] = mode
        return settings

    def load_from_json(self, data):
        if not data:
            return
        mode = data["mode"]
        self.radio.set_value("create")
        if mode == "create":
            self.new_project_name.set_value(data.get("project_name", ""))
        elif mode == "iou_merge":
            self.iou_merge_threshold.value = data.get("iou_merge_threshold", 0)


class Preview:
    def __init__(self, gui: "PredictAppGui"):
        self.gui = gui
        self._preview_dir = os.path.join(self.gui.static_dir, "preview")
        os.makedirs(self._preview_dir, exist_ok=True)
        self._preview_path = os.path.join(self._preview_dir, "preview.jpg")
        self._peview_url = f"/static/preview/preview.jpg"

        self.preview_button = Button("Preview", icon="zmdi zmdi-eye")

        self.gallery = GridGallery(
            2,
            sync_views=True,
            enable_zoom=True,
            resize_on_zoom=True,
            empty_message="Click 'Preview' to see the model output.",
        )
        self.container = Container(
            widgets=[self.preview_button, self.gallery],
            direction="vertical",
            gap=20,
        )
        self.card = Card(title="Preview", content=self.container)

        @self.preview_button.click
        def preview_button_click():
            self.run_preview()

    def _get_frame_annotation(
        self, video_info: VideoInfo, frame_index: int, project_meta: ProjectMeta
    ) -> Annotation:
        video_annotation = VideoAnnotation.from_json(
            self.gui.api.video.annotation.download(video_info.id, frame_index),
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

    def run_preview(self):
        self.gallery.clean_up()
        self.gallery.loading = True
        try:
            items_settings = self.gui.items.get_item_settings()
            if "video_id" in items_settings:
                video_id = items_settings["video_id"]
                video_info = self.gui.api.video.get_info_by_id(video_id)
                video_frame = random.randint(0, video_info.frames_count - 1)
                self.gui.api.video.frame.download_path(
                    video_info.id, video_frame, self._preview_path
                )
                img_url = self._peview_url
                project_meta = ProjectMeta.from_json(
                    self.gui.api.project.get_meta(video_info.project_id)
                )
                input_ann = self._get_frame_annotation(video_info, video_frame, project_meta)
                output_ann = self.gui.model.model_api.predict(
                    input=self._preview_path, **self.gui.get_inference_settings()
                )[0].annotation
            else:
                if "project_id" in items_settings:
                    project_id = items_settings["project_id"]
                    dataset_infos = self.gui.api.dataset.get_list(project_id)
                    dataset_info = random.choice(dataset_infos)
                elif "dataset_ids" in items_settings:
                    dataset_ids = items_settings["dataset_ids"]
                    dataset_id = random.choice(dataset_ids)
                    dataset_info = self.gui.api.dataset.get_info_by_id(dataset_id)
                else:
                    raise ValueError("No valid item settings found for preview.")
                images = self.gui.api.image.get_list(dataset_info.id)
                image_info = random.choice(images)
                img_url = image_info.preview_url
                # for testing
                self.gui.api.image.download_path(image_info.id, self._preview_path)
                img_url = self._peview_url
                ##
                project_meta = ProjectMeta.from_json(
                    self.gui.api.project.get_meta(dataset_info.project_id)
                )
                input_ann = Annotation.from_json(
                    self.gui.api.annotation.download(image_info.id).annotation,
                    project_meta=project_meta,
                )
                prediction = self.gui.model.model_api.predict(
                    image_id=image_info.id, **self.gui.get_inference_settings()
                )[0]
                output_ann = prediction.annotation

            self.gallery.append(img_url, input_ann, "Input")
            self.gallery.append(img_url, output_ann, "Output")
        finally:
            self.gallery.loading = False


class PredictAppGui:

    def __init__(self, api: Api, static_dir: str = "static"):
        self.api = api
        self.static_dir = static_dir
        self.team_id = env.team_id()
        self.model = DeployModel(api=self.api, team_id=self.team_id)
        self.model.deploy = self._deploy_model

        self.model_card = Card(title="Select Model", description="", content=self.model)
        self.inference_settings = Editor("", language_mode="yaml", height_lines=10)
        self.inference_settings_card = Card(
            title="Inference Settings", content=self.inference_settings
        )
        self.items = SelectItem(self)
        self.output = SelectOutput(self)
        self.preview = Preview(self)
        self.progress = Progress()
        self.run_button = Button("Run", icon="zmdi zmdi-play")

        self.layout = Container(
            widgets=[
                self.model_card,
                self.items.card,
                self.inference_settings_card,
                self.preview.card,
                self.output.card,
                self.run_button,
                self.progress,
            ],
            direction="vertical",
            gap=10,
        )

    def _deploy_model(self) -> None:
        model_api = type(self.model).deploy(self.model)
        inference_settings = model_api.get_settings()
        self.set_inference_settings(inference_settings)
        return model_api

    def get_inference_settings(self):
        return yaml.safe_load(self.inference_settings.get_text())

    def set_inference_settings(self, settings: Dict[str, Any]):
        if isinstance(settings, str):
            self.inference_settings.set_text(settings)
        else:
            self.inference_settings.set_text(yaml.safe_dump(settings))

    def get_run_parameters(self) -> Dict[str, Any]:
        return {
            "model": self.model.get_deploy_parameters(),
            "inference_settings": self.get_inference_settings(),
            "item": self.items.get_item_settings(),
            "output": self.output.get_output_settings(),
        }

    def load_from_json(self, data):
        self.model.load_from_json(data.get("model", {}))
        inference_settings = data.get("inference_settings", "")
        self.set_inference_settings(inference_settings)
        self.items.load_from_json(data.get("items", {}))
        self.output.load_from_json(data.get("output", {}))

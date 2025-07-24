from typing import Any, Dict, List

from supervisely.api.api import Api
from supervisely.api.image_api import ImageInfo
from supervisely.app.widgets import (
    Card,
    Container,
    DeployModel,
    FastTable,
    Field,
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
                RadioGroup.Item("images", "Images", content=self.select_images_container),
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


class PredictAppGui:
    def __init__(self, api: Api):
        self.api = api
        self.team_id = env.team_id()
        self.model = DeployModel(api=self.api, team_id=self.team_id)
        for layout in self.model._modes_layouts.values():
            layout.deploy_button.hide()
        self.model.status

        self.model_card = Card(title="Select Model", description="", content=self.model)
        self.items = SelectItem(self)
        self.output = SelectOutput(self)
        self.progress = Progress()
        self.run_button = Button("Run", icon="zmdi zmdi-play")

        self.layout = Container(
            widgets=[
                self.model_card,
                self.items.card,
                self.output.card,
                self.run_button,
                self.progress,
            ],
            direction="vertical",
            gap=10,
        )

    def get_run_parameters(self) -> Dict[str, Any]:
        return {
            "model": self.model.get_deploy_parameters(),
            "item": self.items.get_item_settings(),
            "output": self.output.get_output_settings(),
        }

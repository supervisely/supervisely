import copy
import time
import uuid
from collections import defaultdict
from typing import List, Optional

import supervisely as sly
from supervisely.annotation.annotation import Annotation
from supervisely.api.annotation_api import AnnotationInfo
from supervisely.app import DataJson
from supervisely.app.content import StateJson
from supervisely.app.widgets import Widget
from supervisely.project.project_meta import ProjectMeta

ann_info = AnnotationInfo(
    **{
        "image_id": 30689223,
        "image_name": "000000575815.jpg",
        "annotation": {
            "description": "",
            "tags": [
                {
                    "id": 24692230,
                    "tagId": 483629,
                    "name": "caption",
                    "value": "A pepperoni pizza sitting on top of a wooden cutting board.",
                    "labelerLogin": "grokhael",
                    "createdAt": "2024-06-04T11:40:25.513Z",
                    "updatedAt": "2024-06-04T11:40:25.513Z",
                },
                {
                    "id": 24692231,
                    "tagId": 483629,
                    "name": "caption",
                    "value": "A pizza with pepperoni served on a wooden tray.",
                    "labelerLogin": "grokhael",
                    "createdAt": "2024-06-04T11:40:25.513Z",
                    "updatedAt": "2024-06-04T11:40:25.513Z",
                },
                {
                    "id": 24692232,
                    "tagId": 483629,
                    "name": "caption",
                    "value": "A pepperoni pizza on a napkin on top of a wooden surface.",
                    "labelerLogin": "grokhael",
                    "createdAt": "2024-06-04T11:40:25.513Z",
                    "updatedAt": "2024-06-04T11:40:25.513Z",
                },
                {
                    "id": 24692233,
                    "tagId": 483629,
                    "name": "caption",
                    "value": "a pepperoni pizza laying on a white piece of parchment paper",
                    "labelerLogin": "grokhael",
                    "createdAt": "2024-06-04T11:40:25.513Z",
                    "updatedAt": "2024-06-04T11:40:25.513Z",
                },
                {
                    "id": 24692234,
                    "tagId": 483629,
                    "name": "caption",
                    "value": "A large pepperoni pizza on a cutting board.",
                    "labelerLogin": "grokhael",
                    "createdAt": "2024-06-04T11:40:25.513Z",
                    "updatedAt": "2024-06-04T11:40:25.513Z",
                },
            ],
            "size": {"height": 451, "width": 640},
            "objects": [
                {
                    "id": 240870668,
                    "classId": 9149150,
                    "objectId": None,
                    "description": "",
                    "geometryType": "polygon",
                    "labelerLogin": "grokhael",
                    "createdAt": "2024-06-04T11:40:25.513Z",
                    "updatedAt": "2024-06-04T11:40:25.513Z",
                    "tags": [],
                    "classTitle": "pizza",
                    "points": {
                        "exterior": [
                            [237, 383],
                            [160, 343],
                            [104, 287],
                            [99, 220],
                            [110, 179],
                            [118, 149],
                            [134, 124],
                            [152, 103],
                            [204, 79],
                            [245, 55],
                            [289, 48],
                            [341, 48],
                            [388, 55],
                            [422, 64],
                            [456, 87],
                            [504, 137],
                            [516, 207],
                            [508, 288],
                            [446, 355],
                            [390, 376],
                            [327, 379],
                            [300, 377],
                            [266, 386],
                        ],
                        "interior": [],
                    },
                },
                {
                    "id": 240870669,
                    "classId": 9149150,
                    "objectId": None,
                    "description": "",
                    "geometryType": "rectangle",
                    "labelerLogin": "grokhael",
                    "createdAt": "2024-06-04T11:40:25.513Z",
                    "updatedAt": "2024-06-04T11:40:25.513Z",
                    "tags": [],
                    "classTitle": "pizza",
                    "points": {"exterior": [[99, 48], [516, 386]], "interior": []},
                },
            ],
        },
        "created_at": "2024-06-04T11:40:25.513Z",
        "updated_at": "2024-06-04T11:40:25.513Z",
    }
)


class GridGalleryV2(Widget):
    class Routes:
        IMAGE_CLICKED = "image_clicked_cb"

    def __init__(
        self,
        columns_number: int,
        fit_on_resize: bool = True,
        enable_zoom: bool = True,
        show_opacity_slider: bool = True,
        show_zoom_slider: bool = True,
        annotations_opacity: float = 0.8,
        enable_pointer_events: bool = True,
        transparent_background: bool = True,
        show_filter: bool = True,
        fill_rectangle: bool = True,
        border_width: int = 3,
        widget_id: str = None,
    ):
        self._data = []
        self._layout = []
        self._annotations = {}

        self.columns_number = columns_number

        self._last_used_column_index = 0
        self._project_meta: ProjectMeta = None
        self._loading = False

        self._bindings_dict = defaultdict(list)

        #############################
        # grid gallery settings
        self._show_preview: bool = True
        self._fill_rectangle: bool = fill_rectangle
        self._border_width: int = border_width

        self._fit_on_resize: bool = fit_on_resize
        self._enable_zoom: bool = enable_zoom
        self._show_opacity_header: bool = show_opacity_slider
        self._show_zoom_header: bool = show_zoom_slider
        self._opacity: float = annotations_opacity
        self._enable_pointer_events: bool = enable_pointer_events
        self._transparent_background: bool = transparent_background
        self._show_filter: bool = show_filter

        # self._show_opacity_header: bool = show_opacity_slider
        # self._opacity: float = annotations_opacity
        # self._enable_zoom: bool = enable_zoom
        # self._sync_views: bool = sync_views
        # self._resize_on_zoom: bool = resize_on_zoom
        # self._show_preview: bool = show_preview
        # self._views_bindings: list = []
        # self._view_height: int = view_height
        # self._empty_message: str = empty_message

        #############################

        super().__init__(widget_id=widget_id, file_path=__file__)

    def _generate_project_meta(self):
        objects_dict = dict()
        obj_tags_dict = dict()

        for cell_data in self._data:
            ann_info: AnnotationInfo = cell_data["annotation_info"]
            project_meta: ProjectMeta = cell_data["project_meta"]

            annotation = Annotation.from_json(ann_info.annotation, project_meta)

            for idx, label in enumerate(annotation.labels):
                objects_dict[label.obj_class.name] = label.obj_class
                for tag in label.tags:
                    obj_tags_dict[tag.name] = project_meta.get_tag_meta(tag.name)
                # if len(label.tags) == 0:
                #     obj_tags_dict["confidence"] = tmp
                #     ann_info.annotation["objects"][idx]["tags"].append(sly.Tag(tmp, 1.0))

            cell_data["annotation_info"] = ann_info

        objects_list = list(objects_dict.values())
        tags_list = list(obj_tags_dict.values())

        self._project_meta = ProjectMeta(obj_classes=objects_list, tag_metas=tags_list)
        return self._project_meta.to_json()

    def get_json_data(self):
        return {
            "content": {
                "projectMeta": self._generate_project_meta(),
                "layout": self._layout,
                "annotations": self._annotations,
            },
            "loading": self._loading,
        }

    def get_json_state(self):
        return {
            "options": {
                "fitOnResize": self._fit_on_resize,
                "enableZoom": self._enable_zoom,
                "showOpacityInHeader": self._show_opacity_header,
                "showZoomInHeader": self._show_zoom_header,
                "opacity": self._opacity,
                "enableObjectsPointerEvents": self._enable_pointer_events,
                "showTransparentBackground": self._transparent_background,
                "showFilter": self._show_filter,
            }
        }

    # return {
    #     "options": {
    #         "showOpacityInHeader": self._show_opacity_header,
    #         "opacity": self._opacity,
    #         "enableZoom": self._enable_zoom,
    #         "syncViews": self._sync_views,
    #         "syncViewsBindings": self._views_bindings,
    #         "resizeOnZoom": self._resize_on_zoom,
    #         "fillRectangle": self._fill_rectangle,
    #         "borderWidth": self._border_width,
    #         "selectable": False,
    #         "viewHeight": self._view_height,
    #         "showPreview": self._show_preview,
    #     },
    #     "selectedImage": None,
    #     "activeFigure": None,
    # }

    def get_column_index(self, incoming_value):
        if incoming_value is not None and 0 > incoming_value > self.columns_number:
            raise ValueError(f"column index == {incoming_value} is out of bounds")

        if incoming_value is None:
            incoming_value = self._last_used_column_index
            self._last_used_column_index = (self._last_used_column_index + 1) % self.columns_number
        else:
            self._last_used_column_index = incoming_value

        return incoming_value

    def append(
        self,
        image_url: str,
        annotation_info: AnnotationInfo,
        project_meta: ProjectMeta,
        title: str = "",
        column_index: int = None,
        # zoom_to: int = None,
        # zoom_factor: float = 1.2,
        # title_url=None,
    ):
        column_index = self.get_column_index(incoming_value=column_index)
        cell_uuid = str(
            uuid.uuid5(
                namespace=uuid.NAMESPACE_URL,
                name=f"{image_url}_{title}_{column_index}_{time.time()}",
            ).hex
        )

        self._data.append(
            {
                "image_url": image_url,
                "annotation_info": annotation_info,
                "project_meta": project_meta,
                # "annotation": (
                #     supervisely.Annotation((1, 1)) if annotation is None else annotation.clone()
                # ),
                "column_index": column_index,
                # "title": (
                #     title if title_url is None else title + ' <i class="zmdi zmdi-open-in-new"></i>'
                # ),
                "cell_uuid": cell_uuid,
                # "zoom_to": zoom_to,
                # "zoom_factor": zoom_factor,
                # "title_url": title_url,
            }
        )

        self._update()
        return cell_uuid

    def clean_up(self):
        self._data = []
        self._layout = []
        self._annotations = {}
        self._update()
        self.update_data()

    def _update_layout(self):
        layout = [[] for _ in range(self.columns_number)]

        for cell_data in self._data:
            layout[cell_data["column_index"]].append(cell_data["cell_uuid"])

        self._layout = copy.deepcopy(layout)
        DataJson()[self.widget_id]["content"]["layout"] = self._layout

    def _update_annotations(self):
        annotations = {}
        for cell_data in self._data:
            annotations[cell_data["cell_uuid"]] = {
                "imageUrl": cell_data["image_url"],
                "annotation": cell_data["annotation_info"]._asdict(),
                # "title": cell_data["title"],
                # "title_url": cell_data["title_url"],
            }
            # if not cell_data["zoom_to"] is None:
            #     zoom_params = {
            #         "figureId": cell_data["zoom_to"],
            #         "factor": cell_data["zoom_factor"],
            #     }
            #     annotations[cell_data["cell_uuid"]]["zoomToFigure"] = zoom_params

        self._annotations = copy.deepcopy(annotations)
        DataJson()[self.widget_id]["content"]["annotations"] = self._annotations

    def _update_object_bindings(self):
        object_bindings = []
        for cell_data in self._data:
            ann_json = cell_data["annotation_info"].annotation
            for obj in ann_json["objects"]:
                self._bindings_dict[obj["classTitle"]].append(
                    {"id": obj["id"], "annotationKey": cell_data["cell_uuid"]}
                )

        def filter_unique_ids(dict_list):
            seen_ids = set()
            seen_anns = set()
            unique_dicts = []
            for item in dict_list:
                if item["id"] not in seen_ids:
                    seen_ids.add(item["id"])
                    unique_dicts.append(item)
            return unique_dicts

        for class_name, bindings in self._bindings_dict.items():
            object_bindings.append(filter_unique_ids(bindings))
        DataJson()[self.widget_id]["content"]["objectsBindings"] = object_bindings

    def _update_project_meta(self):
        DataJson()[self.widget_id]["content"]["projectMeta"] = self._generate_project_meta()

    def _update(self):
        self._update_layout()
        self._update_annotations()
        self._update_object_bindings()
        self._update_project_meta()

        DataJson().send_changes()
        StateJson().send_changes()

    @property
    def loading(self):
        return self._loading

    @loading.setter
    def loading(self, value: bool):
        self._loading = value
        DataJson()[self.widget_id]["loading"] = self._loading
        DataJson().send_changes()

    # def sync_images(self, image_ids: List[str]):
    #     self._views_bindings.append(image_ids)
    #     StateJson()[self.widget_id]["options"]["syncViewsBindings"] = self._views_bindings
    #     StateJson().send_changes()

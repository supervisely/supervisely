import copy
import time
import uuid
from collections import defaultdict
from typing import List, Union

from supervisely.project.project_meta import ProjectMeta
from supervisely.annotation.annotation import Annotation
from supervisely.api.annotation_api import AnnotationInfo
from supervisely.app import DataJson
from supervisely.app.content import StateJson
from supervisely.app.widgets import Widget
from supervisely.project.project_meta import ProjectMeta


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
        enable_panning=False,
        border_width: int = 3,
        default_tag_filters: List[Union[str, dict]] = None,
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

        self._enablePan = enable_panning

        self._filters_tags = default_tag_filters
        #############################

        self._filters = []
        self._object_bindings = []

        super().__init__(widget_id=widget_id, file_path=__file__)

    def _generate_project_meta(self):
        objects_dict = dict()
        obj_tags_dict = dict()

        for cell_data in self._data:
            ann_info: AnnotationInfo = cell_data["annotation_info"]
            project_meta: ProjectMeta = cell_data["project_meta"]

            # tmp = project_meta
            # if self._project_meta is not None:  # TODO
            #     project_meta = self._project_meta

            annotation = Annotation.from_json(ann_info.annotation, project_meta)
            for idx, label in enumerate(annotation.labels):
                objects_dict[label.obj_class.name] = label.obj_class
                for tag in label.tags:
                    obj_tags_dict[tag.name] = project_meta.get_tag_meta(tag.name)

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
            "filters": self._filters,
            "objectBindings": self._object_bindings,
            "options": {
                "fitOnResize": self._fit_on_resize,
                "enableZoom": self._enable_zoom,
                "showOpacityInHeader": self._show_opacity_header,
                "showZoomInHeader": self._show_zoom_header,
                "opacity": self._opacity,
                "enableObjectsPointerEvents": self._enable_pointer_events,
                "showTransparentBackground": self._transparent_background,
                "showFilter": self._show_filter,
                "enablePan": self._enablePan,
                "lineWidth": self._border_width,
            },
        }


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
        ignore_tags_filtering: Union[bool, List[str]] = False,
        call_update: bool = True,
    ):
        column_index = self.get_column_index(incoming_value=column_index)
        cell_uuid = str(
            uuid.uuid5(
                namespace=uuid.NAMESPACE_URL,
                name=f"{image_url}_{title}_{column_index}_{time.time()}",
            ).hex
        )

        if ignore_tags_filtering is True:
            pass

        self._data.append(
            {
                "image_url": image_url,
                "annotation_info": annotation_info,
                "project_meta": project_meta,
                "column_index": column_index,
                "cell_uuid": cell_uuid,
                "skipObjectTagsFiltering": ignore_tags_filtering,

            }
        )

        if call_update:
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
            tmp = cell_data["cell_uuid"]
            skip_filters: Union[bool, List[str]] = cell_data.get("skipObjectTagsFiltering")
            if skip_filters is True or isinstance(skip_filters, list):
                tmp = {
                    "layoutDataKey": cell_data["cell_uuid"],
                    "options": {"skipObjectTagsFiltering": skip_filters},
                }

            layout[cell_data["column_index"]].append(tmp)

        self._layout = copy.deepcopy(layout)
        DataJson()[self.widget_id]["content"]["layout"] = self._layout

    def _update_annotations(self):
        annotations = {}
        for cell_data in self._data:
            annotations[cell_data["cell_uuid"]] = {
                "imageUrl": cell_data["image_url"],
                "annotation": cell_data["annotation_info"]._asdict(),
            }
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

        self._object_bindings = object_bindings
        DataJson()[self.widget_id]["content"]["objectsBindings"] = object_bindings

    def _update_project_meta(self):
        DataJson()[self.widget_id]["content"]["projectMeta"] = self._generate_project_meta()

    def _update_filters(self):
        filters = []
        if self._filters_tags is not None:
            for filters_tag in self._filters_tags:
                tmp = list(filters_tag.items())[0]
                filters.append({"type": "tag", "tagId": tmp[0], "value": tmp[1]})
        self._filters = filters
        StateJson()[self.widget_id]["filters"] = filters

    def _update(self):
        self._update_layout()
        self._update_annotations()
        self._update_object_bindings()
        self._update_project_meta()
        self._update_filters()

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

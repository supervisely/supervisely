from typing import Union
import supervisely_lib as sly
from supervisely_lib.project.project_meta import ProjectMeta
from supervisely_lib.api.api import Api
from supervisely_lib.annotation.annotation import Annotation


class Gallery:
    def __init__(self, task_id, api: Api, v_model, project_meta: ProjectMeta, col_number: int, preview_info=False,
                 enable_zoom=False, resize_on_zoom=False,
                 sync_views=False, show_preview=True, selectable=False, opacity=0.5, show_opacity_header=True,
                 fillRectangle=False, borderWidth=3):
        self._task_id = task_id
        self._api = api
        self._v_model = v_model
        self._project_meta = project_meta.clone()
        self._data = {}
        self.col_number = col_number
        self.preview_info = preview_info
        self._need_zoom = False
        if not isinstance(self.col_number, int):
            raise ValueError("Columns number must be integer, not {}".format(type(self.col_number).__name__))

        self._options = {
            "enableZoom": enable_zoom,
            "syncViews": sync_views,
            "resizeOnZoom": resize_on_zoom,
            "showPreview": show_preview,
            "selectable": selectable,
            "opacity": opacity,
            "showOpacityInHeader": show_opacity_header,
            "fillRectangle": fillRectangle,
            "borderWidth": borderWidth
        }
        self._options_initialized = False

    def add_item(self, title, image_url, ann: Union[Annotation, dict] = None, col_index=None, info_dict=None,
                 zoom_to_figure=None):

        if col_index is not None:
            if col_index <= 0 or col_index > self.col_number:
                raise ValueError("Column number is not correct, check your input data")

        res_ann = Annotation((1, 1))
        if ann is not None:
            if type(ann) is dict:
                res_ann = Annotation.from_json(ann, self._project_meta)
            else:
                res_ann = ann.clone()

        self._data[title] = [image_url, res_ann, col_index]

        if zoom_to_figure is not None:
            self._data[title].append(zoom_to_figure)
            self._need_zoom = True

        if self.preview_info:
            if info_dict is not None:
                self._data[title].append(info_dict)

    def add_item_by_id(self, image_id, with_ann=True, col_index=None, info_dict=None,
                 zoom_to_figure=None):
        image_info = self._api.image.get_info_by_id(image_id)
        if with_ann:
            ann_info = self._api.annotation.download(image_id)
            ann = sly.Annotation.from_json(ann_info.annotation, self._project_meta)
        else:
            ann = None

        self.add_item(image_info.name, image_info.full_storage_url, ann, col_index, info_dict, zoom_to_figure)

    def _get_item_annotation(self, name):
        if self.preview_info:
            return {
                "url": self._data[name][0],
                "figures": [label.to_json() for label in self._data[name][1].labels],
                "title": name,
                "info": self._data[name][3]
            }
        else:
            return {
                "url": self._data[name][0],
                "figures": [label.to_json() for label in self._data[name][1].labels],
                "title": name,
            }

    def update(self, options=True):
        if len(self._data) == 0:
            raise ValueError("Items list is empty")
        if self._need_zoom:
            gallery_json = self._zoom_to_figure()
        else:
            gallery_json = self.to_json()
        if options is True or self._options_initialized is False:
            if self._need_zoom:
                self._options["resizeOnZoom"] = True
            self._api.task.set_field(self._task_id, self._v_model, gallery_json)
            self._options_initialized = True
        else:
            self._api.task.set_field(self._task_id, f"{self._v_model}.content", gallery_json["content"])

    def _zoom_to_figure(self):
        gallery_json = self.to_json()
        items = self._data.items()
        zoom_to_figure_name = "zoomToFigure"
        for item in items:
            curr_image_name = item[0]
            curr_image_data = item[1]

            if len(curr_image_data) < 4:
                raise ValueError("Option zoom_to_figure not set for {} image".format(curr_image_name))

            elif type(curr_image_data[3]) is not tuple:
                raise ValueError("Option zoom_to_figure not set for {} image".format(curr_image_name))

            zoom_params = {
                "figureId": curr_image_data[3][0],
                "factor": curr_image_data[3][1]
            }
            gallery_json["content"]["annotations"][curr_image_name][zoom_to_figure_name] = zoom_params

        return gallery_json

    def to_json(self):
        annotations = {}
        layout = [[] for _ in range(self.col_number)]
        index_in_layout = 0

        for curr_data_name, curr_url_ann_index in self._data.items():
            annotations[curr_data_name] = self._get_item_annotation(curr_data_name)

            curr_col_index = curr_url_ann_index[2]
            if curr_col_index is not None:
                layout[curr_col_index - 1].append(curr_data_name)
            else:
                if index_in_layout == self.col_number:
                    index_in_layout = 0
                layout[index_in_layout].append(curr_data_name)
                index_in_layout += 1

        return {
            "content": {
                "projectMeta": self._project_meta.to_json(),
                "layout": layout,
                "annotations": annotations
            },
            "options": {
                **self._options
            }
        }
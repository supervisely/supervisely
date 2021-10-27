from typing import Union
import supervisely_lib as sly
from supervisely_lib.project.project_meta import ProjectMeta
from supervisely_lib.api.api import Api
from supervisely_lib.annotation.annotation import Annotation


class Gallery:

    def __init__(self, task_id, api: Api, v_model, project_meta: ProjectMeta, col_number: int, with_info=False,
                 enable_zoom=True,
                 sync_views=True, show_preview=False, selectable=False, opacity=0.5, show_opacity_header=True):
        self._task_id = task_id
        self._api = api
        self._v_model = v_model
        self._project_meta = project_meta.clone()
        self._data = {}
        self.col_number = col_number
        self.with_info = with_info
        if not isinstance(self.col_number, int):
            raise ValueError("Columns number must be integer, not {}".format(type(self.col_number).__name__))

        self._options = {
            "enableZoom": enable_zoom,
            "syncViews": sync_views,
            "showPreview": show_preview,
            "selectable": selectable,
            "opacity": opacity,
            "showOpacityInHeader": show_opacity_header
        }
        self._options_initialized = False

    def add_item(self, title, image_url, ann: Union[Annotation, dict] = None, col_index = None):

        if col_index is not None:
            if col_index <=0 or col_index > self.col_number:
                raise ValueError("Column number is not correct, check your input data")

        res_ann = Annotation((1,1))
        if ann is not None:
            if type(ann) is dict:
                res_ann = Annotation.from_json(ann, self._project_meta)
            else:
                res_ann = ann.clone()

        self._data[title] = [image_url, res_ann, col_index]

        if self.with_info:
            preview_data = {}
            preview_data["objects"] = len(ann.labels)
            labelers_cnt = []
            for label in ann.labels:
                if label.geometry.labeler_login not in labelers_cnt:
                    labelers_cnt.append(label.geometry.labeler_login)
            preview_data["labelers"] = len(labelers_cnt)

            self._data[title].append(preview_data)


    def add_item_by_id(self, image_id, with_ann = True, col_index = None):
        image_info = self._api.image.get_info_by_id(image_id)
        if with_ann:
            ann_info = self._api.annotation.download(image_id)
            ann = sly.Annotation.from_json(ann_info.annotation, self._project_meta)
        else:
            ann = None

        self.add_item(image_info.name, image_info.full_storage_url, ann, col_index)

    def _get_item_annotation(self, name):
        if self.with_info:
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

        gallery_json = self.to_json()
        if options is True or self._options_initialized is False:
            self._api.task.set_field(self._task_id, self._v_model, gallery_json)
            self._options_initialized = True
        else:
            self._api.task.set_field(self._task_id, f"{self._v_model}.content", gallery_json["content"])

    def to_json(self):

        annotations = {}
        layout = []
        index_in_layout = 0

        for _ in range(self.col_number):
            layout.append([])

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

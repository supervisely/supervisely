from typing import Union
import supervisely_lib as sly
from supervisely_lib.project.project_meta import ProjectMeta
from supervisely_lib.api.api import Api
from supervisely_lib.annotation.annotation import Annotation
import cv2
from skimage import io


class Gallery:

    def __init__(self, task_id, api: Api, v_model, project_meta: ProjectMeta, col_number: int, with_info=False,
                 enable_zoom=True,
                 sync_views=True, show_preview=False, selectable=False, opacity=0.5, show_opacity_header=True):
        self._task_id = task_id
        self._api = api
        self._v_model = v_model
        self._project_meta = project_meta.clone()
        self._data = {}
        self._zoom_data = {}
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


    def add_item_by_class_name(self, title, image_url, class_name, ann: Union[Annotation, dict] = None, col_index = None):
        if col_index is not None:
            if col_index <=0 or col_index > self.col_number:
                raise ValueError("Column number is not correct, check your input data")

        res_ann = Annotation((1,1))
        if ann is not None:
            if type(ann) is dict:
                res_ann = Annotation.from_json(ann, self._project_meta)
            else:
                res_ann = ann.clone()

        name_idx = 1
        for label in res_ann.labels:
            if label.obj_class.name == class_name:
                res_ann_one_label = res_ann.clone(labels=[label])

                curr_title = str(name_idx) + '_' + title
                name_idx += 1

                self._data[curr_title] = [image_url, res_ann_one_label, col_index]

                if self.with_info:
                    preview_data = {}
                    preview_data["objects"] = len(ann.labels)
                    labelers_cnt = []
                    for label in ann.labels:
                        if label.geometry.labeler_login not in labelers_cnt:
                            labelers_cnt.append(label.geometry.labeler_login)
                    preview_data["labelers"] = len(labelers_cnt)

                    self._data[curr_title].append(preview_data)


    def add_item_by_id(self, image_id, with_ann = True, col_index = None):
        image_info = self._api.image.get_info_by_id(image_id)
        if with_ann:
            ann_info = self._api.annotation.download(image_id)
            ann = sly.Annotation.from_json(ann_info.annotation, self._project_meta)
        else:
            ann = None

        self.add_item(image_info.name, image_info.full_storage_url, ann, col_index)

    def zoom_to_figures(self, temp_ds_id, crop_padding):

        crop_img_nps = []
        crop_anns = []
        crop_im_names = []
        col_indexes = []

        for im_name, url_ann in self._data.items():
            if len(url_ann[1].labels) == 0:
                continue
            crop_im_names.append(im_name)
            image_filename = url_ann[0]
            img_np = io.imread(image_filename)
            class_name = url_ann[1].labels[0].obj_class.name
            crop_data = sly.aug.instance_crop(img_np, url_ann[1], class_name, False, padding_config=crop_padding)
            crop_img_nps.append(crop_data[0][0])
            crop_anns.append(crop_data[0][1])
            col_indexes.append(url_ann[2])

        new_img_infos = self._api.image.upload_nps(temp_ds_id, crop_im_names, crop_img_nps)
        new_img_urls = [im_info.full_storage_url for im_info in new_img_infos]

        for im_name, image_url, res_ann, col_idx in zip(crop_im_names, new_img_urls, crop_anns, col_indexes):
            self._zoom_data[im_name] = [image_url, res_ann, col_idx]

        self.update_zoom()

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

    def _get_item_annotation_zoom(self, name):
            return {
                "url": self._zoom_data[name][0],
                "figures": [label.to_json() for label in self._zoom_data[name][1].labels],
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


    def update_zoom(self, options=True):
        if len(self._zoom_data) == 0:
            raise ValueError("Items list is empty")

        gallery_json = self.zoom_to_json()
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

    def zoom_to_json(self):

        annotations = {}
        layout = []
        index_in_layout = 0

        for _ in range(self.col_number):
            layout.append([])

        for curr_data_name, curr_url_ann_index in self._zoom_data.items():
            annotations[curr_data_name] = self._get_item_annotation_zoom(curr_data_name)

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

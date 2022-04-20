import copy
import uuid

import supervisely
from supervisely import Annotation
from supervisely.app import DataJson
from supervisely.app.widgets import Widget


class GridGallery(Widget):
    def __init__(self, columns_number: int,

                 preview_info: bool = False,
                 enable_zoom: bool = False,
                 resize_on_zoom: bool = False,
                 sync_views: bool = False,
                 show_preview: bool = True,
                 selectable: bool = False,

                 opacity: float = 0.5,
                 show_opacity_header: bool = True,
                 fill_rectangle: bool = False,
                 border_width: float = 3,
                 widget_id: str = None):

        self._data = []
        self._layout = []
        self._annotations = {}

        self.columns_number = columns_number

        self.preview_info = preview_info
        self._with_title_url = False

        self._last_used_column_index = 0

        #############################
        # grid gallery settings
        self.enable_zoom: bool = enable_zoom
        self.sync_views: bool = sync_views
        self.resize_on_zoom: bool = resize_on_zoom
        self.show_preview: bool = show_preview
        self.selectable: bool = selectable
        self.opacity: float = opacity
        self.show_opacity_header: bool = show_opacity_header
        self.fill_rectangle: bool = fill_rectangle
        self.border_width: float = border_width
        #############################

        super().__init__(widget_id=widget_id, file_path=__file__)

    def generate_project_meta(self):
        objects_set = set()
        for cell_data in self._data:
            annotation: supervisely.Annotation = cell_data.get['annotation']
            for label in annotation.labels:
                objects_set.add(label.obj_class)

        objects_list = list(objects_set)
        objects_collection = supervisely.ObjClassCollection(objects_list) if len(objects_list) > 0 else None

        return supervisely.ProjectMeta(obj_classes=objects_collection)

    def get_json_data(self):
        return {
            "content": {
                "projectMeta": self.generate_project_meta().to_json(),
                "layout": self._layout,
                "annotations": self._annotations
            },
            "options": {
                "enableZoom": self.enable_zoom,
                "syncViews": self.sync_views,
                "resizeOnZoom": self.resize_on_zoom,
                "showPreview": self.show_preview,
                "selectable": self.selectable,
                "opacity": self.opacity,
                "showOpacityInHeader": self.show_opacity_header,
                "fillRectangle": self.fill_rectangle,
                "borderWidth": self.border_width,
                "viewHeight": '400px'
            }
        }

    def get_json_state(self):
        return None

    def get_column_index(self, incoming_value):
        if incoming_value is not None and 0 > incoming_value > self.columns_number:
            raise ValueError(f'column index == {incoming_value} is out of bounds')

        if incoming_value is None:
            incoming_value = self._last_used_column_index
            self._last_used_column_index = (self._last_used_column_index + 1) % self.columns_number
        else:
            self._last_used_column_index = incoming_value

        return incoming_value

    def append_column_by_image_url(self, image_url: str,
                                   annotation: Annotation = None,
                                   title: str = None,
                                   column_index: int = None,
                                   custom_info: dict = None,
                                   zoom_to_figure=None,
                                   title_url=None):

        cell_uuid = str(uuid.uuid5(namespace=uuid.NAMESPACE_URL, name=image_url).hex)
        column_index = self.get_column_index(incoming_value=column_index)

        self._data.append({
            "image_url": image_url,
            "annotation": Annotation((1, 1)) if annotation is None else annotation.clone(),
            "column_index": column_index,
            "title": title if title is not None else "",
            "cell_uuid": cell_uuid
        })

        self._update_layout()
        self._update_annotations()

        # if zoom_to_figure is not None:
        #     self._data[cell_uuid]["zoom_to_figure"] = zoom_to_figure
        #     self._need_zoom = True

        # if title_url is not None:
        #     self.preview_info = True
        #     self._with_title_url = True
        #     self._data[cell_uuid]["labelingUrl"] = title_url

        # if self.preview_info:
        #     if custom_info is not None:
        #         self._data[cell_uuid]["info"] = custom_info
        #     else:
        #         self._data[cell_uuid]["info"] = None
    #
    # def update(self, options=True):
    #     gallery_json = self.to_json()

    # def _zoom_to_figure(self, annotations):
    #     items = self._data.items()
    #     zoom_to_figure_name = "zoomToFigure"
    #     for item in items:
    #         curr_image_name = item[0]
    #         curr_image_data = item[1]
    #
    #         if type(curr_image_data["zoom_to_figure"]) is not tuple:
    #             raise ValueError("Option zoom_to_figure not set for {} image".format(curr_image_name))
    #         elif type(curr_image_data["zoom_to_figure"]) is None:
    #             raise ValueError("Option zoom_to_figure not set for {} image".format(curr_image_name))
    #
    #         zoom_params = {
    #             "figureId": curr_image_data["zoom_to_figure"][0],
    #             "factor": curr_image_data["zoom_to_figure"][1]
    #         }
    #         annotations[curr_image_name][zoom_to_figure_name] = zoom_params
    #
    # def _add_info(self, annotations):
    #     items = self._data.items()
    #     for item in items:
    #         curr_image_name = item[0]
    #         curr_image_data = item[1]
    #
    #         annotations[curr_image_name]["info"] = curr_image_data["info"]

    # def to_json(self):
    #     annotations = {}
    #     layout = [[] for _ in range(self.col_number)]
    #     index_in_layout = 0
    #
    #     for curr_image_name, curr_image_data in self._data.items():
    #         annotations[curr_image_name] = self._get_item_annotation(curr_image_name)
    #
    #         curr_col_index = curr_image_data["col_index"]
    #         if curr_col_index is not None:
    #             layout[curr_col_index - 1].append(curr_image_name)
    #         else:
    #             if index_in_layout == self.col_number:
    #                 index_in_layout = 0
    #             layout[index_in_layout].append(curr_image_name)
    #             index_in_layout += 1

    # if self._need_zoom:
    #     self._zoom_to_figure(annotations)
    #
    # if self.preview_info:
    #     self._add_info(annotations)

    def _update_layout(self):
        layout = [[] for _ in range(self.columns_number)]

        for cell_data in self._data:
            layout[cell_data['column_index']].append(cell_data['cell_uuid'])

        self._layout = copy.deepcopy(layout)
        DataJson()[self.widget_id]['content']['layout'] = self._layout

    def _update_annotations(self):
        annotations = {}

        for cell_data in self._data:
            annotations[cell_data['cell_uuid']] = {
                "url": cell_data["image_url"],
                "figures": [label.to_json() for label in cell_data["annotation"].labels],
                "title": cell_data["title"], #?
                "info": cell_data.get("info", None),
                "labelingUrl": cell_data.get("labelingUrl", None)
            }

        self._annotations = copy.deepcopy(annotations)
        DataJson()[self.widget_id]['content']['annotations'] = self._annotations

# @property
#   def title(self):
#       return self._title
#
#   @title.setter
#   def title(self, value):
#       self._title = value
#       DataJson()[self.widget_id]['title'] = self._title
#
#   @property
#   def description(self):
#       return self._description
#
#   @description.setter
#   def description(self, value):
#       self._description = value
#       DataJson()[self.widget_id]['description'] = self._description


# def add_item_by_id(self, image_id, with_ann=True, col_index=None, info_dict=None,
#                    zoom_to_figure=None, title_url=None):
#     image_info = self._api.image.get_info_by_id(image_id)
#     if with_ann:
#         ann_info = self._api.annotation.download(image_id)
#         ann = supervisely.Annotation.from_json(ann_info.annotation, self._project_meta)
#     else:
#         ann = None
#
#     self.add_item(image_info.name, image_info.full_storage_url, ann, col_index, info_dict, zoom_to_figure,
#                   title_url)

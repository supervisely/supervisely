import supervisely as sly
from supervisely.app import DataJson, StateJson
from supervisely.app.widgets import Widget
from supervisely.app.widgets_context import JinjaWidgets


class ImageRegionSelector(Widget):
    class Routes:
        BBOX_CHANGED = "bbox_changed"

    def __init__(self, 
        image_info: sly.ImageInfo, 
        widget_id: str = None, 
        disabled: bool = False,
        widget_width: str = "100%",
        widget_height: str = "100%",
    ):
        self.image_link = None
        self.image_name = None
        self.image_url = None
        self.image_hash = None
        self.image_size = None
        self.image_id = None
        self.dataset_id = None
        self.original_bbox = None
        self.scaled_bbox = None
        self._disabled = disabled
        self.widget_width = widget_width
        self.widget_height = widget_height

        super().__init__(widget_id=widget_id, file_path=__file__)
        self.image_update(image_info)
        script_path = "./sly/css/app/widgets/image_region_selector/script.js"
        JinjaWidgets().context["__widget_scripts__"][self.__class__.__name__+'1'] = "https://cdn.jsdelivr.net/npm/svg.js@2.7.1/dist/svg.min.js"
        JinjaWidgets().context["__widget_scripts__"][self.__class__.__name__+'2'] = "https://cdn.jsdelivr.net/npm/svg.select.js@3.0.1/dist/svg.select.min.js"
        JinjaWidgets().context["__widget_scripts__"][self.__class__.__name__+'3'] = "https://cdn.jsdelivr.net/npm/svg.resize.js@1.4.3/dist/svg.resize.min.js"
        JinjaWidgets().context["__widget_scripts__"][self.__class__.__name__+'4'] = "https://cdn.jsdelivr.net/npm/svg.draggable.js@2.2.2/dist/svg.draggable.min.js"
        JinjaWidgets().context["__widget_scripts__"][self.__class__.__name__+'5'] = "https://cdn.jsdelivr.net/npm/svg.panzoom.js@1.2.3/dist/svg.panzoom.min.js"
        JinjaWidgets().context["__widget_scripts__"][self.__class__.__name__+'6'] = "https://rawgit.com/nodeca/pako/1.0.11/dist/pako.min.js"
        JinjaWidgets().context["__widget_scripts__"][self.__class__.__name__+'7'] = "https://cdnjs.cloudflare.com/ajax/libs/uuid/8.3.2/uuidv4.min.js"
        JinjaWidgets().context["__widget_scripts__"][self.__class__.__name__] = script_path

    def image_update(self, image_info: sly.ImageInfo):
        self.image_link = image_info.full_storage_url
        self.image_name = image_info.name

        self.image_url = image_info.full_storage_url
        self.image_hash = image_info.hash
        self.image_size = image_info.size

        self.image_id = image_info.id
        self.dataset_id = image_info.dataset_id

        self.original_bbox = [[int(image_info.width * 0.1), int(image_info.height * 0.1)],
                              [int(image_info.width * 0.9), int(image_info.height * 0.9)]]
        self.scaled_bbox = self.original_bbox
        
        StateJson()[self.widget_id].update(self.get_json_state())
        StateJson().send_changes()

    def bbox_changed(self, func):
            route_path = self.get_route_path(ImageRegionSelector.Routes.BBOX_CHANGED)
            server = self._sly_app.get_server()
            self._changes_handled = True

            @server.post(route_path)
            def _click():
                self.bbox_update()
                res = self.scaled_bbox
                func(res)

            return _click
    
    def bbox_update(self):
        self.scaled_bbox = StateJson()[self.widget_id]['scaledBbox']
        bboxes_padding = 0

        scaled_width, scaled_height = self.get_bbox_size(self.scaled_bbox)
        original_width, original_height = int(scaled_width / (1 + bboxes_padding)), int(
            scaled_height / (1 + bboxes_padding))

        div_width, div_height = (scaled_width - original_width) // 2, (scaled_height - original_height) // 2

        self.original_bbox[0][0] = self.scaled_bbox[0][0] + div_width
        self.original_bbox[0][1] = self.scaled_bbox[0][1] + div_height
        self.original_bbox[1][0] = self.scaled_bbox[1][0] - div_width
        self.original_bbox[1][1] = self.scaled_bbox[1][1] - div_height
        StateJson().send_changes()

    def get_json_data(self):
        return {
        }

    def get_json_state(self):
        return {
            'imageLink': self.image_link,
            'imageName': self.image_name,
            'imageUrl': self.image_url,
            'imageHash': self.image_hash,
            'imageSize': self.image_size,
            'imageId': self.image_id,
            'datasetId': self.dataset_id,
            'originalBbox': self.original_bbox,
            'scaledBbox': self.scaled_bbox,
            'disabled': self._disabled,
            'widget_width': self.widget_width,
            'widget_height': self.widget_height,
            'widget_id': self.widget_id,
        }

    @property
    def is_empty(self):
        if len(self.original_bbox) > 0:
            return False
        return True

    def get_bbox_size(self, current_bbox):
        box_width = current_bbox[1][0] - current_bbox[0][0]
        box_height = current_bbox[1][1] - current_bbox[0][1]
        return box_width, box_height

    def add_bbox_padding(self, padding_coefficient=0):
        padding_coefficient /= 100

        original_w, original_h = self.get_bbox_size(current_bbox=self.original_bbox)
        additional_w, additional_h = int(original_w * padding_coefficient // 2), int(original_h * padding_coefficient // 2),

        self.scaled_bbox[0][0] = self.original_bbox[0][0] - additional_w if self.original_bbox[0][0] - additional_w > 0 else 0
        self.scaled_bbox[0][1] = self.original_bbox[0][1] - additional_h if self.original_bbox[0][1] - additional_h > 0 else 0
        self.scaled_bbox[1][0] = self.original_bbox[1][0] + additional_w if self.original_bbox[1][0] + additional_w < self.image_size[0] else self.image_size[0] - 1
        self.scaled_bbox[1][1] = self.original_bbox[1][1] + additional_h if self.original_bbox[1][1] + additional_h < self.image_size[1] else self.image_size[1] - 1
        StateJson.send_changes()

    def get_relative_coordinates(self, abs_coordinates):
        box_width, box_height = self.get_bbox_size(current_bbox=self.scaled_bbox)
        return {
            'x': (abs_coordinates['position'][0][0] - self.scaled_bbox[0][0]) / box_width,
            'y': (abs_coordinates['position'][0][1] - self.scaled_bbox[0][1]) / box_height,
        }

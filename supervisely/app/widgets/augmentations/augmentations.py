import os
import json
import random
from typing import List, Optional, Dict
from collections import namedtuple

import supervisely as sly
from supervisely.app import StateJson, DataJson
from supervisely.app.widgets import Container, Widget, RadioTabs, Editor, GridGallery, Select, Input, Field, Button

import src.sly_globals as g


class Augmentations(Widget):
    def __init__(
        self,
        image_info = None, 
        task_type: str = None,
        remote_preview_path: str = '/temp/preview_augs.jpg',
        widget_id=None
    ):  
        self._image_info = image_info
        self._task_type = task_type
        self._remote_preview_path = remote_preview_path

        self._template_path  = None
        self._pipeline = None
        self._py_code = None

        self._editor = Editor(language_mode='python', height_px=250, readonly=True)
        self._grid_gallery = GridGallery(
            columns_number=2,
            annotations_opacity=0.3,
            show_opacity_slider=True,
            enable_zoom=True,
            resize_on_zoom=False,
            sync_views=True,
            fill_rectangle=False,
        )
        self._grid_gallery.hide()
        self._button_preview = Button('Preview on random image')
        self._content = Container([self._editor, self._grid_gallery, self._button_preview])
        
        @self._button_preview.click
        def update_preview():
            self.preview_augs()
        super().__init__(widget_id=widget_id, file_path=__file__)
    
    def preview_augs(self, image_info = None):
        if not image_info:
            ds_name, item_name = self.get_random_item()
            self._image_info = self.get_image_info_from_cache(ds_name, item_name)

        img = g.api.image.download_np(self._image_info.id)
        ann_json = g.api.annotation.download(self._image_info.id).annotation

        image_ann = sly.Annotation.from_json(ann_json, g.project_meta)
        meta = g.project_meta
        if self._task_type == "detection":
            image_ann, meta = self.convert_ann_to_bboxes(image_ann, g.project_meta)

        _, res_img, res_ann = sly.imgaug_utils.apply(self._pipeline, meta, img, image_ann)
        local_image_path = os.path.join(g.data_dir, "preview_augs.jpg")
        sly.image.write(local_image_path, res_img)
        if g.api.file.exists(g.team.id, self._remote_preview_path):
            g.api.file.remove(g.team.id, self._remote_preview_path)
        file_info = g.api.file.upload(g.team.id, local_image_path, self._remote_preview_path)
        
        self._grid_gallery.clean_up()
        self._grid_gallery.append(
            title=f"Original", image_url=self._image_info.full_storage_url, annotation=image_ann
        )
        self._grid_gallery.append(
            title=f"Augmented", image_url=file_info.full_storage_url, annotation=res_ann
        )
        self._grid_gallery.show()
    
    def get_random_item(self):
        all_ds_names = g.project_fs.datasets.keys()
        ds_name = random.choice(all_ds_names)
        ds = g.project_fs.datasets.get(ds_name)
        items = list(ds)
        item_name = random.choice(items)
        return ds_name, item_name
    
    def get_image_info_from_cache(self, dataset_name, item_name):
        dataset_fs = g.project_fs.datasets.get(dataset_name)
        img_info_path = dataset_fs.get_img_info_path(item_name)
        image_info_dict = sly.json.load_json_file(img_info_path)
        ImageInfo = namedtuple('ImageInfo', image_info_dict)
        info = ImageInfo(**image_info_dict)
        return info

    def update_augmentations(self, path_or_data: str = None, string_format: str = 'python'):
        if path_or_data.endswith('.json'):
            self._template_path = path_or_data
            config = sly.json.load_json_file(self._template_path)
            self._pipeline, self._py_code = self.load_augs_template(config)
        elif path_or_data.endswith('.py'):
            # TODO 
            raise NotImplementedError('.py files not supported yet.')
        else:
            if string_format == 'json':
                config = json.loads(path_or_data)
                self._pipeline, self._py_code = self.load_augs_template(config)
            elif string_format == 'python':
                # TODO create func to conversion augs  
                locals = {}
                exec(path_or_data, {}, locals)
                pipeline = locals['seq']
                sly.imgaug_utils.aug_to_python(pipeline)
                self._py_code = path_or_data
                raise NotImplementedError('Raw python augmentations not supported yet.')
            else:
                raise ValueError('Supported values for "string_format" is "python" or "json"')
        self._editor.set_text(text=self._py_code)
    
    def get_augmentations(self):
        return self._pipeline, self._py_code

    def get_json_data(self) -> Dict:
        return {}

    def get_json_state(self) -> Dict:
        return {}
    
    @staticmethod
    def convert_ann_to_bboxes(image_ann, project_meta):
        meta = project_meta.clone()
        for obj_class in meta.obj_classes:
            if obj_class.geometry_type == "rectangle":
                continue
            class_obj = sly.ObjClass(obj_class.name, sly.Rectangle, obj_class.color)
            meta = meta.delete_obj_class(obj_class.name)
            meta = meta.add_obj_class(class_obj)
        new_ann_json = {
            "size": {
                "height": image_ann.img_size[0],
                "width": image_ann.img_size[1]
            },
            "tags": [],
            "objects": []
        }
        new_ann = sly.Annotation.from_json(new_ann_json, meta)
        for label in image_ann.labels:
            if label.geometry.geometry_name == "rectangle":
                new_ann = image_ann.add_label(label)
                continue
            class_obj = sly.ObjClass(label.obj_class.name, sly.Rectangle, label.obj_class.color)
            updated_label = label.convert(class_obj)[0]
            new_ann = new_ann.add_label(updated_label)
        return new_ann, meta

    @staticmethod
    def load_augs_template(config: dict):
        pipeline = sly.imgaug_utils.build_pipeline(config["pipeline"], random_order=config["random_order"]) # to validate
        py_code = sly.imgaug_utils.pipeline_to_python(config["pipeline"], config["random_order"])
        return pipeline, py_code

    def disable(self):
        self._disabled = True
        self._editor.disable()
        self._grid_gallery.disable()
        self._button_preview.disable()
        DataJson()[self.widget_id]["disabled"] = self._disabled
        DataJson().send_changes()

    def enable(self):
        self._disabled = False
        self._editor.enable()
        self._grid_gallery.enable()
        self._button_preview.enable()
        DataJson()[self.widget_id]["disabled"] = self._disabled
        DataJson().send_changes()


class AugmentationsWithTabs(Widget):
    def __init__(
        self,
        image_info = None, 
        templates: Optional[List[Dict[str, str]]] = None,
        task_type: str = None,
        remote_preview_path: str = '/temp/preview_augs.jpg',
        widget_id=None
    ):  
        self._image_info = image_info
        self._templates = templates
        self._task_type = task_type
        self._remote_preview_path = remote_preview_path
        
        self._augs1 = Augmentations(image_info=self._image_info, task_type=self._task_type)
        self._augs2 = Augmentations(image_info=self._image_info, task_type=self._task_type)
        self._augs2._editor.hide()
        self._augs2._button_preview.hide()
        self._template_path_input = Input(placeholder="Path to .json file in Team Files")
        self._template_selector = Select(
            items=[Select.Item(value=t['value'], label=t['label']) for t in templates], 
            filterable=True, 
            placeholder="select me"
        )
        self._button_template_update = Button('Load template from file')
        self._radio_tabs = RadioTabs(
            titles=["From template", "Custom pipeline"],
            contents=[
                Field(
                    title='Template', 
                    content=Container([self._template_selector, self._augs1]),
                ),
                Field(
                    title='Path to JSON configuration', 
                    description='Copy path in Team Files', 
                    content=Container([self._template_path_input, self._button_template_update, self._augs2])
                )
            ],
            descriptions=[
                "Choose one of the prepared templates",
                "Use ImgAug Studio appto configure and save custom augmentations",
            ],
        )
        
        self._current_augs = self._augs1
        self._current_augs.update_augmentations(templates[0]['value'])

        @self._radio_tabs.value_changed
        def tab_toggle(tab_title):
            if tab_title == 'Custom pipeline':
                self._current_augs = self._augs2
            else:
                self._current_augs = self._augs1

        @self._template_selector.value_changed
        def selector_value_changed(value = None):
            self._current_augs.update_augmentations(value)

        @self._button_template_update.click
        def update_template():
            custom_template_path = self._template_path_input.get_value()
            if g.api.file.exists(team_id=g.team.id, remote_path=custom_template_path.strip()):
                self.load_existing_pipeline(custom_template_path)
                self._augs2._editor.show()
                self._augs2._button_preview.show()
            else:
                raise FileExistsError("Team files does't contains file by passed path.")

        self._content = Container([self._radio_tabs])
        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self) -> Dict:
        return {}

    def get_json_state(self) -> Dict:
        return {}
  
    def get_augmentations(self):
        return self._current_augs.get_augmentations()
    
    def load_existing_pipeline(self, remote_path):
        custom_pipeline_path = os.path.join(g.data_dir, sly.fs.get_file_name_with_ext(remote_path))
        g.api.file.download(g.team.id, remote_path, custom_pipeline_path)
        self._current_augs.update_augmentations(custom_pipeline_path)

    def disable(self):
        self._disabled = True
        self._current_augs.disable()
        self._radio_tabs.disable()
        DataJson()[self.widget_id]["disabled"] = self._disabled
        DataJson().send_changes()

    def enable(self):
        self._disabled = False
        self._current_augs.enable()
        self._radio_tabs.enable()
        DataJson()[self.widget_id]["disabled"] = self._disabled
        DataJson().send_changes()
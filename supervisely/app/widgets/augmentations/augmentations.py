import os
import json
from typing import List, Optional, Dict

import supervisely as sly
from supervisely.app import StateJson
from supervisely.app.widgets import Container, Widget, RadioTabs, Editor, GridGallery, Select, Input, Field, Button


SAMPLE_AUGS = """import imgaug.augmenters as iaa

seq = iaa.Sequential([
	iaa.Sometimes(0.2, iaa.blur.GaussianBlur(sigma=(0, 3))),
	iaa.Sometimes(0.5, iaa.contrast.GammaContrast(gamma=(0.8, 1.2), per_channel=True)),
], random_order=False)
"""


class Augmentations(Widget):
    def __init__(
        self,
        templates: Optional[list[dict]] = None,
        images: Optional[list] = None, 
        widget_id=None
    ):  
        self._content = []
        self._editor = Editor(
            initial_text=SAMPLE_AUGS,
            language_mode='python',
            height_px=250
        )

        if templates:
            self._template_path_input = Input(placeholder="Path to .json file in Team Files")
            self._template_selector = Select(
                items=[Select.Item(value=t['value'], label=t['label']) for t in templates], 
                filterable=True, 
                placeholder="select me"
            )
            self._button_template_update = Button('Load template from file')
            self._radio_tabs = RadioTabs(
                titles=["Default template", "Custom pipeline"],
                contents=[
                    Field(
                        title='Template', 
                        content=self._template_selector,
                    ),
                    Field(
                        title='Path to JSON configuration', 
                        description='Copy path in Team Files', 
                        content=Container([self._template_path_input, self._button_template_update])
                    )
                ],
                descriptions=[
                    "Choose one of the prepared templates",
                    "Use ImgAug Studio appto configure and save custom augmentations",
                ],
            )
            self._content.append(Container([self._radio_tabs]))

            @self._template_selector.value_changed
            def selector_value_changed(value = None):
                print(f'New values: {value}')
                self.update_augmentations(value)

            @self._button_template_update.click
            def update_template():
                self.update_augmentations(self._template_path_input)
        self._content.append(Container([self._editor]))

        self._button_preview = Button('Preview on random image')
        self._grid_gallery = GridGallery(columns_number=2)
        self._grid_gallery.hide()
        self._content.append(Container([self._button_preview, self._grid_gallery]))
        
        @self._button_preview.click
        def update_preview():
            # TODO 
            # self._grid_gallery.update_data()
            self._grid_gallery.show()
            # self._grid_gallery.update_data()
            
        self._content = Container(self._content)

        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self) -> Dict:
        return {}

    def get_json_state(self) -> Dict:
        return {}
    
    def update_augmentations(self, path_or_data: str = None, string_format: str = 'python'):
        if path_or_data is None:
            if self._radio_tabs.get_active_tab() == 'Custom pipeline':
                self._template_path = self._template_path_input.get_value()
            else:
                self._template_path = self._template_selector.get_value()
            config = sly.json.load_json_file(self._template_path)
            self._pipeline, self._py_code = self.load_augs_template(config)
        elif path_or_data.endswith('.json'):
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
                # # TODO create func to conversion augs  
                # locals = {}
                # exec(path_or_data, {}, locals)
                # pipeline = locals['seq']
                # self._py_code = path_or_data
                raise NotImplementedError('.py files not supported yet.')
            else:
                raise ValueError('Supported values for "string_format" is "python" or "json"')
        self._editor.set_text(text=self._py_code)
    
    @staticmethod
    def load_augs_template(config: dict):
        pipeline = sly.imgaug_utils.build_pipeline(config["pipeline"], random_order=config["random_order"]) # to validate
        py_code = sly.imgaug_utils.pipeline_to_python(config["pipeline"], config["random_order"])
        return pipeline, py_code

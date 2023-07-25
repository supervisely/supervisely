from typing import List, Optional, Dict

import supervisely as sly
from supervisely.app import StateJson, DataJson
from supervisely.app.widgets import Button, GridGallery, Slider, Widget


class PredictionsGallery(Widget):
    def __init__(
        self,
        slider_title: Optional[str] = "epochs",
        enable_zoom: Optional[bool] = False,
        opacity: Optional[float] = 0.4,
        widget_id=None,
    ):

        # save data for gallery
        self._image_url = None
        self._gt_ann = None
        self._gt_title = None
        self._data = []
        self._titles = []


        # save gallery state
        self._columns = 2
        self._current_grid = 0
        self._total_grids = 0
        self._slider_title = slider_title

        self._first_button = Button(
            "", icon="zmdi zmdi-skip-previous", plain=True, button_size="mini"
        )
        self._prev_button = Button(
            "", icon="zmdi zmdi-chevron-left", plain=True, button_size="mini"
        )
        self._next_button = Button(
            "", icon="zmdi zmdi-chevron-right", plain=True, button_size="mini"
        )
        self._last_button = Button("", icon="zmdi zmdi-skip-next", plain=True, button_size="mini")

        self._slider = Slider(show_stops=True, min=1, max=1, step=1, value=1)

        self._grid_gallery = GridGallery(
            columns_number=self._columns,
            annotations_opacity=opacity,
            show_opacity_slider=True,
            enable_zoom=enable_zoom,
            resize_on_zoom=False,
            sync_views=True,
            fill_rectangle=False,
        )
        self._grid_gallery.hide()

        @self._slider.value_changed
        def on_slider_change(value):
            self._update_gallery(int(value))

        @self._first_button.click
        def on_first_button_click():
            if self._current_grid > 1:
                self._update_gallery(1)

        @self._prev_button.click
        def on_prev_button_click():
            if self._current_grid > 1:
                self._update_gallery(self._current_grid - 1)

        @self._next_button.click
        def on_next_button_click():
            if self._current_grid < self._total_grids:
                self._update_gallery(self._current_grid + 1)

        @self._last_button.click
        def on_last_button_click():
            if self._current_grid < self._total_grids:
                self._update_gallery(self._total_grids)

        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self) -> Dict:
        return {}

    def get_json_state(self) -> Dict:
        return {
            "currentGrid": self._current_grid,
            "totalGrids": self._total_grids,
            "sliderTitle": f" {self._slider_title}",
        }

    def set_ground_truth(
        self,
        image_url: str,
        annotation: sly.Annotation,
        title: str = "Ground truth",
    ):
        self._image_url = image_url
        self._gt_ann = annotation
        self._gt_title = title
        self._update_data()
        # self.add_prediction(annotation=annotation, title=title)

    def add_predictions(self, annotations: List[sly.Annotation], titles: List[str] = None):
        titles = titles if titles is not None else [""] * len(annotations)
        self._data.extend(annotations)
        self._titles.extend(titles)
        self._update_data()

    def add_prediction(self, annotation: sly.Annotation, title: str = None):
        self._data.append(annotation)
        title = title if title is not None else ""
        self._titles.append(title)
        self._update_data()

    def _update_data(self):
        self._total_grids = max(len(self._data), 1)
        last_page = False
        if self._slider.get_value() == self._slider.get_max():
            last_page = True

        self._slider.set_max(self._total_grids)

        if last_page:
            self._update_gallery(self._total_grids)

        StateJson()[self.widget_id]["totalGrids"] = self._total_grids
        StateJson().send_changes()

    def _update_gallery(self, page: int):
        if self._grid_gallery.is_hidden():
            self._grid_gallery.show()
        self._slider.set_value(page)
        self._current_grid = page
        StateJson()[self.widget_id]["currentGrid"] = page
        StateJson().send_changes()

        self._grid_gallery.clean_up()
        self._grid_gallery.append(self._image_url, self._gt_ann, self._gt_title) # set gt
        for idx in range(self._columns - 1):
            ann_index = (page - 1) * (self._columns - 1) + idx
            title_index = (page - 1) * (self._columns - 1) + idx
            self._grid_gallery.append(
                title=self._titles[title_index] if title_index < len(self._titles) else None,
                image_url=self._image_url,
                annotation=self._data[ann_index] if ann_index < len(self._data) else None,
            )
        DataJson().send_changes()

    def disable(self):
        self._disabled = True
        self._grid_gallery.disable()
        self._first_button.disable()
        self._prev_button.disable()
        self._next_button.disable()
        self._last_button.disable()
        self._slider.disable()
        DataJson()[self.widget_id]["disabled"] = self._disabled
        DataJson().send_changes()

    def enable(self):
        self._disabled = False
        self._grid_gallery.enable()
        self._first_button.enable()
        self._prev_button.enable()
        self._next_button.enable()
        self._last_button.enable()
        self._slider.enable()
        DataJson()[self.widget_id]["disabled"] = self._disabled
        DataJson().send_changes()

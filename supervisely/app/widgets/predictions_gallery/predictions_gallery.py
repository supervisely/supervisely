from typing import List, Optional, Dict

import supervisely as sly
from supervisely.app import StateJson, DataJson
from supervisely.app.widgets import Button, GridGallery, Slider, Widget


class PredictionsGallery(Widget):
    def __init__(
        self,
        columns_number=2,
        slider_title: Optional[str] = "epochs",
        single_image_mode: Optional[bool] = True,
        enable_zoom: Optional[bool] = False,
        opacity: Optional[float] = 0.3,
        widget_id=None,
    ):
        self._columns = columns_number

        # save data for gallery
        self._image_urls = []
        self._data = []
        self._titles = []

        self._slider_title = slider_title
        self._single_image_mode = single_image_mode

        # save gallery state
        self._current_grid = 0
        self._total_grids = 0

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
            self._update_data()
            self._update_gallery(int(value))

        @self._first_button.click
        def on_first_button_click():
            self._update_gallery(1)

        @self._prev_button.click
        def on_prev_button_click():
            self._update_gallery(self._current_grid - 1)

        @self._next_button.click
        def on_next_button_click():
            self._update_gallery(self._current_grid + 1)

        @self._last_button.click
        def on_last_button_click():
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

    def add_images(
        self,
        image_urls: List[str],
        annotations: List[sly.Annotation] = None,
        titles: List[str] = None,
    ):
        self._image_urls.extend(image_urls)
        if titles is not None:
            self._titles.extend(titles)
        else:
            self._titles.extend([""] * len(image_urls))
        for idx in range(self._columns):
            image_index = (
                idx
                if self._single_image_mode is True
                else (self._total_grids - 1) * self._columns + idx
            )
            self._grid_gallery.append(
                title=self._titles[idx],
                image_url=self._image_urls[image_index],
            )
        if annotations is not None:
            self.add_annotations(annotations=annotations)

        # self._update_data()
        # self._grid_gallery._update()

    def add_image(
        self,
        image_url: str,
        annotation: sly.Annotation = None,
        title: str = None,
    ):
        annotations = [annotation] if annotation is not None else None
        titles = [title] if title is not None else None
        self._add_images(image_urls=[image_url], annotations=annotations, titles=titles)

    def add_annotations(self, annotations: List[sly.Annotation]):
        self._data.extend(annotations)
        self._update_data()

    def add_annotation(self, annotation: sly.Annotation):
        self._add_annotations(annotations=[annotation])

    def _update_data(self):
        pages_cnt = len(self._data) // self._columns + len(self._data) % self._columns
        self._total_grids = pages_cnt
        last_page = False
        if self._slider.get_value() == self._slider.get_max():
            last_page = True

        self._slider.set_max(pages_cnt)

        if last_page:
            self._update_gallery(pages_cnt)
        elif self._current_grid == 0:
            self._update_gallery(1)

        StateJson()[self.widget_id]["totalGrids"] = pages_cnt
        StateJson().send_changes()

    def _update_gallery(self, page: int):
        if page < 1 or page > self._total_grids or page == self._current_grid:
            return
        if self._grid_gallery.is_hidden():
            self._grid_gallery.show()
        self._slider.set_value(page)
        self._current_grid = page
        StateJson()[self.widget_id]["currentGrid"] = page
        StateJson().send_changes()

        self._grid_gallery.clean_up()
        for idx in range(self._columns):
            image_index = (
                idx if self._single_image_mode is True else (page - 1) * self._columns + idx
            )
            self._grid_gallery.append(
                title=self._titles[idx],
                image_url=self._image_urls[image_index],
                annotation=self._data[(page - 1) * self._columns + idx],
            )

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

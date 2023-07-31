import os
import json
from typing import List, Optional, Tuple

import supervisely as sly
from supervisely.app import StateJson, DataJson
from supervisely.app.widgets import Button, FolderThumbnail, GridGallery, Slider, Widget
from supervisely.app import get_data_dir


class ImagePairsSequence(Widget):
    def __init__(
        self,
        opacity: Optional[float] = 0.4,
        enable_zoom: Optional[bool] = False,
        sync_views: Optional[bool] = True,
        slider_title: Optional[str] = "pairs",
        widget_id=None,
    ):
        self._api = sly.Api.from_env()
        self._team_id = sly.env.team_id()

        # init data for gallery
        self._left_data = []
        self._right_data = []
        self._info = {"left": [], "right": []}

        # init gallery options
        self._columns = 2
        self._current_grid = 0
        self._total_grids = 0
        self._slider_title = slider_title
        self._need_update = False

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

        self._folder_thumbnail = FolderThumbnail()

        self._grid_gallery = GridGallery(
            columns_number=self._columns,
            annotations_opacity=opacity,
            show_opacity_slider=True,
            enable_zoom=enable_zoom,
            resize_on_zoom=False,
            sync_views=sync_views,
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

    def get_json_data(self) -> dict:
        return {}

    def get_json_state(self) -> dict:
        return {
            "currentGrid": self._current_grid,
            "totalGrids": self._total_grids,
            "sliderTitle": f" {self._slider_title}",
        }

    def set_left(self, url: str, ann: sly.Annotation = None, title: str = None):
        data = [url, ann, title]
        self._add_with_check("left", [data])
        self._update_data()

    def set_right(self, url: str, ann: sly.Annotation = None, title: str = None):
        data = [url, ann, title]
        self._add_with_check("right", [data])
        self._update_data()

    def set_left_batch(
        self, urls: List[str], anns: List[sly.Annotation] = None, titles: List[str] = None
    ):
        anns = [None] * len(urls) if anns is None else anns
        titles = [None] * len(urls) if titles is None else titles
        data = list(zip(urls, anns, titles))
        self._add_with_check("left", data)
        self._update_data()

    def set_right_batch(
        self, urls: List[str], anns: List[sly.Annotation] = None, titles: List[str] = None
    ):
        anns = [None] * len(urls) if anns is None else anns
        titles = [None] * len(urls) if titles is None else titles
        data = list(zip(urls, anns, titles))
        self._add_with_check("right", data)
        self._update_data()

    def set_pair(
        self,
        left: Tuple[str, Optional[sly.Annotation], Optional[str]],
        right: Tuple[str, Optional[sly.Annotation], Optional[str]],
    ):
        self._add_with_check("left", [left])
        self._add_with_check("right", [right])
        self._update_data()

    def set_pairs_batch(self, left: List[Tuple], right: List[Tuple]):
        self._add_with_check("left", left)
        self._add_with_check("right", right)
        self._update_data()

    def clean_up(self):
        self._left_data = []
        self._right_data = []
        self._grid_gallery.hide()
        self._slider.set_value(1)
        self._slider.set_max(1)
        self._need_update = False
        self._current_grid = 0
        self._total_grids = 0
        StateJson()[self.widget_id]["totalGrids"] = 0
        StateJson()[self.widget_id]["currentGrid"] = 0
        StateJson().send_changes()

    def _add_with_check(self, side, data):
        data: List[Tuple[str, Optional[sly.Annotation], Optional[str]]]
        total_grids = max(len(self._left_data), len(self._right_data), 1)
        if self._total_grids != total_grids:
            self._need_update = True

        has_empty_before = any([len(self._left_data) == 0, len(self._right_data) == 0])

        if side == "left":
            self._left_data.extend(data)
        elif side == "right":
            self._right_data.extend(data)

        has_empty_after = any([len(self._left_data) == 0, len(self._right_data) == 0])
        if has_empty_before and not has_empty_after:
            self._need_update = True

    def _dump_data(self, side, data, dir_path):
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        images = []
        anntations = []
        drawed = []

        def _generate_name(template, names):
            name = sly.generate_free_name(names, template, with_ext=True)
            names.append(name)
            return os.path.join(dir_path, name)

        for d in data:
            url, ann, title = d
            ext = sly.fs.get_file_ext(url)

            local_image_path = _generate_name("image" + ext, images)
            local_ann_path = _generate_name("annotation.json", anntations)
            annotated_path = _generate_name("drawed" + ext, drawed)

            sly.fs.download(url, local_image_path)
            img = sly.image.read(local_image_path)
            img_size = img.shape[0], img.shape[1]
            ann = ann if ann is not None else sly.Annotation(img_size)

            ann.draw_pretty(img, output_path=annotated_path)
            with open(local_ann_path, "w") as f:
                json.dump(ann.to_json(), f)

            self._info[side].append(
                {"imageName": images[-1], "url": url, "ann": ann.to_json(), "title": title or ""}
            )

    def dump_data(self, remote_dir="tmp/data/image pairs sequence/"):
        new_suffix = 1
        res_name = remote_dir.strip("/")
        while self._api.file.dir_exists(self._team_id, "/" + res_name):
            res_name = "{}_{:02d}".format(remote_dir.rstrip("/"), new_suffix)
            new_suffix += 1
        remote_dir = res_name
        local_dir = os.path.join(get_data_dir(), os.path.basename(remote_dir))
        if not os.path.exists(local_dir):
            os.makedirs(local_dir)
        left_data_dir = os.path.join(local_dir, "left")
        right_data_dir = os.path.join(local_dir, "right")

        self._dump_data("left", self._left_data, left_data_dir)
        self._dump_data("right", self._right_data, right_data_dir)

        with open(os.path.join(local_dir, "info.json"), "w") as f:
            json.dump(self._info, f)

        self._api.file.upload_directory(self._team_id, local_dir, remote_dir)

        return remote_dir

    def _update_data(self):
        self._total_grids = max(len(self._left_data), len(self._right_data), 1)
        min_len = min(len(self._left_data), len(self._right_data))
        if self._slider.get_value() == self._slider.get_max():
            self._need_update = True

        self._slider.set_max(self._total_grids)

        if self._need_update and min_len > 0:
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

        len_left = len(self._left_data)
        len_right = len(self._right_data)

        if len_left > 0 and len_right > 0:
            left = self._left_data[page - 1] if page <= len_left else self._left_data[-1]
            self._grid_gallery.append(*left)  # set left

            right = self._right_data[page - 1] if page <= len_right else self._right_data[-1]
            self._grid_gallery.append(*right)  # set right

        DataJson().send_changes()
        self._need_update = False

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

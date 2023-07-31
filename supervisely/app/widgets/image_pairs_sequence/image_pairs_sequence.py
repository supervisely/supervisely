import os
import pathlib
from pathlib import Path
from typing import List, Optional, Tuple

import supervisely as sly
from supervisely.app import StateJson, DataJson
from supervisely.app.widgets import Button, FolderThumbnail, GridGallery, Slider, Widget


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

        self._dump_image_to_offline_sessions_file([x[0] for x in data])

    def _dump_image_to_offline_sessions_file(self, urls: List[str]):
        if sly.is_production():
            task_id = self._api.task_id
            remote_dir = pathlib.Path(
                "/",
                "offline-sessions",
                str(task_id),
                "app-template",
                "sly",
                "css",
                "app",
                "widgets",
                "image_pairs_sequence",
            )
            dst_paths = [remote_dir.joinpath(pathlib.Path(url).name).as_posix() for url in urls]
            local_paths = [
                os.path.join(sly.app.get_data_dir(), sly.fs.get_file_name_with_ext(url)) for url in urls
            ]
            for remote, local in zip(urls, local_paths):
                self._download_image(remote, local)
                # files = self._api.task.get_import_files_list(task_id)
                # self._api.task.download_import_file(task_id, os.path.basename(remote), local)

            res_remote_dir: str = self._api.file.upload_bulk(
                team_id=self._team_id,
                src_paths=local_paths,
                dst_paths=dst_paths,
            )
            sly.logger.info(f"File stored in {res_remote_dir} for offline usage")
        else:
            sly.logger.info("Debug mode: files are not stored for offline usage")

    def _download_image(self, url: str, save_path: str):
        filepath = None
        if url.startswith("/static"):
            app = self._sly_app.get_server()
            static_dir = Path(app.get_static_dir())
            filepath = url.lstrip("/").removeprefix("static/")
            filepath = static_dir.joinpath(filepath)
        else:
            filepath = url

        sly.fs.download(filepath, save_path)
        return save_path

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

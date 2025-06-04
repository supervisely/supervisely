from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Literal, Optional, Tuple
from uuid import uuid4

import supervisely.app.widgets as w
from supervisely.api.api import Api
from supervisely.api.image_api import ImageInfo
from supervisely.app.content import DataJson, StateJson
from supervisely.app.widgets import Widget
from supervisely.sly_logger import logger
from supervisely.solutions.components.card import Card
from supervisely.solutions.smart_sampling.functions import (
    SamplingMode,
    SamplingSettings,
    copy_to_new_project,
    sample,
)


class SmartSampling(Widget):
    def __init__(
        self,
        project_id: int,
        x: int,
        y: int,
        random_sampling: bool = True,
        diverse_sampling: bool = False,
        ai_search_sampling: bool = False,
        team_id: int = None,
        widget_id: str = None,
    ):
        """
        Smart Sampling widget for Supervisely app.
        :param project_id: int, ID of the project to sample from
        """
        self.project_id = project_id
        self.api = Api.from_env()
        self._x = x
        self._y = y
        self.project = self.api.project.get_info_by_id(project_id)
        self.team_id = team_id or self.project.team_id
        self._has_random_sampling = random_sampling
        self._has_diverse_sampling = diverse_sampling
        self._has_ai_search_sampling = ai_search_sampling
        self.total_num = self.project.items_count

        # Initialize widgets
        self._total_text = None
        self._total_num_text = None
        self._diff_text = None
        self._diff_num_text = None
        self._sampling_text = None

        # Initialize UI containers
        self._settings_modal = None
        self._tasks_modal = None
        self._automation_modal = None
        self._card = None
        self._settings_btn = None
        self._automate_ok_btn = None
        self._run_sample_btn = None

        # Initialize UI for sampling tabs
        self._sampling_tabs = None
        self._random_input = None
        self._diverse_input = None
        self._ai_search_input = None
        self._ai_search_limit_input = None
        self._ai_search_thrs_input = None

        # Initialize automation UI
        self._automate_checkbox = None
        self._automate_input = None
        self._automate_period_select = None

        # Initialize preview UI
        self._preview_btn = None
        self._preview_gallery = None
        self._preview_collapse = None
        self._preview_cont = None

        # Initialize tasks table
        self._tasks_table = None

        super().__init__(widget_id=widget_id, file_path=__file__)
        self.diff_num = self.get_differences_count()
        self.save_differences_count(self.diff_num)

        # callback to be called when sampling ends
        self._sampling_end_callback = None
        self._create_layout()

    # UI PROPERTIES
    @property
    def tasks_modal(self) -> w.Dialog:
        """Get the tasks modal dialog."""
        return self._tasks_modal

    @property
    def settings_modal(self) -> w.Dialog:
        """Get the settings modal dialog."""
        return self._settings_modal

    @property
    def automation_modal(self) -> w.Dialog:
        """Get the automate modal dialog."""
        return self._automation_modal

    @property
    def card(self) -> Card:
        """Get the card widget."""
        return self._card

    @property
    def node(self) -> str:
        """Get the Solution Node block associated with specific coordinates."""
        return self._card.node

    @property
    def settings_btn(self) -> w.Button:
        """Get the settings button"""
        return self._settings_btn

    @property
    def automate_btn(self) -> w.Button:
        """Get the automate OK button"""
        return self._automate_ok_btn

    @property
    def run_sample_btn(self) -> w.Button:
        """Get the run sample button"""
        return self._run_sample_btn

    # STATE and DATA
    def get_json_state(self) -> dict:
        """State of the Sampling block will be used in the Solution app GUI.
        Possible keys:
        - "differences_count": int, number of differences in the input project
        - "sampling_settings": dict, settings for sampling
        - "automation_details": dict, details for automation
        """
        return {}

    def get_json_data(self) -> dict:
        """
        Data will be used to save information for Solution app.
        For example, to save sampling settings or history.
        Possible keys
        - "sampling_history": list of dicts with sampling tasks details
        - "sampled_images": dict with sampled images
        """
        return {}

    # UI CREATION METHODS
    def _create_settings_modal(self) -> Widget:
        self._preview_btn = w.Button("Preview", icon="zmdi zmdi-eye", plain=True)
        self._preview_gallery = w.GridGallery(columns_number=3)
        self._preview_collapse = w.Collapse(
            items=[w.Collapse.Item("preview", "Preview", content=self._preview_gallery)]
        )
        preview_btn_cont = w.Container([self._preview_btn], style="align-items: flex-end")
        self._preview_cont = w.Container(
            [preview_btn_cont, self._preview_collapse], style="margin-bottom: 5px"
        )
        self._total_text = w.Text("Total images in the input project:")
        self._total_num_text = w.Text(f"{self.project.items_count} images")
        self._diff_text = w.Text("Available images for sampling:")
        self._diff_num_text = w.Text(f"{self.diff_num} images")
        self._sampling_text = w.Text(f" of {self.diff_num} images")

        tabs = []
        # random sampling
        if self._has_random_sampling:
            random_text_info = w.Text(
                "<strong>Random sampling</strong> is a simple way to select a random subset of images from the input project. <br> <strong>Note:</strong> The sampling is performed only on the images which were not copied to the labeling project yet.",
            )
            self._random_input = w.InputNumber()
            random_input_cont = w.Container(
                [self._random_input, self._sampling_text],
                direction="horizontal",
                gap=3,
                style="align-items: center",
            )
            random_nums_cont = w.Container(
                [self._total_num_text, self._total_num_text, random_input_cont]
            )
            random_sampling_text = w.Text("Sampling random:")
            random_texts = w.Container([self._total_text, self._diff_text, random_sampling_text])

            random_cont = w.Container(
                [random_texts, random_nums_cont, w.Empty()], direction="horizontal", gap=5
            )
            random_tab_cont = w.Container([random_text_info, random_cont, self._preview_cont])
            tabs.append(random_tab_cont)

        # diverse sampling
        if self._has_diverse_sampling:
            diverse_text_info = w.Text(
                "<strong>Diversity sampling strategy:</strong> Sampling most diverse images using k-means clustering. By default, embeddings are computed using the OpenAI CLIP model. <br> <strong>Note:</strong> The sampling is performed only on the images which were not copied to the labeling project yet.",
            )
            self._diverse_input = w.InputNumber()
            diverse_input_cont = w.Container(
                [self._diverse_input, self._sampling_text],
                direction="horizontal",
                gap=3,
                style="align-items: center",
            )
            diverse_nums_cont = w.Container(
                [self._total_num_text, self._total_num_text, diverse_input_cont]
            )
            diverse_sampling_text = w.Text("Sample size:")
            diverse_texts = w.Container([self._total_text, self._diff_text, diverse_sampling_text])
            diverse_cont = w.Container(
                [diverse_texts, diverse_nums_cont, w.Empty()], direction="horizontal", gap=5
            )
            diverse_tab_cont = w.Container([diverse_text_info, diverse_cont, self._preview_cont])
            tabs.append(diverse_tab_cont)

        # AI Search sampling
        if self._has_ai_search_sampling:
            ai_search_info = w.Text(
                "Sampling images from the project using <strong>AI Search</strong> by user-defined prompt to find the most suitable images. <br> <strong>Note:</strong> The sampling is performed only on the images which were not copied to the labeling project yet.",
            )
            ai_search_total = w.Container(
                [self._total_text, self._total_num_text], direction="horizontal", fractions=[2, 1]
            )
            ai_search_diff = w.Container(
                [self._diff_text, self._total_num_text], direction="horizontal", fractions=[2, 1]
            )
            ai_search_limit_text = w.Text(f"Limit: ")
            self._ai_search_limit_input = w.InputNumber(
                min=1, max=self.diff_num, value=self.diff_num
            )
            ai_search_limit = w.Container(
                [ai_search_limit_text, self._ai_search_limit_input],
                direction="horizontal",
                fractions=[2, 1],
            )

            self._ai_search_thrs_input = w.InputNumber(min=0, max=1, value=0.05, step=0.01)
            ai_search_thrs_text = w.Text(f"Threshold: ")
            ai_search_thrs = w.Container(
                [ai_search_thrs_text, self._ai_search_thrs_input],
                direction="horizontal",
                fractions=[2, 1],
            )

            ai_search_input_text = w.Text(f"Search query: ")
            self._ai_search_input = w.Input(placeholder="e.g. 'cat', 'dog', 'car'")
            ai_search_prompt = w.Flexbox([ai_search_input_text, self._ai_search_input])

            ai_search_tab_cont = w.Container(
                [
                    ai_search_info,
                    ai_search_total,
                    ai_search_diff,
                    ai_search_limit,
                    ai_search_thrs,
                    ai_search_prompt,
                    self._preview_cont,
                ]
            )
            tabs.append(ai_search_tab_cont)

        self._settings_btn = w.Button("Confirm settings", plain=True)
        sample_btn_cont = w.Container([self._settings_btn], style="align-items: flex-end")

        self._sampling_tabs = w.Tabs(
            labels=["Random", "Diverse", "AI Search"],
            contents=tabs,
            type="card",
        )
        run_sample_content = w.Container([self._sampling_tabs, sample_btn_cont])

        self._settings_modal = w.Dialog(
            title="Sampling Settings",
            content=run_sample_content,
            size="tiny",
        )

    def _create_tasks_modal(self) -> w.FastTable:
        """
        Create a table for displaying sampling tasks history.
        :return: FastTable instance
        """
        sampling_table_columns = [
            "#",
            "Mode",
            "Date and Time",
            "Items Count",
            "Settings",
            "Status",
        ]
        columns_options = [
            {},
            {"postfix": "mode", "tooltip": "Sampling mode used for this task"},
            {"tooltip": "description text"},
            {"postfix": "images", "tooltip": "Number of sampled images"},
            {"tooltip": "Settings used for sampling"},
            {},
        ]
        self._tasks_table = w.FastTable(
            columns=sampling_table_columns,
            sort_column_idx=0,
            fixed_columns=1,
            sort_order="asc",
            columns_options=columns_options,
        )
        self._tasks_table.hide()
        self._tasks_modal = w.Dialog(content=w.Container([self._tasks_table]))

    def _get_tasks_data(self) -> List[List[str]]:
        sampling_history = self.get_sampling_tasks()
        rows = []
        for idx, history_item in enumerate(sampling_history, start=1):
            settings = history_item.get("settings")
            row = [
                idx,
                history_item.get("mode"),
                history_item.get("timestamp"),
                history_item.get("items_count"),
                str(settings) if settings else "-",
                history_item.get("status"),
            ]
            rows.append(row)
        return rows

    def _create_automate_modal(self) -> w.Dialog:
        """
        Create a modal dialog for automating sampling.
        :return: w.Dialog instance
        """
        self._automate_checkbox = w.Checkbox(content="Run every", checked=False)
        self._automate_input = w.InputNumber(
            min=1, value=60, debounce=1000, controls=False, size="mini"
        )
        self._automate_input.disable()
        self._automate_period_select = w.Select(
            [
                w.Select.Item("min", "minutes"),
                w.Select.Item("h", "hours"),
                w.Select.Item("d", "days"),
            ],
            size="mini",
        )
        self._automate_period_select.disable()
        automate_cont = w.Container(
            [
                self._automate_checkbox,
                self._automate_input,
                self._automate_period_select,
                w.Empty(),
            ],
            direction="horizontal",
            gap=3,
            fractions=[1, 1, 1, 1],
            style="align-items: center",
        )
        self._automate_ok_btn = w.Button("OK", button_size="mini")
        automate_ok_btn_cont = w.Container([self._automate_ok_btn], style="align-items: flex-end")

        text_1 = w.Text(
            "Schedule automatic sampling data from the input project to the labeling project. <br> <strong>Note:</strong> The sampling is performed only on the images which were not copied to the labeling project yet.",
            status="text",
            color="gray",
        )
        automate_content = w.Container([text_1, automate_cont, automate_ok_btn_cont])
        self._automation_modal = w.Dialog(
            title="Automate Sampling", size="tiny", content=automate_content
        )

        @self._automate_checkbox.value_changed
        def on_automate_checkbox_change(is_checked: bool) -> None:
            if is_checked:
                self._automate_input.enable()
                self._automate_period_select.enable()
            else:
                self._automate_input.disable()
                self._automate_period_select.disable()

    def _create_card(self):
        sampling_tasks_modal_btn = w.Button(
            "Sampling tasks history", icon="zmdi zmdi-view-list-alt", button_size="mini"
        )
        sampling_tooltip_btn_automate = w.Button(
            "Automate sampling", icon="zmdi zmdi-flash-auto", button_size="mini"
        )
        btn_text = f"Run sampling (0 of {self.diff_num})"
        self._run_sample_btn = w.Button(
            btn_text, plain=True, button_size="mini", icon="zmdi zmdi-play"
        )

        self._card = Card(
            x=self._x,  # 635,
            y=self._y,  # g.base_y_step_1 + 330,
            title="Smart sampling",
            description="Selects a data sample from the input project and copies it to the labeling project. Supports various sampling strategies: random, k-means clustering, diversity-based, or using embeddings precomputed by the “AI Index” node for smarter selection.",
            tooltip_widgets=[
                sampling_tasks_modal_btn,
                sampling_tooltip_btn_automate,
                self._run_sample_btn,
            ],
            width=250,
        )

        @self._card.card.click
        def on_sampling_setup_btn_click():
            """Show the sampling settings modal."""
            self.settings_modal.show()

        @sampling_tasks_modal_btn.click
        def on_sampling_tasks_modal_btn_click():
            self.tasks_modal.show()
            self._tasks_table.clear()
            rows = self._get_tasks_data()
            for row in rows:
                self._tasks_table.insert_row(row)
            self._tasks_table.show()

        @self._sampling_tabs.click
        def on_sampling_tabs_click(value: str):
            self._preview_collapse.set_active_panel([])

        @self._preview_btn.click
        def preview_btn_clicked():
            """
            Show or hide the preview modal.
            :param active: bool, True if the preview is shown, False otherwise
            """
            self._preview_gallery.loading = True
            self._preview_collapse.set_active_panel(["preview"])
            sampling_settings = self.get_sample_settings()
            # infos = g.session.preview_sampled_images(sample_settings=sampling_settings)  # ! TODO:
            infos = []
            urls = [img.full_storage_url for img in infos]

            ai_metas = [img.ai_search_meta for img in infos]
            self._preview_gallery.clean_up()

            for idx, (url, ai_meta) in enumerate(zip(urls, ai_metas)):
                title = None
                if ai_meta is not None:
                    title = f"Score: {ai_meta.get('score'):.3f}"
                column = idx % 3
                self._preview_gallery.append(column_index=column, image_url=url, title=title)
            self._preview_gallery.loading = False

        @sampling_tooltip_btn_automate.click
        def on_sampling_automate_btn_click():
            self.automation_modal.show()

    def _create_layout(self):
        """
        Create the layout for the Smart Sampling node.
        This includes setting up the sampling settings modal and tasks history modal.
        """
        self._create_settings_modal()
        self._create_tasks_modal()
        self._create_automate_modal()
        self._create_card()

    # METHODS FOR DATA MANAGEMENT
    def get_differences_count(self) -> int:
        src_datasets = self.api.dataset.get_list(self.project_id, recursive=True)
        sampled_items = self.get_sampled_images()

        total_differences = 0
        for ds_info in src_datasets:
            total_differences += ds_info.items_count
            total_differences -= len(sampled_items.get(ds_info.id, []))

        self.save_differences_count(total_differences)
        return total_differences

    def save_differences_count(self, count: int):
        """
        Set the differences count in the DataJson.
        :param count: int, number of differences
        """
        if count < 0:
            raise ValueError("Differences count cannot be negative.")
        self.diff_num = count
        StateJson()[self.widget_id]["differences_count"] = self.diff_num
        StateJson().send_changes()

    def _get_sampled_images(self) -> dict:
        """
        Get sampled images from the DataJson.
        :return: dict with sampled images
        """
        return DataJson()[self.widget_id].get("sampled_images", {})

    def get_sampled_images(self) -> Dict[int, List[int]]:
        """
        Get sampled images from the DataJson.
        :return: dict with dataset ID as key and list of ImageInfo as value
        """
        sampled_images = self._get_sampled_images()
        if not sampled_images:
            return {}

        images = {}
        for _, items in sampled_images.items():
            for ds_id, item_ids in items.items():
                if ds_id not in images:
                    images[ds_id] = []
                images[ds_id].extend(item_ids)

        return images

    def save_sampled_images(self, sampling_task: str, images: Dict[int, List[ImageInfo]]):
        """
        Add sampled images to the DataJson.
        :param sampling_task: ID of the sampling task
        :type sampling_task: int
        :param images: dict with dataset ID as key and list of ImageInfo as value
        :type images: Dict[int, List[ImageInfo]]
        """
        if "sampled_images" not in DataJson()[self.widget_id]:
            DataJson()[self.widget_id]["sampled_images"] = {}
        flattened_images = {}
        for ds_id, img_list in images.items():
            flattened_images[ds_id] = [img.id for img in img_list]
        DataJson()[self.widget_id]["sampled_images"][sampling_task] = flattened_images
        DataJson().send_changes()

    def get_sampling_tasks(self) -> List[dict]:
        """
        Get the sampling history from the DataJson.
        :return: list of dictionaries with sampling history
        """
        return DataJson()[self.widget_id].get("sampling_history", [])

    def add_sampling_task(self, task_details: dict):
        """
        Add a sampling task_details to the DataJson.
        :param task_details: dict with sampling task details
        """
        if "sampling_history" not in DataJson()[self.widget_id]:
            DataJson()[self.widget_id]["sampling_history"] = []
        if not isinstance(task_details, dict):
            raise TypeError("Task must be a dictionary.")

        for key in ["mode", "status", "timestamp", "items_count", "sampling_id", "settings"]:
            if key not in task_details:
                raise ValueError(f"Sampling History Item must contain '{key}' key.")

        DataJson()[self.widget_id]["sampling_history"].append(task_details)
        DataJson().send_changes()

    def get_sample_settings(self) -> SamplingSettings:
        """
        Get the sample settings from the UI.
        :return: dict with sample settings
        """
        mode = self._sampling_tabs.get_active_tab()
        data = {"mode": mode}
        if mode == "Random" and self._has_random_sampling:
            data["sample_size"] = self._random_input.get_value()
        elif mode == "Diverse" and self._has_diverse_sampling:
            data["sample_size"] = self._diverse_input.get_value()
            data["diversity_mode"] = "centroids"
        elif mode == "AI Search" and self._has_ai_search_sampling:
            data["prompt"] = self._ai_search_input.get_value()
            data["limit"] = self._ai_search_limit_input.get_value()
            data["threshold"] = self._ai_search_thrs_input.get_value()
        self.save_sample_settings(data)
        return data

    def save_sample_settings(self, settings: dict):
        """
        Set the sample settings in the UI.
        :param settings: dict with sample settings
        """
        if not isinstance(settings, dict):
            raise TypeError("Settings must be a dictionary.")
        if "mode" not in settings:
            raise ValueError("Settings must contain 'mode' key.")

        validate_keys = {
            "Random": ["sample_size"],
            "Diverse": ["sample_size"],
            "AI Search": ["prompt", "limit", "threshold"],
        }

        if settings["mode"] not in validate_keys:
            raise ValueError(
                f"Invalid mode: {settings['mode']}. Must be one of {list(validate_keys.keys())}."
            )
        for key in validate_keys[settings["mode"]]:
            if key not in settings:
                raise ValueError(
                    f"Settings for mode '{settings['mode']}' must contain '{key}' key."
                )

        StateJson()[self.widget_id]["sampling_settings"] = settings
        StateJson().send_changes()
        # self.update_sampling_widgets()

    def get_automation_details(self):
        enabled = self._automate_checkbox.is_checked()
        period = self._automate_period_select.get_value()
        interval = self._automate_input.get_value()

        if not enabled:
            return False, None, None, None

        if period == "h":
            sec = interval * 60 * 60
        elif period == "d":
            sec = interval * 60 * 60 * 24
        else:
            sec = interval * 60
        if sec == 0:
            return False, None, None, None
        period, interval = self._get_interval_period(sec)
        self.save_automation_details(enabled, sec)
        return enabled, period, interval, sec

    def save_automation_details(self, enabled: bool, sec: int):
        """
        Save the automation details to the DataJson.
        :param enabled: bool, whether automation is enabled
        :param period: str, period of automation (e.g., "min", "h", "d")
        :param interval: int, interval value for automation
        :param sec: int, seconds for automation
        """
        if not isinstance(enabled, bool):
            raise TypeError("Enabled must be a boolean.")
        if "automation_details" not in StateJson()[self.widget_id]:
            StateJson()[self.widget_id]["automation_details"] = {}
        if not enabled:
            StateJson()[self.widget_id]["automation_details"] = {}
            StateJson().send_changes()
            return

        if not isinstance(sec, int) or sec <= 0:
            raise ValueError("Seconds must be a positive integer.")
        period, interval = self._get_interval_period(sec)
        StateJson()[self.widget_id]["automation_details"] = {
            "enabled": enabled,
            "period": period,
            "interval": interval,
            "seconds": sec,
        }
        StateJson().send_changes()

    def show_automation_info(self, enabled, sec):
        period, interval = self._get_interval_period(sec)
        if enabled is True:
            self.card.show_automation_badge()
            self.card.update_property("Run every", f"{interval} {period}", highlight=True)
        else:
            self.card.hide_automation_badge()
            self.card.remove_property_by_key("Run every")

    def update_automation_widgets(self, enabled, sec):
        if enabled:
            self._automate_checkbox.check()
        else:
            self._automate_checkbox.uncheck()
        period, interval = self._get_interval_period(sec)
        if period is not None and interval is not None:
            self._automate_period_select.set_value(period)
            self._automate_input.value = interval
        self.show_automation_info(enabled, sec)
        self.save_automation_details(enabled, sec)

    def update_sampling_widgets(
        self,
        diff: Optional[int] = None,
        sampling_settings: Optional[dict] = None,
    ):
        """Update the sampling inputs based on the difference."""
        if diff is None:
            diff = self.get_differences_count()

        if sampling_settings is None:
            sampling_settings = self.get_sample_settings()

        min_value = 0 if diff == 0 else 1
        if len(sampling_settings) == 0:
            sampling_settings = {
                "mode": "Random",
                "sample_size": min_value,
            }
        mode = sampling_settings.get("mode")
        max_value = diff if diff > 0 else 0
        value = min(sampling_settings.get("sample_size", min_value), diff)
        if mode == SamplingMode.AI_SEARCH.value:
            value = min(sampling_settings.get("limit", max_value), diff)
        self._sampling_text.text = f" of {diff} images"
        if self._has_random_sampling:
            self._random_input.min = min_value
            self._random_input.max = max_value
            self._random_input.value = value
        if self._has_diverse_sampling:
            self._diverse_input.min = min_value
            self._diverse_input.max = max_value
            self._diverse_input.value = value
        if self._has_ai_search_sampling:
            self._ai_search_input.set_value(sampling_settings.get("prompt", ""))
            self._ai_search_limit_input.value = value
            self._ai_search_limit_input.min = min_value
            self._ai_search_limit_input.max = max_value
            self._ai_search_thrs_input.value = sampling_settings.get("threshold", 0.05)
        self._sampling_tabs.set_active_tab(mode)

        text = f"Run sampling ({mode} {value} of {diff})"
        if mode == SamplingMode.AI_SEARCH.value:
            text = f"Run sampling ({mode})"
        self._run_sample_btn.text = text
        if diff == 0:
            self.card.remove_badge_by_key("Difference:")
        else:
            self.card.update_badge_by_key("Difference:", str(diff), "info")
        self.card.update_property("Difference:", str(diff))

    @staticmethod
    def _get_interval_period(sec):
        if sec is None:
            return None, None
        if sec // 60 < 60:
            period = "min"
            interval = sec // 60
        elif sec // 3600 < 24:
            period = "h"
            interval = sec // 3600
        else:
            period = "d"
            interval = sec // 86400
        return period, interval

    def sample(
        self, dst_project_id: int, settings: Optional[SamplingSettings] = None
    ) -> Optional[Dict[int, List[ImageInfo]]]:
        """Sample images from the input project and copy them to the labeling project.
        :param dst_project_id: int, ID of the destination project
        :return: dict with sampled images or None if no new items to copy
        """
        try:
            if settings is None:
                settings = self.get_sample_settings()
            sampled_images = self.get_sampled_images()
            return sample(
                api=self.api,
                team_id=self.team_id,
                src_project_id=self.project_id,
                dst_project_id=dst_project_id,
                sampled_images=sampled_images,
                settings=settings,
            )

        except Exception as e:
            logger.error(f"Error during sampling: {e}")
            self.add_record_to_history(
                status="error",
                total_items=0,
                items=None,
                mode=settings["mode"],
                sampling_id=uuid4().hex,
            )

    def copy_to_new_project(
        self,
        dst_project_id: int,
        images: Dict[int, List[ImageInfo]],
    ) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
        """
        Copy or move images from the source project to the destination project.

        :param dst_project_id: int, ID of the destination project.
        :type dst_project_id: int
        :param images: dict, Dictionary of sampled images where keys are source dataset IDs and values are lists of ImageInfo objects.
        :type images: dict
        :return: Tuple containing two dictionaries:
            - Source dataset IDs to lists of source image IDs.
            - Destination dataset IDs to lists of destination image IDs.
        :rtype: Tuple[Dict[int, List[int]], Dict[int, List[int]]]
        """

        try:
            src, dst, images_count = copy_to_new_project(
                api=self.api,
                src_project_id=self.project_id,
                dst_project_id=dst_project_id,
                images=images,
            )
            return src, dst, images_count
        except Exception as e:
            logger.error(f"Error during copying to new project: {e}")

    def sample_and_copy(self, dst_project_id: int) -> Optional[int]:
        """
        Sample images from the source project and copy them to the destination project.
        :param dst_project_id: int, ID of the destination project
        :return: int, total number of sampled images or None if no new items to copy
        """
        settings = self.get_sample_settings()
        sampled_images = self.sample(dst_project_id, settings)
        if sampled_images is None:
            return None

        src, dst, images_count = self.copy_to_new_project(dst_project_id, sampled_images)
        if images_count is None:
            return None

        self.add_record_to_history(
            sampling_id=uuid4().hex,
            status="success",
            total_items=images_count,
            settings=settings,
            items=sampled_images,
        )
        return src, dst, images_count

    def add_record_to_history(
        self,
        sampling_id: str,
        status: Literal["success", "error"],
        total_items: int,
        settings: dict,
        items: Optional[Dict[int, List[ImageInfo]]] = None,
    ):
        """
        Add a record to the sampling history.
        :param sampling_id: str, unique ID for the sampling task
        :type sampling_id: str
        :param status: status of the sampling task (e.g., "success", "error")
        :type status: str
        :param total_items: int, total number of items sampled
        :type total_items: int
        :param settings: dict, settings used for sampling
        :type settings: dict
        :param items: dict, sampled images by dataset ID
        :type items: Optional[Dict[int, List[ImageInfo]]]
        """
        if sampling_id is None:
            sampling_id = uuid4().hex

        settings_copy = settings.copy()
        mode = settings_copy.pop("mode", None)
        history_item = {
            "sampling_id": sampling_id,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "mode": mode,
            "settings": settings_copy,
            "items_count": total_items,
            "status": status,
        }

        self.add_sampling_task(history_item)
        if items is not None:
            self.save_sampled_images(sampling_id, items)

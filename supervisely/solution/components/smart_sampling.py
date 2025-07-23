from __future__ import annotations

import random
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple
from uuid import uuid4

from typing_extensions import NotRequired, TypedDict

import supervisely.app.widgets as w
from supervisely.api.api import Api
from supervisely.api.image_api import ImageInfo

# from supervisely.api.entities_collection_api import CollectionTypeFilter, AiSearchThresholdDirection
from supervisely.api.project_api import ProjectInfo
from supervisely.app.content import DataJson, StateJson
from supervisely.app.widgets import (
    Button,
    Checkbox,
    Collapse,
    Container,
    Dialog,
    Empty,
    FastTable,
    Flexbox,
    GridGallery,
    Input,
    InputNumber,
    Select,
    SolutionCard,
    Tabs,
    Text,
    Widget,
)
from supervisely.project.image_transfer_utils import (
    compare_dataset_structure,
    compare_projects,
    copy_structured_images,
)
from supervisely.sly_logger import logger
from supervisely.solution.base_node import Automation, SolutionCardNode, SolutionElement
from supervisely.solution.utils import get_interval_period

# from supervisely.solution.components.card import Card
# from supervisely.solution.smart_sampling.functions import (
#     SamplingMode,
#     SamplingSettings,
#     copy_to_new_project,
#     sample,
# )


class SamplingSettings(TypedDict, total=False):
    mode: str
    sample_size: NotRequired[int]
    diversity_mode: NotRequired[str]
    prompt: NotRequired[str]
    limit: NotRequired[int]


class SamplingMode(Enum):
    RANDOM = "Random"
    DIVERSE = "Diverse"
    AI_SEARCH = "AI Search"


class SmartSamplingGUI(Widget):
    APP_SLUG = "c1241a06bfa0adbaa863c0ed37fdcf42/embeddings-generator"
    JOB_ID = "smart_sampling_job"

    def __init__(
        self,
        project: ProjectInfo,
        dst_project_id: int,
        widget_id: Optional[str] = None,
    ):
        self.api = Api.from_env()
        self.project = project
        self.project_id = project.id
        self.workspace_id = self.project.workspace_id
        self.team_id = self.project.team_id
        self.dst_project_id = dst_project_id
        self._tasks = []
        self._sampled_images = {}
        self._diff_num = 0
        self._items_count = self.project.items_count
        super().__init__(widget_id=widget_id)
        self._create_gui(random_sampling=True, diverse_sampling=True, ai_search_sampling=True)

        # if settings is not None:
        #     self.save_sample_settings(settings)

    @property
    def tasks(self) -> List:
        self._tasks = DataJson()[self.widget_id].get("tasks", [])
        if not isinstance(self._tasks, list):
            raise TypeError("Tasks must be a list.")
        return self._tasks

    @property
    def items_count(self) -> int:
        self._items_count = DataJson()[self.widget_id].get("items_count", 0)
        return self._items_count

    @property
    def diff_num(self) -> int:
        self._diff_num = DataJson()[self.widget_id].get("differences_count", 0)
        return self._diff_num

    @property
    def sampled_images(self) -> Dict[int, List[int]]:
        """
        Get sampled images from DataJson.
        :return: dict, sampled images by dataset ID
        """
        self._sampled_images = DataJson()[self.widget_id].get("sampled_images", {})
        return self._sampled_images

    def get_json_data(self):
        return {
            "src_project_id": self.project_id,
            "dst_project_id": self.dst_project_id,
            "differences_count": self._diff_num,
            "items_count": self._items_count,
            "tasks": self._tasks,
            "sampled_images": self._sampled_images,
            "sampling_settings": {},
        }

    def get_json_state(self):
        return {}

    def _create_gui(
        self,
        random_sampling: bool = True,
        diverse_sampling: bool = False,
        ai_search_sampling: bool = False,
    ):
        self.preview_btn = Button("Preview", icon="zmdi zmdi-eye", plain=True)
        self.preview_gallery = GridGallery(columns_number=3)
        self.preview_collapse = Collapse(
            items=[Collapse.Item("preview", "Preview", content=self.preview_gallery)]
        )
        preview_btn_cont = Container([self.preview_btn], style="align-items: flex-end")
        preview_cont = Container(
            [preview_btn_cont, self.preview_collapse], style="margin-bottom: 5px"
        )
        total_text = Text("Total images in the input project:")
        self.total_num_text = Text(f"{self.items_count} images")
        diff_text = Text("Available images for sampling:")
        self.diff_num_text = Text(f"{self.diff_num} images")
        self.sampling_text = Text(f" of {self.diff_num} images")

        tabs = []
        # random sampling
        if random_sampling:
            random_text_info = Text(
                "<strong>Random sampling</strong> is a simple way to select a random subset of images from the input project. <br> <strong>Note:</strong> The sampling is performed only on the images which were not copied to the labeling project yet.",
            )
            self.random_input = InputNumber()
            random_input_cont = Container(
                [self.random_input, self.sampling_text],
                direction="horizontal",
                gap=3,
                style="align-items: center",
            )
            random_nums_cont = Container(
                [self.total_num_text, self.total_num_text, random_input_cont]
            )
            random_sampling_text = Text("Sampling random:")
            random_texts = Container([total_text, diff_text, random_sampling_text])

            random_cont = Container(
                [random_texts, random_nums_cont, Empty()], direction="horizontal", gap=5
            )
            random_tab_cont = Container([random_text_info, random_cont, preview_cont])
            tabs.append(random_tab_cont)

        # diverse sampling
        if diverse_sampling:
            diverse_text_info = Text(
                "<strong>Diversity sampling strategy:</strong> Sampling most diverse images using k-means clustering. By default, embeddings are computed using the OpenAI CLIP model. <br> <strong>Note:</strong> The sampling is performed only on the images which were not copied to the labeling project yet.",
            )
            self.diverse_input = InputNumber()
            diverse_input_cont = Container(
                [self.diverse_input, self.sampling_text],
                direction="horizontal",
                gap=3,
                style="align-items: center",
            )
            diverse_nums_cont = Container(
                [self.total_num_text, self.total_num_text, diverse_input_cont]
            )
            diverse_sampling_text = Text("Sample size:")
            diverse_texts = Container([total_text, diff_text, diverse_sampling_text])
            diverse_cont = Container(
                [diverse_texts, diverse_nums_cont, Empty()], direction="horizontal", gap=5
            )
            diverse_tab_cont = Container([diverse_text_info, diverse_cont, preview_cont])
            tabs.append(diverse_tab_cont)

        # AI Search sampling
        if ai_search_sampling:
            ai_search_info = Text(
                "Sampling images from the project using <strong>AI Search</strong> by user-defined prompt to find the most suitable images. <br> <strong>Note:</strong> The sampling is performed only on the images which were not copied to the labeling project yet.",
            )
            ai_search_total = Container(
                [total_text, self.total_num_text], direction="horizontal", fractions=[2, 1]
            )
            ai_search_diff = Container(
                [diff_text, self.total_num_text], direction="horizontal", fractions=[2, 1]
            )
            ai_search_limit_text = Text(f"Limit: ")
            self.ai_search_limit_input = InputNumber(min=1, max=self.diff_num, value=self.diff_num)
            ai_search_limit = Container(
                [ai_search_limit_text, self.ai_search_limit_input],
                direction="horizontal",
                fractions=[2, 1],
            )

            self.ai_search_thrs_input = InputNumber(min=0, max=1, value=0.05, step=0.01)
            ai_search_thrs_text = Text(f"Threshold: ")
            ai_search_thrs = Container(
                [ai_search_thrs_text, self.ai_search_thrs_input],
                direction="horizontal",
                fractions=[2, 1],
            )

            ai_search_input_text = Text(f"Search query: ")
            self.ai_search_input = Input(placeholder="e.g. 'cat', 'dog', 'car'")
            ai_search_prompt = Flexbox([ai_search_input_text, self.ai_search_input])

            ai_search_tab_cont = Container(
                [
                    ai_search_info,
                    ai_search_total,
                    ai_search_diff,
                    ai_search_limit,
                    ai_search_thrs,
                    ai_search_prompt,
                    preview_cont,
                ]
            )
            tabs.append(ai_search_tab_cont)

        self.save_settings_btn = Button("Confirm settings", plain=True)
        self.run_btn = Button("Run sampling", plain=True, icon="zmdi zmdi-play")
        sample_btn_cont = Container(
            [self.save_settings_btn, self.run_btn],
            direction="horizontal",
            style="align-items: center; justify-content: space-between; margin-top: 10px;",
        )

        self.sampling_tabs = Tabs(
            labels=["Random", "Diverse", "AI Search"],
            contents=tabs,
            type="card",
        )
        self.content = Container([self.sampling_tabs, sample_btn_cont])

        @self.sampling_tabs.click
        def on_sampling_tabs_click(value: str):
            self.preview_collapse.set_active_panel([])

        @self.preview_btn.click
        def preview_btn_clicked():
            """
            Show or hide the preview modal.
            :param active: bool, True if the preview is shown, False otherwise
            """
            self.preview_gallery.loading = True
            self.preview_collapse.set_active_panel(["preview"])
            sampling_settings = self.get_sample_settings()
            if sampling_settings.get("sample_size", 0) > 6:
                sampling_settings["sample_size"] = 6
            if sampling_settings.get("limit", 0) > 6:
                sampling_settings["limit"] = 6

            sampled_images = self._sample(sampling_settings)

            infos = []
            for _, imgs in sampled_images.items():
                infos.extend(imgs)
            urls = [img.full_storage_url for img in infos]

            # ai_metas = [img.ai_search_meta for img in infos]
            ai_metas = [None for img in infos]
            self.preview_gallery.clean_up()

            for idx, (url, ai_meta) in enumerate(zip(urls, ai_metas)):
                title = None
                if ai_meta is not None:
                    title = f"Score: {ai_meta.get('score'):.3f}"
                column = idx % 3
                self.preview_gallery.append(column_index=column, image_url=url, title=title)
            self.preview_gallery.loading = False

    def to_html(self):
        return self.content.to_html()

    def get_sample_settings(self) -> SamplingSettings:
        """
        Get the sample settings from the UI.
        :return: dict with sample settings
        """
        mode = self.sampling_tabs.get_active_tab()
        data = {"mode": mode}
        if mode == "Random":
            data["sample_size"] = self.random_input.get_value()
        elif mode == "Diverse":
            data["sample_size"] = self.diverse_input.get_value()
            data["diversity_mode"] = "centroids"
        elif mode == "AI Search":
            data["prompt"] = self.ai_search_input.get_value()
            data["limit"] = self.ai_search_limit_input.get_value()
            data["threshold"] = self.ai_search_thrs_input.get_value()
        return data

    def save_sample_settings(self, settings: SamplingSettings):
        mode = settings.get("mode")
        self.sampling_tabs.set_active_tab(mode)
        if mode == "Random":
            self.random_input.value = settings.get("sample_size", 0)
        elif mode == "Diverse":
            self.diverse_input.value = settings.get("sample_size", 0)
        elif mode == "AI Search":
            self.ai_search_input.value = settings.get("prompt", "")
            self.ai_search_limit_input.value = settings.get("limit", 0)
            self.ai_search_thrs_input.value = settings.get("threshold", 0.05)
        self.preview_gallery.clear()
        self.preview_collapse.set_active_panel([])

    def update_sampling_widgets(
        self,
        diff: int,
        sampling_settings: Optional[dict] = None,
    ):
        """Update the sampling inputs based on the difference."""
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
        self.sampling_text.text = f" of {diff} images"
        if hasattr(self, "random_input"):
            self.random_input.min = min_value
            self.random_input.max = max_value
            self.random_input.value = value
        if hasattr(self, "diverse_input"):
            self.diverse_input.min = min_value
            self.diverse_input.max = max_value
            self.diverse_input.value = value
        if hasattr(self, "ai_search_input"):
            self.ai_search_input.set_value(sampling_settings.get("prompt", ""))
            self.ai_search_limit_input.value = value
            self.ai_search_limit_input.min = min_value
            self.ai_search_limit_input.max = max_value
            self.ai_search_thrs_input.value = sampling_settings.get("threshold", 0.05)
        self.sampling_tabs.set_active_tab(mode)

    def _filter_diffs(
        self, diffs: Dict[int, List[ImageInfo]], sampled_images: Dict[int, List[int]]
    ) -> Dict[int, List[ImageInfo]]:
        """
        Filter out already sampled images from the differences.
        :param diffs: dict, differences by dataset ID
        :param sampled_images: dict, sampled images by dataset ID
        :return: dict, filtered differences
        """
        filtered_diffs = {}
        for ds_id, imgs in diffs.items():
            ignore_ids = {img for img in sampled_images.get(ds_id, [])}
            filtered_diffs[ds_id] = [img for img in imgs if img.id not in ignore_ids]
        return filtered_diffs

    def _sample(self, settings: SamplingSettings) -> Optional[Dict[int, List[ImageInfo]]]:
        try:
            diffs = self.calculate_differences()
            total_diffs = self.calculate_diff_count(diffs)

            # If there are no differences
            if total_diffs == 0:
                logger.warning("No new items to copy to the labeling project")
                return None

            # If the sample size is greater than the total differences, return all images
            sample_size = settings.get("sample_size", None)
            if sample_size and sample_size >= total_diffs:
                logger.warning(
                    f"Sample size ({sample_size}) is greater than total differences ({total_diffs}). "
                    "Returning all images."
                )
                return diffs

            mode = settings.get("mode", SamplingMode.RANDOM.value)

            # Calculate the sample size for each dataset
            samples_per_dataset = {}
            remaining = sample_size or total_diffs
            for ds_id, imgs in diffs.items():
                # Calculate proportional size and round down
                if sample_size is not None:
                    ds_sample = int((len(imgs) / total_diffs) * sample_size)
                else:
                    ds_sample = len(imgs)
                samples_per_dataset[ds_id] = ds_sample
                remaining -= ds_sample

            # Distribute any remaining samples randomly
            if remaining > 0:
                datasets_with_space = [
                    ds_id for ds_id, imgs in diffs.items() if len(imgs) > samples_per_dataset[ds_id]
                ]
                while remaining > 0 and datasets_with_space:
                    ds_id = random.choice(datasets_with_space)
                    if len(diffs[ds_id]) > samples_per_dataset[ds_id]:
                        samples_per_dataset[ds_id] += 1
                        remaining -= 1
                    else:
                        datasets_with_space.remove(ds_id)

            if mode == SamplingMode.RANDOM.value:
                new_sampled_images = {}
                for ds_id, sample_count in samples_per_dataset.items():
                    if sample_count > 0:
                        new_sampled_images[ds_id] = random.sample(diffs[ds_id], sample_count)
            elif mode in [SamplingMode.DIVERSE.value, SamplingMode.AI_SEARCH.value]:
                all_diffs_flat = []
                for ds_id, imgs in diffs.items():
                    all_diffs_flat.extend([img.id for img in imgs])
                logger.info(f"Sample mode: {mode}. Settings: {settings}")

                method = "diverse" if mode == SamplingMode.DIVERSE.value else "search"
                data = {"project_id": self.project_id}
                data["image_ids"] = all_diffs_flat

                if mode == SamplingMode.AI_SEARCH.value:
                    # AI search mode
                    prompt = settings.get("prompt", None)
                    if prompt is None:
                        logger.error("Prompt is required for AI search mode.")
                        return None
                    data["prompt"] = prompt
                    data["limit"] = settings.get("limit", None)
                    data["threshold"] = settings.get("threshold", 0.05)
                elif mode == SamplingMode.DIVERSE.value:
                    # Diverse mode
                    data["sample_size"] = sample_size
                    data["num_clusters"] = sample_size
                    data["clustering_method"] = "kmeans"
                    data["sampling_method"] = "centroids"
                else:
                    logger.error(f"Unknown sampling mode: {mode}")
                    return None
                # Send request to the API
                module_info = self.api.app.get_ecosystem_module_info(slug=self.APP_SLUG)
                sessions = self.api.app.get_sessions(
                    self.team_id, module_info.id, statuses=[self.api.task.Status.STARTED]
                )
                if len(sessions) == 0:
                    logger.error("No active sessions found for embeddings generator.")
                    return None
                session = sessions[0]
                # api.app.wait(session.task_id, target_status=api.task.Status.STARTED)
                logger.info(f"Embeddings generator session: {session.task_id}")
                res = self.api.app.send_request(session.task_id, method, data=data)
                if isinstance(res, dict):
                    if "collection_id" in res:
                        collection_id = res["collection_id"]

                        # if mode == SamplingMode.AI_SEARCH.value:
                        #     all_sampled_images = self.api.entities_collection.get_items(
                        #         collection_id=collection_id,
                        #         collection_type=CollectionTypeFilter.AI_SEARCH,
                        #         ai_search_threshold=data.get("threshold", 0.05),
                        #         ai_search_threshold_direction=AiSearchThresholdDirection.ABOVE,
                        #     )
                        # else:
                        #     all_sampled_images = self.api.entities_collection.get_items(
                        #         collection_id=collection_id,
                        #         collection_type=CollectionTypeFilter.DEFAULT,
                        #     )
                        new_sampled_images = {}
                        # for img in all_sampled_images:
                        #     ds_id = img.dataset_id
                        #     if ds_id not in new_sampled_images:
                        #         new_sampled_images[ds_id] = []
                        #     new_sampled_images[ds_id].append(img)
                    elif "message" in res:
                        logger.error(f"Error during sampling: {res['message']}")
                        return None
                elif isinstance(res, list):
                    res_ids = {img["id"] for img in res}
                    new_sampled_images = {
                        ds_id: [img for img in diffs[ds_id] if img.id in res_ids]
                        for ds_id in diffs.keys()
                    }
                else:
                    logger.error(f"Error during sampling: {res}")
                    return None

            else:
                logger.error(f"Unknown sampling mode: {mode}")
                return None

            return new_sampled_images

        except Exception as e:
            logger.error(f"Error during sampling: {e}")

    def _copy_to_new_project(
        self,
        images: Dict[int, List[ImageInfo]],
    ) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
        try:
            src, dst, images_count = copy_structured_images(
                api=self.api,
                src_project_id=self.project_id,
                dst_project_id=self.dst_project_id,
                images=images,
            )
            return src, dst, images_count
        except Exception as e:
            logger.error(f"Error during copying to new project: {e}")

    def _add_task(self, task: Dict):
        """
        Add a task to the DataJson.
        :param task_id: str, unique ID for the task
        """
        if "tasks" not in DataJson()[self.widget_id]:
            DataJson()[self.widget_id]["tasks"] = []
        DataJson()[self.widget_id]["tasks"].append(task)
        DataJson().send_changes()

    def _add_sampled_images(
        self,
        sampling_id: str,
        images: Dict[int, List[ImageInfo]],
    ):
        """
        Save sampled images to DataJson.
        :param sampling_id: str, unique ID for the sampling task
        :param images: dict, sampled images by dataset ID
        """
        if "sampled_images" not in DataJson()[self.widget_id]:
            DataJson()[self.widget_id]["sampled_images"] = {}
        DataJson()[self.widget_id]["sampled_images"][sampling_id] = images
        DataJson().send_changes()

    def get_all_sampled_images(self) -> Dict[int, List[int]]:
        """
        Get sampled images from DataJson.
        :return: dict, sampled images by dataset ID
        """
        res = {}
        for _, images in self.sampled_images.items():
            for ds_id, img_list in images.items():
                if ds_id not in res:
                    res[ds_id] = []
                res[ds_id].extend(img_list)
        return res

    def _add_record_to_history(
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

        self._add_task(history_item)
        if items is not None:
            _items = {}
            for ds_id, img_list in items.items():
                _items[ds_id] = [img.id for img in img_list]
            self._add_sampled_images(sampling_id, _items)

    def run(self) -> Optional[int]:
        """
        Sample images from the source project and copy them to the destination project.
        :param dst_project_id: int, ID of the destination project
        :return: int, total number of sampled images or None if no new items to copy
        """
        uuid = uuid4().hex
        settings = self.get_sample_settings()
        sampled_images = self._sample(settings)
        if sampled_images is None:
            return None

        src, dst, images_count = self._copy_to_new_project(sampled_images)
        if images_count is None or images_count == 0:
            return None

        self._add_record_to_history(
            sampling_id=uuid,
            status="success",
            total_items=images_count,
            settings=settings,
            items=sampled_images,
        )
        return src, dst, images_count

    def calculate_differences(self) -> Tuple[int, Dict[int, List[ImageInfo]]]:
        """
        Calculate the differences between the source and destination projects.
        :return: Tuple with total number of differences and a dict of differences by dataset ID
        """
        diffs = compare_projects(
            api=self.api,
            src_project_id=self.project_id,
            dst_project_id=self.dst_project_id,
        )
        if not diffs:
            return {}

        sampled_images = self.get_all_sampled_images()
        filtered_diffs = self._filter_diffs(diffs, sampled_images)
        return filtered_diffs

    def calculate_diff_count(self, diffs: Dict = None) -> Tuple[int, Dict[int, List[ImageInfo]]]:
        """
        Calculate the differences between the source and destination projects.
        :return: Tuple with total number of differences and a dict of differences by dataset ID
        """
        if not diffs:
            diffs = self.calculate_differences()
        total_diffs = sum(len(imgs) for imgs in diffs.values())
        return total_diffs


class SmartSamplingAutomation(Automation):

    def __init__(self, func: Callable):
        super().__init__()
        self.apply_btn = Button("Apply", plain=True)
        self.widget = self._create_widget()
        self.job_id = self.widget.widget_id
        self.func = func

    def apply(self):
        enabled, _, _, sec = self.get_automation_details()
        if not enabled:
            if self.scheduler.is_job_scheduled(self.job_id):
                self.scheduler.remove_job(self.job_id)
        else:
            self.scheduler.add_job(
                self.func, interval=sec, job_id=self.job_id, replace_existing=True
            )

    def _create_widget(self):
        self.enabled_checkbox = Checkbox(content="Run every", checked=False)
        self.num_input = InputNumber(min=1, value=60, debounce=1000, controls=False, size="mini")
        self.num_input.disable()
        self.period_select = Select(
            [
                Select.Item("min", "minutes"),
                Select.Item("h", "hours"),
                Select.Item("d", "days"),
            ],
            size="mini",
        )
        self.period_select.disable()
        automate_cont = Container(
            [
                self.enabled_checkbox,
                self.num_input,
                self.period_select,
                Empty(),
            ],
            direction="horizontal",
            gap=3,
            fractions=[1, 1, 1, 1],
            style="align-items: center",
        )
        apply_btn = Container([self.apply_btn], style="align-items: flex-end")

        text = Text(
            "Schedule automatic sampling data from the input project to the labeling project. <br> <strong>Note:</strong> The sampling is performed only on the images which were not copied to the labeling project yet.",
            status="text",
            color="gray",
        )

        @self.enabled_checkbox.value_changed
        def on_automate_checkbox_change(is_checked: bool) -> None:
            if is_checked:
                self.num_input.enable()
                self.period_select.enable()
            else:
                self.num_input.disable()
                self.period_select.disable()

        return Container([text, automate_cont, apply_btn])

    def get_automation_details(self) -> Tuple[bool, str, int, int]:
        """
        Get the automation details from the widget.
        :return: Tuple with (enabled, period, interval, seconds)
        """
        enabled = self.enabled_checkbox.is_checked()
        period = self.period_select.get_value()
        interval = self.num_input.get_value()

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
        return enabled, period, interval, sec

    def save_automation_details(self, enabled: bool, sec: int):

        if enabled is False:
            self.enabled_checkbox.uncheck()
        else:
            self.enabled_checkbox.check()
            period, interval = get_interval_period(sec)
            self.num_input.value = interval
            self.period_select.set_value(period)


class SmartSampling(SolutionElement):

    def __init__(
        self,
        api: Api,
        project_id: int,
        dst_project: int,
        x: int,
        y: int,
        *args,
        **kwargs,
    ):
        """
        Smart Sampling widget for Supervisely app.

        :param project_id: ID of the project to sample from
        :type project_id: int
        :param x: x-coordinate for the widget position
        :type x: int
        :param y: y-coordinate for the widget position
        :type y: int
        """
        self.api = api
        self.project_id = project_id
        self.project = self.api.project.get_info_by_id(project_id)
        # self.sampling_settings = SamplingSettings()
        self.main_widget = SmartSamplingGUI(project=self.project, dst_project_id=dst_project)
        self.automation = SmartSamplingAutomation(self.run)
        self.card = self._create_card()
        self.node = SolutionCardNode(content=self.card, x=x, y=y)

        self.modals = [self.tasks_modal, self.main_modal, self.automation_modal]
        self.update_sampling_widgets()

        @self.main_widget.save_settings_btn.click
        def on_save_settings_click():
            self.main_modal.hide()
            self.update_sampling_widgets()

        @self.automation_btn.click
        def on_automation_btn_click():
            self.apply_automation()

        super().__init__(*args, **kwargs)

    # UI PROPERTIES
    @property
    def tasks_modal(self) -> Dialog:
        """Get the tasks modal dialog."""
        if not hasattr(self, "_tasks_modal"):
            self._tasks_modal = self._create_tasks_modal()
        return self._tasks_modal

    @property
    def tasks_table(self) -> FastTable:
        """Get the tasks table widget."""
        if not hasattr(self, "_tasks_table"):
            self._tasks_table = self._create_tasks_table()
        return self._tasks_table

    @property
    def main_modal(self) -> Dialog:
        """Get the settings modal dialog."""
        if not hasattr(self, "_main_modal"):
            self._main_modal = self._create_main_modal()
        return self._main_modal

    @property
    def automation_modal(self) -> Dialog:
        """Get the automate modal dialog."""
        if not hasattr(self, "_automation_modal"):
            self._automation_modal = self._create_automate_modal()
        return self._automation_modal

    @property
    def run_btn(self) -> Button:
        """Get the run sample button"""
        return self.main_widget.run_btn

    @property
    def automation_btn(self) -> Button:
        """Get the apply annotations button"""
        return self.automation.apply_btn
    
    def run(self):
        self.main_widget.run()

    def apply_automation(self):
        enabled, _, _, sec = self.automation.get_automation_details()
        self.show_automation_info(enabled, sec)
        self.automation.apply()

    # UI CREATION METHODS
    def _create_main_modal(self) -> Widget:
        return Dialog(title="Sampling Settings", content=self.main_widget.content, size="tiny")

    def _create_tasks_table(self) -> FastTable:
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
        tasks_table = FastTable(
            columns=sampling_table_columns,
            sort_column_idx=0,
            fixed_columns=1,
            sort_order="asc",
            columns_options=columns_options,
        )
        tasks_table.hide()
        return tasks_table

    def _create_tasks_modal(self) -> FastTable:
        """
        Create a table for displaying sampling tasks history.
        :return: FastTable instance
        """
        return Dialog(content=Container([self.tasks_table]))

    def _get_tasks_data(self) -> List[List[str]]:
        sampling_history = self.main_widget.tasks
        if not sampling_history:
            return []
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

    def _create_automate_modal(self) -> Dialog:
        """
        Create a modal dialog for automating sampling.
        :return: Dialog instance
        """
        # self.automation.apply_btn.click(self.update_automation_widgets())
        return Dialog(title="Automate Sampling", size="tiny", content=self.automation.widget)

    def _create_tooltip(self) -> SolutionCard.Tooltip:
        sampling_tasks_modal_btn = Button(
            "Sampling tasks history",
            icon="zmdi zmdi-view-list-alt",
            button_size="mini",
            button_type="text",
            plain=True,
        )
        sampling_tooltip_btn_automate = Button(
            "Automate sampling",
            icon="zmdi zmdi-flash-auto",
            button_size="mini",
            button_type="text",
            plain=True,
        )
        # btn_text = f"Run sampling (0 of {self.main_widget.diff_num})"

        @sampling_tasks_modal_btn.click
        def on_sampling_tasks_modal_btn_click():
            self.tasks_modal.show()
            self.tasks_table.clear()
            rows = self._get_tasks_data()
            for row in rows:
                self.tasks_table.insert_row(row)
            self.tasks_table.show()

        @sampling_tooltip_btn_automate.click
        def on_sampling_automate_btn_click():
            self.automation_modal.show()

        return SolutionCard.Tooltip(
            description="Selects a data sample from the input project and copies it to the labeling project. Supports various sampling strategies: random, k-means clustering, diversity-based, or using embeddings precomputed by the “AI Index” node for smarter selection.",
            content=[sampling_tasks_modal_btn, sampling_tooltip_btn_automate],
        )

    def _create_card(self):
        card = SolutionCard(
            title="Smart Sampling",
            tooltip=self._create_tooltip(),
            width=250,
        )

        @card.click
        def on_sampling_setup_btn_click():
            """Show the sampling settings modal."""
            self.main_modal.show()

        return card

    # METHODS FOR DATA MANAGEMENT
    def show_automation_info(self, enabled, sec):
        period, interval = get_interval_period(sec)
        if enabled is True:
            self.node.show_automation_badge()
            self.card.update_property("Run every", f"{interval} {period}", highlight=True)
        else:
            self.node.hide_automation_badge()
            self.card.remove_property_by_key("Run every")

    def update_automation_widgets(self):
        enabled, _, _, sec = self.automation.get_automation_details()
        self.automation.save_automation_details(enabled, sec)
        self.show_automation_info(enabled, sec)

    def update_sampling_widgets(
        self,
        diff: Optional[int] = None,
        sampling_settings: Optional[dict] = None,
    ):
        """Update the sampling inputs based on the difference."""
        if diff is None:
            diff = self.main_widget.calculate_diff_count()

        self.main_widget.update_sampling_widgets(diff, sampling_settings)

        if diff == 0:
            self.card.remove_badge_by_key("Difference:")
        else:
            self.card.update_badge_by_key("Difference:", str(diff), "info")
        self.card.update_property("Difference:", str(diff))

        sampling_settings = sampling_settings or self.main_widget.get_sample_settings()
        mode = sampling_settings.get("mode", SamplingMode.RANDOM.value)
        self.card.update_property("mode", mode)
        if mode in [SamplingMode.RANDOM.value, SamplingMode.DIVERSE.value]:
            sample_size = sampling_settings.get("sample_size", 0)
            self.card.update_property("Sample size", str(sample_size))
        elif mode == SamplingMode.AI_SEARCH.value:
            prompt = sampling_settings.get("prompt", "")
            limit = sampling_settings.get("limit", 0)
            threshold = sampling_settings.get("threshold", 0.05)
            self.card.update_property("Prompt", prompt)
            self.card.update_property("Limit", str(limit))
            self.card.update_property("Threshold", f"{threshold:.2f}")

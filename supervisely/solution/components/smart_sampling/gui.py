from __future__ import annotations

import random
from datetime import datetime
from typing import Dict, List, Literal, Optional, Tuple
from uuid import uuid4
from enum import Enum

from supervisely.api.api import Api
from supervisely.api.image_api import ImageInfo
from typing_extensions import NotRequired, TypedDict

from supervisely.api.entities_collection_api import CollectionTypeFilter
from supervisely.api.project_api import ProjectInfo
from supervisely.app.content import DataJson, StateJson
from supervisely.app.widgets import (
    Button,
    Collapse,
    Container,
    Dialog,
    Empty,
    Flexbox,
    GridGallery,
    Input,
    InputNumber,
    Tabs,
    Text,
    Widget,
)
from supervisely.project.image_transfer_utils import compare_projects, copy_structured_images
from supervisely.sly_logger import logger


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

    # ------------------------------------------------------------------
    # Properties -------------------------------------------------------
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Base Widget Methods ----------------------------------------------
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # GUI --------------------------------------------------------------
    # ------------------------------------------------------------------
    def _create_gui(
        self,
        random_sampling: bool = True,
        diverse_sampling: bool = False,
        ai_search_sampling: bool = False,
    ):
        # -- main widgets ----------------------------------------------------
        self.preview_button = Button("Preview", icon="zmdi zmdi-eye", plain=True)
        self.preview_gallery = GridGallery(columns_number=3)
        self.preview_collapse = Collapse(
            items=[Collapse.Item("preview", "Preview", content=self.preview_gallery)]
        )
        preview_button_container = Container([self.preview_button], style="align-items: flex-end")
        preview_container = Container(
            [preview_button_container, self.preview_collapse], style="margin-bottom: 5px"
        )
        total_text = Text("Total images in the input project:")
        self.total_num_text = Text(f"{self.items_count} images")
        diff_text = Text("Available images for sampling:")
        self.diff_num_text = Text(f"{self.diff_num} images")
        self.sampling_text = Text(f" of {self.diff_num} images")

        tabs = []
        # random sampling
        if random_sampling:
            random_tab_container = self._create_random_sampling_tab(
                total_text, diff_text, preview_container
            )
            tabs.append(random_tab_container)

        # diverse sampling
        if diverse_sampling:
            diverse_tab_container = self._create_diverse_sampling_tab(
                total_text, diff_text, preview_container
            )
            tabs.append(diverse_tab_container)

        # AI Search sampling
        if ai_search_sampling:
            ai_search_tab_container = self._create_ai_search_tab(
                total_text, diff_text, preview_container
            )
            tabs.append(ai_search_tab_container)

        self.status_text = Text("", status="text")
        self.status_text.hide()
        self.save_settings_button = Button("Confirm settings", plain=True)
        self.run_button = Button("Run sampling", plain=True, icon="zmdi zmdi-play")
        sample_button_container = Container(
            [self.save_settings_button, self.run_button],
            direction="horizontal",
            style="align-items: center; justify-content: space-between; margin-top: 10px;",
        )

        self.sampling_tabs = Tabs(
            labels=["Random", "Diverse", "AI Search"],
            contents=tabs,
            type="card",
        )
        self.content = Container([self.sampling_tabs, self.status_text, sample_button_container])

        @self.sampling_tabs.click
        def on_sampling_tabs_click(value: str):
            self.preview_collapse.set_active_panel([])

        @self.preview_button.click
        def preview_button_clicked():
            """
            Show or hide the preview modal.
            :param active: bool, True if the preview is shown, False otherwise
            """
            self.status_text.hide()
            self.preview_gallery.loading = True
            self.preview_collapse.set_active_panel(["preview"])
            sampling_settings = self.get_settings()
            if sampling_settings.get("sample_size", 0) > 6:
                sampling_settings["sample_size"] = 6
            if sampling_settings.get("limit", 0) > 6:
                sampling_settings["limit"] = 6

            sampled_images = self._sample(sampling_settings)
            if sampled_images is None:
                self.preview_gallery.loading = False
                return

            infos = []
            for _, imgs in sampled_images.items():
                infos.extend(imgs)
            urls = [img.full_storage_url for img in infos]

            ai_metas = [img.ai_search_meta for img in infos]
            # ai_metas = [None for img in infos]
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

    # ------------------------------------------------------------------
    # GUI Helpers ------------------------------------------------------
    # ------------------------------------------------------------------
    def _create_random_sampling_tab(self, total_text, diff_text, preview_container):
        random_text_info = Text(
            "<strong>Random sampling</strong> is a simple way to select a random subset of images from the input project. <br> <strong>Note:</strong> The sampling is performed only on the images which were not copied to the labeling project yet.",
        )
        self.random_input = InputNumber()
        random_input_container = Container(
            [self.random_input, self.sampling_text],
            direction="horizontal",
            gap=3,
            style="align-items: center",
        )
        random_nums_container = Container(
            [self.total_num_text, self.total_num_text, random_input_container]
        )
        random_sampling_text = Text("Sampling random:")
        random_texts = Container([total_text, diff_text, random_sampling_text])

        random_container = Container(
            [random_texts, random_nums_container, Empty()], direction="horizontal", gap=5
        )
        random_tab_container = Container([random_text_info, random_container, preview_container])
        return random_tab_container

    def _create_diverse_sampling_tab(
        self, total_text: Text, diff_text: Text, preview_container: Container
    ):
        diverse_text_info = Text(
            "<strong>Diversity sampling strategy:</strong> Sampling most diverse images using k-means clustering. By default, embeddings are computed using the OpenAI CLIP model. <br> <strong>Note:</strong> The sampling is performed only on the images which were not copied to the labeling project yet.",
        )
        self.diverse_input = InputNumber()
        diverse_input_container = Container(
            [self.diverse_input, self.sampling_text],
            direction="horizontal",
            gap=3,
            style="align-items: center",
        )
        diverse_nums_container = Container(
            [self.total_num_text, self.total_num_text, diverse_input_container]
        )
        diverse_sampling_text = Text("Sample size:")
        diverse_texts = Container([total_text, diff_text, diverse_sampling_text])
        diverse_container = Container(
            [diverse_texts, diverse_nums_container, Empty()], direction="horizontal", gap=5
        )
        diverse_tab_container = Container([diverse_text_info, diverse_container, preview_container])
        return diverse_tab_container

    def _create_ai_search_tab(
        self, total_text: Text, diff_text: Text, preview_container: Container
    ):
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

        self.ai_search_threshold_input = InputNumber(min=0, max=1, value=0.05, step=0.01)
        ai_search_threshold_text = Text(f"Threshold: ")
        ai_search_threshold = Container(
            [ai_search_threshold_text, self.ai_search_threshold_input],
            direction="horizontal",
            fractions=[2, 1],
        )

        ai_search_input_text = Text(f"Search query: ")
        self.ai_search_input = Input(placeholder="e.g. 'cat', 'dog', 'car'")
        ai_search_prompt = Flexbox([ai_search_input_text, self.ai_search_input])

        ai_search_tab_container = Container(
            [
                ai_search_info,
                ai_search_total,
                ai_search_diff,
                ai_search_limit,
                ai_search_threshold,
                ai_search_prompt,
                preview_container,
            ]
        )
        return ai_search_tab_container

    # ------------------------------------------------------------------
    # Modal ------------------------------------------------------------
    # ------------------------------------------------------------------
    @property
    def modal(self) -> Dialog:
        """Returns the modal for the node GUI."""
        if not hasattr(self, "_modal"):
            self._modal = Dialog(title="Sampling Settings", content=self.content, size="tiny")
        return self._modal

    # ------------------------------------------------------------------
    # Settings --------------------------------------------------------
    # ------------------------------------------------------------------
    def get_settings(self) -> SamplingSettings:
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
            data["threshold"] = self.ai_search_threshold_input.get_value()
        return data

    def save_settings(self, settings: SamplingSettings):
        """
        Save the sample settings to the UI.

        :param settings: dict with sample settings
        """
        mode = settings.get("mode")
        self.sampling_tabs.set_active_tab(mode)
        if mode == "Random":
            self.random_input.value = settings.get("sample_size", 0)
        elif mode == "Diverse":
            self.diverse_input.value = settings.get("sample_size", 0)
        elif mode == "AI Search":
            self.ai_search_input.value = settings.get("prompt", "")
            self.ai_search_limit_input.value = settings.get("limit", 0)
            self.ai_search_threshold_input.value = settings.get("threshold", 0.05)
        self.preview_gallery.clear()
        self.preview_collapse.set_active_panel([])

    def update_widgets(
        self,
        diff: int,
        sampling_settings: Optional[dict] = None,
    ):
        """Update the sampling inputs based on the difference."""
        if sampling_settings is None:
            sampling_settings = self.get_settings()

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
            self.ai_search_threshold_input.value = sampling_settings.get("threshold", 0.05)
        self.sampling_tabs.set_active_tab(mode)

    # ------------------------------------------------------------------
    # Run --------------------------------------------------------
    # ------------------------------------------------------------------
    def run(self) -> Optional[Tuple[Dict[int, List[int]], Dict[int, List[int]], int]]:
        """
        Sample images from the source project and copy them to the destination project.

        :return: Tuple with source, destination and images count or None if no new items to copy
        """
        uuid = uuid4().hex
        settings = self.get_settings()
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

    # ------------------------------------------------------------------
    # Differences Helpers ----------------------------------------------
    # ------------------------------------------------------------------
    def calculate_differences(self) -> Dict[int, List[ImageInfo]]:
        """
        Calculate the differences between the source and destination projects.

        :return: dict of differences by dataset ID
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

    def calculate_diff_count(self, diffs: Dict = None) -> int:
        """
        Calculate the differences between the source and destination projects.

        :return: int, total number of differences
        """
        if not diffs:
            diffs = self.calculate_differences()
        total_diffs = sum(len(imgs) for imgs in diffs.values())
        return total_diffs

    # ------------------------------------------------------------------
    # Sampling Helpers -------------------------------------------------
    # ------------------------------------------------------------------
    def _filter_diffs(
        self, diffs: Dict[int, List[ImageInfo]], sampled_images: Dict[int, List[int]]
    ) -> Dict[int, List[ImageInfo]]:
        """
        Filter out already sampled images from the differences.

        :param diffs: dict, differences by dataset ID
        :type diffs: Dict[int, List[ImageInfo]]
        :param sampled_images: dict, sampled images by dataset ID
        :type sampled_images: Dict[int, List[int]]
        :return: dict, filtered differences
        :rtype: Dict[int, List[ImageInfo]]
        """
        filtered_diffs = {}
        for ds_id, imgs in diffs.items():
            ignore_ids = {img for img in sampled_images.get(ds_id, [])}
            filtered_diffs[ds_id] = [img for img in imgs if img.id not in ignore_ids]
        return filtered_diffs

    def _sample(self, settings: SamplingSettings) -> Optional[Dict[int, List[ImageInfo]]]:
        """
        Sample images from the source project and copy them to the destination project.

        :param settings: dict with sample settings
        :type settings: SamplingSettings
        :return: dict with sampled images by dataset ID or None if no new items to copy
        :rtype: Optional[Dict[int, List[ImageInfo]]]
        """
        try:
            diffs = self.calculate_differences()  # dict with diffs by datasets
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
                sampling_method = None
                clustering_method = None
                num_clusters = None
                prompt = settings.get("prompt", None)
                limit = settings.get("limit", None)
                threshold = settings.get("threshold", 0.05)
                if mode == SamplingMode.AI_SEARCH.value:
                    # AI search mode
                    if prompt is None:
                        logger.error("Prompt is required for AI search mode.")
                        return None
                elif mode == SamplingMode.DIVERSE.value:
                    # Diverse mode
                    clustering_method = "kmeans"
                    if clustering_method == "kmeans":
                        num_clusters = 8
                    sampling_method = "centroids"
                else:
                    logger.error(f"Unknown sampling mode: {mode}")
                    return None

                # Send request to the API
                self.api.project.enable_embeddings(self.project_id)
                self.api.project.calculate_embeddings(self.project_id)

                collection_id = self.api.project.perform_ai_search(
                    project_id=self.project_id,
                    prompt=prompt,
                    method=sampling_method,
                    limit=limit or sample_size,
                    clustering_method=clustering_method,
                    num_clusters=num_clusters,
                    image_id_scope=all_diffs_flat,
                    threshold=threshold,
                )
                # module_info = self.api.app.get_ecosystem_module_info(slug=self.APP_SLUG)
                # sessions = self.api.app.get_sessions(
                #     self.team_id, module_info.id, statuses=[self.api.task.Status.STARTED]
                # )
                # if len(sessions) == 0:
                #     logger.error("No active sessions found for embeddings generator.")
                #     return None
                # session = sessions[0]
                # # api.app.wait(session.task_id, target_status=api.task.Status.STARTED)
                # logger.info(f"Embeddings generator session: {session.task_id}")
                # res = self.api.app.send_request(session.task_id, method, data=data)
                if isinstance(collection_id, int):
                    all_sampled_images = self.api.entities_collection.get_items(
                        collection_id=collection_id,
                        collection_type=CollectionTypeFilter.AI_SEARCH,
                        project_id=self.project_id,
                    )
                    new_sampled_images = {}
                    for img in all_sampled_images:
                        ds_id = img.dataset_id
                        if ds_id not in new_sampled_images:
                            new_sampled_images[ds_id] = []
                        new_sampled_images[ds_id].append(img)
                # elif collection_id is None:
                #     logger.error(f"Error during sampling")
                #     return None
                # elif isinstance(res, list):
                #     res_ids = {img["id"] for img in res}
                #     new_sampled_images = {
                #         ds_id: [img for img in diffs[ds_id] if img.id in res_ids]
                #         for ds_id in diffs.keys()
                #     }
                else:
                    self.status_text.show()
                    self.status_text.set(
                        "Error during sampling. Check that AI Search have calculated embeddings and is ready to use on project page.",
                        status="error",
                    )
                    logger.error(f"Error during sampling")
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

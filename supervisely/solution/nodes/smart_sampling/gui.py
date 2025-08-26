from __future__ import annotations

import random
from datetime import datetime
from enum import Enum
from typing import Dict, List, Literal, Optional, Tuple
from uuid import uuid4

from typing_extensions import NotRequired, TypedDict

from supervisely.api.api import Api
from supervisely.api.entities_collection_api import CollectionTypeFilter
from supervisely.api.image_api import ImageInfo
from supervisely.api.project_api import ProjectInfo
from supervisely.app.content import DataJson, StateJson
from supervisely.app.widgets import (
    Button,
    Card,
    Collapse,
    Container,
    Dialog,
    Empty,
    Field,
    Flexbox,
    GridGallery,
    Icons,
    Input,
    InputNumber,
    NotificationBox,
    OneOf,
    RadioGroup,
    Tabs,
    Text,
    Widget,
)
from supervisely.project.image_transfer_utils import (
    compare_projects,
    copy_structured_images,
)
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
        self.content = self._create_gui()

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
    def _create_gui(self):
        """
        Create the GUI for the Smart Sampling node.
        :param random_sampling: bool, if True, random sampling tab will be created
        :param diverse_sampling: bool, if True, diverse sampling tab will be created
        :param ai_search_sampling: bool, if True, AI Search sampling tab will be created
        """
        # --- Field with radio buttons to select sampling mode ------
        sampling_mode_field = self._create_sampling_mode_field()

        # --- Field with info text about abavailable images --------
        images_field = self._create_images_info_field()

        # --- Field with OneOf to input sampling settings and info text -----
        one_of_widget = self._create_one_of_widget()

        # --- Gallery in Card with preview button -------------------
        gallery_card = self._create_gallery_card()

        # --- Status text for showing messages ----------------------
        status_text = self._create_status_text()

        # --- Field with Save Settings and Run buttons ---------------
        buttons = self._create_main_buttons()

        @self.preview_button.click
        def preview_button_clicked():
            self.hide_status_text()
            self.uncollapse_preview()
            self.preview_gallery.clean_up()
            self.preview_gallery.loading = True
            sampling_settings = self.get_settings()
            if sampling_settings.get("sample_size", 0) > 6:
                sampling_settings["sample_size"] = 6
            if sampling_settings.get("limit", 0) > 6:
                sampling_settings["limit"] = 6

            sampled_images = self._sample(sampling_settings)
            if sampled_images is None:
                self.preview_gallery.loading = False
                self.set_status_text("No images to preview.", "warning")
                self.show_status_text()
                return

            infos = []
            for _, imgs in sampled_images.items():
                infos.extend(imgs)
            urls = [img.full_storage_url for img in infos]
            ai_metas = [img.ai_search_meta for img in infos]

            for idx, (url, ai_meta) in enumerate(zip(urls, ai_metas)):
                title = None
                if ai_meta is not None:
                    title = f"Score: {ai_meta.get('score'):.3f}"
                column = idx % 3
                self.preview_gallery.append(column_index=column, image_url=url, title=title)
            self.preview_gallery.loading = False

        return Container(
            [
                sampling_mode_field,
                images_field,
                one_of_widget,
                gallery_card,
                status_text,
                buttons,
            ],
            style="gap: 10px; margin-top: 10px;",
        )

    # ------------------------------------------------------------------
    # GUI Helpers ------------------------------------------------------
    # ------------------------------------------------------------------
    def _create_images_info_field(self) -> Field:
        get_total_text = lambda x: f"Total images in project: <strong>{x}</strong>"
        total_text = Text(get_total_text(self.items_count))
        self.set_total_num_text = lambda x: total_text.set(text=get_total_text(x), status="text")

        get_diff_text = lambda x: f"Available images for sampling: <strong>{x}</strong>"
        diff_text = Text(get_diff_text(self.diff_num))
        self.set_diff_num_text = lambda x: diff_text.set(text=get_diff_text(x), status="text")

        description = "Information about the total number of images in the input project and the number of available images for sampling."
        return Field(
            title="Images Info",
            description=description,
            content=Container([total_text, diff_text]),
            icon=Field.Icon(
                zmdi_class="zmdi zmdi-collection-image",
                color_rgb=[25, 118, 210],
                bg_color_rgb=[227, 242, 253],
            ),
        )

    def _create_sampling_mode_field(self) -> Field:
        """
        Create a field with radio buttons to select the sampling mode.
        """
        items = [
            RadioGroup.Item(
                value=SamplingMode.RANDOM.value, content=self._create_random_mode_content()
            ),
            RadioGroup.Item(
                value=SamplingMode.DIVERSE.value, content=self._create_diverse_mode_content()
            ),
            RadioGroup.Item(
                value=SamplingMode.AI_SEARCH.value, content=self._create_ai_search_mode_content()
            ),
        ]
        self.sampling_mode = RadioGroup(items=items, direction="vertical")

        # --- Notification box for info about sampling --------
        notification_box = self._create_notification()

        return Field(
            title="Sampling Mode",
            description="Select the sampling mode to use for sampling images.",
            content=Container([self.sampling_mode, notification_box]),
            icon=Field.Icon(
                zmdi_class="zmdi zmdi-settings",
                color_rgb=[25, 118, 210],
                bg_color_rgb=[227, 242, 253],
            ),
        )

    def _create_one_of_widget(self) -> OneOf:
        if not hasattr(self, "sampling_mode"):
            raise ValueError("Sampling mode must be created before creating OneOf widget.")
        return OneOf(conditional_widget=self.sampling_mode)

    def _create_notification(self) -> NotificationBox:
        description = "Sampling is performed only on images that have not been copied to the labeling project yet."
        return NotificationBox(description=description)

    def _create_random_mode_content(self) -> Field:
        """Create the content for the Random sampling mode."""
        num_input = InputNumber(value=1, min=1, max=self.diff_num)
        get_text = lambda x: f" of {x} images"
        text = Text(get_text(self.diff_num))
        container = Flexbox([num_input, text], vertical_alignment="center")

        # --- Methods -------------------------------------------------
        self.get_random_input_value = lambda: num_input.get_value()
        self.set_random_text = lambda x: text.set(text=get_text(x), status="text")
        self.set_random_input_value = lambda x: num_input.set_value(x)
        self.set_random_input_max = lambda x: num_input.set_max(x)
        self.set_random_input_min = lambda x: num_input.set_min(x)

        return Field(
            title="Sample Size",
            description="Select the number of images to sample randomly from the input project.",
            content=container,
            icon=Field.Icon(
                zmdi_class="zmdi zmdi-collection-image-o",
                color_rgb=[25, 118, 210],
                bg_color_rgb=[227, 242, 253],
            ),
        )

    def _create_diverse_mode_content(self) -> Field:
        """Create the content for the Diverse sampling mode."""
        num_input = InputNumber(value=1, min=1, max=self.diff_num)
        get_text = lambda x: f" of {x} images"
        text = Text(get_text(self.diff_num))
        container = Container(
            [num_input, text],
            direction="horizontal",
            gap=5,
            style="align-items: center",
        )

        # --- Methods -------------------------------------------------
        self.set_diverse_text = lambda x: text.set(text=get_text(x), status="text")
        self.get_diverse_input_value = lambda: num_input.get_value()
        self.set_diverse_input_value = lambda x: num_input.set_value(x)
        self.set_diverse_input_max = lambda x: num_input.set_max(x)
        self.set_diverse_input_min = lambda x: num_input.set_min(x)

        return Field(
            title="Sample Size",
            description="Select the number of images to sample using the diversity sampling strategy.",
            content=container,
            icon=Field.Icon(
                zmdi_class="zmdi zmdi-collection-image-o",
                color_rgb=[25, 118, 210],
                bg_color_rgb=[227, 242, 253],
            ),
        )

    def _create_ai_search_mode_content(self) -> Container:
        """Create the content for the AI Search sampling mode."""
        prompt_input = Input(placeholder="e.g. 'cat', 'dog', 'car'")
        limit_input = InputNumber(value=self.diff_num, min=1, max=self.diff_num)
        threshold_input = InputNumber(value=0.05, min=0, max=1, step=0.01)

        # --- Methods -------------------------------------------------
        self.get_ai_search_limit = lambda: limit_input.get_value()
        self.set_ai_search_limit_max = lambda x: limit_input.set_max(x)
        self.set_ai_search_limit_min = lambda x: limit_input.set_min(x)
        self.set_ai_search_limit_value = lambda x: limit_input.set_value(x)
        self.get_ai_search_threshold = lambda: threshold_input.get_value()
        self.set_ai_search_threshold = lambda x: threshold_input.set_value(x)
        self.get_ai_search_prompt = lambda: prompt_input.get_value()
        self.set_ai_search_prompt = lambda x: prompt_input.set_value(x)

        return Container(
            [
                Field(
                    title="Search Query",
                    description="Enter a search query to find the most suitable images using AI Search.",
                    content=prompt_input,
                    icon=Field.Icon(
                        zmdi_class="zmdi zmdi-search",
                        color_rgb=[25, 118, 210],
                        bg_color_rgb=[227, 242, 253],
                    ),
                ),
                Field(
                    title="Limit",
                    description="Set the maximum number of images to sample.",
                    content=limit_input,
                    icon=Field.Icon(
                        zmdi_class="zmdi zmdi-collection-image-o",
                        color_rgb=[25, 118, 210],
                        bg_color_rgb=[227, 242, 253],
                    ),
                ),
                Field(
                    title="Threshold",
                    description="Set the threshold for filtering images based on AI Search scores.",
                    content=threshold_input,
                    icon=Field.Icon(
                        zmdi_class="zmdi zmdi-code-setting",
                        color_rgb=[25, 118, 210],
                        bg_color_rgb=[227, 242, 253],
                    ),
                ),
            ]
        )

    def _create_gallery_card(self) -> Card:
        """Create a Card with a GridGallery for previewing sampled images."""
        self.preview_button = Button(
            "Preview", icon="zmdi zmdi-eye", plain=True, button_size="mini"
        )
        self.preview_gallery = GridGallery(columns_number=3)

        card = Card(
            title="Sampled Images Preview",
            description="Example of sampled images based on the selected sampling mode. Click 'Preview' to see the images.",
            collapsable=True,
            content=self.preview_gallery,
            content_top_right=self.preview_button,
            style="margin-top: 20px",
        )
        self.collapse_preview = lambda: card.collapse()
        self.uncollapse_preview = lambda: card.uncollapse()
        self.collapse_preview()
        return card

    def _create_status_text(self) -> Text:
        text = Text("", status="text")
        text.hide()
        self.show_status_text = lambda: text.show()
        self.hide_status_text = lambda: text.hide()
        self.set_status_text = lambda x, y: text.set(text=x, status=y)
        return text

    def _create_main_buttons(self) -> Container:
        """
        Create a container with Save Settings and Run buttons.
        """
        self.save_settings_button = Button("Save Settings", plain=True, icon="zmdi zmdi-save")
        self.run_button = Button("Run Sampling")

        return Container(
            [Flexbox([self.save_settings_button, self.run_button])],
            style="align-items: flex-end; margin-top: 10px;",
        )

    def update_widgets(
        self,
        diff: int,
        sampling_settings: Optional[dict] = None,
    ):
        """Update the sampling inputs based on the difference."""
        if sampling_settings is None:
            sampling_settings = self.get_settings()

        min_value = 0 if diff == 0 else 1
        mode = sampling_settings.get("mode")
        max_value = diff if diff > 0 else 0
        value = min(sampling_settings.get("sample_size", min_value), diff)

        # info texts
        self.set_diff_num_text(diff)
        project = self.api.project.get_info_by_id(self.project_id)
        self.set_total_num_text(project.items_count if project.items_count else 0)

        # random
        self.set_random_input_min(min_value)
        self.set_random_input_max(max_value)
        self.set_random_input_value(value)

        # diverse
        self.set_diverse_input_min(min_value)
        self.set_diverse_input_max(max_value)
        self.set_diverse_input_value(value)

        # ai search
        value = min(sampling_settings.get("limit", max_value), diff)
        self.set_ai_search_limit_min(min_value)
        self.set_ai_search_limit_max(max_value)
        self.set_ai_search_limit_value(value)
        self.set_ai_search_threshold(sampling_settings.get("threshold", 0.05))
        self.set_ai_search_prompt(sampling_settings.get("prompt", ""))

        self.sampling_mode.set_value(mode)
        self._diff_num = value
        DataJson()[self.widget_id]["differences_count"] = value
        DataJson().send_changes()

    # ------------------------------------------------------------------
    # Modal ------------------------------------------------------------
    # ------------------------------------------------------------------
    @property
    def modal(self) -> Dialog:
        """Returns the modal for the node GUI."""
        if not hasattr(self, "_modal"):
            self._modal = Dialog(title="Sampling Settings", content=self.content)
        return self._modal

    # ------------------------------------------------------------------
    # Settings --------------------------------------------------------
    # ------------------------------------------------------------------
    def get_settings(self) -> SamplingSettings:
        """
        Get the sample settings from the UI.

        :return: dict with sample settings
        """
        mode = self.sampling_mode.get_value()
        data = {"mode": mode}
        if mode == SamplingMode.RANDOM.value:
            data["sample_size"] = self.get_random_input_value()
        elif mode == SamplingMode.DIVERSE.value:
            data["sample_size"] = self.get_diverse_input_value()
            data["diversity_mode"] = "centroids"
        elif mode == SamplingMode.AI_SEARCH.value:
            data["prompt"] = self.get_ai_search_prompt()
            data["limit"] = self.get_ai_search_limit()
            data["threshold"] = self.get_ai_search_threshold()
        else:
            raise ValueError(f"Unknown sampling mode: {mode}")
        return data

    def save_settings(self, settings: SamplingSettings):
        """
        Save the sample settings to the UI.

        :param settings: dict with sample settings
        """
        mode = settings.get("mode")
        self.sampling_mode.set_value(mode)
        if mode == SamplingMode.RANDOM.value:
            self.set_random_input_value(settings.get("sample_size", 0))
        elif mode == SamplingMode.DIVERSE.value:
            self.set_diverse_input_value(settings.get("sample_size", 0))
        elif mode == SamplingMode.AI_SEARCH.value:
            self.set_ai_search_prompt(settings.get("prompt", ""))
            self.set_ai_search_limit_value(settings.get("limit", 0))
            self.set_ai_search_threshold(settings.get("threshold", 0.05))
        self.preview_gallery.clean_up()
        self.collapse_preview()

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
            return {}, {}, 0

        res = self._copy_to_new_project(sampled_images)
        if res is None:
            logger.warning("No images to copy to the labeling project.")
            return {}, {}, 0
        src, dst, images_count = res
        if images_count is None or images_count == 0:
            return {}, {}, 0

        self._add_record_to_history(
            sampling_id=uuid,
            status="success",
            total_items=images_count,
            settings=settings,
            items=sampled_images,
        )
        self.update_widgets(diff=self.calculate_diff_count(), sampling_settings=settings)
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

                # Send request to the API to perform AI search or diverse sampling
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
                else:
                    self.show_status_text()
                    self.set_status_text(
                        "Error during sampling. Check that AI Search have calculated embeddings and is ready to use on project page.",
                        "error",
                    )
                    logger.error(f"Error during sampling")
                    return None

            else:
                logger.error(f"Unknown sampling mode: {mode}")
                return None

            return new_sampled_images

        except Exception as e:
            logger.error(f"Error during sampling: {repr(e)}")

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
            for ds_id, img_list in items.items():
                _items[ds_id] = [img.id for img in img_list]
            self._add_sampled_images(sampling_id, _items)

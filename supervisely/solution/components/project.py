from typing import Callable, List, Optional, Tuple

from supervisely._utils import abs_url
from supervisely.api.api import Api
from supervisely.api.dataset_api import DatasetInfo
from supervisely.api.project_api import ProjectInfo
from supervisely.app.widgets import Button, SolutionProject
from supervisely.sly_logger import logger
from supervisely.solution.base_node import (
    Automation,
    SolutionElement,
    SolutionProjectNode,
)


class ProjectRefresh(Automation):
    """
    Automation for refreshing labeling queue information periodically
    """

    def __init__(self, project_id: int, func: Optional[Callable[[], None]] = None):
        super().__init__()
        self.job_id = f"refresh_project_{project_id}"
        self.project_id = project_id
        self.func = func

    def apply(self, sec: int, *args) -> None:
        self.scheduler.add_job(
            self.func, interval=sec, job_id=self.job_id, replace_existing=True, *args
        )

    def schedule_refresh(self, func: Callable[[], None], interval_sec: int = 5) -> None:
        """
        Schedule a job to refresh labeling queue info.
        """
        self.scheduler.add_job(
            func, interval=interval_sec, job_id=self.job_id, replace_existing=True
        )
        logger.info(
            f"Scheduled refresh for labeling queue {self.project_id} every {interval_sec} seconds"
        )

    def unschedule_refresh(self) -> None:
        """
        Unschedule the job that refreshes labeling queue info.
        """
        if self.scheduler.is_job_scheduled(self.job_id):
            self.scheduler.remove_job(self.job_id)
            logger.info(f"Unscheduled refresh for labeling queue {self.project_id}")


class ProjectGUI:
    def __init__(
        self,
        title: str,
        project: ProjectInfo,
        is_training: bool = False,
        dataset: Optional[DatasetInfo] = None,
    ):
        self.title = title
        self.project = project
        self.is_training = is_training
        self.dataset = dataset
        self.card = self._create_card()

    def _create_card(self) -> SolutionProject:
        """Creates the SolutionProject card with appropriate settings"""
        stats_url = self.project.url.replace("datasets", "stats/datasets")
        tooltip = self._create_tooltip(stats_url)

        if self.is_training:
            items_count, preview_url = [0, 0], [None, None]
        else:
            items_count = (
                self.project.items_count if self.dataset is None else self.dataset.items_count
            )
            preview_url = (
                self.project.image_preview_url
                if self.dataset is None
                else self.dataset.image_preview_url
            )
            items_count, preview_url = [items_count or 0], [preview_url]
            if preview_url[0] == abs_url("/"):
                preview_url[0] = None

        return SolutionProject(
            title=self.title,
            preview_url=preview_url,
            items_count=items_count,
            project_id=self.project.id,
            width=270 if self.is_training else 180,
            tooltip=tooltip,
            tooltip_position="left",
        )

    def _create_tooltip(self, stats_url: str) -> SolutionProject.Tooltip:
        """Creates the tooltip for the SolutionProject card"""
        tooltip_widgets = [
            Button(
                "Open project",
                icon="zmdi zmdi-open-in-new",
                button_size="mini",
                plain=True,
                link=self.project.url,
                button_type="text",
            ),
            Button(
                "QA stats",
                icon="zmdi zmdi-open-in-new",
                button_size="mini",
                plain=True,
                link=stats_url,
                button_type="text",
            ),
        ]

        return SolutionProject.Tooltip(
            description="Each import creates a dataset folder in the Input Project, centralising all incoming data and easily managing it over time.",
            content=tooltip_widgets,
        )

    def update_preview(self, preview_urls: List[str], items_counts: List[int]) -> None:
        """Update preview URLs and item counts for the project card"""
        self.card.update_preview_url(preview_urls)
        self.card.update_items_count(items_counts)


class ProjectNode(SolutionElement):
    """
    Project node for representing a Supervisely project in a solution.
    This node displays project information and provides links to view the project.
    """

    def __init__(
        self,
        api: Api,
        project_id: int,
        title: str = "Input Project",
        description: str = "Centralizes all incoming data. Data in this project will not be modified.",
        is_training: bool = False,
        x: int = 0,
        y: int = 0,
        dataset_id: Optional[int] = None,
        refresh_interval: int = 30,
        *args,
        **kwargs,
    ):
        """
        Initialize the Project node.

        :param api: Supervisely API instance
        :param project_id: ID of the project to display
        :param title: Title for the project node
        :param description: Description of the project node
        :param is_training: Whether this is a training project (affects display)
        :param x: X coordinate of the node
        :param y: Y coordinate of the node
        :param dataset_id: Optional dataset ID to filter items in the project
        :param args: Additional positional arguments
        """
        self.api = api
        self.project_id = project_id
        self.project = self.api.project.get_info_by_id(project_id)
        self.workspace_id = self.project.workspace_id
        self.dataset_id = dataset_id
        self.dataset = None
        if self.dataset_id is not None:
            self.dataset = self.api.dataset.get_info_by_id(self.dataset_id)
            
        self.title = title
        self.description = description
        self.is_training = is_training

        self.automation = ProjectRefresh(project_id=self.project_id, func=self.update)
        self.gui = ProjectGUI(
            title=self.title,
            project=self.project,
            is_training=self.is_training,
            dataset=self.dataset,
        )

        self.node = SolutionProjectNode(content=self.gui.card, x=x, y=y)
        self.modals = []
        self.refresh_interval = refresh_interval
        self.apply_automation(sec=self.refresh_interval)

        super().__init__(*args, **kwargs)

    def get_json_data(self) -> dict:
        """
        Returns the current data of the Project widget.
        """
        return {
            "project_id": self.project_id,
            "workspace_id": self.workspace_id,
            "is_training": self.is_training,
            "dataset_id": self.dataset_id,
        }

    def get_json_state(self) -> dict:
        """
        Returns the current state of the Project widget.
        """
        return {}

    def update(self, new_items_count: Optional[int] = None) -> None:
        """
        Update the project node with new information.

        :param new_items_count: Optional count of newly added items
        """
        self.project = self.api.project.get_info_by_id(self.project_id)
        items_count = self.project.items_count or 0
        preview_url = self.project.image_preview_url
        if self.dataset_id is not None:
            self.dataset = self.api.dataset.get_info_by_id(self.dataset_id)
            items_count = self.dataset.items_count or 0
            preview_url = self.dataset.image_preview_url

        if new_items_count is not None:
            self.node.update_property(key="Last update", value=f"+{new_items_count}")
            self.node.update_badge_by_key(key="Last update:", label=f"+{new_items_count}")
        self.node.update_property(key="Total", value=f"{items_count} images")

        # Update preview
        if self.is_training:
            train_items, val_items = self._get_train_val_items(dataset_id=self.dataset_id)
            self.gui.update_preview(
                [self._get_random_image_url(train_items), self._get_random_image_url(val_items)],
                [len(train_items), len(val_items)],
            )
        else:
            self.gui.update_preview([preview_url], [items_count or 0])

    def _get_train_val_collections(self) -> Tuple[List[int], List[int]]:
        """
        Returns the training and validation collections for the project.
        :return: Tuple of lists containing training and validation collection IDs.
        """
        train_collections = []
        val_collections = []

        if self.is_training:
            for collection_info in self.api.entities_collection.get_list(self.project_id):
                if collection_info.name.startswith("train_"):
                    train_collections.append(collection_info.id)
                elif collection_info.name.startswith("val_"):
                    val_collections.append(collection_info.id)

            if not train_collections and not val_collections:
                logger.warning("No training or validation collections found in the project.")

        return train_collections, val_collections

    def _get_train_val_items(
        self,
        dataset_id: Optional[int] = None,
    ) -> Tuple[List, List]:
        """
        Returns the items in training and validation collections.
        :param dataset_id: Optional dataset ID to filter items
        :type dataset_id: Optional[int]
        :return: Tuple containing lists of training and validation items.
        """
        train_collections, val_collections = self._get_train_val_collections()
        train_items, val_items = [], []

        for collection_id in train_collections:
            images = self.api.entities_collection.get_items(collection_id, self.project_id)
            train_items.extend(images)

        for collection_id in val_collections:
            images = self.api.entities_collection.get_items(collection_id, self.project_id)
            val_items.extend(images)

        if dataset_id is not None:
            # Filter items by dataset ID if provided
            train_items = [item for item in train_items if item.dataset_id == dataset_id]
            val_items = [item for item in val_items if item.dataset_id == dataset_id]

        return train_items, val_items

    def _get_random_image_url(self, images: List) -> Optional[str]:
        """Get a random image URL from a list of images"""
        import random

        if not images:
            return None

        image = random.choice(images)
        if hasattr(image, "preview_url") and image.preview_url:
            return image.preview_url
        elif hasattr(image, "id"):
            # Try to get the preview URL directly from the API
            return self.api.image.get_preview_url(image.id, self.project_id)

        return None

    def apply_automation(self, sec: int, *args) -> None:
        """
        Apply the automation to refresh the project node periodically.

        :param sec: Interval in seconds for refreshing the project node
        """
        self.automation.apply(sec, *args)

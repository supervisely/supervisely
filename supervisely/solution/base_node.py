from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, TypeVar, Union

from supervisely.api.project_api import ProjectInfo
from supervisely.app import DataJson, StateJson
from supervisely.app.widgets import (
    Button,
    SolutionCard,
    SolutionGraph,
    SolutionProject,
    Text,
    Widget,
)
from supervisely.solution.scheduler import TasksScheduler

T = TypeVar("T", bound=Widget)


# SolugionGraph.Node - only for x, y
# SolutionCard / SolutionProject - has specific GUI and properties (should be used as content)
# Schedulable - for scheduling tasks (in solution blocks)
# ?? where to put state management methods? inside SolutionWidget? (it will be incapsulated in cloud_import.gui but not in the block itself)


# class Schedulable:
#     """Mixin class for schedulable nodes.
#     This class provides a scheduler instance that can be used to schedule tasks.
#     """

#     def __init__(self):
#         """Initialize the Schedulable mixin."""
#         self._scheduler = TasksScheduler()

#     @property
#     def scheduler(self) -> TasksScheduler:
#         """Returns the scheduler instance."""
#         return self._scheduler

#     def add_job(
#         self,
#         job_id: str,
#         func: Callable,
#         sec: int,
#         replace_existing: bool = True,
#         args: Optional[List[Any]] = None,  # (var, )
#     ) -> None:
#         """Add a job to the scheduler with the specified parameters."""
#         return self._scheduler.add_job(job_id, func, sec, replace_existing, args)

#     def remove_job(self, job_id: str) -> bool:
#         """Remove a scheduled job using the provided scheduler."""
#         return self._scheduler.remove_job(job_id)

#     def is_job_scheduled(self, job_id: str) -> bool:
#         """Check if a job is scheduled using the provided scheduler."""
#         return self._scheduler.is_job_scheduled(job_id)


# class SolutionWidget:
#     def __new__(cls, content: T) -> T:
#         content.update_state = cls._update_state
#         return content

#     @staticmethod
#     def _update_state(state: dict) -> None:
#         print(state)


# class SolutionNode(SolutionGraph.Node):
#     pass
# class SolutionNode:
#     def __new__(cls, content: T, x: int = 0, y: int = 0) -> T:
#         content = super().__new__(cls, content)

#         content.x = x
#         content.y = y
#         content.node = SolutionGraph.Node(x=x, y=y, content=content)

#         # content.scheduler = TasksScheduler()
#         return content

# SolutionGraph + SolutionGraph.Node
# SolutionCard
# SolutionProject


# only SolutionCard/SolutionProject can be used as content
class SolutionCardNode(SolutionGraph.Node):
    def __init__(self, content: T, x: int = 0, y: int = 0):
        self.content = content
        self.content.x = x
        self.content.y = y

    def __new__(cls, content: T, x: int = 0, y: int = 0) -> T:
        if not isinstance(content, (SolutionCard, SolutionProject)):
            raise TypeError("Content must be one of SolutionCard or SolutionProject")
        return super().__new__(cls, content, x, y)

    def update_property(self, key: str, value: str, link: str = None, highlight: bool = None):
        for prop in self.content.tooltip_properties:
            if prop["key"] == key:
                self.content.update_property(key, value, link, highlight)
                return
        self.content.add_property(key, value, link, highlight)

    def remove_property_by_key(self, key: str):
        self.content.remove_property_by_key(key)

    def update_badge(
        self,
        idx: int,
        label: str,
        on_hover: str = None,
        badge_type: Literal["info", "success", "warning", "error"] = "info",
    ):
        self.content.update_badge(idx, label, on_hover, badge_type)

    def update_badge_by_key(
        self,
        key: str,
        label: str,
        badge_type: Literal["info", "success", "warning", "error"] = None,
        new_key: str = None,
    ):
        for idx, prop in enumerate(self.content.badges):
            if prop["on_hover"] == key:
                self.content.update_badge(idx, label, new_key, badge_type)
                return
        self.content.add_badge(
            self.card_cls.Badge(
                label=label,
                on_hover=new_key or key,
                badge_type=badge_type or "info",
            )
        )

    def add_badge(self, badge):
        self.content.add_badge(badge)

    def remove_badge(self, idx: int):
        self.content.remove_badge(idx)

    def remove_badge_by_key(self, key: str):
        self.content.remove_badge_by_key(key)

    def update_automation_badge(self, enable: bool) -> None:
        for idx, prop in enumerate(self.content.badges):
            if prop["on_hover"] == "Automation":
                if enable:
                    pass  # already enabled
                else:
                    self.content.remove_badge(idx)
                return

        if enable:  # if not found
            self.content.add_badge(
                SolutionCard.Badge(
                    label="⚡",
                    on_hover="Automation",
                    badge_type="warning",
                    plain=True,
                )
            )

    def show_automation_badge(self) -> None:
        self.update_automation_badge(True)

    def hide_automation_badge(self) -> None:
        self.update_automation_badge(False)


# only SolutionProject can be used as content
class SolutionProjectNode(SolutionCardNode):
    def __new__(cls, content: T, x: int = 0, y: int = 0) -> T:
        if not isinstance(content, SolutionProject):
            raise TypeError("Content must be an instance of SolutionProject")
        return super().__new__(cls, content, x, y)

    def update_preview(self, imgs: List[str], counts: List[int]):
        self.content.update_preview_url(imgs)
        self.content.update_items_count(counts)

    def update(
        self,
        project: ProjectInfo = None,
        new_items_count: int = None,
        urls: List[Union[int, str, None]] = None,
        counts: List[Union[int, None]] = None,
    ):
        if project is not None:
            self.project = project
        if new_items_count is not None:
            self.update_property(key="Last update", value=f"+{new_items_count}")
            self.update_property(key="Total", value=f"{self.project.items_count} images")
            self.update_badge_by_key(key="Last update:", label=f"+{new_items_count}")

        if self.is_training and urls is not None and counts is not None:
            self.update_preview(urls, counts)
        else:
            self.update_preview(
                [self.project.image_preview_url],
                [self.project.items_count],
            )


# # some block implementation (for example, cloud import)
class CloudImportGUI(Widget):
    def __init__(self, project_id: str, widget_id: Optional[str] = None):
        self.project_id = project_id
        self.btn = Button("Import from Cloud")
        self.on_import_callback = None
        super().__init__(widget_id=widget_id, file_path=__file__)

        @self.btn.click
        def on_import_click():
            self.run()

    def run(self):
        task_id = self.run_app()
        DataJson()[self.widget_id]["task_id"].append(task_id)
        DataJson().send_changes()


class SolutionCloudImport:
    def __init__(self, project_id: str, x: int, y: int):
        self.project_id = project_id
        self.gui = CloudImportGUI(project_id)
        self.node = SolutionCardNode(content=self.gui, x=x, y=y)


# # usage example
# cloud_import = SolutionCloudImport()


class BaseSolutionNode(Widget):
    card_cls = None

    """Base class for all nodes in a solution workflow.

    Each node maintains its own state and is responsible for its own actions.
    Nodes can have own content, tooltips, and perform actions.
    """

    def __init__(
        self,
        x: int,
        y: int,
        widget_id: Optional[str] = None,
    ):
        self.x = x
        self.y = y
        self.card = None  # * must be initialized in subclasses
        self._node = None
        self._create_gui()
        self._validate_gui_initialized()

        super().__init__(widget_id=widget_id, file_path=__file__)

    def _validate_gui_initialized(self):
        """Ensures that the card content is initialized before any operations."""
        if self.card is None:
            raise ValueError("Card content is not initialized.")

    def _create_gui(self) -> None:
        """Creates the main widget for the node.

        Example:
        ```python
        class MySolutionNode(SolutionNode):
            def _create_widgets(self):
                text = Text("Some text...")
                path_input = Input()
                run_btn = Button("Run", plain=True)
                run_btn_cont = Container([run_btn])
                content = Container([text, path_input, run_btn_cont])

                run_modal = Dialog(title="Import", content=content)
                open_modal_btn = Button("Open Modal")
                tooltip = self.card_cls.Tooltip(
                    description="This is a tooltip description.",
                    content=[open_modal_btn],
                )
                self.card = self.card_cls(
                    title="My Solution Node",
                    content=content,
                    tooltip=tooltip,
                    width=250,
                )
        ```
        """
        raise NotImplementedError("Subclasses must implement _create_widgets method.")

    def add_job(
        self,
        scheduler: TasksScheduler,  # Replace with actual type if available
        job_id: str,
        func: Callable,
        sec: int,
        replace_existing: bool = True,
        args: Optional[List[Any]] = None,
    ) -> None:
        """Add a job to the scheduler with the specified parameters."""
        raise NotImplementedError("Subclasses must implement add_job method.")

    def _add_job(
        self,
        scheduler: TasksScheduler,  # Replace with actual type if available
        job_id: str,
        func: Callable,
        sec: int,
        replace_existing: bool = True,
        args: Optional[List[Any]] = None,
    ) -> None:
        """Schedule a job using the provided scheduler."""
        if not isinstance(scheduler, TasksScheduler):
            raise TypeError("scheduler must be an instance of TasksScheduler")
        if sec <= 0:
            raise ValueError("Interval must be greater than 0 seconds")
        return scheduler.add_job(job_id, func, sec, replace_existing, args)

    def remove_job(
        self,
        scheduler: TasksScheduler,  # Replace with actual type if available
        job_id: str,
    ) -> bool:
        """Remove a scheduled job using the provided scheduler."""
        raise NotImplementedError("Subclasses must implement remove_job method.")

    def _remove_job(
        self,
        scheduler: TasksScheduler,  # Replace with actual type if available
        job_id: str,
    ) -> bool:
        """Remove a scheduled job using the provided scheduler."""
        if not isinstance(scheduler, TasksScheduler):
            raise TypeError("scheduler must be an instance of TasksScheduler")
        return scheduler.remove_job(job_id)

    def _is_job_scheduled(
        self,
        scheduler: TasksScheduler,  # Replace with actual type if available
        job_id: str,
    ) -> bool:
        """Check if a job is scheduled using the provided scheduler."""
        if not isinstance(scheduler, TasksScheduler):
            raise TypeError("scheduler must be an instance of TasksScheduler")
        return scheduler.is_job_scheduled(job_id)

    @property
    def node(self) -> SolutionGraph.Node:
        """Returns the SolutionGraph.Node representation of the card."""
        if self._node is None:
            if self.card is None:
                raise ValueError("Card content is not initialized.")
            self._node = SolutionGraph.Node(x=self.x, y=self.y, content=self.card)
        return self._node

    @property
    def state(self) -> Any:
        return StateJson()[self.widget_id]

    def send_state_changes(self) -> None:
        """Send changes to the state JSON."""
        StateJson().send_changes()

    @property
    def data(self) -> Any:
        return DataJson()[self.widget_id]

    def send_data_changes(self) -> None:
        """Send changes to the data JSON."""
        DataJson().send_changes()

    def update_in_state(self, update_dict: Dict[str, Any]) -> None:
        """Update the node's state with new values"""
        StateJson()[self.widget_id].update(update_dict)
        StateJson().send_changes()

    def save_to_state(self, key: str, value: Any) -> None:
        """Save a value to the node's state under the specified key"""
        StateJson()[self.widget_id][key] = value
        StateJson().send_changes()

    def append_to_state(self, key: str, value: Any) -> None:
        """Append a new value to the node's state under the specified key"""
        if key not in StateJson()[self.widget_id]:
            StateJson()[self.widget_id][key] = []
        StateJson()[self.widget_id][key].append(value)
        StateJson().send_changes()

    def update_in_data(self, update_dict: Dict[str, Any]) -> None:
        """Update the node's data with new values"""
        DataJson()[self.widget_id].update(update_dict)
        DataJson().send_changes()

    def save_to_data(self, key: str, value: Any) -> None:
        """Save a value to the node's data under the specified key"""
        DataJson()[self.widget_id][key] = value
        DataJson().send_changes()

    def append_to_data(self, key: str, value: Any) -> None:
        """Append a new value to the node's data under the specified key"""
        if key not in DataJson()[self.widget_id]:
            DataJson()[self.widget_id][key] = []
        DataJson()[self.widget_id][key].append(value)
        DataJson().send_changes()

    def update_property(self, key: str, value: str, link: str = None, highlight: bool = None):
        for prop in self.card.tooltip_properties:
            if prop["key"] == key:
                self.card.update_property(key, value, link, highlight)
                return
        self.card.add_property(key, value, link, highlight)

    def remove_property_by_key(self, key: str):
        self.card.remove_property_by_key(key)

    def update_badge(
        self,
        idx: int,
        label: str,
        on_hover: str = None,
        badge_type: Literal["info", "success", "warning", "error"] = "info",
    ):
        self.card.update_badge(idx, label, on_hover, badge_type)

    def update_badge_by_key(
        self,
        key: str,
        label: str,
        badge_type: Literal["info", "success", "warning", "error"] = None,
        new_key: str = None,
    ):
        for idx, prop in enumerate(self.card.badges):
            if prop["on_hover"] == key:
                self.card.update_badge(idx, label, new_key, badge_type)
                return
        self.card.add_badge(
            self.card_cls.Badge(
                label=label,
                on_hover=new_key or key,
                badge_type=badge_type or "info",
            )
        )

    def add_badge(self, badge):
        self.card.add_badge(badge)

    def remove_badge(self, idx: int):
        self.card.remove_badge(idx)

    def remove_badge_by_key(self, key: str):
        self.card.remove_badge_by_key(key)


# card_cls = SolutionGraph.Node


class SolutionNode(BaseSolutionNode):
    card_cls = SolutionCard
    """Base class for solution nodes with a card interface.

    This class extends BaseSolutionNode to provide a card-based interface
    for solution nodes, allowing for easy integration into the solution graph.
    """

    def update_automation_badge(self, enable: bool) -> None:
        for idx, prop in enumerate(self.card.badges):
            if prop["on_hover"] == "Automation":
                if enable:
                    pass  # already enabled
                else:
                    self.card.remove_badge(idx)
                return

        if enable:  # if not found
            self.card.add_badge(
                self.card_cls.Badge(
                    label="⚡",
                    on_hover="Automation",
                    badge_type="warning",
                    plain=True,
                )
            )

    def show_automation_badge(self) -> None:
        self.update_automation_badge(True)

    def hide_automation_badge(self) -> None:
        self.update_automation_badge(False)


class SolutionProjectNode(BaseSolutionNode):
    card_cls = SolutionProject
    """Base class for solution nodes with a card interface.

    This class extends BaseSolutionNode to provide a card-based interface
    for solution nodes, allowing for easy integration into the solution graph.
    """

    def _get_preview_details(self, ids: List[int]):
        preview_urls = []
        image_infos = self.api.image.get_info_by_id_batch(ids)
        for image in image_infos:
            preview_urls.append(image.preview_url)
        return preview_urls

    def update_preview(self, imgs: List[Union[int, str]], counts: List[int]):
        if all(isinstance(i, int) for i in imgs):
            imgs = self._get_preview_details(imgs)
        self.card.update_preview_url(imgs)
        self.card.update_items_count(counts)

    def update(
        self,
        project: ProjectInfo = None,
        new_items_count: int = None,
        urls: List[Union[int, str, None]] = None,
        counts: List[Union[int, None]] = None,
    ):
        if project is not None:
            self.project = project
        if new_items_count is not None:
            self.update_property(key="Last update", value=f"+{new_items_count}")
            self.update_property(key="Total", value=f"{self.project.items_count} images")
            self.update_badge_by_key(key="Last update:", label=f"+{new_items_count}")

        if self.is_training and urls is not None and counts is not None:
            self.update_preview(urls, counts)
        else:
            self.update_preview(
                [self.project.image_preview_url],
                [self.project.items_count],
            )

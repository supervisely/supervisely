from typing import Callable, Dict, List, Optional, Tuple

from supervisely._utils import abs_url
from supervisely.api.api import Api
from supervisely.api.image_api import ImageInfo
from supervisely.app.widgets import (
    Button,
    Checkbox,
    Container,
    Dialog,
    Empty,
    InputNumber,
    Select,
    SolutionCard,
    SolutionGraph,
    Text,
    Widget,
)
from supervisely.project.image_transfer_utils import move_structured_images
from supervisely.sly_logger import logger
from supervisely.solution.base_node import Automation, SolutionCardNode, SolutionElement
from supervisely.solution.utils import (
    get_interval_period,
    get_seconds_from_period_and_interval,
)


class MoveLabeledAuto(Automation):

    def __init__(self):
        super().__init__()
        self.apply_btn = Button("Apply", plain=True)
        self.widget = self._create_widget()
        self.job_id = self.widget.widget_id
        self.func = None

    def apply(self, func: Callable[[], None], *args) -> None:
        self.func = func
        sec, path, _, _ = self.get_automation_details()
        if sec is None or path is None:
            if self.scheduler.is_job_scheduled(self.job_id):
                self.scheduler.remove_job(self.job_id)
        else:
            self.scheduler.add_job(
                self.func, interval=sec, job_id=self.job_id, replace_existing=True, *args
            )

    def _create_widget(self):
        self.enabled_checkbox = Checkbox(content="Run every", checked=False)
        self.automate_input = InputNumber(
            min=1, value=60, debounce=1000, controls=False, size="mini"
        )
        self.automate_input.disable()
        self.automate_period_select = Select(
            [
                Select.Item("min", "minutes"),
                Select.Item("h", "hours"),
                Select.Item("d", "days"),
            ],
            size="mini",
        )
        self.automate_period_select.disable()
        automate_box_1 = Container(
            [self.enabled_checkbox, self.automate_input, self.automate_period_select, Empty()],
            direction="horizontal",
            gap=3,
            fractions=[1, 1, 1, 1],
            style="align-items: center",
        )
        self.automate_min_batch = Checkbox(content="Minimum batch size to copy", checked=False)
        self.automate_min_batch_input = InputNumber(
            min=1, value=1000, debounce=1000, controls=False, size="mini"
        )
        self.automate_min_batch.disable()
        self.automate_min_batch_input.disable()
        automate_box_2 = Container(
            [self.automate_min_batch, self.automate_min_batch_input, Empty(), Empty()],
            direction="horizontal",
            gap=3,
            fractions=[1, 1, 1, 1],
            style="align-items: center",
        )
        self.apply_btn = Button("OK", button_size="mini")
        automate_ok_btn_cont = Container([self.apply_btn], style="align-items: flex-end")

        text_1 = Text(
            "Schedule automatic copying of labeled data (status: finished) from the labeling project to the training project.",
            status="text",
            color="gray",
        )
        text_2 = Text(
            "Set the minimum batch size to copy. The copying will not be performed if the number of new labeled images in the labeling project is less than the specified value.",
            status="text",
            color="gray",
        )

        # self.automate_modal = Dialog(
        #     title="Automate Synchronization", size="tiny", content=automate_content
        # )

        @self.enabled_checkbox.value_changed
        def on_automate_checkbox_change(is_checked):
            if is_checked:
                self.automate_input.enable()
                self.automate_period_select.enable()
                self.automate_min_batch.enable()
            else:
                self.automate_input.disable()
                self.automate_period_select.disable()
                self.automate_min_batch.uncheck()
                self.automate_min_batch.disable()
                self.automate_min_batch_input.disable()

        @self.automate_min_batch.value_changed
        def on_automate_min_batch_change(is_checked):
            if is_checked:
                self.automate_min_batch_input.enable()
            else:
                self.automate_min_batch_input.disable()

        return Container([text_1, automate_box_1, text_2, automate_box_2, automate_ok_btn_cont])

    def get_automation_details(self):
        enabled = self.enabled_checkbox.is_checked()
        period = self.automate_period_select.get_value()
        interval = self.automate_input.get_value()
        min_batch_enabled = self.automate_min_batch.is_checked()
        min_batch = self.automate_min_batch_input.get_value() if min_batch_enabled else None

        if not enabled:
            return None, None, None, None, None

        sec = get_seconds_from_period_and_interval(period, interval)
        if sec == 0:
            logger.warning("Interval must be greater than 0")
            return None, None, None, None, None

        return enabled, period, interval, min_batch, sec

    def update_automation_widgets(self, enabled, sec, min_batch=None):
        if enabled:
            self.enabled_checkbox.check()
        else:
            self.enabled_checkbox.uncheck()
        period, interval = get_interval_period(sec)
        if period is not None and interval is not None:
            self.automate_period_select.set_value(period)
            self.automate_input.value = interval
        if min_batch is not None:
            self.automate_min_batch.check()
            self.automate_min_batch_input.value = min_batch
        else:
            self.automate_min_batch.uncheck()


class MoveLabeled(SolutionElement):
    """
    This class is a placeholder for the MoveLabeled node.
    It is used to move labeled data from one location to another.
    """

    def __init__(
        self,
        api: Api,
        src_project_id: int,
        dst_project_id: int,
        x: int = 0,
        y: int = 0,
        *args,
        **kwargs,
    ):
        self.api = api
        self.src_project_id = src_project_id
        self.dst_project_id = dst_project_id

        self.card = self._create_card()
        self.automation = MoveLabeledAuto()
        self.node = SolutionCardNode(content=self.card, x=x, y=y)
        self.automation_modal = self._create_automation_modal()
        self.modals = [self.automation_modal]

        super().__init__(*args, **kwargs)

    @property
    def pull_btn(self):
        if not hasattr(self, "_pull_btn"):
            self._pull_btn = self._create_pull_button()
        return self._pull_btn

    @property
    def automation_btn(self):
        if not hasattr(self, "_automation_btn"):
            self._automation_btn = self._create_automate_button()
        return self._automation_btn

    def _create_card(self):
        return SolutionCard(
            title="Move Finished images",
            tooltip=self._create_tooltip(),
            width=250,
        )

    def _create_tooltip(self):
        return SolutionCard.Tooltip(
            description="Move labeled and accepted images to the Training Project.",
            content=[
                self.pull_btn,
                self.automation_btn,
            ],
        )

    def _create_pull_button(self):
        """
        Create the button for pulling labeled data.
        """
        return Button(
            "Pull labeled data",
            icon="zmdi zmdi-refresh",
            button_size="mini",
            plain=True,
            button_type="text",
        )

    def _create_automate_button(self):
        """
        Create the button for automating the move of labeled data.
        """
        btn = Button(
            "Automate",
            icon="zmdi zmdi-flash-auto",
            button_size="mini",
            plain=True,
            button_type="text",
        )

        @btn.click
        def on_automate_click():
            self.automation_modal.show()

        return btn

    def apply_automation(self, func: Callable[[], None], *args) -> None:
        """
        Apply the automation function to the MoveLabeled node.
        """
        self.automation.apply(func, *args)
        self.update_automation_details()

    def _create_automation_modal(self):
        # self.automation.apply_btn.click(self.update_automation_details())
        return Dialog(
            title="Automate",
            size="tiny",
            content=self.automation.widget,
        )

    def update_automation_details(self) -> Tuple[int, str, int, str]:
        enabled, _, _, min_batch, sec = self.automation.get_automation_details()
        # self.automation_modal.hide()
        if self.node is not None:
            self.update_automation_properties(enabled, sec, min_batch)
            if enabled:
                self.node.show_automation_badge()
            else:
                self.node.hide_automation_badge()

    def run(self, image_ids: Dict[int, List[ImageInfo]]):
        src, dst, total_moved = move_structured_images(
            self.api,
            self.src_project_id,
            self.dst_project_id,
            images=image_ids,
        )
        return src, dst, total_moved

    def update_automation_properties(self, enabled, sec, min_batch=None):
        period, interval = get_interval_period(sec)
        if enabled is not None:
            self.card.update_property("Run every", f"{interval} {period}", highlight=True)
            if min_batch is not None:
                self.card.update_property("Min batch size", f"{min_batch}", highlight=True)
            else:
                self.card.remove_property_by_key("Min batch size")
        else:
            self.card.remove_property_by_key("Run every")
            self.card.remove_property_by_key("Min batch size")

    def apply_automation(self, func: Callable[[], None], *args) -> None:
        self.automation.apply(func, *args)
        self.update_automation_details()

    def update_automation_widgets(self, enabled, sec, min_batch=None):
        self.automation.update_automation_widgets(enabled, sec, min_batch)
        self.update_automation_properties(enabled, sec, min_batch)

    def add_to_collection(
        self,
        image_ids: List[int],
        split_name: str,
    ) -> None:
        """
        Add the MoveLabeled node to a collection.
        """
        if not image_ids:
            logger.warning("No images to add to collection.")
            return
        collections = self.api.entities_collection.get_list(self.dst_project_id)

        main_collection_name = f"All_{split_name}"
        main_collection = None

        last_batch_index = 0
        for collection in collections:
            if collection.name == main_collection_name:
                main_collection = collection
            elif collection.name.startswith(batch_collection_name):
                last_batch_index = max(last_batch_index, int(collection.name.split("_")[-1]))

        if main_collection is None:
            main_collection = self.api.entities_collection.create(
                self.dst_project_id, main_collection_name
            )
            logger.info(f"Created new collection '{main_collection_name}'")

        self.api.entities_collection.add_items(main_collection.id, image_ids)

        batch_collection_name = f"{split_name}_{last_batch_index + 1}"
        batch_collection = self.api.entities_collection.create(
            self.dst_project_id, batch_collection_name
        )
        logger.info(f"Created new collection '{batch_collection_name}'")

        self.api.entities_collection.add_items(batch_collection.id, image_ids)

        logger.info(f"Added {len(image_ids)} images to {split_name} collections")

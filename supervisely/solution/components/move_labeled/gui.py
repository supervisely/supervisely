from typing import Callable, Dict, List, Literal, Optional, Tuple

from supervisely.app.widgets import (
    Button,
    Checkbox,
    CheckboxField,
    Container,
    Empty,
    Field,
    InputNumber,
    Select,
    Switch,
    Text,
    Flexbox,
)
from supervisely.sly_logger import logger
from supervisely.solution.utils import (
    get_interval_period,
    get_seconds_from_period_and_interval,
)


class MoveLabeledGUI:
    """
    GUI components for the MoveLabeled node.
    """

    def __init__(self):
        self.widget = self._create_widgets()

    def _create_widgets(self) -> Container:
        """
        Creates the GUI widgets for the MoveLabeled node.
        """

        # Info -----------------------------------------------------------------------------
        text = Text(
            "This node is used to move labeled and accepted images to the Training project. After the data is moved, it will be added to the corresponding collections (train / val) in the Training project. <br><br> The node will not move images that have already been moved to the destination project to avoid duplication. <br><br> Note: The node will only move images that have been labeled and accepted. If the images are not labeled or accepted, they will not be moved.",
        )
        info = Field(
            text,
            title="How it works",
            icon=Field.Icon(
                zmdi_class="zmdi zmdi-help",
                color_rgb=(21, 101, 192),
                bg_color_rgb=(227, 242, 253),
            ),
        )

        # Automation settings -------------------------------------------------------------
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
            [self.automate_input, self.automate_period_select, Empty()],
            direction="horizontal",
            gap=3,
            fractions=[1, 1, 1],
            style="align-items: center",
        )
        self.automate_min_batch = Checkbox(content="Minimum batch size to copy", checked=False)
        self.automate_min_batch_input = InputNumber(
            min=1, value=1000, debounce=1000, controls=False, size="mini"
        )
        self.automate_min_batch.disable()
        self.automate_min_batch_input.disable()
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
        automate_box_2 = Container(
            [self.automate_min_batch, self.automate_min_batch_input, Empty(), Empty()],
            direction="horizontal",
            gap=3,
            fractions=[1, 1, 1, 1],
            style="align-items: center",
        )
        automation_field = Field(
            Container(
                [self.automation_switch, text_1, automate_box_1, text_2, automate_box_2],
            ),
            title="Enable Automation",
            description="Enable or disable automation for moving labeled data.",
            icon=Field.Icon(
                zmdi_class="zmdi zmdi-settings",
                color_rgb=(21, 101, 192),
                bg_color_rgb=(227, 242, 253),
            ),
        )

        @self.automation_switch.value_changed
        def on_automate_checkbox_change(is_checked):
            if is_checked:
                self.enable_automation_widgets()
            else:
                self.disable_automation_widgets()

        @self.automate_min_batch.value_changed
        def on_automate_min_batch_change(is_checked):
            if is_checked:
                self.automate_min_batch_input.enable()
            else:
                self.automate_min_batch_input.disable()

        run_btn_cont = Container(
            [Flexbox([self.automation_btn, self.run_btn])],
            style="align-items: flex-end",
        )
        return Container([info, automation_field, run_btn_cont])

    def enable_automation_widgets(self) -> None:
        """
        Enables the automation widgets for the MoveLabeled node.
        This method is called when the automation switch is toggled on.
        """
        self.automate_input.enable()
        self.automate_period_select.enable()
        self.automate_min_batch.enable()

    def disable_automation_widgets(self) -> None:
        """
        Disables the automation widgets for the MoveLabeled node.
        This method is called when the automation switch is toggled off.
        """
        self.automate_input.disable()
        self.automate_period_select.disable()
        self.automate_min_batch.uncheck()
        self.automate_min_batch.disable()
        self.automate_min_batch_input.disable()

    @property
    def automation_switch(self) -> Switch:
        if not hasattr(self, "_automation_switch"):
            self._automation_switch = Switch(switched=False)
        return self._automation_switch

    @property
    def run_btn(self):
        if not hasattr(self, "_run_btn"):
            self._run_btn = Button(
                "Pull labeled data",
                # icon="zmdi zmdi-refresh",
                # button_size="mini",
                # plain=True,
                # button_type="text",
            )
        return self._run_btn

    @property
    def automation_btn(self) -> Button:
        if not hasattr(self, "_automation_btn"):
            self._automation_btn = Button(
                "Save settings",
                icon="zmdi zmdi-check",
                # button_size="mini",
                plain=True,
                # button_type="text",
            )
        return self._automation_btn

    def get_automation_details(self):
        enabled = self.automation_switch.is_switched()
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
            self.automation_switch.check()
        else:
            self.automation_switch.uncheck()
        period, interval = get_interval_period(sec)
        if period is not None and interval is not None:
            self.automate_period_select.set_value(period)
            self.automate_input.value = interval
        if min_batch is not None:
            self.automate_min_batch.check()
            self.automate_min_batch_input.value = min_batch
        else:
            self.automate_min_batch.uncheck()
            self.automate_min_batch_input.value = min_batch

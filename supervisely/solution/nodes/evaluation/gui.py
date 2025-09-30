from typing import Any, Dict, List, Literal, Optional, Union

from supervisely.app.content import DataJson
from supervisely.app.widgets import (
    AgentSelector,
    Button,
    Container,
    Dialog,
    Field,
    Switch,
    Widget,
)
from supervisely.solution.engine.modal_registry import ModalRegistry


class EvaluationReportGUI(Widget):

    def __init__(self, team_id: int, widget_id: Optional[str] = None):
        self.team_id = team_id
        super().__init__(widget_id=widget_id)
        self.content = self._init_gui()

        ModalRegistry().attach_settings_widget(
            owner_id=self.widget_id, widget=self.content, size="tiny"
        )
    
    @property
    def modal(self) -> Dialog:
        return ModalRegistry().settings_dialog_tiny

    def open_modal(self):
        ModalRegistry().open_settings(owner_id=self.widget_id, size="tiny")

    def _init_gui(self):
        agent_selector_field = Field(
            self.agent_selector,
            title="Select Agent for Evaluation",
            description="Select the agent to deploy the model on.",
            icon=Field.Icon(
                zmdi_class="zmdi zmdi-storage",
                color_rgb=(21, 101, 192),
                bg_color_rgb=(227, 242, 253),
            ),
        )
        automation_field = Field(
            self.automation_switch,
            title="Enable Automation",
            description="Enable or disable automatic model re-evaluation after training.",
            icon=Field.Icon(
                zmdi_class="zmdi zmdi-settings",
                color_rgb=(21, 101, 192),
                bg_color_rgb=(227, 242, 253),
            ),
        )

        return Container([agent_selector_field, automation_field], gap=20)

    @property
    def run_btn(self) -> Button:
        if not hasattr(self, "_run_btn"):
            self._run_btn = Button(
                "Run",
                icon="zmdi zmdi-play",
                plain=True,
                button_type="text",
                button_size="mini",
            )

        return self._run_btn

    @property
    def automation_switch(self) -> Switch:
        if not hasattr(self, "_automation_switch"):
            self._automation_switch = Switch(switched=True)
        return self._automation_switch

    @property
    def agent_selector(self) -> AgentSelector:
        if not hasattr(self, "_agent_selector"):
            self._agent_selector = AgentSelector(self.team_id, show_only_gpu=True)
        return self._agent_selector

    def get_json_data(self) -> dict:
        return {
            "enabled": self.automation_switch.is_switched(),
            "agent_id": self.agent_selector.get_value(),
        }

    def get_json_state(self) -> dict:
        return {}

    def save_settings(self, enabled: bool, agent_id: Optional[int] = None):
        DataJson()[self.widget_id]["settings"] = {
            "enabled": enabled,
            "agent_id": agent_id if agent_id is not None else self.agent_selector.get_value(),
        }
        DataJson().send_changes()

    def load_settings(self):
        data = DataJson().get(self.widget_id, {}).get("settings", {})
        enabled = data.get("enabled")
        agent_id = data.get("agent_id")
        self.update_widgets(enabled, agent_id)

    def update_widgets(self, enabled: bool, agent_id: Optional[int] = None):
        if enabled is True:
            self.automation_switch.on()
        elif enabled is False:
            self.automation_switch.off()
        else:
            pass  # do nothing, keep current state
        if agent_id is not None:
            self.agent_selector.set_value(agent_id)
            self.automation_switch.on()
        elif enabled is False:
            self.automation_switch.off()
        else:
            pass  # do nothing, keep current state
        if agent_id is not None:
            self.agent_selector.set_value(agent_id)

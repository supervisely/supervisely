from typing import Optional

from supervisely.app.widgets import (
    AgentSelector,
    Button,
    Container,
    Dialog,
    Field,
    Text,
    Widget,
)
from supervisely.io.env import team_id
from supervisely.solution.engine.modal_registry import ModalRegistry


class MoveLabeledGUI(Widget):
    """
    GUI components for the MoveLabeled node.
    """

    def __init__(self, widget_id: Optional[str] = None):
        self.content = self._create_widget()
        super().__init__(widget_id=widget_id)

        # --- modals -------------------------------------------------------------
        ModalRegistry().attach_settings_widget(
            owner_id=self.widget_id,
            widget=self.content,
            size="tiny",
        )

    def get_json_data(self):
        return {}

    def get_json_state(self):
        return {}

    def open_modal(self):
        ModalRegistry().open_settings(owner_id=self.widget_id, size="tiny")

    def _create_widget(self) -> Container:
        """Creates the GUI widgets for the MoveLabeled node."""
        text = Text(
            """
            This node automates the process of moving labeled images to the Training project and organizing them into train/val collections according to the specified criteria in the previous steps. <br> <br>

            <div style="line-height: 1.4;">
            <strong>What this node does:</strong><br>
            &nbsp;&nbsp;• Moves labeled and accepted images to the Training project<br>
            &nbsp;&nbsp;• Automatically organizes data into train/val collections<br>
            
            <strong>Important notes:</strong><br>
            &nbsp;&nbsp;• Only processes <em>labeled and accepted</em> images<br>
            &nbsp;&nbsp;• Skips images already moved to the Training project<br>
            </div>
            """,
        )

        get_available_items = (
            lambda x: f"<strong>Available items to move:</strong><br>&nbsp;&nbsp;• {x} Images"
        )
        available_items_info = Text(get_available_items(0))

        # Function to update the available items count
        self.set_items_count = lambda x: available_items_info.set(
            text=get_available_items(x), status="text"
        )

        info = Field(
            Container([text, available_items_info], gap=15),
            title="How it works",
            icon=Field.Icon(
                zmdi_class="zmdi zmdi-help",
                color_rgb=(21, 101, 192),
                bg_color_rgb=(227, 242, 253),
            ),
        )
        self.agent_selector = AgentSelector(team_id(), compact=True)
        agent_select_field = Field(
            self.agent_selector,
            title="Agent",
            description="Select the agent to run the Data Commander task.",
            icon=Field.Icon(
                zmdi_class="zmdi zmdi-desktop-mac",
                color_rgb=(21, 101, 192),
                bg_color_rgb=(227, 242, 253),
            ),
        )
        btn_cont = Container([self.run_btn], style="align-items: flex-end")
        return Container([info, agent_select_field, btn_cont], gap=20)

    @property
    def run_btn(self):
        if not hasattr(self, "_run_btn"):
            self._run_btn = Button("Run")
        return self._run_btn

    @property
    def modal(self):
        return ModalRegistry().settings_dialog_tiny

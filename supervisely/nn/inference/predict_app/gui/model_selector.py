from typing import Any, Dict, List

from supervisely.api.api import Api
from supervisely.app.widgets import Button, Card, Container, DeployModel, Text


class ModelSelector:
    title = "Model"
    description = "Connect to deployed model or deploy new model"
    lock_message = "Select previous step to unlock"

    def __init__(self, api: Api, team_id: int):
        # Init Step
        self.api = api
        self.team_id = team_id
        self.display_widgets: List[Any] = []
        # -------------------------------- #

        # Init Base Widgets
        self.validator_text = None
        self.button = None
        self.container = None
        self.card = None
        # -------------------------------- #

        # Init Step Widgets
        self.model: DeployModel = None
        # -------------------------------- #

        # Model Selector
        self.model = DeployModel(api=self.api, team_id=self.team_id)
        # Add widgets to display ------------ #
        self.display_widgets.extend([self.model])
        # ----------------------------------- #

        # Base Widgets
        self.validator_text = Text("")
        self.validator_text.hide()
        # Add widgets to display ------------ #
        self.display_widgets.extend([self.validator_text])
        # ----------------------------------- #

        # Card Layout
        self.container = Container(self.display_widgets)
        self.card = Card(
            title=self.title,
            description=self.description,
            content=self.container,
            lock_message=self.lock_message,
        )
        # ----------------------------------- #

    @property
    def widgets_to_disable(self) -> list:
        return [
            self.model,
            self.model.connect_button,
            self.model.deploy_button,
            self.model.stop_button,
            self.model.disconnect_button,
        ]

    def get_settings(self) -> Dict[str, Any]:
        return self.model.get_deploy_parameters()

    def load_from_json(self, data):
        self.model.load_from_json(data)

    def validate_step(self) -> bool:
        self.validator_text.hide()

        if self.model.model_api is None:
            self.validator_text.set(text="Please connect or deploy a model", status="error")
            self.validator_text.show()
            return False

        return True

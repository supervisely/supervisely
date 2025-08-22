import os
from typing import List, Any, Dict

from supervisely.api.api import Api
from supervisely.app.widgets import Button, Container, Card, Text, GridGallery


class Preview:
    title = "Preview"
    description = "Preview the model output"
    lock_message = None

    def __init__(self, api: Api, static_dir: str):
        # Init Step
        self.api = api
        self.display_widgets: List[Any] = []
        self.static_dir = static_dir
        self.inference_settings = None
        # -------------------------------- #

        # Init Base Widgets
        self.validator_text = None
        self.button = None
        self.container = None
        self.card = None
        # -------------------------------- #

        # Init Step Widgets
        self.gallery = None
        # -------------------------------- #

        # Preview Directory
        self.preview_dir = os.path.join(self.static_dir, "preview")
        os.makedirs(self.preview_dir, exist_ok=True)
        self.preview_path = os.path.join(self.preview_dir, "preview.jpg")
        self.peview_url = f"/static/preview/preview.jpg"
        # ----------------------------------- #

        # Preview Widget
        self.gallery = GridGallery(
            2,
            sync_views=True,
            enable_zoom=True,
            resize_on_zoom=True,
            empty_message="Click 'Preview' to see the model output.",
        )
        # Add widgets to display ------------ #
        self.display_widgets.extend([self.gallery])
        # ----------------------------------- #

        # Base Widgets
        self.validator_text = Text("")
        self.validator_text.hide()
        self.button = Button("Preview", icon="zmdi zmdi-eye")
        # Add widgets to display ------------ #
        self.display_widgets.extend([self.validator_text, self.button])
        # ----------------------------------- #

        # Card Layout
        self.container = Container(self.display_widgets)
        self.card = Card(
            title="Preview",
            content=self.container,
            lock_message=self.lock_message,
        )
        self.card.lock()
        # ----------------------------------- #

        @self.button.click
        def button_click():
            self.run_preview()

    @property
    def widgets_to_disable(self) -> list:
        return [self.gallery]

    def load_from_json(self, data: Dict[str, Any]) -> None:
        return

    def get_settings(self) -> Dict[str, Any]:
        return {
            "preview_path": self.preview_path,
            "preview_url": self.peview_url,
            "inference_settings": self.inference_settings,
        }

    def validate_step(self) -> bool:
        return True

    def run_preview(self) -> None:
        raise NotImplementedError(
            "run_preview must be implemented by subclasses or injected at runtime"
        )

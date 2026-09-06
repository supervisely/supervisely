from supervisely.app.widgets import Widget, ConditionalWidget
from typing import Dict


class OneOf(Widget):
    """Renders exactly one of several child widgets based on conditional state (e.g. radio selection)."""

    def __init__(
        self,
        conditional_widget: ConditionalWidget,
        widget_id: str = None,
    ):
        """:param conditional_widget: Widget with conditional branches (e.g. radio + content map).
        :type conditional_widget: ConditionalWidget
        :param widget_id: Unique widget identifier.
        :type widget_id: str, optional
        """
        self._conditional_widget = conditional_widget
        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self) -> Dict:
        return {}

    def get_json_state(self) -> Dict:
        return {}

from typing import Any, Dict, List, Literal, Optional

from supervisely.app import DataJson
from supervisely.app.widgets import Widget
from supervisely.app.widgets.card.card import Card
from supervisely.app.widgets_context import JinjaWidgets


class CardV2(Card):

    def __init__(
        self,
        title: Optional[str] = None,
        description: Optional[str] = None,
        properties: Optional[Dict[str, List[Any]]] = None,
        properties_layout: Literal["horizontal", "vertical"] = "vertical",
        collapsable: Optional[bool] = False,
        content: Optional[Widget] = None,
        content_top_right: Optional[Widget] = None,
        icon: Optional[Widget] = None,
        lock_message: Optional[str] = "Card content is locked",
        widget_id: Optional[str] = None,
        remove_padding: Optional[bool] = False,
        overflow: Optional[Literal["auto", "unset", "scroll"]] = "auto",
        style: Optional[str] = "",
    ):
        self._title = title
        self._description = description
        self._collapsable = collapsable
        self._collapsed = False
        self._content = content
        self._content_top_right = content_top_right
        self._show_slot = content_top_right is not None
        self._icon = icon
        self._remove_padding = remove_padding
        self._overflow = overflow
        self._style = style
        self._options = {
            "collapsable": self._collapsable,
            "marginBottom": "0px",
            "contentOverflow": self._overflow,
        }
        self._lock_message = lock_message
        self._properties = properties
        self._properties_layout = properties_layout
        self._locked = {"disabled": False, "message": self._lock_message}

        super(Card, self).__init__(widget_id=widget_id, file_path=__file__)
        script_path = "./sly/css/app/widgets/card_v2/script.js"
        JinjaWidgets().context["__widget_scripts__"][self.__class__.__name__] = script_path

    def get_json_data(self) -> Dict[str, Any]:
        data = super().get_json_data()
        if self._properties:
            data["properties"] = self._properties
            data["propertiesLayout"] = self._properties_layout
        data["removePadding"] = self._remove_padding
        data["lockMessage"] = self._locked["message"]
        data["locked"] = self._locked["disabled"]
        data["collapsable"] = self._collapsable
        data["collapsed"] = self._collapsed
        data["overflow"] = self._overflow
        data["hasIcon"] = self._icon is not None
        data["hasContent"] = self._content is not None
        data["hasContentTopRight"] = self._content_top_right is not None
        return data

    def update_properties(self, properties: Dict[str, List[Any]]):
        if not isinstance(self._properties, dict):
            self._properties = properties
        else:
            self._properties.update(properties)
        DataJson()[self.widget_id]["properties"] = self._properties
        DataJson().send_changes()

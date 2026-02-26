from typing import Any, Callable, Dict, List, Literal, Optional, Union

from supervisely.app import StateJson
from supervisely.app.widgets import Widget

SUPPORTED_TAG_WIDGET_TYPES = ["primary", "gray", "success", "warning", "danger"]

# @TODO: fix problem with close transition id


class ElementTagsList(Widget):
    """Displays multiple element tags in a list."""

    class Routes:
        """Route name constants for this widget."""
        CLOSE = "tag_close_cb"

    class Tag:
        """Single tag in ElementTagsList."""

        def __init__(
            self,
            text: str,
            type: Optional[Literal["primary", "gray", "success", "warning", "danger"]] = "primary",
            hit: Optional[bool] = False,
            color: Optional[str] = "",
            closable: Optional[bool] = False,
            close_transition: Optional[bool] = False,
        ):
            """
            :param text: Tag text.
            :type text: str
            :param type: Tag type, one of: primary, gray, success, warning, danger.
            :type type: Optional[Literal["primary", "gray", "success", "warning", "danger"]]
            :param hit: Highlight tag.
            :type hit: Optional[bool]
            :param color: Custom color.
            :type color: Optional[str]
            :param closable: Show close button.
            :type closable: Optional[bool]
            :param close_transition: Animate on close.
            :type close_transition: Optional[bool]
            """
            self._text = text
            self._type = type
            self._hit = hit
            self._color = color
            self._closable = closable
            self._close_transition = close_transition

        @property
        def text(self) -> str:
            """Returns tag text.

            :returns: tag text
            :rtype: str
            """
            return self._text

        @property
        def type(self) -> str:
            """Returns tag type.

            :returns: tag type
            :rtype: str
            """
            return self._type

        @property
        def hit(self) -> bool:
            """Returns True if tag is highlighted, False otherwise.

            :returns: True if tag is highlighted, False otherwise
            :rtype: bool
            """
            return self._hit

        @property
        def color(self) -> str:
            """Returns tag color.

            :returns: tag color
            :rtype: str
            """
            return self._color

        def to_json(self) -> Dict[str, Union[str, bool]]:
            """Returns JSON representation of tag.

            Dictionary contains the following fields:
                - text: Tag text
                - type: Tag type, one of: primary, gray, success, warning, danger
                - hit: If True, tag will be highlighted
                - color: Tag color
                - closable: If True, tag will be closable
                - close_transition: If True, tag will be closable with transition

            :returns: dictionary with tag data
            :rtype: Dict[str, Union[str, bool]]
            """
            return {
                "text": self._text,
                "type": self._type,
                "hit": self._hit,
                "color": self._color,
                "closable": self._closable,
                "close_transition": self._close_transition,
            }

        @classmethod
        def from_json(cls, tag_json: Dict[str, Union[str, bool]]) -> "ElementTagsList.Tag":
            """Creates tag from JSON representation.

            :param tag_json: JSON representation of tag
            :type tag_json: Dict[str, Union[str, bool]]
            :returns: tag
            :rtype: :class:`~supervisely.app.widgets.element_tags_list.element_tags_list.ElementTagsList.Tag`
            """
            return cls(
                tag_json["text"],
                tag_json["type"],
                tag_json["hit"],
                tag_json["color"],
                tag_json["closable"],
                tag_json["close_transition"],
            )

    def __init__(
        self,
        tags: Optional[List[Tag]] = [],
        widget_id: Optional[str] = None,
    ):
        """
        :param tags: List of ElementTagsList.Tag.
        :type tags: Optional[List[ElementTagsList.Tag]]
        :param widget_id: Unique widget identifier.
        :type widget_id: Optional[str]

        :Usage Example:

            .. code-block:: python

                from supervisely.app.widgets import ElementTagsList
                tags = [ElementTagsList.Tag("Tag1", type="primary")]
                el = ElementTagsList(tags=tags)
        """
        self._clicked_tag = None

        self._validate_tags(tags)
        self._tags = tags

        self._close_handled = False

        super().__init__(widget_id=widget_id, file_path=__file__)

    def _validate_tags(self, tags):
        for tag in tags:
            if not isinstance(tag, ElementTagsList.Tag):
                raise ValueError("Tag type should be class ElementTagsList.Tag")

    def get_json_data(self) -> Dict:
        """ElementTagsList widget has no data, the method returns an empty dict."""
        return {}

    def get_json_state(self) -> Dict[str, List[Dict[str, Union[str, bool]]]]:
        """Returns dictionary with widget state.

        The dictionary contains the following fields:
            - tags: List of JSON representations of tags

        :returns: dictionary with widget state
        :rtype: Dict[str, List[Dict[str, Union[str, bool]]]]
        """
        return {"tags": [tag.to_json() for tag in self._tags]}

    def set_tags(self, tags: List[Tag]) -> None:
        """Sets tags for ElementTagsList widget.
        This method overwrites all previous tags.
        To add tags, use add_tags method.

        :param tags: List of tags
        :type tags: List[:class:`~supervisely.annotation.tag.Tag`]]
        """

        self._tags = tags
        self.update_state()
        StateJson().send_changes()

    def get_tags(self) -> List[Tag]:
        """Returns current tags.

        :returns: current tags
        :rtype: List[:class:`~supervisely.annotation.tag.Tag`]
        """
        return [ElementTagsList.Tag.from_json(tag) for tag in StateJson()[self.widget_id]["tags"]]

    def add_tags(self, tags: List[Tag]) -> None:
        """Adds tags to ElementTagsList widget.
        This method adds tags to the end of the list.
        To replace all tags, use set_tags method.

        :param tags: List of tags
        :type tags: List[:class:`~supervisely.annotation.tag.Tag`]]
        """
        self._tags = self.get_tags()
        self._tags.extend(tags)
        self.update_state()
        StateJson().send_changes()

    def close(self, func: Callable[[List[Tag]], Any]) -> Callable[[], None]:
        """Decorator for function that will be called when tag is closed.

        :param func: Function that will be called when tag is closed
        :type func: Callable[[List[:class:`~supervisely.annotation.tag.Tag`]], Any]
        :returns: Decorated function
        :rtype: Callable[[], None]
        """
        route_path = self.get_route_path(ElementTagsList.Routes.CLOSE)
        server = self._sly_app.get_server()
        self._changes_handled = True

        @server.post(route_path)
        async def _click():
            res = self.get_tags()
            func(res)

        return _click

from typing import Any, Callable, Dict, List, Literal, Optional, Union

from supervisely.app import StateJson
from supervisely.app.widgets import Widget

SUPPORTED_TAG_WIDGET_TYPES = ["primary", "gray", "success", "warning", "danger"]

# @TODO: fix problem with close transition id


class ElementTagsList(Widget):
    """ElementTagsList widget in Supervisely is a widget that allows users to display multiple elements tags in the UI.

    Read about it in `Developer Portal <https://developer.supervisely.com/app-development/widgets/text-elements/elementtagslist>`_
        (including screenshots and examples).

    :param tags: List of tags
    :type tags: Optional[List[Tag]]
    :param widget_id: An identifier of the widget.
    :type widget_id: str, optional

    :Usage example:
    .. code-block:: python

            from supervisely.app.widgets import ElementTagsList

            element_tags_list = ElementTagsList(
                tags=[
                    ElementTagsList.Tag(
                        text="Tag",
                        type="primary",
                        hit=True,
                        color="#20a0ff",
                    )
                ]
            )
    """

    class Routes:
        CLOSE = "tag_close_cb"

    class Tag:
        """Represents tag in ElementTagsList widget.

        :param text: Tag text
        :type text: Optional[str]
        :param type: Tag type, one of: primary, gray, success, warning, danger
        :type type: Optional[Literal["primary", "gray", "success", "warning", "danger"]]
        :param hit: If True, tag will be highlighted
        :type hit: Optional[bool]
        :param color: Tag color
        :type color: Optional[str]
        :param closable: If True, tag will be closable
        :type closable: Optional[bool]
        :param close_transition: If True, tag will be closable with transition
        :type close_transition: Optional[bool]
        """

        def __init__(
            self,
            text: str,
            type: Optional[Literal["primary", "gray", "success", "warning", "danger"]] = "primary",
            hit: Optional[bool] = False,
            color: Optional[str] = "",
            closable: Optional[bool] = False,
            close_transition: Optional[bool] = False,
        ):
            self._text = text
            self._type = type
            self._hit = hit
            self._color = color
            self._closable = closable
            self._close_transition = close_transition

        @property
        def text(self) -> str:
            """Returns tag text.

            :return: tag text
            :rtype: str
            """
            return self._text

        @property
        def type(self) -> str:
            """Returns tag type.

            :return: tag type
            :rtype: str
            """
            return self._type

        @property
        def hit(self) -> bool:
            """Returns True if tag is highlighted, False otherwise.

            :return: True if tag is highlighted, False otherwise
            :rtype: bool
            """
            return self._hit

        @property
        def color(self) -> str:
            """Returns tag color.

            :return: tag color
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

            :return: dictionary with tag data
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
            :return: tag
            :rtype: ElementTagsList.Tag
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

        :return: dictionary with widget state
        :rtype: Dict[str, List[Dict[str, Union[str, bool]]]]
        """
        return {"tags": [tag.to_json() for tag in self._tags]}

    def set_tags(self, tags: List[Tag]) -> None:
        """Sets tags for ElementTagsList widget.
        This method overwrites all previous tags.
        To add tags, use add_tags method.

        :param tags: List of tags
        :type tags: List[Tag]]
        """

        self._tags = tags
        self.update_state()
        StateJson().send_changes()

    def get_tags(self) -> List[Tag]:
        """Returns current tags.

        :return: current tags
        :rtype: List[Tag]
        """
        return [ElementTagsList.Tag.from_json(tag) for tag in StateJson()[self.widget_id]["tags"]]

    def add_tags(self, tags: List[Tag]) -> None:
        """Adds tags to ElementTagsList widget.
        This method adds tags to the end of the list.
        To replace all tags, use set_tags method.

        :param tags: List of tags
        :type tags: List[Tag]]
        """
        self._tags = self.get_tags()
        self._tags.extend(tags)
        self.update_state()
        StateJson().send_changes()

    def close(self, func: Callable[[List[Tag]], Any]) -> Callable[[], None]:
        """Decorator for function that will be called when tag is closed.

        :param func: Function that will be called when tag is closed
        :type func: Callable[[List[Tag]], Any]
        :return: Decorated function
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

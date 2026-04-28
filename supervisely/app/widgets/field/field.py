from __future__ import annotations

from typing import Any, Dict, List, Optional

import supervisely.imaging.color as sly_color
from supervisely.app.widgets import Widget


class Field(Widget):
    """Form field with title, description, and optional icon; wraps another widget."""

    class Icon:
        """Material Design Icon or image for Field header."""

        def __init__(
            self,
            zmdi_class: Optional[str] = None,
            color_rgb: Optional[List[int, int, int]] = None,
            bg_color_rgb: Optional[List[int, int, int]] = None,
            image_url: Optional[str] = None,
        ) -> Field.Icon:
            """
            :param zmdi_class: Material Design Icon class name.
            :type zmdi_class: Optional[str]
            :param color_rgb: Icon color [R, G, B].
            :type color_rgb: Optional[List[int, int, int]]
            :param bg_color_rgb: Background color [R, G, B].
            :type bg_color_rgb: Optional[List[int, int, int]]
            :param image_url: Image URL (alternative to zmdi_class).
            :type image_url: Optional[str]
            :raises ValueError: If both or neither zmdi_class/image_url; invalid colors.
            """
            if zmdi_class is None and image_url is None:
                raise ValueError(
                    "One of the arguments has to be defined: zmdi_class or image_url"
                )
            if zmdi_class is not None and image_url is not None:
                raise ValueError(
                    "Only one of the arguments has to be defined: zmdi_class or image_url"
                )

            if image_url is None and color_rgb is None:
                color_rgb = [255, 255, 255]

            if image_url is None and bg_color_rgb is None:
                bg_color_rgb = [0, 154, 255]

            self._zmdi_class = zmdi_class
            self._color = color_rgb
            self._bg_color = bg_color_rgb
            self._image_url = image_url
            if self._color is not None:
                sly_color._validate_color(self._color)
            if self._bg_color is not None:
                sly_color._validate_color(self._bg_color)

        def to_json(self) -> Dict[str, Any]:
            """Returns JSON representation of the icon.

            Dictionary contains the following fields:
                If icon is Material Design Icon:
                    - className: Material Design Icon class name
                    - color: RGB color of the icon
                    - bgColor: RGB color of the icon background
                If icon is image:
                    - imageUrl: URL of the icon image

            :returns: JSON representation of the icon
            :rtype: Dict[str, Any]
            """
            res = {}
            if self._zmdi_class is not None:
                res["className"] = self._zmdi_class
                res["color"] = sly_color.rgb2hex(self._color)
                res["bgColor"] = sly_color.rgb2hex(self._bg_color)
            if self._image_url is not None:
                res["imageUrl"] = self._image_url
                res["bgColor"] = sly_color.rgb2hex(self._bg_color)
            return res

    def __init__(
        self,
        content: Widget,
        title: str,
        description: Optional[str] = None,
        title_url: Optional[str] = None,
        description_url: Optional[str] = None,
        icon: Optional[Field.Icon] = None,
        widget_id: Optional[str] = None,
    ):
        """
        :param content: Child widget.
        :type content: :class:`~supervisely.app.widgets.widget.Widget`
        :param title: Field title.
        :type title: str
        :param description: Field description.
        :type description: Optional[str]
        :param title_url: Link URL for title.
        :type title_url: Optional[str]
        :param description_url: Link URL for description.
        :type description_url: Optional[str]
        :param icon: Field.Icon for header.
        :type icon: Optional[Field.Icon]
        :param widget_id: Widget identifier.
        :type widget_id: Optional[str]

        :Usage Example:

            .. code-block:: python

                from supervisely.app.widgets import Field, Text
                field = Field(Text("Hello"), "Title", "Description", icon=Field.Icon(zmdi_class="zmdi zmdi-bike"))
        """
        self._title = title
        self._description = description
        self._title_url = title_url
        self._description_url = description_url
        self._icon = icon
        self._content = content
        if self._title_url is not None and self._title is None:
            raise ValueError(
                "Title can not be specified only as url without text value"
            )
        if self._description_url is not None and self._description is None:
            raise ValueError(
                "Description can not be specified only as url without text value"
            )

        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self) -> Dict[str, Any]:
        """Returns dictionary with widget data, which defines the appearance and behavior of the widget.
        Dictionary contains the following fields:
            - title: Title of the field
            - description: Description of the field
            - title_url: URL of the title
            - description_url: URL of the description
            - icon: Icon for the field
            If icon is Material Design Icon:
            - icon with the following fields:
                - className: Material Design Icon class name
                - color: RGB color of the icon
                - bgColor: RGB color of the icon background

        :returns: Dictionary with widget data
        :rtype: Dict[str, Any]
        """
        res = {
            "title": self._title,
            "description": self._description,
            "title_url": self._title_url,
            "description_url": self._description_url,
            "icon": None,
        }
        if self._icon is not None:
            res["icon"] = self._icon.to_json()
        return res

    def get_json_state(self) -> None:
        """Field widget does not have state, the method returns None."""
        return None

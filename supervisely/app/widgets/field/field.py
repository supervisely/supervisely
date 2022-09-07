from __future__ import annotations
from typing import List
from supervisely.app.widgets import Widget
from supervisely.api.project_api import ProjectInfo
from supervisely.project.project import Project
import supervisely.imaging.color as sly_color


class Field(Widget):
    class Icon:
        def __init__(
            self,
            zmdi_class=None,
            color_rgb: List[int, int, int] = [255, 255, 255],
            bg_color_rgb: List[int, int, int] = [0, 154, 255],
            image_url=None,
        ) -> Field.Icon:
            if zmdi_class is None and image_url is None:
                raise ValueError(
                    "One of the arguments has to be defined: zmdi_class or image_url"
                )
            if zmdi_class is not None and image_url is not None:
                raise ValueError(
                    "Only one of the arguments has to be defined: zmdi_class or image_url"
                )

            self._zmdi_class = zmdi_class
            self._color = color_rgb
            self._bg_color = bg_color_rgb
            self._image_url = image_url
            sly_color._validate_color(self._color)
            sly_color._validate_color(self._bg_color)

        def to_json(self):
            res = {
                "className": self._zmdi_class,
                "color": sly_color.rgb2hex(self._color),
                "bgColor": sly_color.rgb2hex(self._bg_color),
                "image_url": self._image_url,
            }
            return res

    def __init__(
        self,
        title: str,
        description: str = None,
        title_url: str = None,
        description_url: str = None,
        icon: Field.Icon = None,
        content: Widget = None,
    ):
        self._title = title
        self._description = description
        self._title_url = title_url
        self._description_url = description_url
        self._icon = icon
        self._content = content

        self._info = info
        self._description = f"{self._info.items_count} {self._info.type}"
        self._url = Project.get_url(self._info.id)
        super().__init__(file_path=__file__)

    def get_json_data(self):
        return {
            "id": self._info.id,
            "name": self._info.name,
            "description": self._description,
            "url": self._url,
            "image_preview_url": self._info.image_preview_url,
        }

    def get_json_state(self):
        return None

# coding: utf-8

from supervisely_lib.collection.str_enum import StrEnum


class WidgetType(StrEnum):
    TABLE = 'table'
    PLOTLY = "plotly"
    MARKDOWN = "markdown"
    NOTIFICATION = "notification"
    FORM = "form"
    BUTTON = "button"
    GALLERY = "gallery"
    LINECHART = "line-chart"
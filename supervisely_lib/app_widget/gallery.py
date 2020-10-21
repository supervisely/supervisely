# coding: utf-8
import uuid

from supervisely_lib.api.api import Api
from supervisely_lib.api.module_api import ApiField
from supervisely_lib.app_widget.widget_type import WidgetType
from supervisely_lib.app_widget.base_widget import BaseWidget


class GalleryWidget(BaseWidget):
    widget_type = WidgetType.GALLERY

    @classmethod
    def create(cls, name, description, image_mask_pairs=None, id=None):
        res = dict()
        res[ApiField.TYPE] = str(cls.widget_type)
        res[ApiField.NAME] = name
        res[ApiField.SUBTITLE] = description
        res[ApiField.CONTENT] = image_mask_pairs if image_mask_pairs is not None else []
        res[ApiField.ID] = uuid.uuid4().hex if id is None else id
        return res

    @classmethod
    def add_item(cls, api: Api, app_id, widget_id, img_url, mask_url):
        item = [[img_url, mask_url]]
        return api.report.update_widget(report_id=app_id, widget_id=widget_id, content=item)

    @classmethod
    def replace_items(cls, api: Api, app_id, widget_id, items):
        return api.report.rewrite_widget(report_id=app_id, widget_id=widget_id, widget_type=str(cls.widget_type), content=items)
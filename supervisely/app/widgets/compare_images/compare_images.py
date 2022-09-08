import uuid

from supervisely.annotation.annotation import Annotation
from supervisely.app import DataJson
from supervisely.app.widgets import GridGallery


class CompareImages(GridGallery):
    def __init__(
            self,
            annotations_opacity: float = 0.5,
            show_opacity_slider: bool = True,
            enable_zoom: bool = False,
            resize_on_zoom: bool = False,
            fill_rectangle: bool = True,
            border_width: int = 3,
            widget_id: str = None,
    ):

        super().__init__(
            columns_number=2,
            annotations_opacity=annotations_opacity,
            show_opacity_slider=show_opacity_slider,
            enable_zoom=enable_zoom,
            resize_on_zoom=resize_on_zoom,
            fill_rectangle=fill_rectangle,
            border_width=border_width,
            widget_id=widget_id,
        )

    # def _clean_left(self):
    #     for idx, img in enumerate(self._data):
    #         if img.get("position") == "left":
    #             del self._data[idx]
    #             break
    #     DataJson()[self.widget_id]["content"]["layout"][0] = []
    #     self._update()
    #     self.update_data()
    #
    # def _clean_right(self):
    #     for idx, img in enumerate(self._data):
    #         if img.get("position") == "right":
    #             del self._data[idx]
    #             break
    #     DataJson()[self.widget_id]["content"]["layout"][1] = []
    #     self._update()
    #     self.update_data()


    def _clean_img(self, position):
        col_idx = 0 if position == "left" else 1
        for idx, img in enumerate(self._data):
            if img.get("position") == position:
                del self._data[idx]
                break
        DataJson()[self.widget_id]["content"]["layout"][col_idx] = []
        self._update()
        self.update_data()

    def _update_image(self, image_url: str, position: str, annotation: Annotation = None, title: str = "",
                      column_index: int = None):
        column_index = self.get_column_index(incoming_value=column_index)
        cell_uuid = str(
            uuid.uuid5(
                namespace=uuid.NAMESPACE_URL, name=f"{image_url}_{title}_{column_index}"
            ).hex
        )
        self._data.append(
            {
                "image_url": image_url,
                "annotation": Annotation((1, 1))
                if annotation is None
                else annotation.clone(),
                "column_index": column_index,
                "title": title,
                "cell_uuid": cell_uuid,
                "position": position
            }
        )
        self._update()

    def set_left(self, title, image_url, ann: Annotation = None):
        if len(DataJson()[self.widget_id]["content"]["layout"]) != 0:
            self._clean_img(position="left")
        self._update_image(image_url=image_url, annotation=ann, title=title, column_index=0, position="left")
        DataJson().send_changes()

    def set_right(self, title, image_url, ann: Annotation = None):
        if len(DataJson()[self.widget_id]["content"]["layout"]) != 0:
            self._clean_img(position="right")
        self._update_image(image_url=image_url, annotation=ann, title=title, column_index=1, position="right")
        DataJson().send_changes()

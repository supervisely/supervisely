from pathlib import Path
from typing import List, Union
from urllib.parse import urlparse

import cv2
import numpy as np

from supervisely.annotation.annotation import Annotation
from supervisely.app.content import DataJson, StateJson
from supervisely.app.widgets import Widget
from supervisely.app.widgets_context import JinjaWidgets
from supervisely.imaging.image import write


def mask_to_heatmap(mask, colormap=cv2.COLORMAP_JET, transparent_low=False):
    """
    Convert a (H,W,3) or (H,W) RGB mask to an RGBA heatmap PNG.
    Black pixels (0,0,0) become transparent if transparent_low=True.
    """
    # Convert to grayscale if RGB
    if mask.ndim == 3:
        mask_gray = mask.mean(axis=-1).astype(np.uint8)
    else:
        mask_gray = mask.astype(np.uint8)

    # Normalize to 0-255
    mask_norm = cv2.normalize(mask_gray, None, 0, 255, cv2.NORM_MINMAX)

    mask_norm = cv2.GaussianBlur(mask_norm, (5, 5), 0)

    # Apply colormap (returns BGR)
    heatmap_bgr = cv2.applyColorMap(mask_norm, colormap)

    # Convert BGR to BGRA
    heatmap_bgra = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2BGRA)

    if transparent_low:
        # Set alpha = 0 where mask is 0, else 255
        alpha = np.where(mask_norm == 0, 0, 255).astype(np.uint8)
        heatmap_bgra[..., 3] = alpha

    heatmap_rgba = heatmap_bgra[..., [2, 1, 0, 3]]

    return heatmap_rgba


class Heatmap(Widget):

    def __init__(
        self,
        static_dir: str,
        background_image: Union[str, np.ndarray] = None,
        heatmap_mask: np.ndarray = None,
        width: int = None,
        height: int = None,
        widget_id: str = None,
        file_path: str = __file__,
    ):
        self._background_url = None
        self._heatmap_url = None
        self._width = width
        self._height = height
        self._opacity = 75
        self.static_dir = static_dir
        super().__init__(widget_id, file_path=file_path)

        if background_image:
            self.set_background(background_image)

        if heatmap_mask:
            self.set_heatmap(background_image)

        script_path = "./sly/css/app/widgets/heatmap/script.js"
        JinjaWidgets().context["__widget_scripts__"][self.__class__.__name__] = script_path

    def _save_to_static(self, img: np.ndarray, name: str):
        self.static_path = Path(self.static_dir)
        self.static_path.mkdir(parents=True, exist_ok=True)
        img_path = self.static_path / self.widget_id / name
        write(str(img_path), img, remove_alpha_channel=False)

    def get_json_data(self):
        return {
            "backgroundUrl": self._background_url,
            "heatmapUrl": self._heatmap_url,
            "width": self._width,
            "height": self._height,
        }

    def get_json_state(self):
        return {"opacity": self._opacity}

    def set_background(self, background_image: Union[str, np.ndarray]):
        try:
            if isinstance(background_image, np.ndarray):
                self._save_to_static(background_image, "background.png")
                self._background_url = f"/static/{self.widget_id}/background.png"
        except Exception as e:
            self._background_url = None
        finally:
            DataJson()[self.widget_id]["backgroundUrl"] = self._background_url
            DataJson().send_changes()

    def set_heatmap(self, mask: np.ndarray):
        try:
            heatmap = mask_to_heatmap(mask)
            self._save_to_static(heatmap, name="mask.png")
            self._heatmap_url = f"/static/{self.widget_id}/mask.png"
            self._min_value = mask.min()
            self._max_value = mask.max()
        except Exception as e:
            self._heatmap_url = None
        finally:
            DataJson()[self.widget_id]["heatmapUrl"] = self._heatmap_url
            DataJson()[self.widget_id]["minValue"] = self._min_value
            DataJson()[self.widget_id]["maxValue"] = self._max_value
            DataJson().send_changes()

    @property
    def opacity(self):
        StateJson()[self.widget_id]["opacity"]

    @opacity.setter
    def opacity(self, value: int):
        value = max(0, value)
        value = min(100, value)
        StateJson()[self.widget_id]["opacity"] = value
        StateJson().send_changes()

    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, value):
        self._width = value
        DataJson()[self.widget_id]["width"] = self._width

    @property
    def height(self):
        return self._height

    @height.setter
    def height(self, value):
        self._height = value
        DataJson()[self.widget_id]["height"] = self._height

    def set_heatmap_from_annotations(self, anns: List[Annotation], object_name: str = None):
        if len(anns) == 0:
            raise ValueError("asdasd")
        sizes = [ann.img_size for ann in anns]
        avg_size = (
            sum(size[0] for size in sizes) / len(sizes),
            sum(size[1] for size in sizes) / len(sizes),
        )
        mask = np.zeros(avg_size)
        for ann in anns:
            for label in ann.labels:
                if object_name is None or label.obj_class.name == object_name:
                    label.resize(ann.img_size, avg_size)
                    label.draw(mask)
        self.set_heatmap(mask)

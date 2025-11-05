import shutil
from pathlib import Path
from typing import Any, Callable, List, Union
from urllib.parse import urlparse

import cv2
import numpy as np

from supervisely.annotation.annotation import Annotation
from supervisely.app.content import DataJson, StateJson
from supervisely.app.widgets import Widget
from supervisely.app.widgets_context import JinjaWidgets
from supervisely.imaging.image import write


def mask_to_heatmap(
    mask: np.ndarray, colormap=cv2.COLORMAP_JET, transparent_low=False, vmin=None, vmax=None
):
    if mask.ndim == 3:
        mask_gray = mask.mean(axis=-1)
    else:
        mask_gray = mask.copy()
    mask_gray = mask_gray.astype(np.float64)
    if vmin is None:
        vmin = np.nanmin(mask_gray)
    if vmax is None:
        vmax = np.nanmax(mask_gray)

    if vmax == vmin:
        mask_norm = np.full_like(mask_gray, 128, dtype=np.uint8)
    else:
        mask_norm = ((mask_gray - vmin) / (vmax - vmin) * 255).astype(np.uint8)
    mask_norm = cv2.GaussianBlur(mask_norm, (5, 5), 0)
    heatmap_bgr = cv2.applyColorMap(mask_norm, colormap)
    heatmap_bgra = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2BGRA)

    if transparent_low:
        alpha = np.where(mask_norm == 0, 0, 255).astype(np.uint8)
        heatmap_bgra[..., 3] = alpha
    heatmap_rgba = heatmap_bgra[..., [2, 1, 0, 3]]

    return heatmap_rgba


def colormap_to_hex_list(colormap=cv2.COLORMAP_JET, n=5):
    values = np.linspace(0, 255, n, dtype=np.uint8)
    colors_bgr = cv2.applyColorMap(values[:, None], colormap)
    colors_rgb = colors_bgr[:, 0, ::-1]
    return [f"#{r:02X}{g:02X}{b:02X}" for r, g, b in colors_rgb]


class Heatmap(Widget):
    """
    Supervisely widget that displays an interactive heatmap overlay
    on top of a background image.
    """

    class Routes:
        CLICK = "heatmap_clicked_cb"

    def __init__(
        self,
        static_dir: str,
        background_image: Union[str, np.ndarray] = None,
        heatmap_mask: np.ndarray = None,
        vmin: Any = None,
        vmax: Any = None,
        transparent_low: bool = False,
        colormap: int = cv2.COLORMAP_JET,
        width: int = None,
        height: int = None,
        widget_id: str = None,
        file_path: str = __file__,
    ):
        """
        Initializes the Heatmap widget.

        :param static_dir: Path to the directory where static files (images, CSS, etc.) are stored.
        :type static_dir: str
        :param background_image: Background image to display under the heatmap. Can be a path to an image file or a NumPy array.
        :type background_image: Union[str, np.ndarray], optional
        :param heatmap_mask: NumPy array representing the heatmap mask values.
        :type heatmap_mask: np.ndarray, optional
        :param vmin: Minimum value for normalizing the heatmap. If None, it is inferred from the mask.
        :type vmin: Any, optional
        :param vmax: Maximum value for normalizing the heatmap. If None, it is inferred from the mask.
        :type vmax: Any, optional
        :param transparent_low: Whether to make low values in the heatmap transparent.
        :type transparent_low: bool, optional
        :param colormap: OpenCV colormap used to colorize the heatmap (e.g., cv2.COLORMAP_JET).
        :type colormap: int, optional
        :param width: Width of the output heatmap in pixels.
        :type width: int, optional
        :param height: Height of the output heatmap in pixels.
        :type height: int, optional
        :param widget_id: Unique identifier for the widget instance.
        :type widget_id: str, optional
        :param file_path: Path to the file where the widget is defined (used for static resource resolution).
        :type file_path: str, optional
        """
        self._background_url = None
        self._heatmap_url = None
        self._mask_data = None
        self._vmin = vmin
        self._vmax = vmax
        self._transparent_low = transparent_low
        self._colormap = colormap
        self._width = width
        self._height = height
        self._opacity = 70
        self._min_value = 0
        self._max_value = 0
        self.static_dir = static_dir
        super().__init__(widget_id, file_path=file_path)

        if background_image:
            self.set_background(background_image)

        if heatmap_mask:
            self.set_heatmap(heatmap_mask)

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
            "maskData": self._mask_data,
            "width": self._width,
            "height": self._height,
            "minValue": self._min_value,
            "maxValue": self._max_value,
            "legendColors": colormap_to_hex_list(self._colormap),
        }

    def get_json_state(self):
        return {"opacity": self._opacity, "clickedValue": None, "maskX": None, "maskY": None}

    def set_background(self, background_image: Union[str, np.ndarray]):
        try:
            if isinstance(background_image, np.ndarray):
                self._save_to_static(background_image, "background.png")
                self._background_url = f"/static/{self.widget_id}/background.png"
            elif isinstance(background_image, str):
                parsed = urlparse(background_image)
                bg_image_path = Path(background_image)
                if parsed.scheme in ("http", "https") and parsed.netloc:
                    self._background_url = background_image
                elif bg_image_path.exists() and bg_image_path.is_file():
                    img_name = bg_image_path.name
                    dst_path = self.static_path / self.widget_id / img_name
                    shutil.copyfile(bg_image_path, dst_path)
                    self._background_url = f"/static/{self.widget_id}/{img_name}"
                else:
                    raise ValueError(f"Unable to find image at {background_image}")
            else:
                raise ValueError(f"Unsupported background_image type: {type(background_image)}")
        except Exception as e:
            self._background_url = None
        finally:
            DataJson()[self.widget_id]["backgroundUrl"] = self._background_url
            DataJson().send_changes()

    def set_heatmap(self, mask: np.ndarray):
        try:
            heatmap = mask_to_heatmap(
                mask,
                colormap=self._colormap,
                vmin=self._vmin,
                vmax=self._vmax,
                transparent_low=self._transparent_low,
            )
            self._save_to_static(heatmap, name="mask.png")
            self._heatmap_url = f"/static/{self.widget_id}/mask.png"
            self._min_value = mask.min()
            self._max_value = mask.max()
            self._mask_data = mask.tolist()
        except Exception as e:
            self._heatmap_url = None
            self._min_value = None
            self._max_value = None
            self._mask_data = None
        finally:
            DataJson()[self.widget_id]["heatmapUrl"] = self._heatmap_url
            DataJson()[self.widget_id]["minValue"] = self._min_value
            DataJson()[self.widget_id]["maxValue"] = self._max_value
            DataJson()[self.widget_id]["maskData"] = self._mask_data
            DataJson().send_changes()

    def set_heatmap_from_annotations(self, anns: List[Annotation], object_name: str = None):
        if len(anns) == 0:
            raise ValueError("Annotations list should have at least one element")
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

    @property
    def opacity(self):
        return StateJson()[self.widget_id]["opacity"]

    @opacity.setter
    def opacity(self, value: int):
        value = max(0, value)
        value = min(100, value)
        StateJson()[self.widget_id]["opacity"] = value
        StateJson().send_changes()

    @property
    def colormap(self):
        return self._colormap

    @colormap.setter
    def colormap(self, value: int):
        self._colormap = value
        DataJson()[self.widget_id]["legendColors"] = colormap_to_hex_list(self._colormap)

    @property
    def vmin(self):
        return self._vmin

    @vmin.setter
    def vmin(self, value):
        self._vmin = value

    @property
    def vmax(self):
        return self._vmax

    @vmax.setter
    def vmax(self, value):
        self._vmax = value

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

    @property
    def click_x(self):
        return StateJson()[self.widget_id]["maskX"]

    @property
    def click_y(self):
        return StateJson()[self.widget_id]["maskY"]

    @property
    def click_value(self):
        return StateJson()[self.widget_id]["clickedValue"]

    def click(self, func: Callable[[int, int, float], None]) -> Callable[[], None]:
        """
        Registers a callback for heatmap click events.

        The callback receives coordinates in NumPy order (y, x, value),
        where:
            - y: row index (height axis)
            - x: column index (width axis)
            - value: clicked pixel value
        """
        route_path = self.get_route_path(self.Routes.CLICK)
        server = self._sly_app.get_server()
        self._click_handled = True

        @server.post(route_path)
        def _click():
            x = StateJson()[self.widget_id]["maskX"]
            y = StateJson()[self.widget_id]["maskY"]
            clicked_value = StateJson()[self.widget_id]["clickedValue"]
            func(y, x, clicked_value)

        return _click

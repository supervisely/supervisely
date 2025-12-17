from pathlib import Path
from typing import Any, Callable, List, Union
from urllib.parse import urlparse

import cv2
import numpy as np

from supervisely._utils import logger
from supervisely.annotation.annotation import Annotation
from supervisely.app.content import DataJson, StateJson
from supervisely.app.widgets import Widget
from supervisely.app.widgets_context import JinjaWidgets
from supervisely.imaging.image import np_image_to_data_url_backup_rgb, read


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


def to_json_safe(val):
    if val is None:
        return None
    if isinstance(val, (np.integer, int)):
        return int(val)
    if isinstance(val, (np.floating, float)):
        return float(val)
    return str(val)


class Heatmap(Widget):
    """
    Supervisely widget that displays an interactive heatmap overlay on top of a background image.

    :param background_image: Background image to display under the heatmap. Can be a path to an image file or a NumPy array
    :type background_image: Union[str, np.ndarray], optional
    :param heatmap_mask: NumPy array representing the heatmap mask values
    :type heatmap_mask: np.ndarray, optional
    :param vmin: Minimum value for normalizing the heatmap. If None, it is inferred from the mask
    :type vmin: Any, optional
    :param vmax: Maximum value for normalizing the heatmap. If None, it is inferred from the mask
    :type vmax: Any, optional
    :param transparent_low: Whether to make low values in the heatmap transparent
    :type transparent_low: bool, optional
    :param colormap: OpenCV colormap used to colorize the heatmap (e.g., cv2.COLORMAP_JET)
    :type colormap: int, optional
    :param width: Width of the output heatmap in pixels
    :type width: int, optional
    :param height: Height of the output heatmap in pixels
    :type height: int, optional
    :param widget_id: Unique identifier for the widget instance
    :type widget_id: str, optional

    This widget provides an interactive visualization for numerical data as colored overlays.
    Users can click on the heatmap to get exact values at specific coordinates.
    The widget supports various colormaps, transparency controls, and value normalization.

    :Usage example:

     .. code-block:: python

        import numpy as np
        from supervisely.app.widgets import Heatmap

        # Create temperature heatmap
        temp_data = np.random.uniform(-20, 40, size=(100, 100))
        heatmap = Heatmap(
            background_image="/path/to/background.jpg",
            heatmap_mask=temp_data,
            vmin=-20,
            vmax=40,
            colormap=cv2.COLORMAP_JET
        )

        @heatmap.click
        def handle_click(y: int, x: int, value: float):
            print(f"Temperature at ({x}, {y}): {value:.1f}°C")
    """

    class Routes:
        CLICK = "heatmap_clicked_cb"

    def __init__(
        self,
        background_image: Union[str, np.ndarray] = None,
        heatmap_mask: np.ndarray = None,
        vmin: Any = None,
        vmax: Any = None,
        transparent_low: bool = False,
        colormap: int = cv2.COLORMAP_JET,
        width: int = None,
        height: int = None,
        widget_id: str = None,
    ):
        self._background_url = None
        self._heatmap_url = None
        self._mask_data = None  # Store numpy array for efficient value lookup
        self._click_callback = None  # Optional user callback
        self._vmin = vmin
        self._vmax = vmax
        self._transparent_low = transparent_low
        self._colormap = colormap
        self._width = width
        self._height = height
        self._opacity = 70
        self._min_value = 0
        self._max_value = 0
        super().__init__(widget_id, file_path=__file__)

        if background_image is not None:
            self.set_background(background_image)

        if heatmap_mask is not None:
            self.set_heatmap(heatmap_mask)

        script_path = "./sly/css/app/widgets/heatmap/script.js"
        JinjaWidgets().context["__widget_scripts__"][self.__class__.__name__] = script_path

        # Register default click handler to update value from server-side mask
        self._register_click_handler()

    def get_json_data(self):
        # Get mask dimensions if available
        mask_height, mask_width = 0, 0
        if self._mask_data is not None:
            mask_height, mask_width = self._mask_data.shape[:2]

        return {
            "backgroundUrl": self._background_url,
            "heatmapUrl": self._heatmap_url,
            "width": self._width,
            "height": self._height,
            "maskWidth": mask_width,
            "maskHeight": mask_height,
            "minValue": self._min_value,
            "maxValue": self._max_value,
            "legendColors": colormap_to_hex_list(self._colormap),
        }

    def get_json_state(self):
        return {"opacity": self._opacity, "clickedValue": None, "maskX": None, "maskY": None}

    def set_background(self, background_image: Union[str, np.ndarray]):
        """
        Sets the background image that will be displayed under the heatmap overlay.

        :param background_image: Background image source. Can be a file path, URL, or NumPy array
        :type background_image: Union[str, np.ndarray]
        :raises ValueError: If the background image type is unsupported or file path doesn't exist
        :raises Exception: If there's an error during image processing or file operations

        This method handles three types of background images:
            1. **NumPy array**: Converts to PNG and encodes as data URL
            2. **HTTP/HTTPS URL**: Uses the URL directly for remote images
            3. **Local file path**: Reads file and encodes as data URL

        All images are converted to data URLs for efficient in-memory serving.

        :Usage example:

         .. code-block:: python

            from supervisely.app.widgets.heatmap import Heatmap
            import numpy as np
            heatmap = Heatmap()

            # Using a local file path
            heatmap.set_background("/path/to/image.jpg")

            # Using a NumPy array (RGB image)
            bg_array = np.random.randint(0, 255, size=(480, 640, 3), dtype=np.uint8)
            heatmap.set_background(bg_array)

            # Using a remote URL
            heatmap.set_background("https://example.com/background.png")
        """
        try:
            if isinstance(background_image, np.ndarray):
                self._background_url = np_image_to_data_url_backup_rgb(background_image)
            elif isinstance(background_image, str):
                parsed = urlparse(background_image)
                bg_image_path = Path(background_image)
                if parsed.scheme in ("http", "https") and parsed.netloc:
                    self._background_url = background_image
                elif parsed.scheme == "data":
                    self._background_url = background_image
                elif bg_image_path.exists() and bg_image_path.is_file():
                    np_image = read(bg_image_path, remove_alpha_channel=False)
                    self._background_url = np_image_to_data_url_backup_rgb(np_image)
                else:
                    raise ValueError(f"Unable to find image at {background_image}")
            else:
                raise ValueError(f"Unsupported background_image type: {type(background_image)}")
        except Exception as e:
            logger.error(f"Error setting background: {e}", exc_info=True)
            self._background_url = None
            raise
        finally:
            DataJson()[self.widget_id]["backgroundUrl"] = self._background_url
            DataJson().send_changes()

    def set_heatmap(self, mask: np.ndarray):
        """
        Sets the heatmap mask data and generates a colorized PNG overlay.

        :param mask: NumPy array representing the heatmap values to be displayed
        :type mask: np.ndarray

        :raises Exception: If there's an error during heatmap generation

        The heatmap is converted to a data URL for efficient in-memory serving.

        :Usage example:

         .. code-block:: python

            from supervisely.app.widgets.heatmap import Heatmap
            import numpy as np

            heatmap = Heatmap()

            # Create probability heatmap (0.0 to 1.0)
            probability_mask = np.random.uniform(0.0, 1.0, size=(100, 100))
            heatmap.set_heatmap(probability_mask)

            # Create temperature heatmap (-50 to 150)
            temp_mask = np.random.uniform(-50, 150, size=(200, 300))
            heatmap.set_heatmap(temp_mask)
        """
        try:
            heatmap = mask_to_heatmap(
                mask,
                colormap=self._colormap,
                vmin=self._vmin,
                vmax=self._vmax,
                transparent_low=self._transparent_low,
            )
            self._heatmap_url = np_image_to_data_url_backup_rgb(heatmap)
            self._min_value = to_json_safe(mask.min())
            self._max_value = to_json_safe(mask.max())

            # Store mask as numpy array for efficient server-side value lookup
            self._mask_data = mask.copy()

        except Exception as e:
            logger.error(f"Error setting heatmap: {e}", exc_info=True)
            self._heatmap_url = None
            self._min_value = None
            self._max_value = None
            self._mask_data = None
            raise
        finally:
            DataJson()[self.widget_id]["heatmapUrl"] = self._heatmap_url
            DataJson()[self.widget_id]["minValue"] = self._min_value
            DataJson()[self.widget_id]["maxValue"] = self._max_value

            # Update mask dimensions
            if self._mask_data is not None:
                h, w = self._mask_data.shape[:2]
                DataJson()[self.widget_id]["maskWidth"] = w
                DataJson()[self.widget_id]["maskHeight"] = h
            else:
                DataJson()[self.widget_id]["maskWidth"] = 0
                DataJson()[self.widget_id]["maskHeight"] = 0

            # Don't send maskData - will be fetched on-demand when user clicks
            DataJson().send_changes()

    def set_heatmap_from_annotations(self, anns: List[Annotation], object_name: str = None):
        """
        Creates and sets a heatmap from Supervisely annotations showing object density/overlaps.

        :param anns: List of Supervisely annotations to convert to heatmap
        :type anns: List[Annotation]
        :param object_name: Name of the object class to filter annotations by. If None, all objects are included
        :type object_name: str, optional
        :raises ValueError: If the annotations list is empty

        This method creates a density heatmap mask by:
            1. Using widget dimensions (width/height) if specified, calculating missing dimension from aspect ratio
            2. Creating a zero-filled mask of the target size
            3. Drawing each matching label onto the mask, accumulating values
            4. Areas with overlapping objects will have higher values (brighter in heatmap)
            5. Setting the resulting density mask as the heatmap

        :Usage example:

         .. code-block:: python

            from supervisely.annotation.annotation import Annotation

            ann1 = Annotation.load_json_file("/path/to/ann1.json")
            ann2 = Annotation.load_json_file("/path/to/ann2.json")
            ann3 = Annotation.load_json_file("/path/to/ann3.json")
            annotations = [ann1, ann2, ann3]
            heatmap.set_heatmap_from_annotations(annotations, object_name="person")

        """
        if len(anns) == 0:
            raise ValueError("Annotations list should have at least one element")

        # Use widget dimensions if specified, otherwise calculate average from annotations
        if self._width is not None and self._height is not None:
            # Both dimensions specified - use them directly
            target_size = (self._height, self._width)
        elif self._width is not None or self._height is not None:
            # Only one dimension specified - calculate the other from annotations aspect ratio
            sizes = [ann.img_size for ann in anns]
            avg_height = sum(size[0] for size in sizes) / len(sizes)
            avg_width = sum(size[1] for size in sizes) / len(sizes)
            aspect_ratio = avg_width / avg_height

            if self._width is not None:
                # Width specified, calculate height
                target_height = int(round(self._width / aspect_ratio / 2) * 2)
                target_size = (target_height, self._width)
            else:
                # Height specified, calculate width
                target_width = int(round(self._height * aspect_ratio / 2) * 2)
                target_size = (self._height, target_width)
        else:
            # No dimensions specified - calculate average size from annotations and round to even numbers
            sizes = [ann.img_size for ann in anns]
            target_size = (
                int(round(sum(size[0] for size in sizes) / len(sizes) / 2) * 2),
                int(round(sum(size[1] for size in sizes) / len(sizes) / 2) * 2),
            )

        # Count matching labels to determine max possible value
        total_labels = 0
        for ann in anns:
            for label in ann.labels:
                if object_name is None or label.obj_class.name == object_name:
                    total_labels += 1

        if total_labels == 0:
            raise ValueError(f"No labels found for object_name='{object_name}'")

        # Create density mask that accumulates overlapping objects
        mask = np.zeros(target_size, dtype=np.float32)

        for ann in anns:
            for label in ann.labels:
                if object_name is None or label.obj_class.name == object_name:
                    # Create a resized label for the target mask size
                    resized_label = label.resize(ann.img_size, target_size)

                    # Create temporary mask for this label
                    temp_mask = np.zeros(target_size, dtype=np.float32)
                    resized_label.draw(temp_mask, color=1.0)

                    # Add to accumulating density mask (overlaps will sum up)
                    mask += temp_mask

        logger.info(
            f"Created density heatmap: {total_labels} labels, "
            f"target size: {target_size}, "
            f"max density: {mask.max():.1f}, "
            f"avg density: {mask.mean():.3f}"
        )

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

    def _register_click_handler(self):
        """Register internal click handler to update value from server-side mask."""
        route_path = self.get_route_path(self.Routes.CLICK)
        server = self._sly_app.get_server()

        @server.post(route_path)
        def _click():
            x = StateJson()[self.widget_id]["maskX"]
            y = StateJson()[self.widget_id]["maskY"]

            logger.debug(
                f"Heatmap click: x={x}, y={y}, _mask_data shape={self._mask_data.shape if self._mask_data is not None else None}"
            )

            # Get value from server-side mask data
            clicked_value = None
            if self._mask_data is not None and x is not None and y is not None:
                h, w = self._mask_data.shape[:2]
                if 0 <= y < h and 0 <= x < w:
                    clicked_value = float(self._mask_data[y, x])
                    # Update state with the value
                    StateJson()[self.widget_id]["clickedValue"] = clicked_value
                    StateJson().send_changes()
                    logger.debug(f"Heatmap click value: {clicked_value}")
                else:
                    logger.warning(f"Coordinates out of bounds: x={x}, y={y}, shape=({h}, {w})")
            else:
                if self._mask_data is None:
                    logger.warning("Mask data is None")
                if x is None:
                    logger.warning("x coordinate is None")
                if y is None:
                    logger.warning("y coordinate is None")

            # Call user callback if registered
            if self._click_callback is not None:
                self._click_callback(y, x, clicked_value)

    def click(self, func: Callable[[int, int, float], None]) -> Callable[[], None]:
        """
        Registers a callback for heatmap click events.

        :param func: Callback function that receives click coordinates and value
        :type func: Callable[[int, int, float], None]
        :returns: The registered callback function
        :rtype: Callable[[], None]

        The callback receives coordinates in NumPy order (y, x, value), where:
            - y: row index (height axis)
            - x: column index (width axis)
            - value: clicked pixel value (fetched from server-side mask)

        :Usage example:

         .. code-block:: python

            @heatmap.click
            def handle_click(y: int, x: int, value: float):
                print(f"Clicked at row {y}, col {x}, value: {value}")

        """
        self._click_callback = func
        return func

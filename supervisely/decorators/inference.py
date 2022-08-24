import os
import numpy
import functools
from supervisely.sly_logger import logger
from supervisely.io.fs import silent_remove
from supervisely.geometry.bitmap import Bitmap
from supervisely.imaging import image as sly_image
from supervisely.geometry.rectangle import Rectangle
from supervisely._utils import rand_str as sly_rand_str
from supervisely.annotation.annotation import Annotation
from supervisely.geometry.point_location import PointLocation
from supervisely.geometry.sliding_windows_fuzzy import (
    SlidingWindowsFuzzy,
    SlidingWindowBorderStrategy,
)


def process_image_roi(func):
    """
    Decorator for processing annotation labels before and after inference.

    Crops input image before inference if kwargs['state']['rectangle_crop'] provided
    and then scales annotation back to original image size.

    Keyword arguments:

    :param image_np: Image in numpy.ndarray format (use image_path or image_np, not both)
    :type image_np: numpy.ndarray
    :param image_path: Path to image (use image_path or image_np, not both)
    :type image_path: str
    :param project_meta: ProjectMeta of the current project
    :type project_meta: ProjectMeta
    :param state: Application state
    :type state: dict
    :raises: :class:`ValueError`, if image_np or image_path invalid or not provided
    :return: Annotation in json format
    :rtype: :class:`dict`
    """

    def process_image_path(image_path, sly_rectangle):
        image = sly_image.read(image_path)
        image_size = image.shape[:2]
        image_base_dir = os.path.dirname(image_path)
        image_name, image_ext = os.path.splitext(os.path.basename(image_path))

        image_crop = sly_image.crop(image, sly_rectangle)
        image_crop_path = os.path.join(
            image_base_dir, sly_rand_str(10) + "_" + image_name + "_crop" + image_ext
        )
        sly_image.write(image_crop_path, image_crop)
        return image_crop_path, image_size

    def process_image_np(image_np, sly_rectangle):
        image_crop = sly_image.crop(image_np, sly_rectangle)
        image_size, image_crop_size = image_np.shape[:2], image_crop.shape[:2]
        return image_crop, image_size

    def scale_ann_to_original_size(
        ann_json, project_meta, original_size, sly_rectangle
    ):
        ann = Annotation.from_json(ann_json, project_meta)
        updated_labels = []
        for label in ann.labels:
            if type(label.geometry) is Rectangle:
                updated_geometry = Rectangle(
                    top=label.geometry.top + sly_rectangle.top,
                    left=label.geometry.left + sly_rectangle.left,
                    bottom=label.geometry.bottom + sly_rectangle.top,
                    right=label.geometry.right + sly_rectangle.left,
                )

            if type(label.geometry) is Bitmap:
                bitmap_data = label.geometry.data
                bitmap_origin = PointLocation(
                    label.geometry.origin.row + sly_rectangle.top,
                    label.geometry.origin.col + sly_rectangle.left,
                )

                updated_geometry = Bitmap(data=bitmap_data, origin=bitmap_origin)
            updated_labels.append(label.clone(geometry=updated_geometry))

        ann = ann.clone(img_size=original_size, labels=updated_labels)
        ann_json = ann.to_json()
        return ann_json

    @functools.wraps(func)
    def wrapper_inference(*args, **kwargs):
        project_meta = kwargs["project_meta"]
        state = kwargs["state"]
        rectangle_json = state.get("rectangle")

        if rectangle_json is None:
            ann_json = func(*args, **kwargs)
            return ann_json

        rectangle = Rectangle.from_json(rectangle_json)
        if "image_np" in kwargs.keys():
            image_np = kwargs["image_np"]
            if not isinstance(image_np, numpy.ndarray):
                raise ValueError("Invalid input. Image path must be numpy.ndarray")
            image_crop_np, image_size = process_image_np(image_np, rectangle)
            kwargs["image_np"] = image_crop_np
            ann_json = func(*args, **kwargs)
            ann_json = scale_ann_to_original_size(
                ann_json, project_meta, image_size, rectangle
            )
        elif "image_path" in kwargs.keys():
            image_path = kwargs["image_path"]
            if not isinstance(image_path, str):
                raise ValueError("Invalid input. Image path must be str")
            image_crop_path, image_size = process_image_path(image_path, rectangle)
            kwargs["image_path"] = image_crop_path
            ann_json = func(*args, **kwargs)
            ann_json = scale_ann_to_original_size(
                ann_json, project_meta, image_size, rectangle
            )
            silent_remove(image_crop_path)
        else:
            raise ValueError("image_np or image_path not provided!")

        return ann_json

    return wrapper_inference


def process_image_sliding_window(func):
    @functools.wraps(func)
    def wrapper_inference(*args, **kwargs):
        project_meta = kwargs["project_meta"]
        state = kwargs["state"]
        settings = kwargs["settings"]
        inference_mode = settings.get("inference_mode", "full_image")
        if inference_mode != "sliding_window":
            ann_json = func(*args, **kwargs)
            return ann_json

        sliding_window_params = settings["sliding_window_params"]
        image_path = kwargs["image_path"]
        img = sly_image.read(image_path)
        img_h, img_w = img.shape[:2]
        windowHeight = sliding_window_params.get("windowHeight", img_h)
        windowWidth = sliding_window_params.get("windowWidth", img_w)
        overlapY = sliding_window_params.get("overlapY", 0)
        overlapX = sliding_window_params.get("overlapX", 0)
        borderStrategy = sliding_window_params.get("borderStrategy", "shift_window")

        slider = SlidingWindowsFuzzy(
            [windowHeight, windowWidth], [overlapY, overlapX], borderStrategy
        )
        rectangles = []
        for window in slider.get(img.shape[:2]):
            rectangles.append(window)

        ann = Annotation.from_img_path(image_path)
        results = []
        for rect in rectangles:
            state["rectangle"] = rect.to_json()
            ann_json = func(*args, **kwargs)
            slice_ann = Annotation.from_json(ann_json, project_meta)
            ann = ann.add_labels(slice_ann.labels)
            results.append(
                {
                    "rectangle": rect.to_json(),
                    "labels": [l.to_json() for l in slice_ann.labels],
                }
            )
        return {"annotation": ann.to_json(), "data": {"slides": results}}

    return wrapper_inference

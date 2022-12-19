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
from supervisely.annotation.label import Label
from supervisely.annotation.tag import Tag
from supervisely.geometry.point_location import PointLocation
from supervisely.geometry.sliding_windows_fuzzy import (
    SlidingWindowsFuzzy,
    SlidingWindowBorderStrategy,
)


def _process_image_path(image_path, sly_rectangle):
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

def _process_image_np(image_np, sly_rectangle):
    image_crop = sly_image.crop(image_np, sly_rectangle)
    image_size, image_crop_size = image_np.shape[:2], image_crop.shape[:2] # TODO: second param?
    return image_crop, image_size

def _scale_ann_to_original_size(
    ann, original_size, sly_rectangle
):
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
    return ann

def _apply_agnostic_nms(labels, iou_thres=0.5):
    import torch
    import torchvision
    # TODO: where we can get iou_th and conf_th?
    boxes = []
    scores = []
    for label in labels:
        label: Label
        label_rect: Rectangle = label.geometry.to_bbox()
        boxes.append([
            float(label_rect.left), 
            float(label_rect.top), 
            float(label_rect.right), 
            float(label_rect.bottom)
        ])
        conf_score: Tag = label.tags.get("confidence", None)
        if conf_score is None:
            raise ValueError("Label don't have confidence score tag named 'confidence'.")
        scores.append(conf_score.value)
    boxes = torch.tensor(boxes)
    scores = torch.tensor(scores)
    saved_inds = torchvision.ops.nms(boxes, scores, iou_thres)
    saved_labels = []
    for ind in saved_inds:
        saved_labels.append(labels[ind])
    return saved_labels


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
    :raises: :class:`ValueError`, if image_np or image_path invalid or not provided
    :return: Annotation in json format
    :rtype: :class:`dict`
    """
    @functools.wraps(func)
    def wrapper_inference(*args, **kwargs):
        settings = kwargs["settings"]
        rectangle_json = settings.get("rectangle")

        if rectangle_json is None:
            ann = func(*args, **kwargs)
            return ann

        rectangle = Rectangle.from_json(rectangle_json)
        if "image_np" in kwargs.keys():
            image_np = kwargs["image_np"]
            if not isinstance(image_np, numpy.ndarray):
                raise ValueError("Invalid input. Image path must be numpy.ndarray")
            image_crop_np, image_size = _process_image_np(image_np, rectangle)
            kwargs["image_np"] = image_crop_np
            ann = func(*args, **kwargs)
            ann = _scale_ann_to_original_size(
                ann, image_size, rectangle
            )
        elif "image_path" in kwargs.keys():
            image_path = kwargs["image_path"]
            if not isinstance(image_path, str):
                raise ValueError("Invalid input. Image path must be str")
            image_crop_path, image_size = _process_image_path(image_path, rectangle)
            kwargs["image_path"] = image_crop_path
            ann = func(*args, **kwargs)
            ann = _scale_ann_to_original_size(
                ann, image_size, rectangle
            )
            silent_remove(image_crop_path)
        else:
            raise ValueError("image_np or image_path not provided!")

        return ann

    return wrapper_inference


def process_image_sliding_window(func):
    @functools.wraps(func)
    def wrapper_inference(*args, **kwargs):
        settings = kwargs["settings"]
        data_to_return = kwargs["data_to_return"]
        inference_mode = settings.get("inference_mode", "full_image")
        if inference_mode != "sliding_window":
            ann = func(*args, **kwargs)
            return ann
        sliding_window_mode = settings.get("sliding_window_mode", "basic")
        if sliding_window_mode == "none":
            ann = func(*args, **kwargs)
            return ann

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

        data_to_return["slides"] = []
        all_labels = []
        for rect in rectangles:
            image_crop_path, image_size = _process_image_path(image_path, rect)
            kwargs["image_path"] = image_crop_path
            slice_ann: Annotation = func(*args, **kwargs)
            slice_ann = _scale_ann_to_original_size(
                slice_ann, image_size, rect
            )
            data_to_return["slides"].append(
                {
                    "rectangle": rect.to_json(),
                    "labels": [l.to_json() for l in slice_ann.labels],
                }
            )
            all_labels.extend(slice_ann.labels)

        all_json_labels = []
        for slide in data_to_return["slides"]:
            all_json_labels.extend(slide["labels"])

        full_rect = Rectangle(0, 0, img_h, img_w)
        all_labels_slide = {
            "rectangle": full_rect.to_json(),
            "labels": all_json_labels
        }
        data_to_return["slides"].append(all_labels_slide) # for visualization
        ann = Annotation.from_img_path(image_path)

        if sliding_window_mode == "advanced":
            labels_after_nms = _apply_agnostic_nms(all_labels)
            ann = ann.add_labels(labels_after_nms)
            all_labels_after_nms_slide = {
                "rectangle": full_rect.to_json(),
                "labels": [l.to_json() for l in labels_after_nms]
            }
            data_to_return["slides"].append(all_labels_after_nms_slide)
        else:
            ann = ann.add_labels(all_labels)
            data_to_return["slides"].append(all_labels_slide)
        return ann

    return wrapper_inference

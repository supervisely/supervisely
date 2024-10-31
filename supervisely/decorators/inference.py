import functools
import os
from typing import List, Optional, Tuple

import numpy as np

from supervisely._utils import rand_str as sly_rand_str
from supervisely.annotation.annotation import Annotation
from supervisely.annotation.label import Label
from supervisely.annotation.tag import Tag
from supervisely.geometry.bitmap import Bitmap
from supervisely.geometry.graph import GraphNodes, Node
from supervisely.geometry.point_location import PointLocation
from supervisely.geometry.rectangle import Rectangle
from supervisely.geometry.sliding_windows_fuzzy import (
    SlidingWindowBorderStrategy,
    SlidingWindowsFuzzy,
)
from supervisely.imaging import image as sly_image
from supervisely.io.fs import silent_remove
from supervisely.sly_logger import logger


def _process_image_path(image_path: str, rect: Rectangle) -> Tuple[str, Tuple[int, int]]:
    image = sly_image.read(image_path)
    image_size = image.shape[:2]
    image_base_dir = os.path.dirname(image_path)
    image_name, image_ext = os.path.splitext(os.path.basename(image_path))

    image_crop = sly_image.crop(image, rect)
    image_crop_path = os.path.join(
        image_base_dir, sly_rand_str(10) + "_" + image_name + "_crop" + image_ext
    )
    sly_image.write(image_crop_path, image_crop)
    return image_crop_path, image_size


def _scale_ann_to_original_size(
    ann: Annotation, original_size: Tuple[int, int], rect: Rectangle
) -> Annotation:
    updated_labels = []
    for label in ann.labels:
        updated_geometry = label.geometry
        if type(label.geometry) is Rectangle:
            updated_geometry = Rectangle(
                top=label.geometry.top + rect.top,
                left=label.geometry.left + rect.left,
                bottom=label.geometry.bottom + rect.top,
                right=label.geometry.right + rect.left,
            )

        if type(label.geometry) is Bitmap:
            bitmap_data = label.geometry.data
            bitmap_origin = PointLocation(
                label.geometry.origin.row + rect.top,
                label.geometry.origin.col + rect.left,
            )

            updated_geometry = Bitmap(data=bitmap_data, origin=bitmap_origin)

        if type(label.geometry) is GraphNodes:
            new_nodes = []
            for id, node in label.geometry.nodes.items():
                new_nodes.append(
                    Node(
                        label=id,
                        row=node.location.row + rect.top,
                        col=node.location.col + rect.left,
                    )
                )

            updated_geometry = GraphNodes(new_nodes)

        updated_labels.append(label.clone(geometry=updated_geometry))

    ann = ann.clone(img_size=original_size, labels=updated_labels)
    return ann


def _apply_agnostic_nms(labels: List[Label], iou_thres: Optional[float] = 0.5) -> List[Label]:
    # pylint: disable=import-error
    import torch

    # pylint: disable=import-error
    import torchvision

    # TODO: where we can get iou_th and conf_th?
    boxes = []
    scores = []
    for label in labels:
        label: Label
        label_rect: Rectangle = label.geometry.to_bbox()
        boxes.append(
            [
                float(label_rect.left),
                float(label_rect.top),
                float(label_rect.right),
                float(label_rect.bottom),
            ]
        )
        conf_score: Tag = label.tags.get("confidence", None)
        if conf_score is None:
            raise ValueError("Label don't have confidence score tag named 'confidence'.")
        scores.append(conf_score.value)
    if len(boxes) > 0 and len(scores) > 0:
        boxes = torch.tensor(boxes)
        scores = torch.tensor(scores)
        saved_inds = torchvision.ops.nms(boxes, scores, iou_thres)
    else:
        saved_inds = []
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
            if not isinstance(image_np, np.ndarray):
                raise ValueError("Invalid input. Image path must be numpy.ndarray")
            original_image_size = image_np.shape[:2]
            image_crop_np = sly_image.crop(image_np, rectangle)
            kwargs["image_np"] = image_crop_np
            ann = func(*args, **kwargs)
            ann = _scale_ann_to_original_size(ann, original_image_size, rectangle)
        elif "image_path" in kwargs.keys():
            image_path = kwargs["image_path"]
            if not isinstance(image_path, str):
                raise ValueError("Invalid input. Image path must be str")
            image_crop_path, original_image_size = _process_image_path(image_path, rectangle)
            kwargs["image_path"] = image_crop_path
            ann = func(*args, **kwargs)
            ann = _scale_ann_to_original_size(ann, original_image_size, rectangle)
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

        assert isinstance(data_to_return, dict)
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
            image_crop_path, original_image_size = _process_image_path(image_path, rect)
            kwargs["image_path"] = image_crop_path
            slice_ann: Annotation = func(*args, **kwargs)
            slice_ann = _scale_ann_to_original_size(slice_ann, original_image_size, rect)
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
        all_labels_slide = {"rectangle": full_rect.to_json(), "labels": all_json_labels}
        data_to_return["slides"].append(all_labels_slide)  # for visualization
        ann = Annotation.from_img_path(image_path)

        if sliding_window_mode == "advanced":
            labels_after_nms = _apply_agnostic_nms(all_labels)
            ann = ann.add_labels(labels_after_nms)
            all_labels_after_nms_slide = {
                "rectangle": full_rect.to_json(),
                "labels": [l.to_json() for l in labels_after_nms],
            }
            data_to_return["slides"].append(all_labels_after_nms_slide)
        else:
            ann = ann.add_labels(all_labels)
            data_to_return["slides"].append(all_labels_slide)
        return ann

    return wrapper_inference


def process_images_batch_roi(func):
    @functools.wraps(func)
    def wrapper_inference(*args, **kwargs):
        """
        Process images batch with ROI cropping before inference and scaling back to original size after inference.
        Pass settings with rectangle or rectangles to crop images and annotations.
        If rectangle provided, crop all images with the same rectangle.
        If rectangles provided, crop each image with corresponding rectangle.
        """
        source = kwargs["source"]
        settings = kwargs["settings"]
        if "rectangles" in settings:
            rectangles = [Rectangle.from_json(rect_json) for rect_json in settings["rectangles"]]
        elif "rectangle" in settings:
            rectangles = [Rectangle.from_json(settings["rectangle"]) for _ in source]
        else:
            return func(*args, **kwargs)

        original_images_sizes = [image_np.shape[:2] for image_np in source]
        images_crops_nps = [
            sly_image.crop(image_np, rect) for image_np, rect in zip(source, rectangles)
        ]
        kwargs["source"] = images_crops_nps
        anns = func(*args, **kwargs)
        anns = [
            _scale_ann_to_original_size(ann, original_image_size, rect)
            for ann, original_image_size, rect in zip(anns, original_images_sizes, rectangles)
        ]
        return anns

    return wrapper_inference


def process_images_batch_sliding_window(func):
    @functools.wraps(func)
    def wrapper_inference(*args, **kwargs):
        settings = kwargs["settings"]
        data_to_return = kwargs["data_to_return"]
        assert isinstance(data_to_return, list)
        inference_mode = settings.get("inference_mode", "full_image")
        sliding_window_mode = settings.get("sliding_window_mode", "basic")
        if inference_mode != "sliding_window" or sliding_window_mode == "none":
            anns = func(*args, **kwargs)
            for i in range(len(anns)):
                data_to_return.append({})
            return anns

        sliding_window_params = settings["sliding_window_params"]
        source: List[np.ndarray] = kwargs["source"]
        result_anns = []
        for img in source:
            if isinstance(img, str):
                img = sly_image.read(img)
            img_h, img_w = img.shape[:2]
            original_image_size = (img_h, img_w)
            windowHeight = sliding_window_params.get("windowHeight", img_h)
            windowWidth = sliding_window_params.get("windowWidth", img_w)
            overlapY = sliding_window_params.get("overlapY", 0)
            overlapX = sliding_window_params.get("overlapX", 0)
            borderStrategy = sliding_window_params.get("borderStrategy", "shift_window")

            slider = SlidingWindowsFuzzy(
                [windowHeight, windowWidth], [overlapY, overlapX], borderStrategy
            )
            
            rects = []
            crops = []
            for rect in slider.get(img.shape[:2]):
                rects.append(rect)
                crops.append(sly_image.crop(img, rect))

            # Inference
            kwargs["source"] = crops
            slice_anns: List[Annotation] = func(*args, **kwargs)

            all_labels = []
            all_json_labels = []
            slides = []
            for rect, slice_ann in zip(rects, slice_anns):
                slice_ann = _scale_ann_to_original_size(slice_ann, original_image_size, rect)
                json_labels = [l.to_json() for l in slice_ann.labels]
                slides.append({
                    "rectangle": rect.to_json(),
                    "labels": json_labels,
                })
                all_labels += slice_ann.labels
                all_json_labels += json_labels
            
            # Add full image slide
            full_rect = Rectangle(0, 0, img_h, img_w)
            all_labels_slide = {"rectangle": full_rect.to_json(), "labels": all_json_labels}
            slides.append(all_labels_slide)  # for visualization

            # Apply NMS
            ann = Annotation((img_h, img_w))
            if sliding_window_mode == "advanced":
                labels_after_nms = _apply_agnostic_nms(all_labels)
                ann = ann.add_labels(labels_after_nms)
                slides.append({
                    "rectangle": full_rect.to_json(),
                    "labels": [l.to_json() for l in labels_after_nms],
                })
            else:
                ann = ann.add_labels(all_labels)
                slides.append(all_labels_slide)

            result_anns.append(ann)
            data_to_return.append({"slides": slides})

        return result_anns

    return wrapper_inference

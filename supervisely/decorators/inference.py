import os
import numpy
import functools
import supervisely_lib as sly


def process_image_path(image_path, state):
    image = sly.image.read(image_path)
    image_base_dir = os.path.dirname(image_path)
    image_name, image_ext = os.path.splitext(os.path.basename(image_path))

    selected_figure_bbox = state["rectangle_crop"]
    sly_rect = sly.Rectangle.from_json(selected_figure_bbox)

    image_crop = sly.image.crop(image, sly_rect)
    image_crop_path = os.path.join(image_base_dir, sly.rand_str(10) + "_" + image_name + "_crop" + image_ext)
    sly.image.write(image_crop_path, image_crop)
    return image_crop_path


def process_image_np(image_np, state):
    selected_figure_bbox = state["rectangle_crop"]
    sly_rect = sly.Rectangle.from_json(selected_figure_bbox)
    image_crop = sly.image.crop(image_np, sly_rect)
    return image_crop


def crop_input_before_inference_and_scale_back_to_original_size(func):
    """Crops input image before inference if kwargs['state']['rectangle_crop'] provided and then scales annotation back to original image size. Image must be path or numpy array."""
    @functools.wraps(func)
    def wrapper_inference(*args, **kwargs):
        project_meta = kwargs["project_meta"]
        state = kwargs["state"]
        app_logger = kwargs["app_logger"]

        if "rectangle_crop" not in state.keys():
            ann_json = func(*args, **kwargs)
            return ann_json

        if "image_path" in kwargs.keys():
            image_path = kwargs["image_path"]
            if not isinstance(image_path, str):
                app_logger.warn("Invalid input. Image path must be str")
            image_path = process_image_path(image_path, state)
            sly.fs.silent_remove(image_path)

        if "image_np" in kwargs.keys():
            image_np = kwargs["image_np"]
            if not isinstance(image_np, numpy.ndarray):
                app_logger.warn("Invalid input. Image must be numpy.ndarray")
            image_np = process_image_np(image_np, state)

        ann_json = func(*args, **kwargs)
        ann = sly.Annotation.from_json(ann_json, project_meta)

        return ann_json
    return wrapper_inference

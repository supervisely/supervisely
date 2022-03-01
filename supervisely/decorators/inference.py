import os
import numpy
import functools
import supervisely as sly


def crop_input_before_inference_and_scale_ann(func):
    """Decorator for processing annotation labels before and after inference.

    Crops input image before inference if kwargs['state']['rectangle_crop'] provided
    and then scales annotation back to original image size.
    Image must be path or numpy array.
    """

    def process_image_path(image_path, sly_rectangle):
        image = sly.image.read(image_path)
        image_size = image.shape[:2]
        image_base_dir = os.path.dirname(image_path)
        image_name, image_ext = os.path.splitext(os.path.basename(image_path))

        image_crop = sly.image.crop(image, sly_rectangle)
        image_crop_path = os.path.join(image_base_dir, sly.rand_str(10) + "_" + image_name + "_crop" + image_ext)
        sly.image.write(image_crop_path, image_crop)
        return image_crop_path, image_size

    def process_image_np(image_np, sly_rectangle):
        image_crop = sly.image.crop(image_np, sly_rectangle)
        image_size, image_crop_size = image_np.shape[:2], image_crop.shape[:2]
        return image_crop, image_size

    def scale_ann_to_original_size(ann_json, project_meta, original_size, sly_rectangle):
        ann = sly.Annotation.from_json(ann_json, project_meta)
        updated_labels = []
        for label in ann.labels:
            if type(label.geometry) is sly.Rectangle:
                updated_geometry = sly.Rectangle(
                    top=label.geometry.top + sly_rectangle.top,
                    left=label.geometry.left + sly_rectangle.left,
                    bottom=label.geometry.bottom + sly_rectangle.top,
                    right=label.geometry.right + sly_rectangle.left)

            if type(label.geometry) is sly.Bitmap:
                bitmap_data = label.geometry.data
                bitmap_origin = sly.PointLocation(label.geometry.origin.col + sly_rectangle.top,
                                                  label.geometry.origin.row + sly_rectangle.left)

                updated_geometry = sly.Bitmap(data=bitmap_data, origin=bitmap_origin)
            updated_labels.append(label.clone(geometry=updated_geometry))

        ann = ann.clone(img_size=original_size, labels=updated_labels)
        ann_json = ann.to_json()
        return ann_json

    @functools.wraps(func)
    def wrapper_inference(*args, **kwargs):
        project_meta = kwargs["project_meta"]
        state = kwargs["state"]
        rectangle_crop = state.get("rectangle_crop")

        if rectangle_crop is None:
            ann_json = func(*args, **kwargs)
            return ann_json

        rectangle = sly.Rectangle.from_json(rectangle_crop)
        if "image_np" in kwargs.keys():
            image_np = kwargs["image_np"]
            if not isinstance(image_np, numpy.ndarray):
                supervisely.logger.warn("Invalid input. Image must be numpy.ndarray")
            image_crop_np, image_size = process_image_np(image_np, rectangle)
            kwargs["image_np"] = image_crop_np
            ann_json = func(*args, **kwargs)
            ann_json = scale_ann_to_original_size(ann_json, project_meta, image_size, rectangle)

        elif "image_path" in kwargs.keys():
            image_path = kwargs["image_path"]
            if not isinstance(image_path, str):
                supervisely.logger.warn("Invalid input. Image path must be str")
            image_crop_path, image_size = process_image_path(image_path, rectangle)
            kwargs["image_path"] = image_crop_path
            image_path = image_crop_path
            ann_json = func(*args, **kwargs)
            ann_json = scale_ann_to_original_size(ann_json, project_meta, image_size, rectangle)
            sly.fs.silent_remove(image_path)
        return ann_json
    return wrapper_inference

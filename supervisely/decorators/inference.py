import os
import supervisely as sly


def crop_input_before_inference(func):
    """Crops input image before inference and then scales annotation back to original image size"""
    def wrapper_inference(image_path, context, state, app_logger):
        assert isinstance(image_path, str)
        image = sly.image.read(image_path)
        if "rectangle_crop" not in state.keys() or state["rectangle_crop"] is None:
            ann_json = func(image_path, context, state, app_logger)
            return ann_json

        image_base_dir = os.path.dirname(image_path)
        image_name, image_ext = os.path.splitext(os.path.basename(image_path))

        selected_figure_bbox = state["rectangle_crop"]
        sly_rect = sly.Rectangle.from_json(selected_figure_bbox)

        image_crop = sly.image.crop(image, sly_rect)
        image_crop_path = os.path.join(image_base_dir, sly.rand_str(10) + "_" + image_name + "_crop" + image_ext)
        sly.image.write(image_crop_path, image_crop)

        ann_json = func(image_crop_path, context, state, app_logger)
        
        original_height, original_width = image.shape[:2]
        ann_json["size"]["height"], ann_json["size"]["width"] = original_height, original_width
        for object in ann_json["objects"]:
            object_ext_points = object["points"]["exterior"]
            object_ext_points[0][0] += sly_rect.left
            object_ext_points[0][1] += sly_rect.top
            object_ext_points[1][0] += sly_rect.left
            object_ext_points[1][1] += sly_rect.top

        sly.fs.silent_remove(image_crop_path)
        return ann_json
    return wrapper_inference

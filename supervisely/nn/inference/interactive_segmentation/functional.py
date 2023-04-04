import os
from copy import deepcopy
import numpy as np

import supervisely as sly
from supervisely.io.fs import silent_remove


def download_volume_slice_as_np(
    volume_id: int,
    slice_index: int,
    normal: dict,
    window_center: float,
    window_width: int,
    api: sly.Api,
) -> np.ndarray:
    # x - 'sagittal', y - 'coronal', z - 'axial'
    if normal["x"] == 1:
        plane = sly.Plane.SAGITTAL
    elif normal["y"] == 1:
        plane = sly.Plane.CORONAL
    elif normal["z"] == 1:
        plane = sly.Plane.AXIAL
    img = api.volume.download_slice_np(
        volume_id, slice_index, plane, window_center, window_width
    )
    return img


def get_image_by_hash(hash, save_path, api: sly.Api):
    api.image.download_paths_by_hashes([hash], [save_path])
    base_image = sly.image.read(save_path)
    silent_remove(save_path)
    return base_image


def download_image_from_context(context: dict, api: sly.Api, output_dir: str):
    if "image_id" in context:
        return api.image.download_np(context["image_id"])
    elif "image_hash" in context:
        img_path = os.path.join(output_dir, "base_image.png")
        return get_image_by_hash(context["image_hash"], img_path, api=api)
    elif "volume" in context:
        volume_id = context["volume"]["volume_id"]
        slice_index = context["volume"]["slice_index"]
        normal = context["volume"]["normal"]
        window_center = context["volume"]["window_center"]
        window_width = context["volume"]["window_width"]
        return download_volume_slice_as_np(
            volume_id=volume_id,
            slice_index=slice_index,
            normal=normal,
            window_center=window_center,
            window_width=window_width,
            api=api,
        )
    elif "video" in context:
        return api.video.frame.download_np(
            context["video"]["video_id"], context["video"]["frame_index"]
        )
    else:
        raise Exception("Project type is not supported")


def crop_image(crop, image_np):
    x1, y1 = crop[0]["x"], crop[0]["y"]
    x2, y2 = crop[1]["x"], crop[1]["y"]
    bbox = sly.Rectangle(y1, x1, y2, x2)
    img_crop = sly.image.crop(image_np, bbox)
    return img_crop


def transform_clicks_to_crop(crop, clicks: list):
    clicks = deepcopy(clicks)
    for click in clicks:
        click["x"] -= crop[0]["x"]
        click["y"] -= crop[0]["y"]
    return clicks


def validate_click_bounds(crop, clicks: list):
    x_max = crop[1]["x"] - crop[0]["x"]  # width
    y_max = crop[1]["y"] - crop[0]["y"]  # height
    for click in clicks:
        is_in_bbox = (
            click["x"] >= 0
            and click["y"] >= 0
            and click["x"] <= x_max
            and click["y"] <= y_max
        )
        if not is_in_bbox:
            return False
    return True


def format_bitmap(bitmap: sly.Bitmap, crop):
    bitmap_json = bitmap.to_json()["bitmap"]
    bitmap_origin = bitmap_json["origin"]
    bitmap_origin = {
        "x": crop[0]["x"] + bitmap_origin[0],
        "y": crop[0]["y"] + bitmap_origin[1],
    }
    bitmap_data = bitmap_json["data"]
    return bitmap_origin, bitmap_data

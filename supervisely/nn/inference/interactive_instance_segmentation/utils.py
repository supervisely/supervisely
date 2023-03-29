import functools
import os
import numpy as np

import supervisely as sly
from supervisely.io.fs import silent_remove


def download_volume_slice_as_np(
    volume_id: int,
    slice_index: int,
    normal: dict,
    window_center: float,
    window_width: int,
    api: sly.Api
) -> np.ndarray:
    data = {
        "volumeId": volume_id,
        "sliceIndex": slice_index,
        "normal": {"x": normal["x"], "y": normal["y"], "z": normal["z"]},
        "windowCenter": window_center,
        "windowWidth": window_width,
    }

    image_bytes = api.post(
        method="volumes.slices.images.download", data=data, stream=True
    ).content

    return sly.image.read_bytes(image_bytes)


@functools.lru_cache(maxsize=100)
def get_image_by_hash(hash, save_path, api: sly.Api):
    api.image.download_paths_by_hashes([hash], [save_path])
    base_image = sly.image.read(save_path)
    silent_remove(save_path)
    return base_image


@functools.lru_cache(maxsize=100)
def get_image_by_id(image_id, api: sly.Api):
    return api.image.download_np(image_id)


def download_image_from_context(context: dict, api: sly.Api, output_dir: str):
    if "image_id" in context:
        return api.image.download_np(context["image_id"])
    elif "image_hash" in context:
        img_path = os.path.join(output_dir, "base_image.png")
        return get_image_by_hash(context["image_hash"], img_path)
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


def transform_clicks_to_crop(crop, clicks: dict):
    clicks = clicks.copy()
    for click in clicks:
        click["x"] -= crop[0]["x"]
        click["y"] -= crop[0]["y"]
        assert click["x"] >= 0 and click["y"] >= 0, "Invalid click coords: below zero"
    return clicks


def get_hash(d: dict):
    hash = ''.join([str(v) for v in d.values()])
    return hash


def get_new_clicks(current_clicks, incoming_clicks):
    current_clicks_hashed = {get_hash(click) : click for click in current_clicks}
    incoming_clicks_hashed = {get_hash(click) : click for click in incoming_clicks}
    base_diff = set(current_clicks_hashed) - set(incoming_clicks_hashed)
    new_diff = set(incoming_clicks_hashed) - set(current_clicks_hashed)
    if len(new_diff) == 1 and len(base_diff) == 0:
        # exactly 1 new click added
        new_click_hash = next(iter(new_diff))
        new_click = incoming_clicks_hashed[new_click_hash]
        return [new_click]
    else:
        return None


def format_bitmap(bitmap: sly.Bitmap, crop: dict):
    bitmap_json = bitmap.to_json()["bitmap"]
    bitmap_origin = bitmap_json["origin"]
    bitmap_origin = {"x": crop[0]["x"] + bitmap_origin[0], "y": crop[0]["y"] + bitmap_origin[1]}
    bitmap_data = bitmap_json["data"]
    return bitmap_origin, bitmap_data

import mimetypes
from pathlib import Path

import magic
import numpy as np
from PIL import Image
from typing import Union, List

from supervisely import Rectangle, Label, logger
from supervisely.imaging.image import read, write
from supervisely.io.fs import (
    get_file_ext,
    get_file_name,
    get_file_name_with_ext,
    silent_remove,
)

Image.MAX_IMAGE_PIXELS = None
EXT_TO_CONVERT = [".heic", ".avif", ".heif", ".jfif"]


def validate_image(path: str) -> tuple:
    """Validate image by mimetypes, ext, and remove alpha channel."""

    try:
        ext = get_file_ext(path)
        if ext.lower() == ".nrrd":
            return path
        if ext.lower() in EXT_TO_CONVERT:
            path = convert_to_jpg(path)
            return path
        if ext.lower() != ".mpo":
            name = get_file_name_with_ext(path)
            new_name = validate_mimetypes(name, path)
            if new_name != name:
                img = read(path, remove_alpha_channel=False)
                silent_remove(path)
                path = Path(path).with_name(new_name).as_posix()
                write(path, img, remove_alpha_channel=False)
        return path
    except Exception as e:
        logger.warning(f"Skip image: {repr(e)}", extra={"file_path": path})


def validate_mimetypes(name: str, path: str) -> list:
    """Validate mimetypes for images."""

    mimetypes.add_type("image/webp", ".webp")  # to extend types_map
    mimetypes.add_type("image/jpeg", ".jfif")  # to extend types_map

    with open(path, "rb") as f:
        mimetype = magic.from_buffer(f.read(), mime=True)
    file_ext = get_file_ext(path).lower()
    if file_ext in mimetypes.guess_all_extensions(mimetype):
        return name

    new_img_ext = mimetypes.guess_extension(mimetype)
    if new_img_ext == ".bin" or new_img_ext is None:
        new_img_ext = ".jpeg"
    new_img_name = f"{get_file_name(name)}{new_img_ext}"
    logger.info(f"Image {name} with mimetype {mimetype} will be converted to {new_img_ext}")

    return new_img_name


def convert_to_jpg(path) -> tuple:
    """Convert image to jpg."""

    # * do not remove folllowing imports, it is used to register avif/heic formats
    import pillow_avif  # pylint: disable=import-error
    from pillow_heif import register_heif_opener  # pylint: disable=import-error

    register_heif_opener()

    new_path = Path(path).with_suffix(".jpeg").as_posix()
    with Image.open(path) as image:
        image.convert("RGB").save(new_path)
    silent_remove(path)
    return new_path


def read_tiff_image(path: str) -> Union[np.ndarray, None]:
    """
    Read tiff image.
    Method will transpose image if it has shape (C, H, W) to (H, W, C).
    """

    import tifffile

    logger.debug(f"Found tiff file: {path}.")
    image = tifffile.imread(path)
    name = get_file_name_with_ext(path)
    if image is not None:
        tiff_shape = image.shape
        if image.ndim == 3:
            if tiff_shape[0] < tiff_shape[1] and tiff_shape[0] < tiff_shape[2]:
                image = image.transpose(1, 2, 0)
                logger.warning(f"{name}: transposed shape from {tiff_shape} to {image.shape}")

    return image


def validate_image_bounds(labels: List[Label], img_rect: Rectangle) -> List[Label]:
    """
    Check if labels are localed inside the image canvas, print a warning and skip them if not.
    """
    new_labels = [label for label in labels if img_rect.contains(label.geometry.to_bbox())]
    if new_labels != labels:
        logger.warning(
            f"{len(labels) - len(new_labels)} annotation objects are out of image bounds. Skipping..."
        )
    return new_labels

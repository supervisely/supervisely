import mimetypes
import os
import re
from pathlib import Path
from typing import List, Union

import magic
import numpy as np
from PIL import Image, ImageFile

from supervisely import Label, Rectangle, logger
from supervisely.geometry.oriented_bbox import OrientedBBox
from supervisely.imaging.image import read, write
from supervisely.io.fs import (
    get_file_ext,
    get_file_name,
    get_file_name_with_ext,
    silent_remove,
)

Image.MAX_IMAGE_PIXELS = None
EXT_TO_CONVERT = [".heic", ".avif", ".heif"]
_HEIF_EXTS = [".heic", ".heif"]


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
                new_path = _free_path(path, get_file_ext(new_name))
                try:
                    write(new_path, img, remove_alpha_channel=False)
                except Exception:
                    silent_remove(new_path)  # keep the original, leave no partial file
                    raise
                silent_remove(path)
                path = new_path
            elif ext.lower() == ".jfif":
                # JFIF is JPEG: keep the bytes, only give it the extension the platform stores
                new_path = _free_path(path, ".jpeg")
                try:
                    os.rename(path, new_path)
                    path = new_path
                except OSError as e:  # e.g. a read-only folder: upload it as .jfif
                    logger.debug(f"Could not rename {path} to .jpeg: {repr(e)}")
        return path
    except Exception as e:
        logger.warning(f"Skip image: {repr(e)}", extra={"file_path": path})


def validate_mimetypes(name: str, path: str) -> list:
    """Validate mimetypes for images."""

    mimetypes.add_type("image/webp", ".webp")  # to extend types_map
    mimetypes.add_type("image/jpeg", ".jfif")  # to extend types_map

    # libmagic reads only the file header; resolve symlinks, which it does not follow
    mimetype = magic.from_file(os.path.realpath(path), mime=True)
    file_ext = get_file_ext(path).lower()
    if file_ext in mimetypes.guess_all_extensions(mimetype):
        return name

    new_img_ext = mimetypes.guess_extension(mimetype)
    if new_img_ext == ".bin" or new_img_ext is None:
        new_img_ext = ".jpeg"
    new_img_name = f"{get_file_name(name)}{new_img_ext}"
    logger.info(f"Image {name} with mimetype {mimetype} will be converted to {new_img_ext}")

    return new_img_name


def _register_heif_opener() -> None:
    """Register the HEIC/HEIF opener from pillow-heif, which Pillow does not read on its own."""
    try:
        from pillow_heif import register_heif_opener  # pylint: disable=import-error
    except ImportError as e:
        raise ImportError(
            "HEIC/HEIF support is not installed. Install the 'pillow-heif' package."
        ) from e
    register_heif_opener()


def _register_avif_opener() -> None:
    """Use pillow-avif-plugin when it is installed, otherwise Pillow's built-in AVIF support."""
    try:
        import pillow_avif  # pylint: disable=import-error,unused-import

        return
    except ImportError:
        pass

    from PIL import features

    try:
        has_avif = features.check_module("avif")
    except ValueError:  # Pillow older than 11.2 does not know the module
        has_avif = False
    if not has_avif:
        raise ImportError(
            "AVIF support is not installed. Install Pillow 11.3+ (Python 3.9+) "
            "or the 'pillow-avif-plugin' package."
        )


def _free_path(path: str, ext: str) -> str:
    """Path with extension `ext` next to `path` that does not overwrite an existing file."""
    new_path = Path(path).with_suffix(ext)
    i = 0
    while new_path.exists():
        i += 1
        suffix = ext if i == 1 else f"_{i}{ext}"
        new_path = Path(f"{path}{suffix}")
    return new_path.as_posix()


def convert_to_jpg(path) -> str:
    """Convert a HEIC, HEIF or AVIF image to JPEG."""

    ext = get_file_ext(path).lower()
    if ext in _HEIF_EXTS:
        _register_heif_opener()
    elif ext == ".avif":
        _register_avif_opener()

    new_path = _free_path(path, ".jpeg")
    # the SDK loads truncated images by default; pillow-heif then returns an all-black
    # picture for a truncated HEIC, so fail on it instead and keep the original
    load_truncated = ImageFile.LOAD_TRUNCATED_IMAGES
    ImageFile.LOAD_TRUNCATED_IMAGES = False
    try:
        with Image.open(path) as image:
            image.convert("RGB").save(new_path, format="JPEG")
    except Exception:
        silent_remove(new_path)  # do not leave a half-written JPEG behind
        raise
    finally:
        ImageFile.LOAD_TRUNCATED_IMAGES = load_truncated
    silent_remove(path)
    return new_path


def read_tiff_image(path: str) -> Union[np.ndarray, None]:
    """
    Read tiff image.
    Method will transpose image if it has shape (C, H, W) to (H, W, C).
    """

    import tifffile

    logger.debug(f"Found tiff file: {path}.")
    try:
        image = tifffile.imread(path)
    except Exception as e:
        logger.warning(
            f"tifffile failed to read TIFF, trying Pillow fallback: {repr(e)}",
            extra={"file_path": path},
        )
        image = _read_tiff_image_fallback(path)
    name = get_file_name_with_ext(path)
    if image is not None:
        tiff_shape = image.shape
        if image.ndim == 3:
            if tiff_shape[0] < tiff_shape[1] and tiff_shape[0] < tiff_shape[2]:
                image = image.transpose(1, 2, 0)
                logger.warning(f"{name}: transposed shape from {tiff_shape} to {image.shape}")

    return image


def _read_tiff_image_fallback(path: str) -> Union[np.ndarray, None]:
    """
    Fallback method to read tiff image using Pillow.
    """
    from PIL import ImageSequence

    try:
        with Image.open(path) as pil_img:
            frames = [np.asarray(frame) for frame in ImageSequence.Iterator(pil_img)]
        if not frames:
            return None
        if len(frames) == 1:
            return frames[0]

        if all(frame.shape == frames[0].shape for frame in frames):
            return np.stack(frames, axis=0)

        logger.warning(
            "TIFF has multiple pages with different shapes; using the first page only.",
            extra={"file_path": path},
        )
        return frames[0]
    except Exception as e:
        logger.warning(
            f"Pillow failed to read TIFF: {repr(e)}",
            extra={"file_path": path},
        )
        return None


def validate_image_bounds(labels: List[Label], img_rect: Rectangle) -> List[Label]:
    """
    Check if labels are localed inside the image canvas, print a warning and skip them if not.
    """
    new_labels = []
    for label in labels:
        if isinstance(label.geometry, OrientedBBox):
            new_labels.append(label)
        elif img_rect.contains(label.geometry.to_bbox()):
            new_labels.append(label)
    if new_labels != labels:
        logger.warning(
            f"{len(labels) - len(new_labels)} annotation objects are out of image bounds. Skipping..."
        )
    return new_labels

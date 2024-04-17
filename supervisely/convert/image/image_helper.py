import mimetypes
import os

import magic
from PIL import Image

from supervisely import logger
from supervisely.imaging.image import read, write
from supervisely.io.fs import (
    get_file_ext,
    get_file_name,
    get_file_name_with_ext,
    silent_remove,
)

EXT_TO_CONVERT = [".heic", ".avif"]


def validate_image(path: str) -> tuple:
    """
    Validate image by mimetypes, ext, normalize exif and remove alpha channel.
    """
    try:
        ext = get_file_ext(path)
        if ext == ".nrrd":
            return path
        name = get_file_name_with_ext(path)
        if ext.lower() in EXT_TO_CONVERT:
            path = convert_to_jpg(path)
            return path
        if ext.lower() != ".mpo":
            name = validate_mimetypes(name, path)
            path = normalize_exif_and_remove_alpha_channel(name, path)
        return path
    except Exception as e:
        logger.warning(f"Skip image {name}: {repr(e)}", extra={"file_path": path})


def normalize_exif_and_remove_alpha_channel(name: str, path: str) -> tuple:
    """Normalize exif and remove alpha channel."""
    img = read(path)
    if name != get_file_name_with_ext(path):
        silent_remove(path)
        path = os.path.join(os.path.dirname(path), name)
    write(path, img)
    return path


def validate_mimetypes(name: str, path: str) -> list:
    """Validate mimetypes for images."""

    mimetypes.add_type("image/webp", ".webp")  # to extend types_map
    mimetypes.add_type("image/jpeg", ".jfif")  # to extend types_map

    mime = magic.Magic(mime=True)
    mimetype = mime.from_file(path)
    file_ext = get_file_ext(path).lower()
    if file_ext in mimetypes.guess_all_extensions(mimetype):
        return name

    new_img_ext = mimetypes.guess_extension(mimetype)
    new_img_name = f"{get_file_name(name)}{new_img_ext}"
    logger.warn(
        f"Image {name} extension doesn't have correct mimetype {mimetype}. "
        f"Image will be converted to {new_img_ext}"
    )

    return new_img_name


def convert_to_jpg(path) -> tuple:
    """
    Convert image to jpg.
    """
    # * do not remove folllowing imports, it is used to register avif/heic formats
    import pillow_avif  # pylint: disable=unused-import
    from pillow_heif import register_heif_opener

    register_heif_opener()

    name = get_file_name(path)
    new_name = f"{name}.jpeg"
    dirname = os.path.dirname(path)
    new_path = os.path.join(dirname, new_name)
    with Image.open(path) as image:
        image.convert("RGB").save(new_path)
    silent_remove(path)
    return new_path

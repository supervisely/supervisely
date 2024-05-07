import mimetypes
from pathlib import Path

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

EXT_TO_CONVERT = [".heic", ".avif", ".heif"]


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

    mime = magic.Magic(mime=True)
    mimetype = mime.from_file(path)
    file_ext = get_file_ext(path).lower()
    if file_ext in mimetypes.guess_all_extensions(mimetype):
        return name

    new_img_ext = mimetypes.guess_extension(mimetype)
    if new_img_ext == ".bin" or new_img_ext is None:
        new_img_ext = ".jpeg"
    new_img_name = f"{get_file_name(name)}{new_img_ext}"
    logger.info(
        f"Image {name} with mimetype {mimetype} will be converted to {new_img_ext}"
    )

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

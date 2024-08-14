import os

import cv2
import nrrd
import numpy as np

from supervisely._utils import generate_free_name
from supervisely.imaging.image import fliplr
from supervisely.io.fs import get_file_ext

def read_high_color_images(image_path: str) -> np.ndarray:
    """Read high color images"""

    ext = get_file_ext(image_path).lower()
    if ext in [".exr", ".hdr"]:
        os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
        image = cv2.imread(image_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    elif ext in [".tiff", ".tif"]:
        import tifffile

        image = tifffile.imread(image_path)
    else:
        image = cv2.imread(image_path)

    return image

def convert_to_nrrd(image: np.ndarray) -> np.ndarray:
    """Convert image to nrrd format"""
    from PIL import Image

    image = Image.fromarray(image)
    image = image.rotate(90, expand=True)
    image = fliplr(np.asarray(image))
    return image


def save_nrrd(image: np.ndarray, save_path: str) -> str:
    """Save numpy image as nrrd file"""

    image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    image = fliplr(image)

    used_names = {os.path.basename(save_path)}

    name = os.path.basename(save_path)
    while os.path.exists(save_path):
        name = generate_free_name(used_names, name, True, True)
        save_path = os.path.join(os.path.dirname(save_path), name)

    nrrd.write(save_path, image)

    return save_path

# coding: utf-8

import os.path
from pkg_resources import parse_version
import base64
import requests
import numpy as np

import cv2
from PIL import ImageDraw, ImageFile, ImageFont, Image as PILImage
import numpy as np
from enum import Enum
import skimage.transform

from supervisely_lib.io.fs import ensure_base_path, get_file_ext
from supervisely_lib.geometry.rectangle import Rectangle
from supervisely_lib.geometry.image_rotator import ImageRotator
from supervisely_lib.imaging.font import get_font
from supervisely_lib._utils import get_bytes_hash

ImageFile.LOAD_TRUNCATED_IMAGES = True

#@TODO: refactoring image->img
KEEP_ASPECT_RATIO = -1  # TODO: need move it to best place

# Do NOT use directly for image extension validation. Use is_valid_ext() /  has_valid_ext() below instead.
SUPPORTED_IMG_EXTS = ['.jpg', '.jpeg', '.mpo', '.bmp', '.png', '.webp']
DEFAULT_IMG_EXT = '.png'


class CornerAnchorMode:
    TOP_LEFT = "tl"
    TOP_RIGHT = "tr"
    BOTTOM_LEFT = 'bl'
    BOTTOM_RIGHT = 'br'


class RotateMode(Enum):
    KEEP_BLACK = 0
    CROP_BLACK = 1
    SAVE_ORIGINAL_SIZE = 2


class ImageExtensionError(Exception):
    pass


class UnsupportedImageFormat(Exception):
    pass


class ImageReadException(Exception):
    pass

def is_valid_ext(ext: str) -> bool:
    '''
    The function is_valid_ext checks file extension for list of supported images extensions('.jpg', '.jpeg', '.mpo', '.bmp', '.png', '.webp')
    :param ext: file extention
    :return: True if file extention in list of supported images extensions, False - in otherwise
    '''
    return ext.lower() in SUPPORTED_IMG_EXTS


def has_valid_ext(path: str) -> bool:
    '''
    The function has_valid_ext checks if a given file has a supported extension('.jpg', '.jpeg', '.mpo', '.bmp', '.png', '.webp')
    :param path: the path to the input file
    :return: True if a given file has a supported extension, False - in otherwise
    '''
    return is_valid_ext(os.path.splitext(path)[1])


def validate_ext(path):
    '''
    The function validate_ext generate exception error if file extention is not in list of supported images
    extensions('.jpg', '.jpeg', '.mpo', '.bmp', '.png', '.webp')
    :param path: the path to the input file
    '''
    _, ext = os.path.splitext(path)
    if not is_valid_ext(ext):
        raise ImageExtensionError(
            'Unsupported image extension: {!r} for file {!r}. Only the following extensions are supported: {}.'.format(
                ext, path, ', '.join(SUPPORTED_IMG_EXTS)))


def validate_format(path):
    '''
    The function validate_format checks input file format. It generate exception error(ImageReadException) if error
    has occured trying to read image and generate exception error(UnsupportedImageFormat) if file extention is not in
    list of supported images extensions('.jpg', '.jpeg', '.mpo', '.bmp', '.png', '.webp')
    :param path: the path to the input file
    '''
    try:
        pil_img = PILImage.open(path)
        pil_img.load()  # Validate image data. Because 'open' is lazy method.
    except OSError as e:
        raise ImageReadException(
            'Error has occured trying to read image {!r}. Original exception message: {!r}'.format(path, str(e)))

    img_format = pil_img.format
    img_ext = '.' + img_format
    if not is_valid_ext('.' + img_format):
        raise UnsupportedImageFormat(
            'Unsupported image format {!r} for file {!r}. Only the following formats are supported: {}'.format(
                img_ext, path, ', '.join(SUPPORTED_IMG_EXTS)))


def read(path, remove_alpha_channel=True) -> np.ndarray:
    '''
    The function read loads an image from the specified file and returns it in RGB format. If the image cannot be read
    it generate exception error(ImageReadException) if error has occured trying to read image and generate exception
    error(UnsupportedImageFormat) if file extention is not in list of supported images
    extensions('.jpg', '.jpeg', '.mpo', '.bmp', '.png', '.webp')
    :param path: the path to the input file
    :return: image in RGB format(numpy matrix)
    '''
    validate_format(path)
    if remove_alpha_channel is True:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise IOError("OpenCV can not open the file {!r}".format(path))
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise IOError("OpenCV can not open the file {!r}".format(path))
        cnt_channels = img.shape[2]
        if cnt_channels == 4:
            return cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        elif cnt_channels == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif cnt_channels == 1:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            raise ValueError("image has {} channels. Please, contact support...".format(cnt_channels))


def read_bytes(image_bytes, keep_alpha=False) -> np.ndarray:
    '''
    The function read_bytes loads an byte image and returns it in RGB format.
    :param image_bytes: byte image
    :return: image in RGB format(numpy matrix)
    '''
    image_np_arr = np.asarray(bytearray(image_bytes), dtype="uint8")
    if keep_alpha is True:
        img = cv2.imdecode(image_np_arr, cv2.IMREAD_UNCHANGED)
        if len(img.shape) == 2:
            img = np.expand_dims(img, 2)
        cnt_channels = img.shape[2]
        if cnt_channels == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        elif cnt_channels == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif cnt_channels == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        return img
    else:
        img = cv2.imdecode(image_np_arr, cv2.IMREAD_COLOR)  # cv2.imdecode returns BGR always
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def write(path, img, remove_alpha_channel=True):
    '''
    The function write saves the image to the specified file. It create directory from path if the directory for this
    path does not exist. It generate exception error(UnsupportedImageFormat) if file extention is not in list
    of supported images extensions('.jpg', '.jpeg', '.mpo', '.bmp', '.png', '.webp').
    :param path: the path to the output file
    :param img: image in RGB format(numpy matrix)
    '''
    ensure_base_path(path)
    validate_ext(path)
    res_img = img.copy()
    if len(img.shape) == 2:
        res_img = np.expand_dims(img, 2)
    cnt_channels = res_img.shape[2]
    if cnt_channels == 4:
        if remove_alpha_channel is True:
            res_img = cv2.cvtColor(res_img.astype(np.uint8), cv2.COLOR_RGBA2BGR)
        else:
            res_img = cv2.cvtColor(res_img.astype(np.uint8), cv2.COLOR_RGBA2BGRA)
    elif cnt_channels == 3:
        res_img = cv2.cvtColor(res_img.astype(np.uint8), cv2.COLOR_RGB2BGR)
    elif cnt_channels == 1:
        res_img = cv2.cvtColor(res_img.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    return cv2.imwrite(path, res_img)


def draw_text_sequence(bitmap: np.ndarray,
                       texts: list,
                       anchor_point: tuple,
                       corner_snap: CornerAnchorMode = CornerAnchorMode.TOP_LEFT,
                       col_space: int = 12,
                       font: ImageFont.FreeTypeFont = None,
                       fill_background: bool = True) -> None:
    """
    Draws text labels on bitmap from left to right with `col_space` spacing between labels.

    Args:
        bitmap: target image (canvas)
        texts: texts sequence for drawing
        anchor_point: start anchor point (row, column)
        corner_snap: control, how to draw text around `anchor_point`
        col_space: horizontal space between text labels in pixels.
        font: True-Type font object
        fill_background: draw background or not.
    """
    col_offset = 0
    for text in texts:
        position = anchor_point[0], anchor_point[1] + col_offset
        _, text_width = draw_text(bitmap, text, position, corner_snap, font, fill_background)
        col_offset += text_width + col_space


def draw_text(bitmap: np.ndarray,
              text: str,
              anchor_point: tuple,
              corner_snap: CornerAnchorMode=CornerAnchorMode.TOP_LEFT,
              font: ImageFont.FreeTypeFont = None,
              fill_background=True) -> tuple:
    """
    Draws given text on bitmap image.
    Args:
        bitmap: target image (canvas)
        text: text for drawing
        anchor_point: start anchor point (row, column)
        corner_snap: control, how to draw text around `anchor_point`
        font: True-Type font object
        fill_background: draw background or not.

    Returns:
        Calculated (text_height, text_width) tuple. It may be helpful for some calculations
    """

    if font is None:
        font = get_font()

    source_img = PILImage.fromarray(bitmap)
    source_img = source_img.convert("RGBA")

    canvas = PILImage.new('RGBA', source_img.size, (0, 0, 0, 0))
    drawer = ImageDraw.Draw(canvas, "RGBA")
    text_width, text_height = drawer.textsize(text, font=font)
    rect_top, rect_left = anchor_point

    if corner_snap == CornerAnchorMode.TOP_LEFT:
        pass  # Do nothing
    elif corner_snap == CornerAnchorMode.TOP_RIGHT:
        rect_left -= text_width
    elif corner_snap == CornerAnchorMode.BOTTOM_LEFT:
        rect_top -= text_height
    elif corner_snap == CornerAnchorMode.BOTTOM_RIGHT:
        rect_top -= text_height
        rect_left -= text_width

    if fill_background:
        rect_right = rect_left + text_width
        rect_bottom = rect_top + text_height
        drawer.rectangle(((rect_left, rect_top), (rect_right+1, rect_bottom)), fill=(255, 255, 255, 128))
    drawer.text((rect_left+1, rect_top), text, fill=(0, 0, 0, 255), font=font)

    source_img = PILImage.alpha_composite(source_img, canvas)
    source_img = source_img.convert("RGB")
    bitmap[:,:,:] = np.array(source_img, dtype=np.uint8)

    return (text_height, text_width)


#@TODO: not working with alpha channel
def write_bytes(img, ext) -> np.ndarray:
    '''
    The function compresses the image and stores it in the byte object. It generate exception
    error(UnsupportedImageFormat) if file extention is not in list of supported images
    extensions('.jpg', '.jpeg', '.mpo', '.bmp', '.png', '.webp'). Function generate exception error(RuntimeError) if it
    can't encode input image.
    :param img: image in RGB format(numpy matrix) to be written
    :param ext: file extension that defines the output format
    :return: the byte object
    '''
    ext = ('.' + ext).replace('..', '.')
    if not is_valid_ext(ext):
        raise UnsupportedImageFormat(
            'Unsupported image format {!r}. Only the following formats are supported: {}'.format(
                ext, ', '.join(SUPPORTED_IMG_EXTS)))
    img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR)
    encode_status, img_array = cv2.imencode(ext, img)
    if encode_status is True:
        return img_array.tobytes()
    raise RuntimeError('Can not encode input image')


#@TODO: not working with alpha channel
def get_hash(img, ext):
    '''
    The function get_hash hash input image with sha256 algoritm and encode result by using Base64
    :param img: image in RGB format(numpy matrix)
    :param ext: file extension that defines the output format
    :return: the byte object
    '''
    return get_bytes_hash(write_bytes(img, ext))


def crop(img: np.ndarray, rect: Rectangle) -> np.ndarray:
    '''
    The function crop cut out part of the image with rectangle size. If rectangle for crop is out of image area it
    generates exception error(ValueError).
    :param img: image(numpy matrix) to be cropped
    :param rect: class object Rectangle of a certain size
    :return: cropped image
    '''
    img_rect = Rectangle.from_array(img)
    if not img_rect.contains(rect):
        raise ValueError('Rectangle for crop out of image area!')
    return rect.get_cropped_numpy_slice(img)


def crop_with_padding(img: np.ndarray, rect: Rectangle) -> np.ndarray:
    '''
    The function crop cut out part of the image with rectangle size. If rectangle for crop is out of image area it
    generates additional padding.
    :param img: image(numpy matrix) to be cropped
    :param rect: class object Rectangle of a certain size
    :return: cropped image
    '''
    img_rect = Rectangle.from_array(img)
    if not img_rect.contains(rect):
        row, col = img.shape[:2]
        new_img = cv2.copyMakeBorder(
            img,
            top=rect.height,
            bottom=rect.height,
            left=rect.width,
            right=rect.width,
            borderType=cv2.BORDER_CONSTANT
        )
        new_rect = rect.translate(drow=rect.height, dcol=rect.width)
        return new_rect.get_cropped_numpy_slice(new_img)

    else:
        return rect.get_cropped_numpy_slice(img)


def restore_proportional_size(in_size: tuple, out_size: tuple = None,
                              frow: float = None, fcol: float = None, f: float = None):
    '''
    The function restore_proportional_size calculate new size of the image. If some parameters are not specified, or
    size dimensions are less than 0 it generate exception error(ValueError).
    :param in_size: size of input image
    :param out_size: new image size
    :param frow: length of output image
    :param fcol: height of output image
    :param f: positive non zero scale factor
    :return: size of output image
    '''
    if out_size is not None and (frow is not None or fcol is not None) and f is None:
        raise ValueError('Must be specified output size or scale factors not both of them.')

    if out_size is not None:
        if out_size[0] == KEEP_ASPECT_RATIO and out_size[1] == KEEP_ASPECT_RATIO:
            raise ValueError('Must be specified at least 1 dimension of size!')

        if (out_size[0] <= 0 and out_size[0] != KEEP_ASPECT_RATIO) or \
                (out_size[1] <= 0 and out_size[1] != KEEP_ASPECT_RATIO):
            raise ValueError('Size dimensions must be greater than 0.')

        result_row = out_size[0] if out_size[0] > 0 else max(1, round(out_size[1] / in_size[1] * in_size[0]))
        result_col = out_size[1] if out_size[1] > 0 else max(1, round(out_size[0] / in_size[0] * in_size[1]))
    else:
        if f is not None:
            if f < 0:
                raise ValueError('"f" argument must be positive!')
            frow = fcol = f

        if (frow < 0 or fcol < 0) or (frow is None or fcol is None):
            raise ValueError('Specify "f" argument for single scale!')

        result_col = round(fcol * in_size[1])
        result_row = round(frow * in_size[0])
    return result_row, result_col


#@TODO: reimplement, to be more convenient
def resize(img: np.ndarray, out_size: tuple=None, frow: float=None, fcol: float=None) -> np.ndarray:
    '''
    The function resize resizes the image img down to or up to the specified size. If some parameters are not specified, or
    size dimensions are less than 0 it generate exception error(ValueError).
    :param img: input image
    :param out_size: new image size
    :param frow: length of output image
    :param fcol: height of output image
    :return: resized image
    '''
    result_height, result_width = restore_proportional_size(img.shape[:2], out_size, frow, fcol)
    return cv2.resize(img, (result_width, result_height), interpolation=cv2.INTER_CUBIC)


def resize_inter_nearest(img: np.ndarray, out_size: tuple=None, frow: float=None, fcol: float=None) -> np.ndarray:
    '''
    The function resize_inter_nearest resize image to match a certain size. Performs interpolation to up-size or
    down-size images. For down-sampling N-dimensional images by applying a function or the arithmetic mean, see
    `skimage.measure.block_reduce` and `skimage.transform.downscale_local_mean`, respectively.
    If some parameters are not specified, or size dimensions are less than 0 it generate exception
    error(ValueError).
    :param img: input image
    :param out_size: new image size
    :param frow: length of output image
    :param fcol: height of output image
    :return: resized image
    '''
    target_shape = restore_proportional_size(img.shape[:2], out_size, frow, fcol)
    resize_kv_args = dict(order=0, preserve_range=True, mode='constant')
    if parse_version(skimage.__version__) >= parse_version('0.14.0'):
        resize_kv_args['anti_aliasing'] = False
    return skimage.transform.resize(img, target_shape, **resize_kv_args).astype(img.dtype)


def scale(img: np.ndarray, factor: float) -> np.ndarray:
    """
    :param factor: positive non zero scale factor
    :return: scaled by factor image
    """
    return resize(img, (round(img.shape[0] * factor), round(img.shape[1] * factor)))


def fliplr(img: np.ndarray) -> np.ndarray:
    """
    :return: image flipped horizontally (columns order reversed)
    """
    return np.flip(img, 1)


def flipud(img: np.ndarray) -> np.ndarray:
    """
    :return: image flipped vertically (rows order reversed)
    """
    return np.flip(img, 0)


def rotate(img: np.ndarray, degrees_angle: float, mode=RotateMode.KEEP_BLACK) -> np.ndarray:
    """
    Rotate given a NumPy / OpenCV image on selected angle and Extend/Crop by chosen mode.
    :param img: image for rotation
    :param degrees_angle: angle in degrees
    :param mode: one of RotateMode enum values
    :return: rotated and processed image
    """

    rotator = ImageRotator(imsize=img.shape[:2], angle_degrees_ccw=degrees_angle)
    if mode == RotateMode.KEEP_BLACK:
        return rotator.rotate_img(img, use_inter_nearest=False)  # @TODO: order = ???
    elif mode == RotateMode.CROP_BLACK:
        img_rotated = rotator.rotate_img(img, use_inter_nearest=False)
        return rotator.inner_crop.get_cropped_numpy_slice(img_rotated)
    elif mode == RotateMode.SAVE_ORIGINAL_SIZE:
        # TODO Implement this in rotator instead.
        return skimage.transform.rotate(img, degrees_angle, resize=False)
    else:
        raise NotImplementedError('Rotate mode "{0}" not supported!'.format(str(mode)))


# Color augmentations
def _check_contrast_brightness_inputs(min_value, max_value):
    '''
    The function _check_contrast_brightness_inputs checks the input brightness or contrast for correctness.
    '''
    if min_value < 0:
        raise ValueError('Minimum value must be greater than or equal to 0.')
    if min_value > max_value:
        raise ValueError('Maximum value must be greater than or equal to minimum value.')


def random_contrast(image: np.ndarray, min_factor: float, max_factor: float) -> np.ndarray:
    """
    Randomly changes contrast of the input image.

    Args:
        image: Input image array.
        min_factor: Lower bound of contrast range.
        max_factor: Upper bound of contrast range.
    Returns:
        Image array with changed contrast.
    """
    _check_contrast_brightness_inputs(min_factor, max_factor)
    contrast_value = np.random.uniform(min_factor, max_factor)
    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image_mean = round(image_gray.mean())
    image = image.astype(np.float32)
    image = contrast_value * (image - image_mean) + image_mean
    return np.clip(image, 0, 255).astype(np.uint8)


def random_brightness(image: np.ndarray, min_factor: float, max_factor: float) -> np.ndarray:
    """
    Randomly changes brightness of the input image.

    Args:
        image: Input image array.
        min_factor: Lower bound of brightness range.
        max_factor: Upper bound of brightness range.
    Returns:
        Image array with changed brightness.
    """
    _check_contrast_brightness_inputs(min_factor, max_factor)
    brightness_value = np.random.uniform(min_factor, max_factor)
    image = image.astype(np.float32)
    image = image * brightness_value
    return np.clip(image, 0, 255).astype(np.uint8)


def random_noise(image: np.ndarray, mean: float, std: float) -> np.ndarray:
    """
    Adds random Gaussian noise to the input image.

    Args:
        image: Input image array.
        mean: The mean value of noise distribution.
        std: The standard deviation of noise distribution.
    Returns:
        Image array with additional noise.
    """
    image = image.astype(np.float32)
    image += np.random.normal(mean, std, image.shape)

    return np.clip(image, 0, 255).astype(np.uint8)


def random_color_scale(image: np.ndarray, min_factor: float, max_factor: float) -> np.ndarray:
    """
    Changes image colors by randomly scaling each of RGB components. The scaling factors are sampled uniformly from
    the given range.
    Args:
        image: Input image array.
        min_factor: minimum scale factor
        max_factor: maximum scale factor
    Returns:
        Image array with shifted colors.
    """
    image_float = image.astype(np.float64)
    scales = np.random.uniform(low=min_factor, high=max_factor, size=(1,1, image.shape[2]))
    res_image = image_float * scales
    return np.clip(res_image, 0, 255).astype(np.uint8)


# Blurs
def blur(image: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    Blurs an image using the normalized box filter.

    Args:
        image: Input image array.
        kernel_size: Blurring kernel size.
    Returns:
        Blurred image array.
    """
    return cv2.blur(image, (kernel_size, kernel_size))


def median_blur(image: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    Blurs an image using the median filter.

    Args:
        image: Input image array.
        kernel_size: Blurring kernel size.
    Returns:
        Blurred image array.
    """
    return cv2.medianBlur(image, kernel_size)


def gaussian_blur(image: np.ndarray, sigma_min: float, sigma_max: float) -> np.ndarray:
    """
    Blurs an image using a Gaussian filter.

    Args:
        image: Input image array.
        sigma_min: Lower bound of Gaussian kernel standard deviation range.
        sigma_max: Upper bound of Gaussian kernel standard deviation range.
    Returns:
        Blurred image array.
    """
    sigma_value = np.random.uniform(sigma_min, sigma_max)
    return cv2.GaussianBlur(image, (0, 0), sigma_value)


def drop_image_alpha_channel(img: np.ndarray) -> np.ndarray:
    """
    Converts 4-channel image to 3-channel.

    Args:
        img: Input 4-channel image array.
    Returns:
        3-channel image array.
    """
    if img.shape[2] != 4:
        raise ValueError('Only 4-channel RGBA images are supported for alpha channel removal. ' +
                         'Instead got {} channels.'.format(img.shape[2]))
    return cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)


#@TODO: refactor from two separate methods to a single one
#bgra or bgr
def np_image_to_data_url(img):
    encode_status, bgra_result_png = cv2.imencode('.png', img)
    img_png = bgra_result_png.tobytes()
    img_base64 = base64.b64encode(img_png)
    data_url = 'data:image/png;base64,{}'.format(str(img_base64, 'utf-8'))
    return data_url


def data_url_to_numpy(data_url):
    img_base64 = data_url[len('data:image/png;base64,'):]
    img_base64 = base64.b64decode(img_base64)
    image = read_bytes(img_base64)
    return image


#only rgb
def np_image_to_data_url_backup_rgb(img):
    img_base64 = base64.b64encode(write_bytes(img, 'png'))
    data_url = 'data:image/png;base64,{}'.format(str(img_base64, 'utf-8'))
    return data_url
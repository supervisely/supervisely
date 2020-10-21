# coding: utf-8
import random
import colorsys


def _validate_color(color):
    """
    Checks input color for compliance with the required format
    :param: color: color (RGB tuple of integers)
    """
    if not isinstance(color, (list, tuple)):
        raise ValueError('Color has to be list, or tuple')
    if len(color) != 3:
        raise ValueError('Color have to contain exactly 3 values: [R, G, B]')
    for channel in color:
        validate_channel_value(channel)


def random_rgb() -> list:
    """
    Generate RGB color with fixed saturation and lightness
    :return: RGB integer values.
    """
    hsl_color = (random.random(), 0.3, 0.8)
    rgb_color = colorsys.hls_to_rgb(*hsl_color)
    return [round(c * 255) for c in rgb_color]


def _normalize_color(color):
    """
    Divide all RGB values by 255.
    :param color: color (RGB tuple of integers)
    """
    return [c / 255. for c in color]


def _color_distance(first_color: list, second_color: list) -> float:
    """
    Calculate distance in HLS color space between Hue components of 2 colors
    :param first_color: first color (RGB tuple of integers)
    :param second_color: second color (RGB tuple of integers)
    :return: Euclidean distance between 'first_color' and 'second_color'
    """
    first_color_hls = colorsys.rgb_to_hls(*_normalize_color(first_color))
    second_color_hls = colorsys.rgb_to_hls(*_normalize_color(second_color))
    hue_distance = min(abs(first_color_hls[0] - second_color_hls[0]),
                       1 - abs(first_color_hls[0] - second_color_hls[0]))
    return hue_distance


def generate_rgb(exist_colors: list) -> list:
    """
    Generate new color which oppositely by exist colors
    :param exist_colors: list of existing colors in RGB format.
    :return: RGB integer values. Example: [80, 255, 0]
    """
    largest_min_distance = 0
    best_color = random_rgb()
    if len(exist_colors) > 0:
        for _ in range(100):
            color = random_rgb()
            current_min_distance = min(_color_distance(color, c) for c in exist_colors)
            if current_min_distance > largest_min_distance:
                largest_min_distance = current_min_distance
                best_color = color
    _validate_color(best_color)
    return best_color


def rgb2hex(color: list) -> str:
    """
    Convert integer color format to HEX string
    :param color: RGB integer values. Example: [80, 255, 0]
    :return: HEX RGB string. Example: "#FF42А4
    """
    _validate_color(color)
    return '#' + ''.join('{:02X}'.format(component) for component in color)


def _hex2color(hex_value: str) -> list:
    """
        Convert HEX RGB string to integer RGB format
        :param hex_value: HEX RGBA string. Example: "#FF02А4
        :return: RGB integer values. Example: [80, 255, 0]
    """
    assert hex_value.startswith('#')
    return [int(hex_value[i:(i + 2)], 16) for i in range(1, len(hex_value), 2)]


def hex2rgb(hex_value: str) -> list:
    """
    Convert HEX RGB string to integer RGB format
    :param hex_value: HEX RGBA string. Example: "#FF02А4
    :return: RGB integer values. Example: [80, 255, 0]
    """
    assert len(hex_value) == 7, "Supported only HEX RGB string format!"
    color = _hex2color(hex_value)
    _validate_color(color)
    return color


def _hex2rgba(hex_value: str) -> list:
    """
    Convert HEX RGBA string to integer RGBA format
    :param hex_value: HEX RGBA string. Example: "#FF02А4CC
    :return: RGBA integer values. Example: [80, 255, 0, 128]
    """
    assert len(hex_value) == 9, "Supported only HEX RGBA string format!"
    return _hex2color(hex_value)


def validate_channel_value(value):
    if 0 <= value <= 255:
        pass
    else:
        raise ValueError('Color channel has to be in range [0; 255]')
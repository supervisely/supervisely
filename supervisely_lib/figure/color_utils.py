# @TODO: wtf, move out, drop, change

import random


def get_random_color(pastel_factor = 0.5):
    return [(x+pastel_factor)/(1.0+pastel_factor) for x in [random.uniform(0,1.0) for i in [1,2,3]]]


def color_distance(c1,c2):
    return sum([abs(x[0]-x[1]) for x in zip(c1,c2)])


def generate_new_color(existing_colors,pastel_factor = 0.5):
    max_distance = None
    best_color = None
    for i in range(0,100):
        color = get_random_color(pastel_factor = pastel_factor)
        if not existing_colors:
            return [c * 255 for c in color]
        best_distance = min([color_distance(color,c) for c in existing_colors])
        if not max_distance or best_distance > max_distance:
            max_distance = best_distance
            best_color = color
    return [c * 255 for c in best_color]


def color2code(color):
    return '#%02x%02x%02x' % (int(color[0]), int(color[1]), int(color[2]))


def gen_new_color():
    return color2code(generate_new_color([]))


def hex2rgb(hex_value):
    bigint = int(hex_value.lstrip('#'), 16)
    res = [bigint >> (i * 8) & 255 for i in range(2, -1, -1)]
    return res


def hex2rgba(hex_value):
    bigint = int(hex_value.lstrip('#'), 16)
    res = [bigint >> (i * 8) & 255 for i in range(3, -1, -1)]
    return res

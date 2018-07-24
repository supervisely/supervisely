# coding: utf-8

import random
import collections

from ..figure import Rect


# performs updating like lhs.update(rhs), but operates recursively on nested dictionaries
def update_recursively(lhs, rhs):
    for k, v in rhs.items():
        if isinstance(v, collections.Mapping):
            lhs[k] = update_recursively(lhs.get(k, {}), v)
        else:
            lhs[k] = v
    return lhs


# @TODO: support float percents?
# settings fmt from export CropLayer
def rect_from_bounds(settings_dct, img_w, img_h, shift_inside=True):
    def get_px_value(dim_name, max_side):
        value = settings_dct.get(dim_name)
        if value is None:
            return 0
        if value.endswith('px'):
            value = int(value[:-len('px')])
        else:
            value = int(value[:-len('%')])
            value = int(max_side * value / 100.0)
        if not shift_inside:
            value *= -1
        return value

    def calc_new_side(old_side, l_name, r_name):
        l_bound = get_px_value(l_name, old_side)
        r_bound = old_side - get_px_value(r_name, old_side)
        return l_bound, r_bound

    left, right = calc_new_side(img_w, 'left', 'right')
    top, bottom = calc_new_side(img_h, 'top', 'bottom')
    res = Rect(left, top, right, bottom)
    return res


# @TODO: support float percents?
# returns rect with ints
def random_rect_from_bounds(settings_dct, img_w, img_h):
    def rand_percent(p_name):
        perc_dct = settings_dct[p_name]
        the_percent = random.uniform(perc_dct['min_percent'], perc_dct['max_percent'])
        return the_percent

    def calc_new_side(old_side, perc):
        new_side = min(int(old_side), int(old_side * perc / 100.0))
        l_bound = random.randint(0, old_side - new_side)  # including [a; b]
        r_bound = l_bound + new_side
        return l_bound, r_bound

    rand_percent_w = rand_percent('width')
    if not settings_dct.get('keep_aspect_ratio', False):
        rand_percent_h = rand_percent('height')
    else:
        rand_percent_h = rand_percent_w
    left, right = calc_new_side(img_w, rand_percent_w)
    top, bottom = calc_new_side(img_h, rand_percent_h)
    res = Rect(left, top, right, bottom)
    return res

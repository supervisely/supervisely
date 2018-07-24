# coding: utf-8

from .figure_point import FigurePoint
from .figure_line import FigureLine
from .figure_rectangle import FigureRectangle
from .figure_polygon import FigurePolygon
from .figure_bitmap import FigureBitmap


class FigureFactory:

    name_to_class = {
        "point": FigurePoint,
        "line": FigureLine,
        "rectangle": FigureRectangle,
        "polygon": FigurePolygon,
        "bitmap": FigureBitmap
    }

    @classmethod
    def create_from_packed(cls, project_meta, packed_obj):
        cls_title = packed_obj['classTitle']
        cls_dct = project_meta.classes[cls_title]
        if cls_dct is None:
            raise RuntimeError('Figure with non-existing class {}'.format(cls_title))
        # if cls_dct['shape'] not in ['bitmap', 'polygon', 'rectangle', 'point']:
        #     return None  # @TODO: rm debug if
        figure_class = cls.name_to_class[cls_dct['shape']]
        figure = figure_class.from_packed(packed_obj)
        return figure

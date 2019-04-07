# coding: utf-8

from copy import deepcopy

from supervisely_lib.io.json import JsonSerializable


# @TODO: use properties instead of field if it makes sense
class Geometry(JsonSerializable):
    @staticmethod
    def geometry_name():
        """
        :return: string with name of geometry
        """
        raise NotImplementedError()

    def crop(self, rect):
        """
        :param rect: Rectangle
        :return: list of Geometry
        """
        raise NotImplementedError()

    def relative_crop(self, rect):
        """ Crops object like "crop" method, but return results with coordinates relative to rect
        :param rect:
        :return: list of Geometry
        """
        return [geom.translate(drow=-rect.top, dcol=-rect.left) for geom in self.crop(rect)]

    def rotate(self, rotator):
        """Rotates around image center -> New Geometry
        :param rotator: ImageRotator
        :return: Geometry
        """
        raise NotImplementedError()

    def resize(self, in_size, out_size):
        """
        :param in_size: (rows, cols)
        :param out_size:
            (128, 256)
            (128, KEEP_ASPECT_RATIO)
            (KEEP_ASPECT_RATIO, 256)
        :return: Geometry
        """
        raise NotImplementedError()

    def scale(self, factor):
        """Scales around origin with a given factor.
        :param: factor (float):
        :return: Geometry
        """
        raise NotImplementedError()

    def translate(self, drow, dcol):
        """
        :param drow: int rows shift
        :param dcol: int cols shift
        :return: Geometry
        """
        raise NotImplementedError()

    def fliplr(self, img_size):
        """
        :param img_size: (rows, cols)
        :return: Geometry
        """
        raise NotImplementedError()

    def flipud(self, img_size):
        """
        :param img_size: (rows, cols)
        :return: Geometry
        """
        raise NotImplementedError()

    def draw(self, bitmap, color, thickness=1):
        """
        :param bitmap: np.ndarray
        :param color: [R, G, B]
        :param thickness: used only in Polyline and Point
        """
        raise NotImplementedError()

    def draw_contour(self, bitmap, color, thickness=1):
        """ Draws the figure contour on a given bitmap canvas
        :param bitmap: np.ndarray
        :param color: [R, G, B]
        :param thickness: (int)
        """
        raise NotImplementedError()

    @property
    def area(self):
        """
        :return: float
        """
        raise NotImplementedError()

    def to_bbox(self):
        """
        :return: Rectangle
        """
        raise NotImplementedError()

    def clone(self):
        return deepcopy(self)

    def validate(self, name, settings):
        if self.geometry_name() != name:
            raise ValueError('Geometry validation error: name is mismatch!') # TODO: Write here good message

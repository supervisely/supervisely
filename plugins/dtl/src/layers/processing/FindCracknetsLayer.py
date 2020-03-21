# coding: utf-8

from copy import deepcopy
from typing import List
import math
import cv2
import numpy as np
from shapely.geometry import Polygon

from legacy_supervisely_lib.figure.figure_line import FigureLine
from legacy_supervisely_lib.figure.figure_polygon import FigurePolygon


from Layer import Layer
from classes_utils import ClassConstants


def simple_distance(p1, p2) -> float:
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def get_simple_poly_area(points):
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def remove_poly_loops(simple_poly: np.ndarray, eps=1) -> list:
    simple_poly = simple_poly.tolist()
    i = 0
    new_polys = []
    while i < len(simple_poly):
        j = i + 1
        while j < len(simple_poly):
            if i != j:
                if simple_distance(simple_poly[i], simple_poly[j]) <= eps:
                    new_poly = simple_poly[i:j+1]
                    del simple_poly[i:j]
                    if len(new_poly) > 2:
                        new_polys.append(new_poly)
                    j = i - 1
            j = j + 1
        i = i + 1
    if len(simple_poly) > 2:
        new_polys.append(simple_poly)
    return new_polys


def combine_clusters(labels, clusters_count=None, X=None):
    X = X or []
    if clusters_count is None:
        clusters_count = len(set(labels)) - (1 if -1 in labels else 0)

    clusters = []

    for i in range(clusters_count):
        clusters.append(list())

    for point_index, label in enumerate(labels):
        if label >= 0:
            clusters[label].append([X[point_index][0], X[point_index][1]])

    return clusters


def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


def intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


def remove_intersections_in_poly(vertices):
    i = 0

    if len(vertices) == 0:
        return []

    while i < (len(vertices) - 1):
        changed = False
        j = i + 2
        while j < (len(vertices) - 1):
            if i != j:
                if intersect(vertices[i], vertices[i + 1], vertices[j], vertices[j + 1]):
                    target_index = i + 1
                    target_jndex = j + 1
                    mid_part = list(reversed(vertices[target_index:target_jndex]))
                    vertices = vertices[:target_index] + mid_part + vertices[target_jndex:]
                    changed = True
            j += 1
        if not changed:
            i += 1
            if i == 1:
                vertices = vertices + [vertices[0]]

    del vertices[-1]
    return vertices



def ExtractCrackNets4(image_size_wh,
                      in_list: List[list],
                      mask_resolution:int=160,
                      loop_devide_eps:int=2,
                      min_area_in_bbox_coef:float=0.3,
                      min_poly_bbox_area_coef:float=0.008,
                      enable_subclustering:bool=True,
                      min_points_in_cluster:int=10
                      ) -> List[list]:
    """
    Принимает набор линий и генерирует по ним полигоны сеток трещин.
    Принцип работы:
        0) Вынимаем из всех линий точки и забываем про линии
        1) Переводим координаты точек в условные значения (mask_resolution)
        2) Кластеризуем если включена опция (enable_subclustering). Если нет - то все точки в одном кластере
        3) Для каждого кластера создаем маску пониженного разрешения и рисуем на ней точки.
        4) Постпроцессинг для масок (убираем дыры и тд)
        5) Находим контуры - это наши новые кластеры 2-го порядка
        5.1) Убираем петли и делаем из них отдельные кластеры.
        6) Фильтруем кластеры 2-го порядка по условным площадям:
            6.1) [площадь кластера]/[площадь описывающего прямоугольник] < (min_area_in_bbox_coef)
            6.2) [площадь кластера]/[вся площадь] < (Параметр: min_poly_bbox_area_coef)
        7) Преобразуем координы в абсолютный масштаб и созраняем кластер как полигон

    :param image_size_wh: размер исходного изображения (ширина, высота)
    :param in_list: Список простых линий - (линия = список точек)
    :param mask_resolution: Разрешение для маски (грубость маски) - желательно: [80-500]
    :param loop_devide_eps: Условное расстояние между двумя точками полигона чтобы определить петлю.
    :param min_area_in_bbox_coef: минимальная площадь заполнения кластером описанного прямоугольника
    :param min_poly_bbox_area_coef: минимальная площадь bbox ко всей площади
    :param enable_subclustering: использовать предварительную кластеризацию
    :param min_points_in_cluster: минимальное количество точек в кластере (предварительная кластеризация)
    :return: список полигонов
    """

    if len(in_list) == 0:
        return []

    points = []
    for line in in_list:
        points.extend(line)

    simple_format_points = np.array(points)

    image_width = image_size_wh[0]
    image_height = image_size_wh[1]

    target_width = mask_resolution
    target_height = int(target_width * (image_height / image_width))
    target_area = float(target_width) * target_height

    width_coef = target_width / float(image_width)
    heigh_coef = target_height / float(image_height)

    simple_format_points[:, 0] = simple_format_points[:, 0] * width_coef
    simple_format_points[:, 1] = simple_format_points[:, 1] * heigh_coef
    simple_format_points = np.array(simple_format_points, np.uint8)
    res = []

    integer_points = simple_format_points.tolist()

    # We need clusteriazation?
    if enable_subclustering and len(simple_format_points) >= min_points_in_cluster:
        import hdbscan
        hdb = hdbscan.HDBSCAN(min_cluster_size=min_points_in_cluster).fit(simple_format_points)
        hdb_labels = hdb.labels_
        points_group = combine_clusters(labels=hdb_labels, X=integer_points)
    else:
        points_group = [integer_points]

    masks = []

    for simple_format_points in points_group:
        mask_image = np.zeros((target_height, target_width), np.uint8)

        for point in simple_format_points:
            point_drawing_size = 1
            cv2.circle(mask_image, (point[0], point[1]), point_drawing_size, (255, 255, 255), -1)

        # im_num = random.randint(1000, 9999)
        # cv2.imwrite("/sly_task_data/tmp/qqq_{}.png".format(im_num), mask_image)

        kernel = np.ones((5, 5), np.uint8)

        # mask_image = cv2.erode(mask_image, kernel, iterations=1)
        # mask_image = cv2.dilate(mask_image, kernel, iterations=1)

        mask_image = cv2.morphologyEx(mask_image, cv2.MORPH_CLOSE, kernel)

        # cv2.imwrite("/sly_task_data/tmp/qqq_{}p.png".format(im_num), mask_image)

        masks.append(mask_image)



    for mask_image in masks:
        contours, hierarchy = cv2.findContours(mask_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        clusters = []
        for contour in contours:
            contour = np.squeeze(contour, axis=1)
            clusters.extend(remove_poly_loops(contour, eps=loop_devide_eps))

        # for i in range(len(clusters)):
        #    clusters[i] = remove_intersections_in_poly(clusters[i])

        for cluster in clusters:
            cluster = np.array(cluster)

            width = np.max(cluster[:, 0]) - np.min(cluster[:, 0])
            height = np.max(cluster[:, 1]) - np.min(cluster[:, 1])

            max_size_ratio_check = 8
            if width == 0 or height == 0 or width / height > max_size_ratio_check or height / width > max_size_ratio_check:
                continue

            poly_area = get_simple_poly_area(cluster.tolist())
            bbox_area = width * height

            # Фильтрация по заполнению
            if poly_area / bbox_area < min_area_in_bbox_coef:
                continue

            # Фильтрация по размеру BBox на изображении
            if bbox_area / target_area < min_poly_bbox_area_coef:
                continue

            cluster = np.array(cluster, np.float)
            cluster[:, 0] = cluster[:, 0] / width_coef
            cluster[:, 1] = cluster[:, 1] / heigh_coef
            cluster = cluster.astype(np.int)

            c_exterior = cluster.tolist()
            poly = Polygon(shell=c_exterior)

            if poly.is_valid == False:
                poly = poly.buffer(0)

            if poly.geom_type == 'MultiPolygon':
                for p in poly:
                    cluster = np.transpose(p.exterior.coords.xy).tolist()
                    res.append(cluster)
            else:
                cluster = np.transpose(poly.exterior.coords.xy).tolist()
                res.append(cluster)
    return res




class FindCracknetsLayer(Layer):
    action = 'find_cracknets'

    layer_settings = {
        "required": ["settings"],
        "properties": {
            "settings": {
                "type": "object",
                "required": ["crack_class",
                             "cracknet_class",
                             "mask_resolution",
                             "loop_devide_eps",
                             "min_area_in_bbox_coef",
                             "min_poly_bbox_area_coef",
                             "enable_subclustering",
                             "min_points_in_cluster"],
                "properties": {
                    "crack_class": {
                        "type": "string",
                        "minLength": 1
                    },
                    "cracknet_class": {
                        "type": "string",
                        "minLength": 1
                    },
                    "mask_resolution": {
                        "type": "integer",
                        "minimum": 1
                    },
                    "loop_devide_eps": {
                        "type": "number",
                        "minimum": 0.001
                    },
                    "min_area_in_bbox_coef": {
                        "type": "number",
                        "minimum": 0.00001
                    },
                    "min_poly_bbox_area_coef": {
                        "type": "number",
                        "minimum": 0.00001
                    },
                    "enable_subclustering": {
                        "type": "boolean"
                    },
                    "min_points_in_cluster": {
                        "type": "integer",
                        "minimum": 1
                    },
                }
            }
        }
    }

    def __init__(self, config):
        Layer.__init__(self, config)

    def define_classes_mapping(self):
        self.cls_mapping[ClassConstants.NEW] = [{'title': self.settings['cracknet_class'], 'shape': 'polygon'}]
        self.cls_mapping[ClassConstants.OTHER] = ClassConstants.DEFAULT

    def process(self, data_el):
        img_desc, ann_orig = data_el
        ann = deepcopy(ann_orig)
        imsize_wh = ann_orig.image_size_wh
        crack_class = self.settings.get('crack_class')
        cracknet_class = self.settings.get('cracknet_class')

        crack_lines = []

        for figure in ann['objects']:
            if figure.class_title == crack_class:
                if not isinstance(figure, FigureLine):
                    raise RuntimeError('Input class must be a Line in find_cracknets layer.')
                packed = figure.pack()
                points = packed['points']['exterior']
                crack_lines.append(points)

        mask_resolution = self.settings.get('mask_resolution')
        loop_devide_eps = self.settings.get('loop_devide_eps')
        min_area_in_bbox_coef = self.settings.get('min_area_in_bbox_coef')
        min_poly_bbox_area_coef = self.settings.get('min_poly_bbox_area_coef')
        enable_subclustering = self.settings.get('enable_subclustering')
        min_points_in_cluster = self.settings.get('min_points_in_cluster')

        simple_polygons = ExtractCrackNets4(image_size_wh=imsize_wh,
                                            in_list=crack_lines,
                                            mask_resolution=mask_resolution,
                                            loop_devide_eps=loop_devide_eps,
                                            min_area_in_bbox_coef=min_area_in_bbox_coef,
                                            min_poly_bbox_area_coef=min_poly_bbox_area_coef,
                                            enable_subclustering=enable_subclustering,
                                            min_points_in_cluster=min_points_in_cluster
                                            )

        # Build sly-polygon from simple-polygon coordinates
        for simple_polygon in simple_polygons:
            ann['objects'].extend(FigurePolygon.from_np_points(cracknet_class, imsize_wh, exterior=simple_polygon, interiors=[]))
        yield img_desc, ann

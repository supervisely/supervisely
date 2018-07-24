# coding: utf-8

from copy import deepcopy
from collections import deque

import numpy as np
import networkx as nx
from supervisely_lib import FigureBitmap, FigureLine

from Layer import Layer
from classes_utils import ClassConstants


# @TODO: check, it may be dirty


def _get_graph(skel):
    h, w = skel.shape
    G = nx.Graph()
    for i in range(0, h):
        for j in range(0, w):
            if skel[i, j] == 0:
                continue
            G.add_node((i, j))
            if i - 1 >= 0 and j - 1 >= 0 and skel[i - 1, j - 1] == 1:
                G.add_edge((i - 1, j - 1), (i, j))
            if i - 1 >= 0 and skel[i - 1, j] == 1:
                G.add_edge((i - 1, j), (i, j))
            if i - 1 >= 0 and j + 1 < w and skel[i - 1, j + 1] == 1:
                G.add_edge((i - 1, j + 1), (i, j))
            if j - 1 >= 0 and skel[i, j - 1] == 1:
                G.add_edge((i, j - 1), (i, j))

    return G


def _bfs_path(G, v):
    path = list(nx.bfs_edges(G, v))
    try:
        curr = path[-1][1]
    except IndexError:
        raise RuntimeError('Someone hasn\'t implemented bfs correctly.')
    i = len(path) - 1

    true_path1 = [curr]
    while curr != v:
        if path[i][1] == curr:
            curr = path[i][0]
            true_path1.append(curr)
        i -= 1

    return true_path1[::-1]


def _get_longest_path(G):
    random_node = list(G)[0]
    path1 = _bfs_path(G, random_node)
    last = path1[-1]
    path2 = _bfs_path(G, last)
    longest_path = max([path1, path2], key=len)

    return longest_path


def _get_all_diameters(G):
    diameters = []

    graphs = list(nx.connected_component_subgraphs(G))
    queue = deque(graphs)
    while queue:
        graph = queue.popleft()
        if graph.number_of_edges() < 3:
            continue
        lpath = _get_longest_path(graph)
        diameters.append(lpath)
        graph.remove_edges_from(list(zip(lpath[:-1], lpath[1:])))
        isolates = list(nx.isolates(graph))
        graph.remove_nodes_from(isolates)
        subgraphs = list(nx.connected_component_subgraphs(graph))
        queue.extend(subgraphs)

    return diameters


# FigureBitmap to FigureLine
class BitmapToLinesLayer(Layer):

    action = 'bitmap2lines'

    layer_settings = {
        "required": ["settings"],
        "properties": {
            "settings": {
                "type": "object",
                "required": ["classes_mapping", "min_points_cnt"],
                "properties": {
                    "classes_mapping": {
                        "type": "object",
                        "patternProperties": {
                            ".*": {"type": "string"}
                        }
                    },
                    "min_points_cnt": {
                        "type": "integer",
                        "minimum": 2
                    }
                }
            }
        }
    }

    def __init__(self, config):
        Layer.__init__(self, config)

    def define_classes_mapping(self):
        for old_class, new_class in self.settings['classes_mapping'].items():
            self.cls_mapping[old_class] = {'title': new_class, 'shape': 'line'}
        self.cls_mapping[ClassConstants.OTHER] = ClassConstants.DEFAULT

    def process(self, data_el):
        img_desc, ann_orig = data_el
        ann = deepcopy(ann_orig)
        img_wh = ann_orig.image_size_wh

        def to_lines(f):
            new_title = self.settings['classes_mapping'].get(f.class_title)
            if new_title is None:
                return [f]
            if not isinstance(f, FigureBitmap):
                raise RuntimeError('Input class must be a Bitmap in bitmap2lines layer.')

            origin, mask = f.get_origin_mask()
            graph = _get_graph(mask)
            graph = nx.minimum_spanning_tree(graph)
            paths = _get_all_diameters(graph)

            res = []
            for coords in paths:
                if len(coords) < self.settings['min_points_cnt']:
                    continue
                coords = np.asarray(coords)
                points = coords[:, ::-1]
                points = points + origin
                res.extend(FigureLine.from_np_points(new_title, img_wh, points))
            return res

        ann.apply_to_figures(to_lines)
        yield img_desc, ann

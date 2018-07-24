# coding: utf-8

import os

from ..utils import json_load, json_dump
from ..project import ProjectMeta
from .dtl_paths import DtlPaths


class DtlHelper:
    def __init__(self):
        self.paths = DtlPaths()
        self.graph = json_load(self.paths.graph_path)

        self.in_project_metas = {}
        self.in_project_dirs = {}
        for pr_dir in self.paths.project_dirs:
            pr_name = os.path.basename(pr_dir)
            self.in_project_metas[pr_name] = ProjectMeta.from_dir(pr_dir)
            self.in_project_dirs[pr_name] = pr_dir

    def save_res_meta(self, meta):
        json_dump(meta.to_py_container(), self.paths.res_meta_path)


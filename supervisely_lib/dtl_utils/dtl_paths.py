# coding: utf-8

import os
import os.path as osp

from ..utils import mkdir
from ..project import ProjectMeta


class DtlPaths:
    task_dir = '/sly_task_data' #'/data/01_export_data' # TODO: for debug

    def __init__(self):
        self._graph_path = osp.join(self.task_dir, 'graph.json')
        self._settings_path = osp.join(self.task_dir, 'task_settings.json')
        self._data_dir = osp.join(self.task_dir, 'data')
        self._results_dir = osp.join(self.task_dir, 'results')
        if os.listdir(self._results_dir):
            raise RuntimeError('Results directory has to be empty')

        self._debug_dir = osp.join(self.task_dir, 'tmp')
        mkdir(self._debug_dir)

        self._project_dirs = [f.path for f in os.scandir(self._data_dir) if f.is_dir()]

        # @TODO: is it needed?
        # if len(self._project_dirs) == 0:
        #     raise RuntimeError('Data folder is empty.')

        self._res_meta_path = ProjectMeta.dir_path_to_fpath(self._results_dir)

    @property
    def graph_path(self):
        return self._graph_path

    @property
    def settings_path(self):
        return self._settings_path

    @property
    def data_dir(self):
        return self._data_dir

    @property
    def results_dir(self):
        return self._results_dir

    @property
    def project_dirs(self):
        return self._project_dirs

    @property
    def debug_dir(self):
        return self._debug_dir

    @property
    def res_meta_path(self):
        return self._res_meta_path

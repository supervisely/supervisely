# coding: utf-8

import os
import os.path as osp


class TaskPaths:
    task_dir = '/sly_task_data'

    def __init__(self, determine_in_project=True):
        self._settings_path = osp.join(self.task_dir, 'task_settings.json')
        self._model_dir = osp.join(self.task_dir, 'model')
        self._results_dir = osp.join(self.task_dir, 'results')
        self._data_dir = osp.join(self.task_dir, 'data')
        self._debug_dir = osp.join(self.task_dir, 'tmp')

        if not determine_in_project:
            self._project_dir = None
        else:
            data_subfolders = [f.path for f in os.scandir(self._data_dir) if f.is_dir()]
            if len(data_subfolders) == 0:
                raise RuntimeError('Data folder is empty.')
            elif len(data_subfolders) > 1:
                raise NotImplementedError('Work with multiple projects is not supported yet.')
            self._project_dir = data_subfolders[0]

    @property
    def settings_path(self):
        return self._settings_path

    @property
    def model_dir(self):
        return self._model_dir

    @property
    def results_dir(self):
        return self._results_dir

    @property
    def project_dir(self):
        return self._project_dir

    @property
    def debug_dir(self):
        return self._debug_dir

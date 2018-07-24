# coding: utf-8

from os import path as osp

from ..utils import mkdir
from ..sly_logger import logger, EventType


class TrainCheckpoints:
    def __init__(self, odir):
        self._common_odir = odir
        self._idx = 0  # the idx doesn't correspond to epoch
        self._dir_from_idx()
        self._last_correct = None

    def _dir_from_idx(self):
        self._subdir = '{:08}'.format(self._idx)
        self._odir = osp.join(self._common_odir, self._subdir)
        mkdir(self._odir)

    def saved(self, is_best, optional_data=None):
        report_best = is_best or (self._idx == 0)
        logger.info('checkpoint', extra={
            'event_type': EventType.CHECKPOINT, 'id': self._idx, 'subdir': self._subdir,
            'best_now': report_best, 'optional': optional_data
        })

        self._last_correct = self._odir
        self._idx += 1
        self._dir_from_idx()

    def get_dir_to_write(self):
        return self._odir

    def get_last_ckpt_dir(self):
        return self._last_correct

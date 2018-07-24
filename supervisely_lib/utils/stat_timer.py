# coding: utf-8
# @TODO: drop from public lib? internal instrument

import os
import time
from threading import Lock

from ..sly_logger import logger


class TinyTimer:
    def __init__(self):
        self.t = time.time()

    def get_sec(self):  # since creation
        now_t = time.time()
        return now_t - self.t


class StatTimer:
    def __init__(self, logging_interval):
        self.logging_interval = logging_interval
        self.lock = Lock()
        self._q_dct = {}

    def add_value(self, name, val_sec):
        if self.logging_interval < 1:
            return  # disabled StatTimer

        self.lock.acquire()

        if name not in self._q_dct:
            self._q_dct[name] = []
        curr_list = self._q_dct[name]
        curr_list.append(val_sec)
        if len(curr_list) >= self.logging_interval:
            msec_per_one = sum(curr_list) / float(len(curr_list)) * 1000.0
            logger.trace('StatTimer {}'.format(name), extra={'msec': msec_per_one})
            curr_list[:] = []  # clear

        self.lock.release()


global_timer = StatTimer(int(os.getenv('STAT_TIMER_LOG_EVERY_RECORDS', '20')))

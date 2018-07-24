# coding: utf-8

import os
import os.path as osp
import shutil
import queue

import supervisely_lib as sly

from . import constants


def create_img_meta_str(img_size_bytes, width, height):
    img_meta = {'size': img_size_bytes, 'width': width, 'height': height}
    res = sly.json_dumps(img_meta)
    return res


def ann_special_fields():
    return 'img_hash', 'img_ext', 'img_size_bytes'


# multithreading
class LogQueue:
    def __init__(self):
        self.q = queue.Queue()  # no limit

    def put_nowait(self, log_line):
        self.q.put_nowait(log_line)

    def _get_batch_nowait(self, batch_limit):
        log_lines = []
        for _ in range(batch_limit):
            try:
                log_line = self.q.get_nowait()
            except queue.Empty:
                break
            log_lines.append(log_line)
        return log_lines

    def get_log_batch_nowait(self):
        res = self._get_batch_nowait(constants.BATCH_SIZE_LOG)
        return res

    def get_log_batch_blocking(self):
        first_log_line = self.q.get(block=True)
        rest_lines = self._get_batch_nowait(constants.BATCH_SIZE_LOG - 1)
        log_lines = [first_log_line] + rest_lines
        return log_lines


class TaskDirCleaner:
    def __init__(self, dir_task):
        self.dir_task = dir_task
        self.marker_fpath = osp.join(self.dir_task, '__do_not_clean.marker')

    def _clean(self):
        for elem in filter(lambda x: 'logs' not in x, os.listdir(self.dir_task)):
            del_path = osp.join(self.dir_task, elem)
            if osp.isfile(del_path):
                os.remove(del_path)
            else:
                shutil.rmtree(del_path)

    def forbid_dir_cleaning(self):
        with open(self.marker_fpath, 'a'):
            os.utime(self.marker_fpath, None)  # touch file

    def allow_cleaning(self):
        if osp.isfile(self.marker_fpath):
            os.remove(self.marker_fpath)

    def clean(self):
        if (not constants.DELETE_TASK_DIR_ON_FINISH) or osp.isfile(self.marker_fpath):
            return
        self._clean()

    def clean_forced(self):
        self._clean()

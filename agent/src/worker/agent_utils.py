# coding: utf-8

import os
import os.path as osp
import shutil
import queue
import json

from worker import constants


def create_img_meta_str(img_size_bytes, width, height):
    img_meta = {'size': img_size_bytes, 'width': width, 'height': height}
    res = json.dumps(img_meta)
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
        res = self._get_batch_nowait(constants.BATCH_SIZE_LOG())
        return res

    def get_log_batch_blocking(self):
        first_log_line = self.q.get(block=True)
        rest_lines = self._get_batch_nowait(constants.BATCH_SIZE_LOG() - 1)
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
        if constants.DELETE_TASK_DIR_ON_FINISH() is False:
            return
        if constants.DELETE_TASK_DIR_ON_FAILURE() is False and osp.isfile(self.marker_fpath):
            return
        self._clean()

    def clean_forced(self):
        self._clean()


#@TODO: remove this method or refactor it in future (dict_name - WTF??)
def get_single_item_or_die(src_dict, key, dict_name):
    results = src_dict.get(key, None)
    if results is None:
        raise ValueError(
            'No values were found for {} in {}. A list with exactly one item is required.'.format(key, dict_name))
    if len(results) != 1:
        raise ValueError(
            'Multiple values ({}) were found for {} in {}. A list with exactly one item is required.'.format(
                len(results), key, dict_name))
    return results[0]


def add_creds_to_git_url(git_url):
    old_str = None
    if 'https://' in git_url:
        old_str = 'https://'
    elif 'http://' in git_url:
        old_str = 'http://'
    res = git_url
    if constants.GIT_LOGIN() is not None and constants.GIT_PASSWORD() is not None:
        res = git_url.replace(old_str, '{}{}:{}@'.format(old_str, constants.GIT_LOGIN(), constants.GIT_PASSWORD()))
        return res
    else:
        return git_url

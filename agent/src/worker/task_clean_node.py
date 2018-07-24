# coding: utf-8

import os
import os.path as osp

import supervisely_lib as sly

from .task_sly import TaskSly
from .agent_utils import TaskDirCleaner
from . import constants


class TaskCleanNode(TaskSly):
    def _rm_checkpoints(self, hashes_to_rm):
        storage = self.data_mgr.storage.nns
        paths_to_rm = [storage.get_storage_path(x['hash']) for x in hashes_to_rm]
        self.logger.info('Will remove specified checkpoints.', extra={'hash_cnt': len(paths_to_rm)})
        removed = [storage.remove_object(st_path) for st_path in paths_to_rm]
        removed = list(filter(None, removed))
        self.logger.info('Specified checkpoints are removed.',
                         extra={'hash_cnt': len(hashes_to_rm), 'removed_cnt': len(removed)})
        # no progress, it shouldn't be slow

    def _clean_tasks_dir(self):
        self.logger.info('Will remove obsolete tasks data.')
        task_dir = constants.AGENT_TASKS_DIR
        task_names = os.listdir(task_dir)
        self.logger.info('Obtained list of subdirs.', extra={'dir_cnt': len(task_names)})

        progress = sly.ProgressCounter(subtask_name='Task dir checking', total_cnt=len(task_names), ext_logger=self.logger)
        for subdir_n in task_names:
            dir_task = osp.join(task_dir, subdir_n)
            try:
                TaskDirCleaner(dir_task).clean_forced()
            except Exception:
                self.logger.warn('Unable to delete task dir.', extra={'dir_task': dir_task}, exc_info=True)
            progress.iter_done_report()

        self.logger.info('Obsolete tasks data has been removed.')

    def _clean_unused_objects(self, name, storage, get_used_hashes_method):
        self.logger.info('Will clean unused {}.'.format(name))
        used_hashes_exts = self.data_mgr.download_object_hashes(get_used_hashes_method)
        persistent_objs = set((storage.get_storage_path(hash_, suffix), suffix)
                              for hash_, suffix in used_hashes_exts)

        self.logger.info('Will scan existing {}.'.format(name))
        found_objs = set(storage.list_objects())

        cnt_extra = {'found_obj_cnt': len(found_objs), 'persist_obj_cnt': len(persistent_objs)}
        self.logger.info('Existing {} are listed.'.format(name), extra=cnt_extra)

        missing_objs = persistent_objs - found_objs
        cnt_extra = {**cnt_extra, 'miss_obj_cnt': len(missing_objs)}
        if len(missing_objs) == 0:
            self.logger.info('There are no missing persistent objects.', extra=cnt_extra)
        else:
            self.logger.warn('Missing persistent objects are found.', extra=cnt_extra)

        temp_objs = found_objs - persistent_objs
        cnt_extra = {**cnt_extra, 'temp_obj_cnt': len(temp_objs)}
        self.logger.info('Temporary objects count is determined.', extra=cnt_extra)

        progress = sly.ProgressCounter(subtask_name='Cleaning {}'.format(name),
                                       total_cnt=len(temp_objs), ext_logger=self.logger)

        removed_cnt = 0
        for st_path, suffix in temp_objs:
            if storage.remove_object(st_path, suffix):
                removed_cnt += 1
            progress.iter_done_report()

        if removed_cnt != len(temp_objs):
            self.logger.warn('Not all objects have been removed.')
        self.logger.info('Unused {} are cleaned.'.format(name),
                         extra={'temp_obj_cnt': len(temp_objs), 'removed_cnt': removed_cnt})

    def _clean_unused_checkpoints(self):
        self._clean_unused_objects('checkpoints', self.data_mgr.storage.nns, 'GetUsedModelList')

    def _clean_unused_images(self):
        self._clean_unused_objects('images', self.data_mgr.storage.images, 'GetUsedImageList')

    def task_main_func(self):
        self.logger.info("CLEAN_NODE_START")

        checkpoints_to_rm = self.info.get('remove_checkpoints', [])
        if len(checkpoints_to_rm) > 0:
            self._rm_checkpoints(checkpoints_to_rm)

        for the_flag, the_method in [
            ('clean_tasks',        self._clean_tasks_dir),
            ('clean_checkpoints',  self._clean_unused_checkpoints),
            ('clean_images',       self._clean_unused_images),
        ]:
            if self.info.get(the_flag, False):
                the_method()

        self.logger.info("CLEAN_NODE_FINISH")

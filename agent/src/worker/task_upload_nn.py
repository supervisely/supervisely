# coding: utf-8

from worker.task_sly import TaskSly


class TaskUploadNN(TaskSly):
    def task_main_func(self):
        if self.info.get('nn_model', None) is None:
            self.logger.critical('TASK_NN_EMPTY')
            raise ValueError('TASK_NN_EMPTY')

        nn_id = self.info['nn_model']['id']
        nn_hash = self.info['nn_model']['hash']

        st_path = self.data_mgr.storage.nns.check_storage_object(nn_hash)
        if st_path is None:
            self.logger.critical("NN_NOT_FOUND", extra={'nn_id': nn_id, 'nn_hash': nn_hash})
            raise RuntimeError("NN_NOT_FOUND")

        self.data_mgr.upload_nn(nn_id, nn_hash)

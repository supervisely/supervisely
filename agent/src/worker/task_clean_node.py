# coding: utf-8

import os
import os.path as osp

import supervisely_lib as sly

from worker.task_sly import TaskSly
from worker.agent_utils import TaskDirCleaner
from worker import constants


class TaskCleanNode(TaskSly):
    def remove_objects(self, storage, spaths):
        removed = []
        for st_path in spaths:
            ret_status = None
            if type(st_path) is str:
                ret_status = storage.remove_object(st_path)
            else:
                ret_status = storage.remove_object(*st_path)
            removed.append(ret_status)
        removed = list(filter(None, removed))
        return removed

    def remove_images(self, storage, proj_structure):
        for key, value in proj_structure.items():
            self.logger.info('Clean dataset', extra={'proj_id': value['proj_id'], 'proj_title': value['proj_title'],
                                                     'ds_id': value['ds_id'], 'ds_title': value['ds_title']})
            #spaths = [spath_ext[0] for spath_ext in value['spaths']]
            removed = self.remove_objects(storage, value['spaths'])
            self.logger.info('Images are removed.', extra={'need_remove_cnt': len(value['spaths']), 'removed_cnt': len(removed)})

    def remove_weights(self, storage, paths):
        removed = self.remove_objects(storage, paths)
        self.logger.info('Weights are removed.', extra={'need_remove_cnt': len(paths), 'removed_cnt': len(removed)})

    def get_dataset_images_hashes(self, dataset_id):
        image_array = self.api.simple_request('GetDatasetImages', sly.api_proto.ImageArray, sly.api_proto.Id(id=dataset_id))
        img_hashes = []

        for batch_img_ids in sly.batched(list(image_array.images), constants.BATCH_SIZE_GET_IMAGES_INFO()):
            images_info_proto = self.api.simple_request('GetImagesInfo', sly.api_proto.ImagesInfo,
                                                        sly.api_proto.ImageArray(images=batch_img_ids))
            img_hashes.extend([(info.hash, info.ext) for info in images_info_proto.infos])
        return img_hashes

    def list_weights_to_remove(self, storage, action, input_weights_hashes):
        if action == "delete_selected":
            return [storage.get_storage_path(hash) for hash in input_weights_hashes]

        if action == "delete_all_except_selected":
            selected_paths = set([storage.get_storage_path(hash) for hash in input_weights_hashes])
            all_paths = set([path_and_suffix[0] for path_and_suffix in storage.list_objects()])
            paths_to_remove = list(all_paths.difference(selected_paths)) # all_paths - selected_paths
            return paths_to_remove

        raise ValueError("Unknown cleanup action", extra={'action': action})

    def list_images_to_remove(self, storage, action, projects):
        img_spaths = {}
        for project in projects:
            for dataset in project["datasets"]:
                ds_id = dataset['id']
                img_spaths[ds_id] = {
                    'proj_id': project['id'],
                    'proj_title': project['title'],
                    'ds_id': ds_id,
                    'ds_title': dataset['title'],
                    'spaths': []
                }
                temp_spaths = [storage.get_storage_path(hash_ext[0], hash_ext[1])
                               for hash_ext in self.get_dataset_images_hashes(ds_id)]
                img_spaths[ds_id]['spaths'] = temp_spaths

        if action == "delete_selected":
            return img_spaths

        if action == "delete_all_except_selected":
            selected_paths=[]
            for key, value in img_spaths.items():
                selected_paths.extend(value['spaths'])
            all_paths = set(storage.list_objects())
            paths_to_remove = all_paths.difference(set(selected_paths)) # all_paths - selected_paths

            result = {}
            result[0] = {'proj_id': -1, 'proj_title': "all cache images", 'ds_id': -1, 'ds_title': "all cache images"}
            result[0]['spaths'] = paths_to_remove
            return result

        raise ValueError("Unknown cleanup action", extra={'action': action})

    def clean_tasks_dir(self):
        self.logger.info('Will remove temporary tasks data.')
        task_dir = constants.AGENT_TASKS_DIR()
        task_names = os.listdir(task_dir)
        for subdir_n in task_names:
            dir_task = osp.join(task_dir, subdir_n)
            TaskDirCleaner(dir_task).clean_forced()
        self.logger.info('Temporary tasks data has been removed.')

    def task_main_func(self):
        self.logger.info("CLEAN_NODE_START")

        if self.info['action'] == "remove_tasks_data":
            self.clean_tasks_dir()
        else:
            if 'projects' in self.info:
                img_storage = self.data_mgr.storage.images
                proj_structure = self.list_images_to_remove(img_storage, self.info['action'], self.info['projects'])
                self.remove_images(img_storage, proj_structure)

            if 'weights' in self.info:
                nns_storage = self.data_mgr.storage.nns
                weights_to_rm = self.list_weights_to_remove(nns_storage, self.info['action'], self.info['weights'])
                self.remove_weights(nns_storage, weights_to_rm)

        self.logger.info("CLEAN_NODE_FINISH")

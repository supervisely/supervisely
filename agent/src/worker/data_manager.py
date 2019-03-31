# coding: utf-8

import os
import supervisely_lib as sly
from worker.agent_storage import AgentStorage
from worker.fs_storages import EmptyStorage
from worker import constants
from collections import defaultdict


class DataManager(object):
    def __init__(self, ext_logger, api, public_api, public_api_context=None):
        self.logger = ext_logger
        self.api = api
        self.public_api = public_api
        if public_api_context is not None:
            self.public_api_context = public_api_context
            self.workspace_id = self.public_api_context['workspace']['id']
        self.storage = AgentStorage()

    def has_nn_storage(self):
        return not isinstance(self.storage.nns, EmptyStorage)

    def has_images_storage(self):
        return not isinstance(self.storage.images, EmptyStorage)

    def download_nn(self, name, parent_dir):
        self.logger.info("DOWNLOAD_MODEL", extra={'nn_title': name})
        model_info = self.public_api.model.get_info_by_name(self.workspace_id, name)
        nn_hash = model_info.hash
        model_dir = os.path.join(parent_dir, name)

        if self.storage.nns.read_object(nn_hash, model_dir):
            self.logger.info('NN has been copied from local storage.')
            return

        model_in_mb = int(float(model_info.size) / 1024 / 1024)
        progress = sly.Progress('Download NN: {!r}'.format(name), model_in_mb)

        self.public_api.model.download_to_dir(self.workspace_id, name, parent_dir, progress.iter_done_report)
        self.logger.info('NN has been downloaded from server.')

        if self.has_nn_storage():
            self.storage.nns.write_object(model_dir, nn_hash)

    def _split_images_by_cache(self, images):
        images_to_download = []
        images_in_cache = []
        images_cache_paths = []
        for image in images:
            cache_path = self.storage.images.check_storage_object(image.hash, image.ext)
            if cache_path is None:
                images_to_download.append(image)
            else:
                images_in_cache.append(image)
                images_cache_paths.append(cache_path)
        return images_to_download, images_in_cache, images_cache_paths

    def download_project(self, parent_dir, name, datasets_whitelist=None):
        self.logger.info("DOWNLOAD_PROJECT", extra={'title': name})
        #@TODO: reimplement and use path without splitting
        project_fs = sly.Project(os.path.join(parent_dir, name), sly.OpenMode.CREATE)
        project_id = self.public_api.project.get_info_by_name(self.workspace_id, name).id
        meta = sly.ProjectMeta.from_json(self.public_api.project.get_meta(project_id))
        project_fs.set_meta(meta)
        for dataset_info in self.public_api.dataset.get_list(project_id):
            dataset_name = dataset_info.name
            dataset_id = dataset_info.id
            need_download = True
            if datasets_whitelist is not None and dataset_name not in datasets_whitelist:
                need_download = False
            if need_download is True:
                dataset = project_fs.create_dataset(dataset_name)
                self.download_dataset(dataset, dataset_id)

    def download_dataset(self, dataset, dataset_id):
        images = self.public_api.image.get_list(dataset_id)
        progress = sly.Progress('Download dataset {!r}: images'.format(dataset.name), len(images), self.logger)

        images_to_download = images
        if self.has_images_storage():
            images_to_download, images_in_cache, images_cache_paths = self._split_images_by_cache(images)
            # copy images from cache to task folder
            for img_info, img_cache_path in zip(images_in_cache, images_cache_paths):
                dataset.add_item_file(img_info.name, img_cache_path)
                progress.iter_done_report()

        # download images from server
        img_ids = []
        img_paths = []
        for img_info in images_to_download:
            img_ids.append(img_info.id)
            # TODO download to a temp file and use dataset api to add the image to the dataset.
            img_paths.append(dataset.deprecated_make_img_path(img_info.name, img_info.ext))

        self.public_api.image.download_batch(img_ids, img_paths, progress.iter_done_report)
        for img_info, img_path in zip(images_to_download, img_paths):
            dataset.add_item_file(img_info.name, img_path)

        if self.has_images_storage():
            progress = sly.Progress('Download dataset {!r}: cache images'.format(dataset.name), len(img_paths), self.logger)
            img_hashes = [img_info.hash for img_info in images_to_download]
            self.storage.images.write_objects(img_paths, img_hashes, progress.iter_done_report)

        # download annotations from server
        img_id_to_name = {image.id: image.name for image in images}
        progress_ann = sly.Progress('Download dataset {!r}: annotations'.format(dataset.name), len(images), self.logger)
        anns = self.public_api.annotation.get_list(dataset_id, progress_cb=progress_ann.iters_done_report)
        for ann in anns:
            img_name = img_id_to_name[ann.image_id]
            dataset.set_ann_dict(img_name, ann.annotation)

    #@TODO: remove legacy stuff
    # @TODO: reimplement and use path without splitting
    def upload_project(self, parent_dir, project_name, new_title, legacy=False, add_to_existing=False):
        # @TODO: reimplement and use path without splitting
        if legacy is False:
            project = sly.Project(os.path.join(parent_dir, project_name), sly.OpenMode.READ)
        else:
            project = sly.Project(parent_dir, sly.OpenMode.READ)

        if add_to_existing is True:
            project_id = self.public_api.project.get_info_by_name(self.workspace_id, project_name).id
            meta_json = self.public_api.project.get_meta(project_id)
            existing_meta = sly.ProjectMeta.from_json(meta_json)
            project.set_meta(sly.ProjectMeta.merge_list([project.meta, existing_meta]))
        else:
            new_project_name = self.public_api.project.get_free_name(self.workspace_id, new_title)
            project_id = self.public_api.project.create(self.workspace_id, new_project_name).id

        self.public_api.project.update_meta(project_id, project.meta.to_json())
        for dataset in project:
            ds_name = dataset.name
            if add_to_existing is True:
                ds_name = self.public_api.dataset.get_free_name(project_id, ds_name)
            dataset_id = self.public_api.dataset.create(project_id, ds_name).id
            self.upload_dataset(dataset, dataset_id)

        self.logger.info('PROJECT_CREATED',extra={'event_type': sly.EventType.PROJECT_CREATED, 'project_id': project_id})

    def upload_dataset(self, dataset, dataset_id):
        progress = None
        items_count = len(dataset)
        hash_to_img_paths = defaultdict(list)
        hash_to_ann_paths = defaultdict(list)
        hash_to_item_names = defaultdict(list)
        for item_name in dataset:
            item_paths = dataset.get_item_paths(item_name)
            img_hash = sly.fs.get_file_hash(item_paths.img_path)
            hash_to_img_paths[img_hash].append(item_paths.img_path)
            hash_to_ann_paths[img_hash].append(item_paths.ann_path)
            hash_to_item_names[img_hash].append(item_name)
            if self.has_images_storage():
                if progress is None:
                    progress = sly.Progress('Dataset {!r}: upload cache images'.format(dataset.name), items_count, self.logger)
                self.storage.images.write_object(item_paths.img_path, img_hash)
                progress.iter_done_report()

        progress_img = sly.Progress('Dataset {!r}: upload images'.format(dataset.name), items_count, self.logger)
        progress_ann = sly.Progress('Dataset {!r}: upload annotations'.format(dataset.name), items_count, self.logger)

        def add_images_annotations(hashes, pb_img_cb, pb_ann_cb):
            names = [name for hash in hashes for name in hash_to_item_names[hash]]
            unrolled_hashes = [hash for hash in hashes for _ in range(len(hash_to_item_names[hash]))]
            ann_paths = [path for hash in hashes for path in hash_to_ann_paths[hash]]
            remote_ids = self.public_api.image.add_batch(dataset_id, names, unrolled_hashes, pb_img_cb)
            self.public_api.annotation.add_batch(remote_ids, ann_paths, pb_ann_cb)

        # add already uploaded images + attach annotations
        remote_hashes = self.public_api.image.check_existing_hashes(list(hash_to_img_paths.keys()))
        add_images_annotations(remote_hashes, progress_img.iter_done_report, progress_ann.iter_done_report)

        # upload new images + add annotations
        new_hashes = list(set(hash_to_img_paths.keys()) - set(remote_hashes))
        img_paths = [path for hash in new_hashes for path in hash_to_img_paths[hash]]
        self.public_api.image.upload_batch(img_paths, progress_img.iter_done_report)
        add_images_annotations(new_hashes, None, progress_ann.iter_done_report)

    def upload_archive(self, task_id, dir_to_archive, archive_name):
        self.logger.info("PACK_TO_ARCHIVE ...")
        local_tar_path = os.path.join(constants.AGENT_TMP_DIR(), sly.rand_str(30) + '.tar')
        sly.fs.archive_directory(dir_to_archive, local_tar_path)

        size_mb = sly.fs.get_file_size(local_tar_path) / 1024.0 / 1024
        progress = sly.Progress("Upload archive", size_mb, ext_logger=self.logger)
        try:
            self.public_api.task.upload_dtl_archive(task_id, local_tar_path, progress.set_current_value)
        finally:
            sly.fs.silent_remove(local_tar_path)

        self.logger.info('ARCHIVE_UPLOADED', extra={'archive_name': archive_name})

    def download_import_files(self, task_id, data_dir):
        import_struct = self.api.simple_request('GetImportStructure', sly.api_proto.ListFiles,
                                                sly.api_proto.Id(id=task_id))
        progress = sly.Progress('Downloading', len(import_struct.files), self.logger)

        def close_fh(fh):
            fpath = fh.file_path
            if fh.close_and_check():
                progress.iter_done_report()
            else:
                self.logger.warning('file was skipped while downloading', extra={'file_path': fpath})

        file_handler = None
        for chunk in self.api.get_stream_with_data('GetImportFiles',
                                                   sly.api_proto.ChunkFile,
                                                   sly.api_proto.ImportRequest(task_id=task_id,
                                                                               files=import_struct.files)):
            new_fpath = chunk.file.path
            if new_fpath:  # non-empty
                if file_handler is not None:
                    close_fh(file_handler)
                real_fpath = os.path.join(data_dir, new_fpath.lstrip('/'))
                self.logger.trace('download import file', extra={'file_path': real_fpath})
                file_handler = sly.ChunkedFileWriter(file_path=real_fpath)

            file_handler.write(chunk.chunk)

        close_fh(file_handler)

    def upload_nn(self, nn_id, nn_hash):
        local_service_log = {'nn_id': nn_id, 'nn_hash': nn_hash}

        storage_nn_dir = self.storage.nns.check_storage_object(nn_hash)
        if storage_nn_dir is None:
            self.logger.critical('NN_NOT_FOUND', extra=local_service_log)
        local_tar_path = os.path.join(constants.AGENT_TMP_DIR(), sly.rand_str(30) + '.tar')
        sly.fs.archive_directory(storage_nn_dir, local_tar_path)

        size_mb = sly.fs.get_file_size(local_tar_path) / 1024.0 / 1024
        progress = sly.Progress("Upload NN weights", size_mb, ext_logger=self.logger)
        try:
            self.public_api.model.upload(nn_hash, local_tar_path, progress.set_current_value)
        finally:
            sly.fs.silent_remove(local_tar_path)

        self.logger.info('ARCHIVE_UPLOADED')
        self.logger.info('NN_UPLOADED', extra=local_service_log)

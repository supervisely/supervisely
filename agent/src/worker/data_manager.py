# coding: utf-8

import os
import supervisely_lib as sly
from supervisely_lib._utils import batched
from worker.agent_storage import AgentStorage
from worker.fs_storages import EmptyStorage
from worker import constants


def _maybe_append_image_extension(name, ext):
    name_split = os.path.splitext(name)
    if name_split[1] == '':
        normalized_ext = ('.' + ext).replace('..', '.')
        result = name + normalized_ext
        sly.image.validate_ext(result)
    else:
        result = name
    return result


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

        model_in_mb = int(float(model_info.size) / 1024 / 1024 + 1)
        progress = sly.Progress('Download NN: {!r}'.format(name), model_in_mb, self.logger)

        self.public_api.model.download_to_dir(self.workspace_id, name, parent_dir, progress.iters_done_report)
        self.logger.info('NN has been downloaded from server.')

        if self.has_nn_storage():
            self.storage.nns.write_object(model_dir, nn_hash)

    def _split_images_by_cache(self, images):
        images_to_download = []
        images_in_cache = []
        images_cache_paths = []
        for image in images:
            _, effective_ext = os.path.splitext(image.name)
            if len(effective_ext) == 0:
                # Fallback for the old format where we were cutting off extensions from image names.
                effective_ext = image.ext
            cache_path = self.storage.images.check_storage_object(image.hash, effective_ext)
            if cache_path is None:
                images_to_download.append(image)
            else:
                images_in_cache.append(image)
                images_cache_paths.append(cache_path)
        return images_to_download, images_in_cache, images_cache_paths

    def download_project(self, parent_dir, name, datasets_whitelist=None):
        self.logger.info("DOWNLOAD_PROJECT", extra={'title': name})
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
        progress_anns = sly.Progress('Dataset {!r}: download annotations'.format(dataset.name), len(images), self.logger)

        images_to_download = images

        # copy images from cache to task folder and download corresponding annotations
        if self.has_images_storage():
            images_to_download, images_in_cache, images_cache_paths = self._split_images_by_cache(images)
            self.logger.info('Dataset {!r}'.format(dataset.name), extra={'total_images': len(images),
                                                                         'images_in_cache': len(images_in_cache),
                                                                         'images_to_download': len(images_to_download)})
            if len(images_to_download) + len(images_in_cache) != len(images):
                raise RuntimeError("Error with images cache during download. Please contact support.")

            if len(images_in_cache) > 0:
                progress_imgs_cache = sly.Progress(
                    'Dataset {!r}: restoring images from cache'.format(dataset.name), len(images_in_cache), self.logger)
                img_cache_ids = [img_info.id for img_info in images_in_cache]
                ann_info_list = self.public_api.annotation.download_batch(
                    dataset_id, img_cache_ids, progress_anns.iters_done_report)
                img_name_to_ann = {ann.image_id: ann.annotation for ann in ann_info_list}
                for img_info, img_cache_path in zip(images_in_cache, images_cache_paths):
                    item_name = _maybe_append_image_extension(img_info.name, img_info.ext)
                    dataset.add_item_file(item_name, img_cache_path, img_name_to_ann[img_info.id], _validate_item=False,
                                          _use_hardlink=True)
                    progress_imgs_cache.iter_done_report()

        # download images from server
        if len(images_to_download) > 0:
            progress_imgs_download = sly.Progress(
                'Dataset {!r}: download images'.format(dataset.name), len(images_to_download), self.logger)
            #prepare lists for api methods
            img_ids = []
            img_paths = []
            for img_info in images_to_download:
                img_ids.append(img_info.id)
                # TODO download to a temp file and use dataset api to add the image to the dataset.
                img_paths.append(
                    os.path.join(dataset.img_dir, _maybe_append_image_extension(img_info.name, img_info.ext)))

            # download annotations
            ann_info_list = self.public_api.annotation.download_batch(
                dataset_id, img_ids, progress_anns.iters_done_report)
            img_name_to_ann = {ann.image_id: ann.annotation for ann in ann_info_list}
            self.public_api.image.download_paths(
                dataset_id, img_ids, img_paths, progress_imgs_download.iters_done_report)
            for img_info, img_path in zip(images_to_download, img_paths):
                dataset.add_item_file(img_info.name, img_path, img_name_to_ann[img_info.id])

            if self.has_images_storage():
                progress_cache = sly.Progress(
                    'Dataset {!r}: cache images'.format(dataset.name), len(img_paths), self.logger)
                img_hashes = [img_info.hash for img_info in images_to_download]
                self.storage.images.write_objects(img_paths, img_hashes, progress_cache.iter_done_report)

    # @TODO: remove legacy stuff
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
        progress_cache = None
        items_count = len(dataset)

        item_names = []
        img_paths = []
        ann_paths = []
        for item_name in dataset:
            item_names.append(item_name)
            item_paths = dataset.get_item_paths(item_name)
            img_paths.append(item_paths.img_path)
            ann_paths.append(item_paths.ann_path)

            if self.has_images_storage():
                if progress_cache is None:
                    progress_cache = sly.Progress('Dataset {!r}: cache images'.format(dataset.name), items_count, self.logger)

                img_hash = sly.fs.get_file_hash(item_paths.img_path)
                self.storage.images.write_object(item_paths.img_path, img_hash)
                progress_cache.iter_done_report()

        progress = sly.Progress('Dataset {!r}: upload images'.format(dataset.name), items_count, self.logger)
        image_infos = self.public_api.image.upload_paths(dataset_id, item_names, img_paths, progress.iters_done_report)

        progress = sly.Progress('Dataset {!r}: upload annotations'.format(dataset.name), items_count, self.logger)
        self.public_api.annotation.upload_paths([info.id for info in image_infos], ann_paths, progress.iters_done_report)

    def upload_tar_file(self, task_id, file_path):
        size_mb = sly.fs.get_file_size(file_path) / 1024.0 / 1024
        progress = sly.Progress("Uploading file", size_mb, ext_logger=self.logger)
        self.public_api.task.upload_dtl_archive(task_id, file_path, progress.set_current_value)

    def upload_archive(self, task_id, dir_to_archive, archive_name):
        self.logger.info("PACK_TO_ARCHIVE ...")
        archive_name = archive_name if len(archive_name) > 0 else sly.rand_str(30)
        local_tar_path = os.path.join(constants.AGENT_TMP_DIR(), archive_name + '.tar')
        sly.fs.archive_directory(dir_to_archive, local_tar_path)

        try:
            self.upload_tar_file(task_id, local_tar_path)
        finally:
            sly.fs.silent_remove(local_tar_path)

        self.logger.info('ARCHIVE_UPLOADED', extra={'archive_name': archive_name})

    def download_import_files(self, task_id, data_dir):
        import_struct = self.api.simple_request('GetImportStructure', sly.api_proto.ListFiles,
                                                sly.api_proto.Id(id=task_id))
        progress = sly.Progress('Downloading', len(import_struct.files), self.logger)

        def maybe_close_fh(fh, pbar, downloaded_paths: set):
            if fh is not None:
                if fh.close_and_check():
                    pbar.iter_done_report()
                    downloaded_paths.add(fh.file_path)
                else:
                    self.logger.warning('file was skipped while downloading', extra={'file_path': fh.file_path})

        files_to_download = list(import_struct.files)
        for batch in batched(files_to_download):
            # Store the file names that have been already downloaded from this batch
            # to avoid rewriting them on transmission retries if connection issues arise.
            downloaded_from_batch = set()
            file_handler = None
            for chunk in self.api.get_stream_with_data('GetImportFiles',
                                                       sly.api_proto.ChunkFile,
                                                       sly.api_proto.ImportRequest(task_id=task_id,
                                                                                   files=batch)):
                new_fpath = chunk.file.path
                if new_fpath:  # non-empty
                    maybe_close_fh(file_handler, progress, downloaded_from_batch)
                    real_fpath = os.path.join(data_dir, new_fpath.lstrip('/'))
                    if real_fpath in downloaded_from_batch:
                        file_handler = None
                    else:
                        self.logger.trace('download import file', extra={'file_path': real_fpath})
                        file_handler = sly.ChunkedFileWriter(file_path=real_fpath)

                if file_handler is not None:
                    file_handler.write(chunk.chunk)

            maybe_close_fh(file_handler, progress, downloaded_from_batch)

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

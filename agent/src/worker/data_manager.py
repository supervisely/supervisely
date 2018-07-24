# coding: utf-8

import os
import os.path as osp
import tarfile

import supervisely_lib as sly
from supervisely_lib.worker_api import ChunkedFileWriter, ChunkedFileReader
import supervisely_lib.worker_proto as api_proto

from .agent_storage import AgentStorage
from .agent_utils import create_img_meta_str, ann_special_fields
from . import constants


class DataManager(object):
    def __init__(self, ext_logger, api):
        self.logger = ext_logger
        self.api = api
        self.storage = AgentStorage()

    def _get_images_from_agent_storage(self, pr_writer, image_id_to_ds, img_infos):
        progress = sly.ProgressCounter('Copying local images', len(img_infos), ext_logger=self.logger)

        dst_paths_hashes = [(pr_writer.get_img_path(image_id_to_ds[info.id], info.title, info.ext), info.hash)
                            for info in img_infos]
        written_img_paths = set(self.storage.images.read_objects(dst_paths_hashes, progress))
        written_hashes = [p_h[1] for p_h in dst_paths_hashes if p_h[0] in written_img_paths]
        return written_hashes

    def _write_images_to_agent_storage(self, src_paths, hashes):
        self.logger.info("WRITE_IMAGES_TO_LOCAL_CACHE", extra={'cnt_images': len(src_paths)})
        progress = sly.ProgressCounter("Add images to local storage", len(src_paths), ext_logger=self.logger)
        self.storage.images.write_objects(zip(src_paths, hashes), progress)

    def _download_images_from_remote(self, pr_writer, image_id_to_ds, img_infos):
        if len(img_infos) == 0:
            return

        infos_with_paths = [(info, pr_writer.get_img_path(image_id_to_ds[info.id], info.title, info.ext))
                            for info in img_infos]
        hash2path = {x[0].hash: x[1] for x in infos_with_paths}  # for unique hashes
        unique_hashes = list(hash2path.keys())

        ready_paths = []
        ready_hashes = []
        progress = sly.ProgressCounter('Download remote images', len(unique_hashes), ext_logger=self.logger)

        def close_fh(fh):
            fpath = fh.file_path
            if fh.close_and_check():
                ready_paths.append(fpath)
                ready_hashes.append(img_hash)
                progress.iter_done_report()
            else:
                self.logger.warning('file was skipped while downloading',
                                    extra={'img_path': fpath, 'img_hash': img_hash})

        # download by unique hashes
        for batch_img_hashes in sly.batched(unique_hashes, constants.BATCH_SIZE_DOWNLOAD_IMAGES):
            file_handler = None
            img_hash = None
            for chunk in self.api.get_stream_with_data('DownloadImages',
                                                       api_proto.ChunkImage,
                                                       api_proto.ImagesHashes(images_hashes=batch_img_hashes)):
                if chunk.image.hash:  # non-empty hash means beginning of new image
                    if file_handler is not None:
                        close_fh(file_handler)
                    img_hash = chunk.image.hash
                    self.logger.trace('download_images', extra={'img_hash': img_hash})
                    dst_fpath = hash2path[img_hash]
                    file_handler = ChunkedFileWriter(file_path=dst_fpath)

                file_handler.write(chunk.chunk)

            close_fh(file_handler)  # must be not None

        # process non-unique hashes
        for info, dst_path in infos_with_paths:
            origin_path = hash2path[info.hash]
            if (origin_path != dst_path) and osp.isfile(origin_path):
                sly.ensure_base_path(dst_path)
                sly.copy_file(origin_path, dst_path)

        self._write_images_to_agent_storage(ready_paths, ready_hashes)

    def _download_images(self, pr_writer, image_id_to_ds):
        # get full info (hash, title, ext)
        img_infos = []
        for batch_img_ids in sly.batched(list(image_id_to_ds.keys()), constants.BATCH_SIZE_GET_IMAGES_INFO):
            images_info_proto = self.api.simple_request('GetImagesInfo',
                                                        api_proto.ImagesInfo,
                                                        api_proto.ImageArray(images=batch_img_ids))
            img_infos.extend(images_info_proto.infos)

        written_hashes = set(self._get_images_from_agent_storage(pr_writer, image_id_to_ds, img_infos))
        img_infos_to_download = [x for x in img_infos if x.hash not in written_hashes]
        self._download_images_from_remote(pr_writer, image_id_to_ds, img_infos_to_download)

    def _download_annotations(self, pr_writer, image_id_to_ds):
        progress = sly.ProgressCounter('Download annotations', len(image_id_to_ds), ext_logger=self.logger)

        for batch_img_ids in sly.batched(list(image_id_to_ds.keys()), constants.BATCH_SIZE_DOWNLOAD_ANNOTATIONS):
            for chunk in self.api.get_stream_with_data('DownloadAnnotations',
                                                       api_proto.ChunkImage,
                                                       api_proto.ImageArray(images=batch_img_ids)):
                img_id = chunk.image.id
                ds_name = image_id_to_ds[img_id]
                self.logger.trace('download_annotations', extra={'img_id': img_id})
                fh = ChunkedFileWriter(file_path=pr_writer.get_ann_path(ds_name, chunk.image.title))
                fh.write(chunk.chunk)
                progress.iter_done_report()
                if not fh.close_and_check():
                    self.logger.warning('ann was skipped while downloading', extra={'img_id': img_id,
                                                                                    'ann_path': fh.file_path})

    def download_project(self, parent_dir, project, datasets, download_images=True):
        project_info = self.api.simple_request('GetProjectMeta', api_proto.Project, api_proto.Id(id=project.id))
        pr_writer = sly.ProjectWriterFS(parent_dir, project_info.title)

        pr_meta = sly.ProjectMeta(sly.json_loads(project_info.meta))
        pr_writer.write_meta(pr_meta)

        image_id_to_ds = {}
        for dataset in datasets:
            image_array = self.api.simple_request('GetDatasetImages', api_proto.ImageArray, api_proto.Id(id=dataset.id))
            image_id_to_ds.update({img_id: dataset.title for img_id in image_array.images})

        if download_images is True:
            self._download_images(pr_writer, image_id_to_ds)
        self._download_annotations(pr_writer, image_id_to_ds)

    # required infos: list of api_proto.Image(hash=, ext=, meta=)
    def upload_images_to_remote(self, fpaths, infos):
        def chunk_generator():
            progress = sly.ProgressCounter('Upload images', len(fpaths), ext_logger=self.logger)

            for batch_paths_infos in sly.batched(list(zip(fpaths, infos)), constants.BATCH_SIZE_UPLOAD_IMAGES):
                for fpath, proto_img_info in batch_paths_infos:
                    self.logger.trace('image upload start', extra={'img_path': fpath})

                    freader = ChunkedFileReader(fpath, constants.NETW_CHUNK_SIZE)
                    for chunk_bytes in freader:
                        current_chunk = api_proto.Chunk(buffer=chunk_bytes, total_size=freader.file_size)
                        yield api_proto.ChunkImage(chunk=current_chunk, image=proto_img_info)

                    self.logger.trace('image uploaded', extra={'img_path': fpath})
                    progress.iter_done_report()

        self.api.put_stream_with_data('UploadImages', api_proto.Empty, chunk_generator())

    def _upload_annotations_to_remote(self, project_id, img_ids, img_names, ann_paths):
        def chunk_generator():
            progress = sly.ProgressCounter('Upload annotations', len(img_ids), ext_logger=self.logger)

            for batch_some in sly.batched(list(zip(img_ids, img_names, ann_paths)),
                                          constants.BATCH_SIZE_UPLOAD_ANNOTATIONS):
                for img_id, img_name, ann_path in batch_some:
                    proto_img = api_proto.Image(id=img_id, title=img_name, project_id=project_id)
                    freader = ChunkedFileReader(ann_path, constants.NETW_CHUNK_SIZE)
                    for chunk_bytes in freader:
                        current_chunk = api_proto.Chunk(buffer=chunk_bytes, total_size=freader.file_size)
                        yield api_proto.ChunkImage(chunk=current_chunk, image=proto_img)
                    self.logger.trace('annotation is uploaded', extra={'img_name': img_name, 'img_path': ann_path})
                    progress.iter_done_report()

        self.api.put_stream_with_data('UploadAnnotations', api_proto.ImageArray, chunk_generator())

    def _create_project(self, project_name, project_meta):
        remote_name = project_name
        for _ in range(3):
            project = self.api.simple_request('CreateProject', api_proto.Id,
                                              api_proto.Project(title=remote_name, meta=project_meta))
            if project.id != 0:  # created
                return project.id, remote_name
            remote_name = "{}_{}".format(project_name, sly.generate_random_string(5))

        raise RuntimeError('Unable to create project with random suffix.')

    def _create_dataset(self, project_id, dataset_name):
        remote_name = dataset_name
        for _ in range(3):
            dataset = self.api.simple_request('CreateDataset',
                                              api_proto.Id,
                                              api_proto.ProjectDataset(project=api_proto.Project(id=project_id),
                                                                       dataset=api_proto.Dataset(title=remote_name)))
            if dataset.id != 0:  # created
                return dataset.id, remote_name
            remote_name = '{}_{}'.format(dataset_name, sly.generate_random_string(5))

        raise RuntimeError('Unable to create dataset with random suffix.')

    @classmethod
    def _construct_project_items_to_upload(cls, project_fs, project_id, ds_names_to_ids):
        project_items = list(project_fs)
        for it in project_items:

            ann = sly.json_load(it.ann_path)
            img_w, img_h = sly.Annotation.get_image_size_wh(ann)

            # get image hash & ext
            if it.img_path and sly.file_exists(it.img_path):
                it.ia_data['image_hash'] = sly.get_image_hash(it.img_path)
                # image_ext is already determined in project_fs
                img_sizeb = sly.get_file_size(it.img_path)
            else:
                for spec_field in ann_special_fields():
                    if spec_field not in ann:
                        raise RuntimeError('Missing spec field in annotation: {}'.format(spec_field))
                it.ia_data['image_hash'] = ann['img_hash']
                it.ia_data['image_ext'] = ann['img_ext']
                img_sizeb = ann['img_size_bytes']

            # construct image info
            img_meta_str = create_img_meta_str(img_sizeb, width=img_w, height=img_h)
            img_proto_info = api_proto.Image(hash=it.ia_data['image_hash'],
                                             title=it.image_name,
                                             ext=it.ia_data['image_ext'],
                                             dataset_id=ds_names_to_ids[it.ds_name],
                                             project_id=project_id,
                                             meta=img_meta_str)
            it.ia_data['img_proto_info'] = img_proto_info

        return project_items

    def upload_project(self, dir_results, pr_name, no_image_files):
        self.logger.info("upload_result_project started")
        root_path, project_name = sly.ProjectFS.split_dir_project(dir_results)
        project_fs = sly.ProjectFS.from_disk(root_path, project_name, by_annotations=True)

        project_meta = sly.ProjectMeta.from_dir(dir_results)
        project_meta_str = project_meta.to_json_str()
        project_id, remote_pr_name = self._create_project(pr_name, project_meta_str)

        ds_name_to_id = {}
        for local_ds_name in project_fs.pr_structure.datasets.keys():
            ds_id, remote_ds_name = self._create_dataset(project_id, local_ds_name)
            ds_name_to_id[local_ds_name] = ds_id

        project_items = self._construct_project_items_to_upload(project_fs, project_id, ds_name_to_id)
        if len(project_items) == 0:
            raise RuntimeError('Empty result project')

        # upload images at first
        if not no_image_files:
            all_img_paths = [it.img_path for it in project_items]
            all_img_hashes = [it.ia_data['image_hash'] for it in project_items]
            self._write_images_to_agent_storage(all_img_paths, all_img_hashes)

            if constants.UPLOAD_RESULT_IMAGES:
                remote_images = self.api.simple_request('FindImagesExist', api_proto.ImagesHashes,
                                                        api_proto.ImagesHashes(images_hashes=all_img_hashes))

                img_hashes_to_upload = set(all_img_hashes) - set(remote_images.images_hashes)
                to_upload_imgs = list(filter(lambda x: x.ia_data['image_hash'] in img_hashes_to_upload, project_items))

                img_paths = [it.img_path for it in to_upload_imgs]
                self.upload_images_to_remote(
                    fpaths=img_paths,
                    infos=[it.ia_data['img_proto_info'] for it in to_upload_imgs]
                )

        # add images to project
        obtained_img_ids = self.api.simple_request('AddImages', api_proto.ImageArray, api_proto.ImagesInfo(
            infos=[it.ia_data['img_proto_info'] for it in project_items]
        ))

        # and upload anns
        self._upload_annotations_to_remote(project_id=project_id,
                                           img_ids=obtained_img_ids.images,
                                           img_names=[it.image_name for it in project_items],
                                           ann_paths=[it.ann_path for it in project_items])

        self.api.simple_request('SetProjectFinished', api_proto.Empty, api_proto.Id(id=project_id))
        return project_id

    def download_import_files(self, task_id, data_dir):
        import_struct = self.api.simple_request('GetImportStructure', api_proto.ListFiles, api_proto.Id(id=task_id))
        progress = sly.ProgressCounter(subtask_name='Downloading',
                                       total_cnt=len(import_struct.files),
                                       ext_logger=self.logger,
                                       report_limit=int(len(import_struct.files) / 10))

        def close_fh(fh):
            fpath = fh.file_path
            if fh.close_and_check():
                progress.iter_done_report()
            else:
                self.logger.warning('file was skipped while downloading', extra={'file_path': fpath})

        file_handler = None
        for chunk in self.api.get_stream_with_data('GetImportFiles',
                                                   api_proto.ChunkFile,
                                                   api_proto.ImportRequest(task_id=task_id, files=import_struct.files)):
            new_fpath = chunk.file.path
            if new_fpath:  # non-empty
                if file_handler is not None:
                    close_fh(file_handler)
                real_fpath = osp.join(data_dir, new_fpath.lstrip('/'))
                self.logger.trace('download import file', extra={'file_path': real_fpath})
                file_handler = ChunkedFileWriter(file_path=real_fpath)

            file_handler.write(chunk.chunk)

        close_fh(file_handler)

    def download_nn(self, nn_id, nn_hash, model_dir):
        if self.storage.nns.read_object(nn_hash, model_dir):
            self.logger.info('NN has been copied from local storage.')
            return

        nn_archive_path = os.path.join(constants.AGENT_TMP_DIR, sly.generate_random_string(30) + '.tar')
        fh = None
        progress = None
        for nn_chunk in self.api.get_stream_with_data('DownloadModel',
                                                      api_proto.Chunk,
                                                      api_proto.ModelDescription(id=nn_id, hash=nn_hash)):
            if fh is None:
                fh = ChunkedFileWriter(file_path=nn_archive_path)
            fh.write(nn_chunk)

            if progress is None:  # fh.total_size may be got from first chunk
                progress = sly.progress_download_nn(fh.total_size, ext_logger=self.logger)
            progress.iters_done_report(len(nn_chunk.buffer))

        if not fh.close_and_check():
            self.logger.critical('file was skipped while downloading', extra={'file_path': fh.file_path})
            raise RuntimeError('Unable to download NN weights.')

        with tarfile.open(nn_archive_path) as archive:
            archive.extractall(model_dir)
        sly.silent_remove(nn_archive_path)
        self.logger.info('NN has been downloaded from server.')

        self.storage.nns.write_object(model_dir, nn_hash)

    def upload_nn(self, nn_id, nn_hash):
        local_service_log = {'nn_id': nn_id, 'nn_hash': nn_hash}

        storage_nn_dir = self.storage.nns.check_storage_object(nn_hash)
        if storage_nn_dir is None:
            self.logger.critical('NN_NOT_FOUND', extra=local_service_log)
        local_tar_path = os.path.join(constants.AGENT_TMP_DIR, sly.generate_random_string(30) + '.tar')
        sly.archive_directory(storage_nn_dir, local_tar_path)

        freader = ChunkedFileReader(local_tar_path, constants.NETW_CHUNK_SIZE)
        progress = sly.ProgressCounter("Upload NN", freader.splitter.chunk_cnt, ext_logger=self.logger)

        def chunk_generator():
            for chunk_bytes in freader:
                current_chunk = api_proto.Chunk(buffer=chunk_bytes, total_size=freader.file_size)
                yield api_proto.ChunkModel(chunk=current_chunk,
                                           model=api_proto.ModelDescription(id=nn_id, hash=nn_hash))
                progress.iter_done_report()

        try:
            self.api.put_stream_with_data('UploadModel', api_proto.Empty, chunk_generator(),
                                          addit_headers={'x-model-hash': nn_hash})
        finally:
            sly.silent_remove(local_tar_path)

        self.logger.info('NN_UPLOADED', extra=local_service_log)

    def upload_archive(self, dir_to_archive, archive_name):
        local_tar_path = os.path.join(constants.AGENT_TMP_DIR, sly.generate_random_string(30) + '.tar')

        self.logger.info("PACK_TO_ARCHIVE ...")
        sly.archive_directory(dir_to_archive, local_tar_path)

        freader = ChunkedFileReader(local_tar_path, constants.NETW_CHUNK_SIZE)
        progress = sly.ProgressCounter("Upload archive", freader.splitter.chunk_cnt, ext_logger=self.logger)

        def chunk_generator():
            for chunk_bytes in freader:
                current_chunk = api_proto.Chunk(buffer=chunk_bytes, total_size=freader.file_size)
                yield current_chunk
                progress.iter_done_report()

        try:
            self.api.put_stream_with_data('UploadArchive', api_proto.Empty, chunk_generator(),
                                          addit_headers={'x-archive-name': archive_name})
        finally:
            sly.silent_remove(local_tar_path)

        self.logger.info('ARCHIVE_UPLOADED', extra={'archive_name': archive_name})

    def download_object_hashes(self, api_method_name):
        self.logger.info('RECEIVE_OBJECT_HASHES')
        res_hashes_exts = [(x.hash, x.ext)  # ok, default ext is ''
                           for x in self.api.get_stream_with_data(api_method_name,
                                                                  api_proto.NodeObjectHash,
                                                                  api_proto.Empty())]
        self.logger.info('OBJECT_HASHES_RECEIVED', extra={'obj_cnt': len(res_hashes_exts)})
        return res_hashes_exts

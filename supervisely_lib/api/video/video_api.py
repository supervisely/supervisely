# coding: utf-8
import json
import urllib.parse
from functools import partial
from requests_toolbelt import MultipartEncoder, MultipartEncoderMonitor

from supervisely_lib.api.module_api import ApiField, RemoveableBulkModuleApi
from supervisely_lib.api.video.video_annotation_api import VideoAnnotationAPI
from supervisely_lib.api.video.video_object_api import VideoObjectApi
from supervisely_lib.api.video.video_figure_api import VideoFigureApi
from supervisely_lib.api.video.video_frame_api import VideoFrameAPI
from supervisely_lib.api.video.video_tag_api import VideoTagApi
from supervisely_lib.sly_logger import logger
from supervisely_lib.io.fs import get_file_hash

from supervisely_lib.io.fs import ensure_base_path
from supervisely_lib._utils import batched
from supervisely_lib.video.video import get_video_streams, gen_video_stream_name

from supervisely_lib.task.progress import Progress


class VideoApi(RemoveableBulkModuleApi):
    def __init__(self, api):
        '''
        :param api: Api class object
        '''
        super().__init__(api)
        self.annotation = VideoAnnotationAPI(api)
        self.object = VideoObjectApi(api)
        self.frame = VideoFrameAPI(api)
        self.figure = VideoFigureApi(api)
        self.tag = VideoTagApi(api)

    @staticmethod
    def info_sequence():
        return [ApiField.ID,
                ApiField.NAME,
                ApiField.HASH,
                ApiField.TEAM_ID,
                ApiField.WORKSPACE_ID,
                ApiField.PROJECT_ID,
                ApiField.DATASET_ID,
                ApiField.PATH_ORIGINAL,
                ApiField.FRAMES_TO_TIMECODES,
                ApiField.FRAMES_COUNT,
                ApiField.FRAME_WIDTH,
                ApiField.FRAME_HEIGHT,
                ApiField.CREATED_AT,
                ApiField.UPDATED_AT,
                ApiField.TAGS,
                ApiField.FILE_META]

    @staticmethod
    def info_tuple_name():
        return 'VideoInfo'

    def url(self, dataset_id, video_id, video_frame=0):
        '''
        :param dataset_id: int
        :param video_id: int
        :param video_frame: int
        :return: url of given video id
        '''
        result = urllib.parse.urljoin(self._api.server_address, f'app/videos/?'
                                                                f'datasetId={dataset_id}&'
                                                                f'videoFrame={video_frame}&'
                                                                f'videoId={video_id}')

        return result

    def _convert_json_info(self, info: dict, skip_missing=True):
        return super(VideoApi, self)._convert_json_info(info, skip_missing=skip_missing)

    def get_list(self, dataset_id, filters=None):
        '''
        :param dataset_id: int
        :param filters: list
        :return: List of the videos from the dataset with given id
        '''
        return self.get_list_all_pages('videos.list',  {ApiField.DATASET_ID: dataset_id, ApiField.FILTER: filters or []})

    def get_info_by_id(self, id):
        '''
        :param id: int
        :return: VideoApi metadata by numeric id
        '''
        return self._get_info_by_id(id, 'videos.info')

    def get_destination_ids(self, id):
        '''
        :param id: int
        :return: id of project and id of dataset from given id
        '''
        dataset_id = self._api.video.get_info_by_id(id).dataset_id
        project_id = self._api.dataset.get_info_by_id(dataset_id).project_id
        return project_id, dataset_id

    def upload_hash(self, dataset_id, name, hash, stream_index=None):
        meta = {}
        if stream_index is not None and type(stream_index) is int:
            meta = {"videoStreamIndex": stream_index}
        return self.upload_hashes(dataset_id, [name], [hash], [meta])[0]

    def upload_hashes(self, dataset_id, names, hashes, metas=None, progress_cb=None):
        return self._upload_bulk_add(lambda item: (ApiField.HASH, item), dataset_id, names, hashes, metas, progress_cb)

    def _upload_bulk_add(self, func_item_to_kv, dataset_id, names, items, metas=None, progress_cb=None):
        if metas is None:
            metas = [{}] * len(items)

        results = []
        if len(names) == 0:
            return results
        if len(names) != len(items):
            raise RuntimeError("Can not match \"names\" and \"items\" lists, len(names) != len(items)")

        for batch in batched(list(zip(names, items, metas))):
            images = []
            for name, item, meta in batch:
                item_tuple = func_item_to_kv(item)
                images.append({'title': name,
                               item_tuple[0]: item_tuple[1],
                               ApiField.META: meta if meta is not None else {}})
            response = self._api.post('videos.bulk.add', {ApiField.DATASET_ID: dataset_id, ApiField.VIDEOS: images})
            if progress_cb is not None:
                progress_cb(len(images))

            results = [self._convert_json_info(item) for item in response.json()]
            name_to_res = {img_info.name: img_info for img_info in results}
            ordered_results = [name_to_res[name] for name in names]

            return ordered_results

    def _download(self, id, is_stream=False):
        '''
        :param id: int
        :param is_stream: bool
        :return: Response object containing video with given id
        '''
        response = self._api.post('videos.download', {ApiField.ID: id}, stream=is_stream)
        return response

    def download_path(self, id, path, progress_cb=None):
        '''
        Download video with given id on the given path
        :param id: int
        :param path: str
        :param progress_cb: SLY Progress Bar callback
        '''
        response = self._download(id, is_stream=True)
        ensure_base_path(path)

        with open(path, 'wb') as fd:
            mb1 = 1024 * 1024
            for chunk in response.iter_content(chunk_size=mb1):
                fd.write(chunk)

                if progress_cb is not None:
                    progress_cb(len(chunk))

    def download_range_by_id(self, id, frame_start, frame_end, is_stream=True):
        '''
        :param id: int
        :param frame_start: int
        :param frame_end: int
        :param is_stream: bool
        :return: Response object containing video with given id between given start and end frames
        '''
        path_original = self.get_info_by_id(id).path_original
        return self.downalod_range_by_path(path_original, frame_start, frame_end, is_stream)

    def downalod_range_by_path(self, path_original, frame_start, frame_end, is_stream=False):
        '''
        :param path_original: str
        :param frame_start: int
        :param frame_end: int
        :param is_stream: bool
        :return: Response object containing video on given path between given start and end frames
        '''
        response = self._api.get(method = 'image-converter/transcode' + path_original,
                                 params={'startFrame': frame_start, 'endFrame': frame_end, "transmux": True},
                                 stream=is_stream,
                                 use_public_api=False)
        return response

    def download_save_range(self, video_id, frame_start, frame_end, save_path):
        '''
        Download video with given id between given start and end frames on the given path
        :param video_id: int
        :param frame_start: int
        :param frame_end: int
        :param save_path:
        :return: str
        '''
        response = self.download_range_by_id(video_id, frame_start, frame_end)
        with open(save_path, 'wb') as fd:
            for chunk in response.iter_content(chunk_size=128):
                fd.write(chunk)
        return save_path

    def notify_progress(self, track_id, video_id, frame_start, frame_end, current, total):
        '''
        :param track_id: int
        :param video_id: int
        :param frame_start: int
        :param frame_end: int
        :param current: int
        :param total: int
        :return: str
        '''
        response = self._api.post('videos.notify-annotation-tool', {"type": "videos:fetch-figures-in-range",
                                                                    "data": {
                                                                        ApiField.TRACK_ID: track_id,
                                                                        ApiField.VIDEO_ID: video_id,
                                                                        ApiField.FRAME_RANGE: [frame_start, frame_end],
                                                                        ApiField.PROGRESS: {
                                                                            ApiField.CURRENT: current,
                                                                            ApiField.TOTAL: total
                                                                        }
                                                                    }
                                                                    })
        return response.json()[ApiField.STOPPED]

    def notify_tracking_error(self, track_id, error, message):
        '''
        :param track_id: int
        :param error: str
        :param message: str
        :return: #TODO nothing to return
        '''
        response = self._api.post('videos.notify-annotation-tool', {"type": "videos:tracking-error",
                                                                    "data": {
                                                                        ApiField.TRACK_ID: track_id,
                                                                        ApiField.ERROR: {
                                                                            ApiField.MESSAGE: "{}: {}".format(error, message)
                                                                        }
                                                                    }
                                                                    })
    # def upload(self):
    #     #"/videos.bulk.upload"
    #     pass
    #
    # def upload_path(self, dataset_id, name, path, meta=None):
    #     metas = None if meta is None else [meta]
    #     return self.upload_paths(dataset_id, [name], [path], metas=metas)[0]

    #@TODO: copypaste from image_api
    def check_existing_hashes(self, hashes):
        results = []
        if len(hashes) == 0:
            return results
        for hashes_batch in batched(hashes, batch_size=900):
            response = self._api.post('images.internal.hashes.list', hashes_batch)
            results.extend(response.json())
        return results

    def upload_paths(self, dataset_id, names, paths, progress_cb=None, metas=None, infos=None, item_progress=None):
        def path_to_bytes_stream(path):
            return open(path, 'rb')

        update_headers = False
        if infos is not None:
            if len(infos) != len(names):
                raise ValueError("Infos have to be None or provided for all videos")
            update_headers = True

        if update_headers:
            self._api.add_header("x-skip-processing", "true")

        video_info_results = []
        hashes = [get_file_hash(x) for x in paths]

        self._upload_data_bulk(path_to_bytes_stream, zip(paths, hashes), progress_cb=progress_cb, item_progress=item_progress)
        if update_headers:
            self.upsert_infos(hashes, infos)
            self._api.pop_header("x-skip-processing")
        metas = self._api.import_storage.get_meta_by_hashes(hashes)
        metas2 = [meta["meta"] for meta in metas]

        for name, hash, meta in zip(names, hashes, metas2):
            try:
                all_streams = meta["streams"]
                video_streams = get_video_streams(all_streams)
                for stream_info in video_streams:
                    stream_index = stream_info["index"]

                    #TODO: check is community
                    # if instance_type == sly.COMMUNITY:
                    #     if _check_video_requires_processing(file_info, stream_info) is True:
                    #         warn_video_requires_processing(file_name)
                    #         continue

                    item_name = name
                    info = self._api.video.get_info_by_name(dataset_id, item_name)
                    if info is not None:
                        item_name = gen_video_stream_name(name, stream_index)
                    res = self.upload_hash(dataset_id, item_name, hash, stream_index)
                    video_info_results.append(res)
            except Exception as e:
                logger.warning("File skipped {!r}: error occurred during processing {!r}".format(name, str(e)))
        return video_info_results

    def _upload_uniq_videos_single_req(self, func_item_to_byte_stream, hashes_items_to_upload, progress_cb=None):
        content_dict = {}
        for idx, (_, item) in enumerate(hashes_items_to_upload):
            content_dict["{}-file".format(idx)] = (str(idx), func_item_to_byte_stream(item), 'video/*')
        encoder = MultipartEncoder(fields=content_dict)

        if progress_cb is not None:
            def _callback(monitor, progress):
                progress(monitor.bytes_read)
            callback = partial(_callback, progress=progress_cb)
            monitor = MultipartEncoderMonitor(encoder, callback)
            resp = self._api.post('videos.bulk.upload', monitor)
        else:
            resp = self._api.post('videos.bulk.upload', encoder)

        resp_list = json.loads(resp.text)
        remote_hashes = [d['hash'] for d in resp_list if 'hash' in d]
        if len(remote_hashes) != len(hashes_items_to_upload):
            problem_items = [(hsh, item, resp['errors'])
                             for (hsh, item), resp in zip(hashes_items_to_upload, resp_list) if resp.get('errors')]
            logger.warn('Not all images were uploaded within request.', extra={
                'total_cnt': len(hashes_items_to_upload), 'ok_cnt': len(remote_hashes), 'items': problem_items})
        return remote_hashes

    def _upload_data_bulk(self, func_item_to_byte_stream, items_hashes, retry_cnt=3, progress_cb=None, item_progress=None):
        hash_to_items = {i_hash: item for item, i_hash in items_hashes}

        unique_hashes = set(hash_to_items.keys())
        remote_hashes = set(self.check_existing_hashes(list(unique_hashes)))  # existing -- from server
        if progress_cb:
            progress_cb(len(remote_hashes))
        #pending_hashes = unique_hashes #- remote_hashes #@TODO: only fo debug!
        pending_hashes = unique_hashes - remote_hashes

        for retry_idx in range(retry_cnt):
            # single attempt to upload all data which is not uploaded yet
            for hashes in batched(list(pending_hashes)):
                pending_hashes_items = [(h, hash_to_items[h]) for h in hashes]
                hashes_rcv = self._upload_uniq_videos_single_req(func_item_to_byte_stream, pending_hashes_items, item_progress)
                pending_hashes -= set(hashes_rcv)
                if set(hashes_rcv) - set(hashes):
                    logger.warn('Hash inconsistency in images bulk upload.',
                                extra={'sent': hashes, 'received': hashes_rcv})
                if progress_cb:
                    progress_cb(len(hashes_rcv))

            if not pending_hashes:
                return

            logger.warn('Unable to upload videos (data).', extra={
                'retry_idx': retry_idx,
                'items': [(h, hash_to_items[h]) for h in pending_hashes]
            })
            # now retry it for the case if it is a shadow server/connection error

        raise RuntimeError("Unable to upload videos (data). "
                           "Please check if videos are in supported format and if ones aren't corrupted.")

    # @TODO: add case to documentation with detailed explanation
    def upsert_info(self, hash, info):
        return self.upsert_infos([hash], [info])

    def upsert_infos(self, hashes, infos, links=None):
        payload = []
        if links is None:
            links = [None] * len(hashes)
        for h, l, info in zip(hashes, links, infos):
            item = {ApiField.HASH: h, ApiField.META: info}
            if l is not None:
                item[ApiField.LINK] = l
            payload.append(item)

        resp = self._api.post('videos.bulk.upsert_file_meta', payload)
        return resp.json()

    def upload_links(self, dataset_id, names, hashes, links, infos, metas=None):
        self.upsert_infos(hashes, infos, links)
        return self._upload_bulk_add(lambda item: (ApiField.LINK, item), dataset_id, names, links, metas)

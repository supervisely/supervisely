# coding: utf-8

from supervisely_lib.api.module_api import ApiField, RemoveableBulkModuleApi
from supervisely_lib.api.video.video_annotation_api import VideoAnnotationAPI
from supervisely_lib.api.video.video_object_api import VideoObjectApi
from supervisely_lib.api.video.video_figure_api import VideoFigureApi
from supervisely_lib.api.video.video_frame_api import VideoFrameAPI
from supervisely_lib.api.video.video_tag_api import VideoTagApi

from supervisely_lib.io.fs import ensure_base_path
from supervisely_lib._utils import batched


class VideoApi(RemoveableBulkModuleApi):
    def __init__(self, api):
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
                ApiField.UPDATED_AT]

    @staticmethod
    def info_tuple_name():
        return 'VideoInfo'

    def _convert_json_info(self, info: dict, skip_missing=True):
        return super(VideoApi, self)._convert_json_info(info, skip_missing=skip_missing)

    def get_list(self, dataset_id, filters=None):
        return self.get_list_all_pages('videos.list',  {ApiField.DATASET_ID: dataset_id, ApiField.FILTER: filters or []})

    def get_info_by_id(self, id):
        return self._get_info_by_id(id, 'videos.info')

    def get_destination_ids(self, id):
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
        response = self._api.post('videos.download', {ApiField.ID: id}, stream=is_stream)
        return response

    def download_path(self, id, path):
        response = self._download(id, is_stream=True)
        ensure_base_path(path)
        with open(path, 'wb') as fd:
            for chunk in response.iter_content(chunk_size=1024*1024):
                fd.write(chunk)

    def download_range_by_id(self, id, frame_start, frame_end, is_stream=True):
        path_original = self.get_info_by_id(id).path_original
        return self.downalod_range_by_path(path_original, frame_start, frame_end, is_stream)

    def downalod_range_by_path(self, path_original, frame_start, frame_end, is_stream=False):
        response = self._api.get(method = 'image-converter/transcode' + path_original,
                                 params={'startFrame': frame_start, 'endFrame': frame_end, "transmux": True},
                                 stream=is_stream,
                                 use_public_api=False)
        return response

    def download_save_range(self, video_id, frame_start, frame_end, save_path):
        response = self.download_range_by_id(video_id, frame_start, frame_end)
        with open(save_path, 'wb') as fd:
            for chunk in response.iter_content(chunk_size=128):
                fd.write(chunk)
        return save_path

    def notify_progress(self, track_id, video_id, frame_start, frame_end, current, total):
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
        response = self._api.post('videos.notify-annotation-tool', {"type": "videos:tracking-error",
                                                                    "data": {
                                                                        ApiField.TRACK_ID: track_id,
                                                                        ApiField.ERROR: {
                                                                            ApiField.MESSAGE: "{}: {}".format(error, message)
                                                                        }
                                                                    }
                                                                    })
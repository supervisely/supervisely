# coding: utf-8

import traceback
import concurrent.futures

import supervisely_lib as sly
from supervisely_lib import EventType
import supervisely_lib.worker_proto as api_proto

from . import constants
from .task_logged import TaskLogged


class ImageStreamer(TaskLogged):
    def __init__(self):
        super().__init__({'task_id': 'image_streamer'})
        self.thread_pool = None

    def init_logger(self):
        super().init_logger()
        sly.change_formatters_default_values(self.logger, 'worker', 'image_streamer')

    def init_additional(self):
        super().init_additional()
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=10)

    def task_main_func(self):
        try:
            self.logger.info('IMAGE_STREAMER_INITIALIZED')
            for gen_event in self.api.get_endless_stream('GetGeneralEventsStream',
                                                         api_proto.GeneralEvent, api_proto.Empty()):
                event_obj = {
                    'request_id': gen_event.request_id,
                    'data': sly.json_loads(gen_event.data.decode('utf-8')),
                }
                self.logger.debug('GET_STREAM_IMAGE_CALL', extra=event_obj)
                self.thread_pool.submit(sly.function_wrapper_nofail, self.stream_image, event_obj)

        except Exception:
            extra = {'event_type': EventType.TASK_CRASHED, 'error': traceback.format_exc().split('\n')}
            self.logger.critical("IMAGE_STREAMER_CRASHED", extra={**extra})

    def stream_image(self, event_obj):
        st_path = self.data_mgr.storage.images.check_storage_object(data_hash=event_obj['data']['hash'],
                                                                    suffix=event_obj['data']['ext'])
        if st_path is None:
            self.logger.error("STREAMER_IMAGE_NOT_FOUND", extra=event_obj)
            return

        file_size = sly.get_file_size(st_path)

        def chunk_generator():
            with open(st_path, 'rb') as file_:
                for chunk_start, chunk_size in sly.ChunkSplitter(file_size, constants.NETW_CHUNK_SIZE):
                    bytes_chunk = file_.read(chunk_size)
                    yield api_proto.Chunk(buffer=bytes_chunk, total_size=file_size)

        self.api.put_stream_with_data('SendGeneralEventData', api_proto.Empty, chunk_generator(),
                                      addit_headers={'x-request-id': event_obj['request_id']})
        self.logger.debug("IMAGE_STREAMED", extra=event_obj)

# coding: utf-8

import concurrent.futures
from queue import Queue
import threading

import cv2
import supervisely_lib as sly
from supervisely_lib import logger
import supervisely_lib.worker_proto as api_proto
from supervisely_lib.worker_api import AgentAPI, SimpleCache, decode_image, \
    download_image_from_remote, download_data_from_remote, send_from_memory_generator

from fast_inference import UnetV2FastApplier


class UnetV2AgentRPC:
    NETW_CHUNK_SIZE = 1048576

    @classmethod
    def _log_timing(cls, tm, msg):
        msecs = tm.get_sec() * 1000
        logger.trace(msg, extra={'durat_msec': msecs})

    def __init__(self, model_creator, conn_settings, cache_limit):
        self.api = AgentAPI(token=conn_settings['token'],
                            server_address=conn_settings['server_address'],
                            ext_logger=logger)
        self.api.add_to_metadata('x-task-id', conn_settings['task_id'])

        self.model_creator = model_creator
        self.model_applier = None
        self.model_inited = threading.Event()
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=10)
        self.in_queue = Queue()
        self.image_cache = SimpleCache(cache_limit)
        logger.info('Created UnetV2Service', extra=conn_settings)

    def _load_image_from_sly(self, req_id, image_hash, src_node_token):
        logger.trace('Will look for image.',
                     extra={'request_id': req_id, 'image_hash': image_hash, 'src_node_token': src_node_token})
        img_data = self.image_cache.get(image_hash)
        if img_data is None:
            img_data_packed = download_image_from_remote(self.api, image_hash, src_node_token, logger)
            img_data = decode_image(img_data_packed)
            self.image_cache.add(image_hash, img_data)

        return img_data

    def _load_arbitrary_image(self, req_id):
        logger.trace('Will load arbitrary image.', extra={'request_id': req_id})
        img_data_packed = download_data_from_remote(self.api, req_id, logger)
        img_data = decode_image(img_data_packed)
        return img_data

    def _load_data(self, event_obj):
        req_id = event_obj['request_id']
        image_hash = event_obj['data'].get('image_hash')
        if image_hash is None:
            img_data = self._load_arbitrary_image(req_id)
        else:
            src_node_token = event_obj['data'].get('src_node_token', '')
            img_data = self._load_image_from_sly(req_id, image_hash, src_node_token)

        # cv2.imwrite('/sly_task_data/last_loaded.png', img_data[:, :, ::-1])  # @TODO: rm debug
        event_obj['data']['image_arr'] = img_data
        self.in_queue.put(item=(event_obj['data'], req_id))
        logger.trace('Input image obtained.', extra={'request_id': req_id})

    def _send_data(self, out_msg, req_id):
        logger.trace('Will send output data.', extra={'request_id': req_id})
        out_bytes = sly.json_dumps(out_msg).encode('utf-8')

        self.api.put_stream_with_data('SendGeneralEventData',
                                      api_proto.Empty,
                                      send_from_memory_generator(out_bytes, self.NETW_CHUNK_SIZE),
                                      addit_headers={'x-request-id': req_id})
        logger.trace('Output data is sent.', extra={'request_id': req_id})

    def _single_img_pipeline(self, in_msg):
        img = in_msg['image_arr']
        if len(img.shape) != 3 or img.shape[2] != 3:
            raise RuntimeError('Expect 3-channel image RGB [0..255].')

        tm = sly.TinyTimer()
        res_ann = self.model_applier.inference(img)
        self._log_timing(tm, 'internal inference')

        res = res_ann.pack()
        return res

    def _sequential_inference(self):
        self.model_applier = self.model_creator()
        self.model_inited.set()
        while True:
            in_msg, req_id = self.in_queue.get(block=True, timeout=None)
            res_msg = sly.function_wrapper_nofail(self._single_img_pipeline, in_msg)
            self.thread_pool.submit(sly.function_wrapper_nofail, self._send_data, res_msg, req_id)  # skip errors

    def run_inf_loop(self):
        def seq_inf_wrapped():
            sly.function_wrapper(self._sequential_inference)  # exit if raised

        processing_thread = threading.Thread(target=seq_inf_wrapped, daemon=True)
        processing_thread.start()
        self.model_inited.wait(timeout=None)
        sly.report_agent_rpc_ready()

        for gen_event in self.api.get_endless_stream('GetGeneralEventsStream',
                                                     api_proto.GeneralEvent, api_proto.Empty()):
            event_obj = {
                'request_id': gen_event.request_id,
                'data': sly.json_loads(gen_event.data.decode('utf-8')),
            }
            logger.debug('GET_INFERENCE_CALL', extra=event_obj)
            sly.function_wrapper_nofail(self._load_data, event_obj)  # sync, in this thread; skip errors


def serve():
    settings = {
        'device_id': 0,
        'cache_limit': 500,
        'connection': {
            'server_address': None,
            'token': None,
            'task_id': None,
        },
    }

    new_settings = sly.json_load(sly.TaskPaths(determine_in_project=False).settings_path)
    logger.info('Input settings', extra={'settings': new_settings})
    sly.update_recursively(settings, new_settings)
    logger.info('Full settings', extra={'settings': settings})

    def model_creator():
        res = UnetV2FastApplier(settings={
            'device_id': settings['device_id']
        })
        return res

    serv_instance = UnetV2AgentRPC(model_creator, settings['connection'], settings['cache_limit'])
    serv_instance.run_inf_loop()


if __name__ == '__main__':
    cv2.setNumThreads(0)
    sly.main_wrapper('UNET_V2_SERVICE', serve)

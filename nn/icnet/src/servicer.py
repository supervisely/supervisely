# coding: utf-8

import cv2
import supervisely_lib as sly
from supervisely_lib import logger
from supervisely_lib.worker_api import AgentRPCServicer, SimpleCache

from fast_inference import ICNetFastApplier


def single_img_pipeline(image, message, model_applier):
    res_ann = model_applier.inference(image)
    res = res_ann.pack()
    return res


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
        res = ICNetFastApplier(settings={
            'device_id': settings['device_id'],
        })
        return res

    image_cache = SimpleCache(settings['cache_limit'])
    serv_instance = AgentRPCServicer(logger=logger,
                                     model_creator=model_creator,
                                     apply_cback=single_img_pipeline,
                                     conn_settings=settings['connection'],
                                     cache=image_cache)
    serv_instance.run_inf_loop()


if __name__ == '__main__':
    cv2.setNumThreads(0)
    sly.main_wrapper('ICNET_SERVICE', serve)

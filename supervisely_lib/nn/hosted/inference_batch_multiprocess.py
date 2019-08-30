# coding: utf-8

import os
import time
from collections import namedtuple
from copy import deepcopy
from threading import Thread

import multiprocessing as mp

from supervisely_lib import logger
from supervisely_lib.annotation.annotation import Annotation
from supervisely_lib.imaging import image as sly_image
from supervisely_lib.nn.config import AlwaysPassingConfigValidator
from supervisely_lib.project.project import Project, read_single_project, OpenMode
from supervisely_lib.project.project_meta import ProjectMeta
from supervisely_lib.task.paths import TaskPaths
from supervisely_lib.task.progress import report_inference_finished
from supervisely_lib.nn.hosted.inference_modes import InferenceModeFactory
from supervisely_lib.nn.hosted.inference_batch import determine_task_inference_mode_config
from supervisely_lib.task.progress import Progress


InferenceRequest = namedtuple('InferenceRequest', ['ds_name', 'item_name', 'item_paths'])
InferenceResponse = namedtuple('InferenceResponse', ['ds_name', 'item_name', 'item_paths', 'ann_json', 'meta_json'])


def single_inference_process_fn(inference_initializer, inference_mode_config, in_project_meta_json, request_queue,
                                response_queue):
    """Loads a separate model, processes requests from request_queue, results go to result_queue.

    None request signals the process to finish.
    """
    single_image_inference = inference_initializer()
    inference_mode = InferenceModeFactory.create(
        inference_mode_config, ProjectMeta.from_json(in_project_meta_json), single_image_inference)
    out_meta_json = inference_mode.out_meta.to_json()

    req = ''
    while req is not None:
        req = request_queue.get()
        if req is not None:
            in_img = sly_image.read(req.item_paths.img_path)
            in_ann = Annotation.load_json_file(req.item_paths.ann_path, inference_mode.out_meta)
            ann = inference_mode.infer_annotate(in_img, in_ann)
            resp = InferenceResponse(ds_name=req.ds_name, item_name=req.item_name, item_paths=req.item_paths,
                                     ann_json=ann.to_json(), meta_json=out_meta_json)
            response_queue.put(resp)
        request_queue.task_done()


def result_writer_thread_fn(in_project, inference_result_queue):
    """Gets inference result annotations from the queue and writes them to the output dataset.

    None result signals the thread to finish.
    """

    out_project = None
    progress_bar = Progress('Model applying: ', in_project.total_items)
    resp = ''
    while resp is not None:
        resp = inference_result_queue.get()
        if resp is not None:
            if out_project is None:
                out_dir = os.path.join(TaskPaths.RESULTS_DIR, in_project.name)
                out_project = Project(out_dir, OpenMode.CREATE)
                out_project.set_meta(ProjectMeta.from_json(resp.meta_json))
            out_dataset = out_project.datasets.get(resp.ds_name)
            if out_dataset is None:
                out_dataset = out_project.create_dataset(resp.ds_name)
            out_dataset.add_item_file(resp.item_name, resp.item_paths.img_path, ann=resp.ann_json)
            progress_bar.iter_done_report()
        inference_result_queue.task_done()


def populate_inference_requests_queue(in_project, inference_processes, request_queue):
    for in_dataset in in_project:
        for in_item_name in in_dataset:
            while True:
                # If any of the inference processes has died, stop populating the requests queue and exit right away.
                # Otherwise a deadlock is possible if no inference processes survive to take requests off the queue.
                if all(p.is_alive() for p in inference_processes):
                    # Do not try to add requests to a full queue to prevent deadlocks if all of the inference processes
                    # die in the interim.
                    if not request_queue.full():
                        logger.trace('Will process image',
                                     extra={'dataset_name': in_dataset.name, 'image_name': in_item_name})
                        in_item_paths = in_dataset.get_item_paths(in_item_name)
                        req = InferenceRequest(ds_name=in_dataset.name, item_name=in_item_name,
                                               item_paths=in_item_paths)
                        request_queue.put(req)
                        break  # out of (while True).
                    else:
                        time.sleep(0.1)
                else:
                    # Early exit, return False to indicate failure.
                    return False
    return True


class BatchInferenceMultiprocessApplier:
    def __init__(self, single_image_inference_initializer, num_processes, default_inference_mode_config: dict,
                 config_validator=None):
        self._config_validator = config_validator or AlwaysPassingConfigValidator()
        self._inference_mode_config = determine_task_inference_mode_config(deepcopy(default_inference_mode_config))

        self._in_project = read_single_project(TaskPaths.DATA_DIR)
        logger.info('Project structure has been read. Samples: {}.'.format(self._in_project.total_items))

        self._inference_request_queue = mp.JoinableQueue(maxsize=2 * num_processes)
        self._inference_result_queue = mp.JoinableQueue(maxsize=2 * num_processes)
        self._inference_processes = [
            mp.Process(
                target=single_inference_process_fn,
                args=(single_image_inference_initializer, self._inference_mode_config, self._in_project.meta.to_json(),
                      self._inference_request_queue, self._inference_result_queue),
                daemon=True)
            for _ in range(num_processes)]
        logger.info('Dataset inference preparation done.')
        for p in self._inference_processes:
            p.start()

    def run_inference(self):
        result_writer_thread = Thread(
            target=result_writer_thread_fn, args=(self._in_project, self._inference_result_queue), daemon=True)
        result_writer_thread.start()

        feed_status = populate_inference_requests_queue(
            self._in_project, self._inference_processes, self._inference_request_queue)
        for _ in self._inference_processes:
            self._inference_request_queue.put(None)
        for p in self._inference_processes:
            p.join()

        if not feed_status or not all(p.exitcode == 0 for p in self._inference_processes):
            raise RuntimeError('One of the inference processes encountered an error.')

        self._inference_result_queue.put(None)
        result_writer_thread.join()
        report_inference_finished()

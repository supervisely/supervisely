import functools
import time
import uuid
from pathlib import Path
from queue import Queue
from threading import Event, Thread
from typing import Any, BinaryIO, Dict, List, Optional

import numpy as np

from supervisely.annotation.annotation import Annotation
from supervisely.annotation.label import Geometry, Label
from supervisely.annotation.obj_class import ObjClass
from supervisely.api.api import Api
from supervisely.api.module_api import ApiField
from supervisely.geometry.helpers import deserialize_geometry
from supervisely.geometry.rectangle import Rectangle
from supervisely.imaging import image as sly_image
from supervisely.nn.inference.tracking.base_tracking import (
    BaseTracking,
    ValidationError,
)
from supervisely.nn.inference.tracking.tracker_interface import TrackerInterface
from supervisely.nn.prediction_dto import Prediction, PredictionBBox
from supervisely.sly_logger import logger
from supervisely.task.progress import Progress


class BBoxTracking(BaseTracking):
    def _deserialize_geometry(self, data: dict):
        geometry_type_str = data["type"]
        geometry_json = data["data"]
        return deserialize_geometry(geometry_type_str, geometry_json)

    def _track(self, api: Api, context: dict, notify_annotation_tool: bool):
        video_interface = TrackerInterface(
            context=context,
            api=api,
            load_all_frames=False,
            frame_loader=self.cache.download_frame,
            frames_loader=self.cache.download_frames,
            should_notify=notify_annotation_tool,
        )

        range_of_frames = [
            video_interface.frames_indexes[0],
            video_interface.frames_indexes[-1],
        ]

        if self.cache.is_persistent:
            self.cache.run_cache_task_manually(
                api,
                None,
                video_id=video_interface.video_id,
            )
        else:
            # if cache is not persistent, run cache task for range of frames
            self.cache.run_cache_task_manually(
                api,
                [range_of_frames],
                video_id=video_interface.video_id,
            )

        api.logger.info("Start tracking.")

        def _upload_loop(q: Queue, stop_event: Event, video_interface: TrackerInterface):
            try:
                while True:
                    items = []
                    while not q.empty():
                        items.append(q.get_nowait())
                    if len(items) > 0:
                        video_interface.add_object_geometries_on_frames(*list(zip(*items)))
                        continue
                    if stop_event.is_set():
                        video_interface._notify(True, task="stop tracking")
                        return
                    time.sleep(1)
            except Exception as e:
                api.logger.error("Error in upload loop: %s", str(e), exc_info=True)
                video_interface._notify(True, task="stop tracking")
                video_interface.global_stop_indicatior = True
                raise

        upload_queue = Queue()
        stop_upload_event = Event()
        Thread(
            target=_upload_loop,
            args=[upload_queue, stop_upload_event, video_interface],
            daemon=True,
        ).start()

        try:
            for fig_id, obj_id in zip(
                video_interface.geometries.keys(),
                video_interface.object_ids,
            ):
                init = False
                for _ in video_interface.frames_loader_generator():
                    geom = video_interface.geometries[fig_id]
                    if not isinstance(geom, Rectangle):
                        stop_upload_event.set()
                        raise TypeError(f"Tracking does not work with {geom.geometry_name()}.")

                    imgs = video_interface.frames
                    target = PredictionBBox(
                        "",  # TODO: can this be useful?
                        [geom.top, geom.left, geom.bottom, geom.right],
                        None,
                    )

                    if not init:
                        self.initialize(imgs[0], target)
                        init = True

                    geometry = self.predict(
                        rgb_image=imgs[-1],
                        prev_rgb_image=imgs[0],
                        target_bbox=target,
                        settings=self.custom_inference_settings_dict,
                    )
                    sly_geometry = self._to_sly_geometry(geometry)
                    upload_queue.put(
                        (sly_geometry, obj_id, video_interface._cur_frames_indexes[-1])
                    )

                    if video_interface.global_stop_indicatior:
                        stop_upload_event.set()
                        return

                api.logger.info(f"Figure #{fig_id} tracked.")
        except Exception:
            stop_upload_event.set()
            raise
        stop_upload_event.set()

    def _track_api(self, api: Api, context: dict, request_uuid: str = None):
        track_t = time.monotonic()
        # unused fields:
        context["trackId"] = "auto"
        context["objectIds"] = []
        context["figureIds"] = []
        if "direction" not in context:
            context["direction"] = "forward"

        input_bboxes: list = context["input_geometries"]

        video_interface = TrackerInterface(
            context=context,
            api=api,
            load_all_frames=False,
            frame_loader=self.cache.download_frame,
            frames_loader=self.cache.download_frames,
            should_notify=False,
        )

        range_of_frames = [
            video_interface.frames_indexes[0],
            video_interface.frames_indexes[-1],
        ]

        if self.cache.is_persistent:
            # if cache is persistent, run cache task for whole video
            self.cache.run_cache_task_manually(
                api,
                None,
                video_id=video_interface.video_id,
            )
        else:
            # if cache is not persistent, run cache task for range of frames
            self.cache.run_cache_task_manually(
                api,
                [range_of_frames],
                video_id=video_interface.video_id,
            )

        predictions = []
        frames_n = video_interface.frames_count
        box_n = len(input_bboxes)
        geom_t = time.monotonic()
        api.logger.info(
            "Start tracking.",
            extra={
                "video_id": video_interface.video_id,
                "frame_range": range_of_frames,
                "geometries_count": box_n,
                "frames_count": frames_n,
                "request_uuid": request_uuid,
            },
        )
        for box_i, input_geom in enumerate(input_bboxes, 1):
            input_bbox = input_geom["data"]
            bbox = Rectangle.from_json(input_bbox)
            predictions_for_object = []
            init = False
            frame_t = time.monotonic()
            for frame_i, _ in enumerate(video_interface.frames_loader_generator(), 1):
                imgs = video_interface.frames
                target = PredictionBBox(
                    "",  # TODO: can this be useful?
                    [bbox.top, bbox.left, bbox.bottom, bbox.right],
                    None,
                )

                if not init:
                    self.initialize(imgs[0], target)
                    init = True

                geometry = self.predict(
                    rgb_image=imgs[-1],
                    prev_rgb_image=imgs[0],
                    target_bbox=target,
                    settings=self.custom_inference_settings_dict,
                )
                sly_geometry = self._to_sly_geometry(geometry)

                predictions_for_object.append(
                    {"type": sly_geometry.geometry_name(), "data": sly_geometry.to_json()}
                )
                api.logger.debug(
                    "Frame processed. Geometry: [%d / %d]. Frame: [%d / %d]",
                    box_i,
                    box_n,
                    frame_i,
                    frames_n,
                    extra={
                        "geometry_index": box_i,
                        "frame_index": frame_i,
                        "processing_time": time.monotonic() - frame_t,
                        "request_uuid": request_uuid,
                    },
                )
                frame_t = time.monotonic()

            predictions.append(predictions_for_object)
            api.logger.info(
                "Geometry processed. Progress: [%d / %d]",
                box_i,
                box_n,
                extra={
                    "geometry_index": box_i,
                    "processing_time": time.monotonic() - geom_t,
                    "request_uuid": request_uuid,
                },
            )
            geom_t = time.monotonic()

        # predictions must be NxK bboxes: N=number of frames, K=number of objects
        predictions = list(map(list, zip(*predictions)))
        api.logger.info(
            "Tracking finished.",
            extra={"tracking_time": time.monotonic() - track_t, "request_uuid": request_uuid},
        )
        return predictions

    def _inference(self, frames: List[np.ndarray], geometries: List[Geometry], settings: dict):
        updated_settings = {
            **self.custom_inference_settings_dict,
            **settings,
        }
        results = [[] for _ in range(len(frames) - 1)]
        for geometry in geometries:
            if not isinstance(geometry, Rectangle):
                raise TypeError(f"Tracking does not work with {geometry.geometry_name()}.")
            target = PredictionBBox(
                "",
                [geometry.top, geometry.left, geometry.bottom, geometry.right],
                None,
            )
            self.initialize(frames[0], target)
            for i in range(len(frames) - 1):
                pred_geometry = self.predict(
                    rgb_image=frames[i + 1],
                    prev_rgb_image=frames[i],
                    target_bbox=target,
                    settings=updated_settings,
                )
                sly_pred_geometry = self._to_sly_geometry(pred_geometry)
                results[i].append(
                    {"type": Rectangle.geometry_name(), "data": sly_pred_geometry.to_json()}
                )
        return results

    def _track_async(self, api: Api, context: dict, request_uuid: str = None):
        api.logger.info("context", extra=context)
        inference_request = self._inference_requests[request_uuid]

        session_id = context.get("session_id", context["sessionId"])
        direct_progress = context.get("useDirectProgressMessages", False)
        frame_index = context["frameIndex"]
        frames_count = context["frames"]
        track_id = context["trackId"]
        video_id = context["videoId"]
        direction = context.get("direction", "forward")
        direction_n = 1 if direction == "forward" else -1
        figures = context["figures"]
        progress: Progress = inference_request["progress"]
        progress_total = frames_count * len(figures)
        progress.total = progress_total

        range_of_frames = [
            frame_index,
            frame_index + frames_count * direction_n,
        ]

        if self.cache.is_persistent:
            self.cache.run_cache_task_manually(
                api,
                None,
                video_id=video_id,
            )
        else:
            # if cache is not persistent, run cache task for range of frames
            self.cache.run_cache_task_manually(
                api,
                [range_of_frames if direction_n == 1 else range_of_frames[::-1]],
                video_id=video_id,
            )

        global_stop_indicatior = False

        def _add_to_inference_request(geometry, object_id, frame_index, figure_id):
            figure_info = api.video.figure._convert_json_info(
                {
                    ApiField.ID: figure_id,
                    ApiField.OBJECT_ID: object_id,
                    "meta": {"frame": frame_index},
                    ApiField.GEOMETRY_TYPE: geometry.geometry_name(),
                    ApiField.GEOMETRY: geometry.to_json(),
                }
            )
            # logger.debug("Acquiring lock for add")
            # with inference_request["lock"]:
            inference_request["pending_results"].append(figure_info)
            # lock
            # logger.debug("Released lock for add")

        def _nofify_loop(q: Queue, stop_event: Event):
            nonlocal global_stop_indicatior
            try:
                while True:
                    items = []  # (geometry, object_id, frame_index)
                    while not q.empty():
                        items.append(q.get_nowait())
                    if len(items) > 0:
                        logger.debug("Got %d items to notify", len(items))
                        items_by_object_id = {}
                        for item in items:
                            items_by_object_id.setdefault(item[1], []).append(item)

                        for object_id, object_items in items_by_object_id.items():
                            frame_range = [
                                min(item[2] for item in object_items),
                                max(item[2] for item in object_items),
                            ]
                            progress.iters_done_report(len(object_items))
                            if direct_progress:
                                api.vid_ann_tool.set_direct_tracking_progress(
                                    session_id,
                                    video_id,
                                    track_id,
                                    frame_range=frame_range,
                                    progress_current=progress.current,
                                    progress_total=progress.total,
                                )
                        logger.debug("Items notified")
                    elif stop_event.is_set():
                        api.logger.debug(f"stop event is set. returning from notify loop")
                        return
                    time.sleep(1)
            except Exception as e:
                api.logger.error("Error in notify loop: %s", str(e), exc_info=True)
                global_stop_indicatior = True
                raise

        def _upload_loop(q: Queue, notify_q: Queue, stop_event: Event, stop_notify_event: Event):
            nonlocal global_stop_indicatior
            try:
                while True:
                    items = []  # (geometry, object_id, frame_index)
                    while not q.empty():
                        items.append(q.get_nowait())
                    if len(items) > 0:
                        logger.debug("Got %d items to upload", len(items))
                        for item in items:
                            figure_id = uuid.uuid5(
                                namespace=uuid.NAMESPACE_URL, name=f"{time.time()}"
                            ).hex
                            _add_to_inference_request(*item, figure_id)
                            if direct_progress:
                                notify_q.put(item)
                        logger.debug("Items added to inference request")
                        time.sleep(0.01)

                    elif stop_event.is_set():
                        stop_notify_event.set()
                        return
            except Exception as e:
                api.logger.error("Error in upload loop: %s", str(e), exc_info=True)
                global_stop_indicatior = True
                stop_notify_event.set()
                raise

        upload_queue = Queue()
        notify_queue = Queue()
        stop_upload_event = Event()
        stop_notify_event = Event()
        upload_thread = Thread(
            target=_upload_loop,
            args=[upload_queue, notify_queue, stop_upload_event, stop_notify_event],
            daemon=True,
        )
        upload_thread.start()
        notify_thread = Thread(
            target=_nofify_loop,
            args=[notify_queue, stop_notify_event],
            daemon=True,
        )
        notify_thread.start()

        api.logger.info("Start tracking.")
        try:
            for figure in figures:
                figure = api.video.figure._convert_json_info(figure)
                if not figure.geometry_type == Rectangle.geometry_name():
                    stop_upload_event.set()
                    raise TypeError(f"Tracking does not work with {figure.geometry_type}.")
                api.logger.info("geometry:", extra={"figure": figure._asdict()})
                sly_geometry: Rectangle = deserialize_geometry(
                    figure.geometry_type, figure.geometry
                )
                api.logger.info("geometry:", extra={"geometry": type(sly_geometry)})
                init = False
                for frame_i in range(*range_of_frames, direction_n):
                    frame_i_next = frame_i + direction_n
                    t = time.time()
                    frame, frame_next = self.cache.download_frames(
                        api,
                        video_id,
                        [frame_i, frame_i_next] if direction_n == 1 else [frame_i_next, frame_i],
                    )
                    api.logger.debug(
                        "Frames %d, %d downloaded. Time: %f",
                        frame_i,
                        frame_i_next,
                        time.time() - t,
                    )
                    if direction_n == -1:
                        frame, frame_next = frame_next, frame

                    target = PredictionBBox(
                        "",  # TODO: can this be useful?
                        [
                            sly_geometry.top,
                            sly_geometry.left,
                            sly_geometry.bottom,
                            sly_geometry.right,
                        ],
                        None,
                    )

                    if not init:
                        self.initialize(frame, target)
                        init = True

                    logger.debug("Start prediction")
                    t = time.time()
                    geometry = self.predict(
                        rgb_image=frame_next,
                        prev_rgb_image=frame,
                        target_bbox=target,
                        settings=self.custom_inference_settings_dict,
                    )
                    logger.debug("Prediction done. Time: %f", time.time() - t)
                    sly_geometry = self._to_sly_geometry(geometry)
                    upload_queue.put((sly_geometry, figure.object_id, frame_i_next))

                    if global_stop_indicatior:
                        stop_upload_event.set()
                        return

                api.logger.info(f"Figure #{figure.id} tracked.")
        except Exception as e:
            if direct_progress:
                api.vid_ann_tool.set_direct_tracking_error(
                    session_id,
                    video_id,
                    track_id,
                    message=f"An error occured during tracking. Error: {e}",
                )
            error = True
            raise
        else:
            error = False
        finally:
            stop_upload_event.set()
            if upload_thread.is_alive():
                upload_thread.join()
            stop_notify_event.set()
            if notify_thread.is_alive():
                notify_thread.join()
            if error:
                progress.message = "Error occured during tracking"
                progress.set(current=0, total=1, report=True)
            else:
                progress.message = "Ready"
                progress.set(current=0, total=1, report=True)

    def track(self, api: Api, state: Dict, context: Dict):
        return self._track(api, context, notify_annotation_tool=True)

    def track_api(self, api: Api, state: Dict, context: Dict):
        inference_request_uuid = uuid.uuid5(namespace=uuid.NAMESPACE_URL, name=f"{time.time()}").hex
        result = self._track_api(api, context, inference_request_uuid)
        logger.info("Track-api request processed.", extra={"request_uuid": inference_request_uuid})
        return result

    def track_api_files(self, files: List[BinaryIO], settings: Dict):
        logger.info(f"Start tracking with settings: {settings}.")
        frame_indexes = list(
            range(settings["frame_index"], settings["frame_index"] + settings["frames"] + 1)
        )
        geometries = map(self._deserialize_geometry, settings["input_geometries"])
        frames = []
        for file, frame_idx in zip(files, frame_indexes):
            img_bytes = file.read()
            frame = sly_image.read_bytes(img_bytes)
            frames.append(frame)
        logger.info("Start tracking.")
        return self._inference(frames, geometries, settings)

    def track_async(self, api: Api, state: Dict, context: Dict):
        batch_size = context.get("batch_size", self.get_batch_size())
        if self.max_batch_size is not None and batch_size > self.max_batch_size:
            raise ValidationError(
                f"Batch size should be less than or equal to {self.max_batch_size} for this model."
            )
        inference_request_uuid = uuid.uuid5(namespace=uuid.NAMESPACE_URL, name=f"{time.time()}").hex
        self._on_inference_start(inference_request_uuid)
        future = self._executor.submit(
            self._handle_error_in_async,
            inference_request_uuid,
            self._track_async,
            api,
            context,
            inference_request_uuid,
        )
        end_callback = functools.partial(
            self._on_inference_end, inference_request_uuid=inference_request_uuid
        )
        future.add_done_callback(end_callback)
        logger.debug(
            "Inference has scheduled from 'track_async' endpoint",
            extra={"inference_request_uuid": inference_request_uuid},
        )
        return {
            "message": "Inference has started.",
            "inference_request_uuid": inference_request_uuid,
        }

    def initialize(self, init_rgb_image: np.ndarray, target_bbox: PredictionBBox) -> None:
        """
        Initializing the tracker with a new object.

        :param init_rgb_image: frame with object
        :type init_rgb_image: np.ndarray
        :param target_bbox: initial bbox
        :type target_bbox: PredictionBBox
        """
        raise NotImplementedError

    def predict(
        self,
        rgb_image: np.ndarray,
        settings: Dict[str, Any],
        prev_rgb_image: np.ndarray,
        target_bbox: PredictionBBox,
    ) -> PredictionBBox:
        """
        SOT prediction

        :param rgb_image: search frame
        :type rgb_image: np.ndarray
        :param settings: model parameters
        :type settings: Dict[str, Any]
        :param init_rgb_image: previous frame with object
        :type init_rgb_image: np.ndarray
        :param target_bbox: bbox added on previous step
        :type target_bbox: PredictionBBox
        :return: predicted annotation
        :rtype: PredictionBBox
        """
        raise NotImplementedError

    def visualize(
        self,
        predictions: List[PredictionBBox],
        images: List[np.ndarray],
        vis_path: str,
        thickness: int = 2,
        classes_whitelist: Optional[List[str]] = None,
    ):
        vis_path = Path(vis_path)

        for i, (pred, image) in enumerate(zip(predictions, images)):
            out_path = vis_path / f"img_{i}.jpg"
            ann = self._predictions_to_annotation(image, [pred], classes_whitelist)
            ann.draw_pretty(
                bitmap=image,
                color=(255, 0, 0),
                thickness=thickness,
                output_path=str(out_path),
                fill_rectangles=False,
            )

    def _to_sly_geometry(self, dto: PredictionBBox) -> Rectangle:
        top, left, bottom, right = dto.bbox_tlbr
        geometry = Rectangle(top=top, left=left, bottom=bottom, right=right)
        return geometry

    def _create_label(self, dto: PredictionBBox) -> Rectangle:
        geometry = self._to_sly_geometry(dto)
        return Label(geometry, ObjClass("", Rectangle))

    def _get_obj_class_shape(self):
        return Rectangle

    def _predictions_to_annotation(
        self,
        image: np.ndarray,
        predictions: List[Prediction],
        classes_whitelist: Optional[List[str]] = None,
    ) -> Annotation:
        labels = []
        for prediction in predictions:
            if (
                not classes_whitelist in (None, "all")
                and prediction.class_name not in classes_whitelist
            ):
                continue
            label = self._create_label(prediction)
            if label is None:
                # for example empty mask
                continue
            if isinstance(label, list):
                labels.extend(label)
                continue
            labels.append(label)

        # create annotation with correct image resolution
        ann = Annotation(img_size=image.shape[:2])
        ann = ann.add_labels(labels)
        return ann

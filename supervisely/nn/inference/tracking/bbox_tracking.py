import functools
import json
import time
import uuid
from pathlib import Path
from queue import Queue
from threading import Event, Lock, Thread
from typing import Any, Dict, List, Optional, Union

import numpy as np
from fastapi import BackgroundTasks, Form, Request, Response, UploadFile, status

import supervisely as sly
import supervisely.nn.inference.tracking.functional as F
from supervisely.annotation.label import Geometry, Label
from supervisely.api.module_api import ApiField
from supervisely.nn.inference import Inference
from supervisely.nn.inference.inference import (
    _convert_sly_progress_to_dict,
    _get_log_extra_for_inference_request,
)
from supervisely.nn.inference.tracking.tracker_interface import TrackerInterface
from supervisely.nn.prediction_dto import Prediction, PredictionBBox


class BBoxTracking(Inference):
    def __init__(
        self,
        model_dir: Optional[str] = None,
        custom_inference_settings: Optional[Union[Dict[str, Any], str]] = None,
    ):
        Inference.__init__(
            self,
            model_dir,
            custom_inference_settings,
            sliding_window_mode=None,
            use_gui=False,
        )

        try:
            self.load_on_device(model_dir, "cuda")
        except RuntimeError:
            self.load_on_device(model_dir, "cpu")
            sly.logger.warn("Failed to load model on CUDA device.")

        sly.logger.debug(
            "Smart cache params",
            extra={"ttl": sly.env.smart_cache_ttl(), "maxsize": sly.env.smart_cache_size()},
        )

    def get_info(self):
        info = super().get_info()
        info["task type"] = "tracking"
        return info

    def _deserialize_geometry(self, data: dict):
        geometry_type_str = data["type"]
        geometry_json = data["data"]
        return sly.deserialize_geometry(geometry_type_str, geometry_json)

    def _on_inference_start(self, inference_request_uuid: str):
        super()._on_inference_start(inference_request_uuid)
        self._inference_requests[inference_request_uuid]["lock"] = Lock()

    def _track(self, api: sly.Api, context: dict, notify_annotation_tool: bool):
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
                    if not isinstance(geom, sly.Rectangle):
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

    def _track_api(self, api: sly.Api, context: dict, request_uuid: str = None):
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
            bbox = sly.Rectangle.from_json(input_bbox)
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
            if not isinstance(geometry, sly.Rectangle):
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
                    {"type": sly.Rectangle.geometry_name(), "data": sly_pred_geometry.to_json()}
                )
        return results

    def _track_api_files(
        self, request: Request, files: List[UploadFile], settings: str = Form("{}")
    ):
        state = json.loads(settings)
        sly.logger.info(f"Start tracking with settings: {state}.")
        video_id = state["video_id"]
        frame_indexes = list(
            range(state["frame_index"], state["frame_index"] + state["frames"] + 1)
        )
        geometries = map(self._deserialize_geometry, state["input_geometries"])
        frames = []
        for file, frame_idx in zip(files, frame_indexes):
            img_bytes = file.file.read()
            frame = sly.image.read_bytes(img_bytes)
            frames.append(frame)
        sly.logger.info("Start tracking.")
        return self._inference(frames, geometries, state)

    def _track_async(self, api: sly.Api, context: dict, request_uuid: str = None):
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
        progress: sly.Progress = inference_request["progress"]
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
                    ApiField.FIGURE_ID: figure_id,
                    ApiField.OBJECT_ID: object_id,
                    "meta": {"frame": frame_index},
                    ApiField.GEOMETRY_TYPE: geometry.geometry_name(),
                    ApiField.GEOMETRY: geometry.to_json(),
                }
            )
            with inference_request["lock"]:
                inference_request["pending_results"].append(figure_info)

        def _nofify_loop(q: Queue, stop_event: Event):
            nonlocal global_stop_indicatior
            try:
                while True:
                    items = []  # (geometry, object_id, frame_index)
                    while not q.empty():
                        items.append(q.get_nowait())
                    if len(items) > 0:
                        api.logger.debug(f"got {len(items)} items to notify")
                        items_by_object_id = {}
                        for item in items:
                            items_by_object_id.setdefault(item[1], []).append(item)

                        for object_id, object_items in items_by_object_id.items():
                            frame_range = [
                                min(item[2] for item in object_items),
                                max(item[2] for item in object_items),
                            ]
                            progress.iters_done(len(object_items))
                            if direct_progress:
                                api.logger.debug(f"notifying")
                                api.vid_ann_tool.set_direct_tracking_progress(
                                    session_id,
                                    video_id,
                                    track_id,
                                    frame_range=frame_range,
                                    progress_current=progress.current,
                                    progress_total=progress.total,
                                )
                        continue
                    if stop_event.is_set():
                        return
                    time.sleep(0.5)
            except Exception as e:
                api.logger.error("Error in notify loop: %s", str(e), exc_info=True)
                global_stop_indicatior = True
                raise

        def _upload_loop(q: Queue, notify_q: Queue, stop_event: Event):
            nonlocal global_stop_indicatior
            try:
                while True:
                    items = []  # (geometry, object_id, frame_index)
                    while not q.empty():
                        items.append(q.get_nowait())
                    if len(items) > 0:
                        for item in items:
                            figure_id = uuid.uuid5(
                                namespace=uuid.NAMESPACE_URL, name=f"{time.time()}"
                            ).hex
                            api.logger.debug(f"_add_to_inference_request")
                            _add_to_inference_request(*item, figure_id)
                            if direct_progress:
                                api.logger.debug(f"put to notify queue")
                                notify_q.put(item)
                        continue
                    if stop_event.is_set():
                        return
                    time.sleep(1)
            except Exception as e:
                api.logger.error("Error in upload loop: %s", str(e), exc_info=True)
                global_stop_indicatior = True
                raise

        upload_queue = Queue()
        notify_queue = Queue()
        stop_upload_event = Event()
        upload_thread = Thread(
            target=_upload_loop,
            args=[upload_queue, notify_queue, stop_upload_event],
            daemon=True,
        )
        upload_thread.start()
        notify_thread = Thread(
            target=_nofify_loop,
            args=[notify_queue, stop_upload_event],
            daemon=True,
        )
        notify_thread.start()

        api.logger.info("Start tracking.")
        try:
            for figure in figures:
                figure = api.video.figure._convert_json_info(figure)
                if not figure.geometry_type == sly.Rectangle.geometry_name():
                    stop_upload_event.set()
                    raise TypeError(f"Tracking does not work with {figure.geometry_type}.")
                api.logger.info("geometry:", extra={"figure": figure._asdict()})
                sly_geometry: sly.Rectangle = sly.deserialize_geometry(
                    figure.geometry_type, figure.geometry
                )
                api.logger.info("geometry:", extra={"geometry": type(sly_geometry)})
                init = False
                for frame_i in range(frame_index, frame_index + frames_count, direction_n):
                    frame_i_next = frame_i + direction_n
                    frame, frame_next = self.cache.download_frames(
                        api,
                        video_id,
                        [frame_i, frame_i_next] if direction_n == 1 else [frame_i_next, frame_i],
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

                    geometry = self.predict(
                        rgb_image=frame,
                        prev_rgb_image=frame_next,
                        target_bbox=target,
                        settings=self.custom_inference_settings_dict,
                    )
                    sly_geometry = self._to_sly_geometry(geometry)
                    upload_queue.put((sly_geometry, figure.object_id, frame_i_next))

                    if global_stop_indicatior:
                        stop_upload_event.set()
                        return

                api.logger.info(f"Figure #{figure.id} tracked.")
        except Exception:
            stop_upload_event.set()
            raise
        stop_upload_event.set()
        upload_thread.join()
        notify_thread.join()

    def serve(self):
        super().serve()
        server = self._app.get_server()
        self.cache.add_cache_endpoint(server)
        self.cache.add_cache_files_endpoint(server)

        def send_error_data(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                value = None
                try:
                    value = func(*args, **kwargs)
                except Exception as exc:
                    request: Request = args[0]
                    context = request.state.context
                    api: sly.Api = request.state.api
                    track_id = context["trackId"]
                    api.logger.error(f"An error occured: {repr(exc)}")

                    api.post(
                        "videos.notify-annotation-tool",
                        data={
                            "type": "videos:tracking-error",
                            "data": {
                                "trackId": track_id,
                                "error": {"message": repr(exc)},
                            },
                        },
                    )
                return value

            return wrapper

        @send_error_data
        def track(request: Request):
            return self._track(
                request.state.api, request.state.context, notify_annotation_tool=True
            )

        @server.post("/track")
        def start_track(request: Request, task: BackgroundTasks):
            task.add_task(track, request)
            return {"message": "Track task started."}

        @server.post("/track-api")
        def track_api(request: Request):
            inference_request_uuid = uuid.uuid5(
                namespace=uuid.NAMESPACE_URL, name=f"{time.time()}"
            ).hex
            sly.logger.info(
                "Received track-api request.", extra={"request_uuid": inference_request_uuid}
            )
            result = self._track_api(
                request.state.api, request.state.context, request_uuid=inference_request_uuid
            )
            sly.logger.info(
                "Track-api request processed.", extra={"request_uuid": inference_request_uuid}
            )
            return result

        @server.post("/track_async")
        def track_async(response: Response, request: Request):
            sly.logger.debug(f"'track_async' request in json format:{request.state.state}")
            # check batch size
            batch_size = request.state.state.get("batch_size", None)
            if batch_size is None:
                batch_size = self.get_batch_size()
            if self.max_batch_size is not None and batch_size > self.max_batch_size:
                response.status_code = status.HTTP_400_BAD_REQUEST
                return {
                    "message": f"Batch size should be less than or equal to {self.max_batch_size} for this model.",
                    "success": False,
                }
            inference_request_uuid = uuid.uuid5(
                namespace=uuid.NAMESPACE_URL, name=f"{time.time()}"
            ).hex
            self._on_inference_start(inference_request_uuid)
            future = self._executor.submit(
                self._handle_error_in_async,
                inference_request_uuid,
                self._track_async,
                request.state.api,
                request.state.context,
                inference_request_uuid,
            )
            end_callback = functools.partial(
                self._on_inference_end, inference_request_uuid=inference_request_uuid
            )
            future.add_done_callback(end_callback)
            sly.logger.debug(
                "Inference has scheduled from 'track_async' endpoint",
                extra={"inference_request_uuid": inference_request_uuid},
            )
            return {
                "message": "Inference has started.",
                "inference_request_uuid": inference_request_uuid,
            }

        @server.post("/pop_tracking_results")
        def pop_tracking_results(request: Request, response: Response):
            context = request.state.context
            inference_request_uuid = context.get("inference_request_uuid", None)
            if inference_request_uuid is None:
                inference_request_uuid = context.get("inference_request_id", None)
            if inference_request_uuid is None:
                response.status_code = status.HTTP_400_BAD_REQUEST
                return {"message": "Error: 'inference_request_uuid' is required."}

            inference_request = self._inference_requests[inference_request_uuid]
            frame_range = context.get("frame_range", None)
            figures = []
            with inference_request["lock"]:
                if frame_range is not None:
                    figure_ids = set()
                    for figure in inference_request["pending_results"]:
                        if (
                            figure.frame_index >= frame_range[0]
                            and figure.frame_index <= frame_range[1]
                        ):
                            figure_ids.add(figure.id)
                            figures.append(figure)
                    inference_request["pending_results"] = [
                        figure
                        for figure in inference_request["pending_results"]
                        if figure.id not in figure_ids
                    ]

                else:
                    figures = inference_request["pending_results"]
                    inference_request["pending_results"].clear()
                inference_request = inference_request.copy()
                inference_request["pending_results"] = figures

            inference_request["pending_results"] = [
                {
                    ApiField.ID: figure.id,
                    ApiField.OBJECT_ID: figure.object_id,
                    ApiField.GEOMETRY_TYPE: figure.geometry_type,
                    ApiField.GEOMETRY: figure.geometry,
                    ApiField.FRAME_INDEX: figure.frame_index,
                }
                for figure in figures
            ]

            inference_request["progress"] = _convert_sly_progress_to_dict(
                inference_request["progress"]
            )

            # Logging
            log_extra = _get_log_extra_for_inference_request(
                inference_request_uuid, inference_request
            )
            sly.logger.debug(f"Sending inference delta results with uuid:", extra=log_extra)
            return inference_request

        @server.post("/track-api-files")
        def track_api_files(
            request: Request,
            files: List[UploadFile],
            settings: str = Form("{}"),
        ):
            return self._track_api_files(request, files, settings)

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

    def _to_sly_geometry(self, dto: PredictionBBox) -> sly.Rectangle:
        top, left, bottom, right = dto.bbox_tlbr
        geometry = sly.Rectangle(top=top, left=left, bottom=bottom, right=right)
        return geometry

    def _create_label(self, dto: PredictionBBox) -> sly.Rectangle:
        geometry = self._to_sly_geometry(dto)
        return Label(geometry, sly.ObjClass("", sly.Rectangle))

    def _get_obj_class_shape(self):
        return sly.Rectangle

    def _predictions_to_annotation(
        self,
        image: np.ndarray,
        predictions: List[Prediction],
        classes_whitelist: Optional[List[str]] = None,
    ) -> sly.Annotation:
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
        ann = sly.Annotation(img_size=image.shape[:2])
        ann = ann.add_labels(labels)
        return ann

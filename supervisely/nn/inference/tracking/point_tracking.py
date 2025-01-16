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
from supervisely.nn.prediction_dto import Prediction, PredictionPoint


class PointTracking(Inference):
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
            extra={
                "ttl": sly.env.smart_cache_ttl(),
                "maxsize": sly.env.smart_cache_size(),
                "path": sly.env.smart_cache_container_dir(),
            },
        )

    def _deserialize_geometry(self, data: dict):
        geometry_type_str = data["type"]
        geometry_json = data["data"]
        return sly.deserialize_geometry(geometry_type_str, geometry_json)

    def _on_inference_start(self, inference_request_uuid: str):
        super()._on_inference_start(inference_request_uuid)
        self._inference_requests[inference_request_uuid]["lock"] = Lock()

    def get_info(self):
        info = super().get_info()
        info["task type"] = "tracking"
        # recommended parameters:
        # info["model_name"] = ""
        # info["checkpoint_name"] = ""
        # info["pretrained_on_dataset"] = ""
        # info["device"] = ""
        return info

    def track_api(self, api: sly.Api, context: dict):
        # unused fields:
        context["trackId"] = "auto"
        context["objectIds"] = []
        context["figureIds"] = []
        if "direction" not in context:
            context["direction"] = "forward"

        input_geometries: list = context["input_geometries"]

        if self.custom_inference_settings_dict.get("load_all_frames"):
            load_all_frames = True
        else:
            load_all_frames = False
        video_interface = TrackerInterface(
            context=context,
            api=api,
            load_all_frames=load_all_frames,
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

        api.logger.info("Start tracking.")

        predictions = []
        for _ in video_interface.frames_loader_generator():
            for input_geom in input_geometries:
                geom = self._deserialize_geometry(input_geom)
                if isinstance(geom, sly.Point):
                    geometries = self._predict_point_geometries(
                        geom,
                        video_interface.frames,
                    )
                elif isinstance(geom, sly.Polygon):
                    if len(geom.interior) > 0:
                        raise ValueError("Can't track polygons with interior.")
                    geometries = self._predict_polygon_geometries(
                        geom,
                        video_interface.frames,
                    )
                elif isinstance(geom, sly.GraphNodes):
                    geometries = self._predict_graph_geometries(
                        geom,
                        video_interface.frames,
                    )
                elif isinstance(geom, sly.Polyline):
                    geometries = self._predict_polyline_geometries(
                        geom,
                        video_interface.frames,
                    )
                else:
                    raise TypeError(f"Tracking does not work with {geom.geometry_name()}.")

                if video_interface.global_stop_indicatior:
                    return

                geometries = [{"type": g.geometry_name(), "data": g.to_json()} for g in geometries]
                predictions.append(geometries)

        # predictions must be NxK figures: N=number of frames, K=number of objects
        predictions = list(map(list, zip(*predictions)))
        return predictions

    def _inference(self, frames: List[np.ndarray], geometries: List[Geometry], settings: dict):
        updated_settings = {
            **self.custom_inference_settings_dict,
            **settings,
        }
        results = [[] for _ in range(len(frames) - 1)]
        for geometry in geometries:
            if isinstance(geometry, sly.Point):
                predictions = self._predict_point_geometries(geometry, frames, updated_settings)
            elif isinstance(geometry, sly.Polygon):
                if len(geometry.interior) > 0:
                    raise ValueError("Can't track polygons with interior.")
                predictions = self._predict_polygon_geometries(
                    geometry,
                    frames,
                    updated_settings,
                )
            elif isinstance(geometry, sly.GraphNodes):
                predictions = self._predict_graph_geometries(
                    geometry,
                    frames,
                    updated_settings,
                )
            elif isinstance(geometry, sly.Polyline):
                predictions = self._predict_polyline_geometries(
                    geometry,
                    frames,
                    updated_settings,
                )
            else:
                raise TypeError(f"Tracking does not work with {geometry.geometry_name()}.")

            for i, prediction in enumerate(predictions):
                results[i].append({"type": geometry.geometry_name(), "data": prediction.to_json()})

        return results

    def track_api_cached(self, request: Request, context: dict):
        sly.logger.info(f"Start tracking with settings: {context}.")
        video_id = context["video_id"]
        frame_indexes = list(
            range(context["frame_index"], context["frame_index"] + context["frames"] + 1)
        )
        geometries = map(self._deserialize_geometry, context["input_geometries"])
        frames = self.cache.get_frames_from_cache(video_id, frame_indexes)
        return self._inference(frames, geometries, context)

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

    def _track_async(self, api: sly.Api, context: dict, inference_request_uuid: str):
        api.logger.info("context", extra=context)
        inference_request = self._inference_requests[inference_request_uuid]

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
                    elif stop_event.is_set():
                        return
                    time.sleep(1)
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
            frames = self.cache.download_frames(
                api, video_id, list(range(*range_of_frames, direction_n))
            )
            for figure in figures:
                figure = api.video.figure._convert_json_info(figure)
                api.logger.info("geometry:", extra={"figure": figure._asdict()})
                sly_geometry: sly.Rectangle = sly.deserialize_geometry(
                    figure.geometry_type, figure.geometry
                )
                api.logger.info("geometry:", extra={"geometry": type(sly_geometry)})
                if isinstance(sly_geometry, sly.Point):
                    geometries = self._predict_point_geometries(
                        sly_geometry,
                        frames,
                    )
                elif isinstance(sly_geometry, sly.Polygon):
                    if len(sly_geometry.interior) > 0:
                        stop_upload_event.set()
                        raise ValueError("Can't track polygons with interior.")
                    geometries = self._predict_polygon_geometries(
                        sly_geometry,
                        frames,
                    )
                elif isinstance(sly_geometry, sly.GraphNodes):
                    geometries = self._predict_graph_geometries(
                        sly_geometry,
                        frames,
                    )
                elif isinstance(sly_geometry, sly.Polyline):
                    geometries = self._predict_polyline_geometries(
                        sly_geometry,
                        frames,
                    )
                else:
                    raise TypeError(f"Tracking does not work with {sly_geometry.geometry_name()}.")

                for i, geometry in enumerate(geometries, 1):
                    upload_queue.put(
                        (
                            geometry,
                            figure.object_id,
                            frame_index + i * direction_n,
                        )
                    )
                api.logger.info(f"Figure #{figure.id} tracked.")

                if global_stop_indicatior:
                    stop_upload_event.set()
                    return
        except Exception as e:
            if direct_progress:
                api.vid_ann_tool.set_direct_tracking_error(
                    session_id,
                    video_id,
                    track_id,
                    message=f"An error occured during tracking. Error: {e}",
                )
            progress.message = "Error occured during tracking"
            progress.set(current=0, total=1, report=True)
            raise
        else:
            progress.message = "Ready"
            progress.set(current=0, total=1, report=True)
        finally:
            stop_upload_event.set()
            if upload_thread.is_alive():
                upload_thread.join()
            if notify_thread.is_alive():
                notify_thread.join()

    def serve(self):
        super().serve()
        server = self._app.get_server()
        self.cache.add_cache_endpoint(server)
        self.cache.add_cache_files_endpoint(server)

        @server.post("/track")
        def start_track(request: Request, task: BackgroundTasks):
            task.add_task(track, request)
            return {"message": "Track task started."}

        @server.post("/track-api")
        def track_api(request: Request):
            return self.track_api(request.state.api, request.state.context)

        @server.post("/track-api-files")
        def track_api_frames_files(
            request: Request,
            files: List[UploadFile],
            settings: str = Form("{}"),
        ):
            return self._track_api_files(request, files, settings)

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
            context = request.state.context
            api: sly.Api = request.state.api

            if self.custom_inference_settings_dict.get("load_all_frames"):
                load_all_frames = True
            else:
                load_all_frames = False
            video_interface = TrackerInterface(
                context=context,
                api=api,
                load_all_frames=load_all_frames,
                frame_loader=self.cache.download_frame,
                frames_loader=self.cache.download_frames,
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
                for _ in video_interface.frames_loader_generator():
                    for (fig_id, geom), obj_id in zip(
                        video_interface.geometries.items(),
                        video_interface.object_ids,
                    ):
                        if isinstance(geom, sly.Point):
                            geometries = self._predict_point_geometries(
                                geom,
                                video_interface.frames_with_notification,
                            )
                        elif isinstance(geom, sly.Polygon):
                            if len(geom.interior) > 0:
                                stop_upload_event.set()
                                raise ValueError("Can't track polygons with interior.")
                            geometries = self._predict_polygon_geometries(
                                geom,
                                video_interface.frames_with_notification,
                            )
                        elif isinstance(geom, sly.GraphNodes):
                            geometries = self._predict_graph_geometries(
                                geom,
                                video_interface.frames_with_notification,
                            )
                        elif isinstance(geom, sly.Polyline):
                            geometries = self._predict_polyline_geometries(
                                geom,
                                video_interface.frames_with_notification,
                            )
                        else:
                            raise TypeError(f"Tracking does not work with {geom.geometry_name()}.")

                        for frame_idx, geometry in zip(
                            video_interface._cur_frames_indexes[1:], geometries
                        ):
                            upload_queue.put(
                                (
                                    geometry,
                                    obj_id,
                                    frame_idx,
                                )
                            )
                        api.logger.info(f"Object #{obj_id} tracked.")

                        if video_interface.global_stop_indicatior:
                            stop_upload_event.set()
                            return
            except Exception:
                stop_upload_event.set()
                raise
            stop_upload_event.set()

        @server.post("/track_async")
        def track_async(response: Response, request: Request):
            sly.logger.debug(f"'track_async' request in json format:{request.state.context}")
            # check batch size
            batch_size = request.state.context.get("batch_size", None)
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
                response.status_code = status.HTTP_400_BAD_REQUEST
                sly.logger.error("Error: 'inference_request_uuid' is required.")
                return {"message": "Error: 'inference_request_uuid' is required."}

            inference_request = self._inference_requests[inference_request_uuid]
            sly.logger.debug(
                "Pop tracking results",
                extra={
                    "inference_request_uuid": inference_request_uuid,
                    "pending_results_len": len(inference_request["pending_results"]),
                    "pending_results": [
                        figure._asdict() for figure in inference_request["pending_results"][:3]
                    ],
                },
            )
            frame_range = context.get("frame_range", None)
            if frame_range is None:
                frame_range = context.get("frameRange", None)
            sly.logger.debug("frame_range: %s", frame_range)
            with inference_request["lock"]:
                inference_request_copy = inference_request.copy()
                inference_request_copy.pop("lock")
                inference_request_copy["progress"] = _convert_sly_progress_to_dict(
                    inference_request_copy["progress"]
                )

                if frame_range is not None:
                    inference_request_copy["pending_results"] = [
                        figure
                        for figure in inference_request_copy["pending_results"]
                        if figure.frame_index >= frame_range[0]
                        and figure.frame_index <= frame_range[1]
                    ]
                    inference_request["pending_results"] = [
                        figure
                        for figure in inference_request["pending_results"]
                        if figure.frame_index < frame_range[0]
                        or figure.frame_index > frame_range[1]
                    ]
                else:
                    inference_request["pending_results"] = []

            sly.logger.debug(
                "inference_request_copy", extra={"inference_request_copy": inference_request_copy}
            )

            inference_request_copy["pending_results"] = [
                {
                    ApiField.ID: figure.id,
                    ApiField.OBJECT_ID: figure.object_id,
                    ApiField.GEOMETRY_TYPE: figure.geometry_type,
                    ApiField.GEOMETRY: figure.geometry,
                    ApiField.META: {ApiField.FRAME: figure.frame_index},
                }
                for figure in inference_request_copy["pending_results"]
            ]

            sly.logger.debug(
                "inference_request_copy", extra={"inference_request_copy": inference_request_copy}
            )

            # Logging
            log_extra = _get_log_extra_for_inference_request(
                inference_request_uuid, inference_request_copy
            )
            sly.logger.debug(f"Sending inference delta results with uuid:", extra=log_extra)
            return inference_request_copy

    def predict(
        self,
        rgb_images: List[np.ndarray],
        settings: Dict[str, Any],
        start_object: Union[PredictionPoint, List[PredictionPoint]],
    ) -> List[PredictionPoint]:
        """
        Track point on given frames.

        :param rgb_images: RGB frames, `m` frames
        :type rgb_images: List[np.array]
        :param settings: model parameters
        :type settings: Dict[str, Any]
        :param start_object: point to track on the initial frame
        :type start_object: PredictionPoint
        :return: predicted points for frame range (0, m]; `m-1` prediction in total
        :rtype: List[PredictionPoint]
        """
        raise NotImplementedError

    def predict_batch(
        self,
        rgb_images: List[np.ndarray],
        settings: Dict[str, Any],
        start_objects: List[PredictionPoint],
    ) -> List[List[PredictionPoint]]:
        """
        Track points on given frames.

        :param rgb_images: RGB frames, `m` frames
        :type rgb_images: List[np.array]
        :param settings: model parameters
        :type settings: Dict[str, Any]
        :param start_objects: points to track on the initial frame
        :type start_objects: List[PredictionPoint]
        :return: predicted points for frame range (0, m]; `m-1` prediction in total
        :rtype: List[List[PredictionPoint]]
        """
        raise NotImplementedError

    def visualize(
        self,
        predictions: List[PredictionPoint],
        images: List[np.ndarray],
        vis_path: str,
        thickness: int = 2,
        classes_whitelist: List[str] = None,
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

    def _create_label(self, dto: PredictionPoint) -> sly.Point:
        geometry = sly.Point(row=dto.row, col=dto.col)
        return Label(geometry, sly.ObjClass("", sly.Point))

    def _get_obj_class_shape(self):
        return sly.Point

    def _predict_point_geometries(
        self,
        geom: sly.Point,
        frames: List[np.ndarray],
        settings: Dict[str, Any] = None,
    ) -> List[sly.Point]:
        if settings is None:
            settings = self.custom_inference_settings_dict
        pp_geom = PredictionPoint("point", col=geom.col, row=geom.row)
        if type(self).predict_batch == PointTracking.predict_batch:
            # if predict_batch is not implemented, we can't use it
            predicted = self.predict(
                frames,
                settings,
                pp_geom,
            )
        else:
            predicted = self.predict_batch(
                frames,
                settings,
                pp_geom,
            )
            predicted = [pred[0] for pred in predicted]
        return F.dto_points_to_sly_points(predicted)

    def _predict_polygon_geometries(
        self,
        geom: sly.Polygon,
        frames: List[np.ndarray],
        settings: Dict[str, Any] = None,
    ) -> List[sly.Polygon]:
        if settings is None:
            settings = self.custom_inference_settings_dict
        polygon_points = F.numpy_to_dto_point(geom.exterior_np, "polygon")

        if type(self).predict_batch == PointTracking.predict_batch:
            # if predict_batch is not implemented, we can't use it
            points = list(
                zip(
                    *[
                        self.predict(frames, settings, polygon_point)
                        for polygon_point in polygon_points
                    ]
                )
            )
        else:
            points: List[List[Prediction]] = self.predict_batch(
                frames,
                settings,
                polygon_points,
            )
        points_loc = [F.dto_points_to_point_location(frame_points) for frame_points in points]
        return F.exteriors_to_sly_polygons(points_loc)

    def _predict_graph_geometries(
        self,
        geom: sly.GraphNodes,
        frames: List[np.ndarray],
        settings: Dict[str, Any] = None,
    ) -> List[sly.GraphNodes]:
        if settings is None:
            settings = self.custom_inference_settings_dict
        points, pids = F.graph_to_dto_points(geom)

        if type(self).predict_batch == PointTracking.predict_batch:
            # if predict_batch is not implemented, we can't use it
            preds = list(zip(*[self.predict(frames, settings, point) for point in points]))
        else:
            preds: List[List[PredictionPoint]] = self.predict_batch(
                frames,
                settings,
                points,
            )

        nodes = []
        for frame_preds in preds:
            frame_nodes = []
            for pred, pid in zip(frame_preds, pids):
                frame_nodes.extend(F.dto_points_to_sly_nodes([pred], pid))
            nodes.append(frame_nodes)

        return F.nodes_to_sly_graph(nodes)

    def _predict_polyline_geometries(
        self,
        geom: sly.Polyline,
        frames: List[np.ndarray],
        settings: Dict[str, Any] = None,
    ) -> List[sly.Polyline]:
        if settings is None:
            settings = self.custom_inference_settings_dict
        polyline_points = F.numpy_to_dto_point(geom.exterior_np, "polyline")
        if type(self).predict_batch == PointTracking.predict_batch:
            # if predict_batch is not implemented, we can't use it
            preds = list(
                zip(
                    *[
                        self.predict(frames, settings, polyline_point)
                        for polyline_point in polyline_points
                    ]
                )
            )
        else:
            preds = self.predict_batch(
                frames,
                settings,
                polyline_points,
            )
        points_loc = [F.dto_points_to_point_location(frame_points) for frame_points in preds]
        return F.exterior_to_sly_polyline(points_loc)

    def _predictions_to_annotation(
        self, image: np.ndarray, predictions: List[Prediction], classes_whitelist: List[str] = None
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

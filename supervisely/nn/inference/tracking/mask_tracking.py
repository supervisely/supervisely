import functools
import json
import time
import uuid
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
from supervisely.nn.prediction_dto import PredictionSegmentation


class MaskTracking(Inference):
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
        return info

    def _track_api(self, api: sly.Api, context: dict):
        # unused fields:
        context["trackId"] = "auto"
        context["objectIds"] = []
        context["figureIds"] = []
        if "direction" not in context:
            context["direction"] = "forward"

        input_geometries: list = context["input_geometries"]

        self.video_interface = TrackerInterface(
            context=context,
            api=api,
            load_all_frames=True,
            notify_in_predict=True,
            per_point_polygon_tracking=False,
            frame_loader=self.cache.download_frame,
            frames_loader=self.cache.download_frames,
            should_notify=False,
        )

        range_of_frames = [
            self.video_interface.frames_indexes[0],
            self.video_interface.frames_indexes[-1],
        ]

        if self.cache.is_persistent:
            # if cache is persistent, run cache task for whole video
            self.cache.run_cache_task_manually(
                api,
                None,
                video_id=self.video_interface.video_id,
            )
        else:
            # if cache is not persistent, run cache task for range of frames
            self.cache.run_cache_task_manually(
                api,
                [range_of_frames],
                video_id=self.video_interface.video_id,
            )

        api.logger.info("Starting tracking process")
        # load frames
        frames = self.video_interface.frames
        # combine several binary masks into one multilabel mask
        i = 0
        label2id = {}

        for input_geom in input_geometries:
            geometry = self._deserialize_geometry(input_geom)
            if not isinstance(geometry, sly.Bitmap) and not isinstance(geometry, sly.Polygon):
                raise TypeError(f"This app does not support {geometry.geometry_name()} tracking")
            # convert polygon to bitmap
            if isinstance(geometry, sly.Polygon):
                polygon_obj_class = sly.ObjClass("polygon", sly.Polygon)
                polygon_label = sly.Label(geometry, polygon_obj_class)
                bitmap_obj_class = sly.ObjClass("bitmap", sly.Bitmap)
                bitmap_label = polygon_label.convert(bitmap_obj_class)[0]
                geometry = bitmap_label.geometry
            if i == 0:
                multilabel_mask = geometry.data.astype(int)
                multilabel_mask = np.zeros(frames[0].shape, dtype=np.uint8)
                geometry.draw(bitmap=multilabel_mask, color=[1, 1, 1])
                i += 1
            else:
                i += 1
                geometry.draw(bitmap=multilabel_mask, color=[i, i, i])
            label2id[i] = {
                "original_geometry": geometry.geometry_name(),
            }

        # run tracker
        tracked_multilabel_masks = self.predict(frames=frames, input_mask=multilabel_mask[:, :, 0])
        tracked_multilabel_masks = np.array(tracked_multilabel_masks)

        predictions = []
        # decompose multilabel masks into binary masks
        for i in np.unique(tracked_multilabel_masks):
            if i != 0:
                predictions_for_label = []
                binary_masks = tracked_multilabel_masks == i
                geometry_type = label2id[i]["original_geometry"]
                for j, mask in enumerate(binary_masks[1:]):
                    # check if mask is not empty
                    if not np.any(mask):
                        api.logger.info(f"Empty mask on frame {context['frameIndex'] + j + 1}")
                        predictions_for_label.append(None)
                    else:
                        # frame_idx = j + 1
                        geometry = sly.Bitmap(mask)
                        predictions_for_label.append(
                            {"type": geometry.geometry_name(), "data": geometry.to_json()}
                        )
                predictions.append(predictions_for_label)

        # predictions must be NxK masks: N=number of frames, K=number of objects
        predictions = list(map(list, zip(*predictions)))
        return predictions

    def _inference(self, frames: List[np.ndarray], geometries: List[Geometry]):
        results = [[] for _ in range(len(frames) - 1)]
        for geometry in geometries:
            if not isinstance(geometry, sly.Bitmap) and not isinstance(geometry, sly.Polygon):
                raise TypeError(f"This app does not support {geometry.geometry_name()} tracking")

            # combine several binary masks into one multilabel mask
            i = 0
            label2id = {}

            original_geometry = geometry.clone()

            # convert polygon to bitmap
            if isinstance(geometry, sly.Polygon):
                polygon_obj_class = sly.ObjClass("polygon", sly.Polygon)
                polygon_label = sly.Label(geometry, polygon_obj_class)
                bitmap_obj_class = sly.ObjClass("bitmap", sly.Bitmap)
                bitmap_label = polygon_label.convert(bitmap_obj_class)[0]
                geometry = bitmap_label.geometry
            if i == 0:
                multilabel_mask = geometry.data.astype(int)
                multilabel_mask = np.zeros(frames[0].shape, dtype=np.uint8)
                geometry.draw(bitmap=multilabel_mask, color=[1, 1, 1])
                i += 1
            else:
                i += 1
                geometry.draw(bitmap=multilabel_mask, color=[i, i, i])
            label2id[i] = {
                "original_geometry": original_geometry.geometry_name(),
            }
            # run tracker
            tracked_multilabel_masks = self.predict(
                frames=frames, input_mask=multilabel_mask[:, :, 0]
            )
            tracked_multilabel_masks = np.array(tracked_multilabel_masks)
            # decompose multilabel masks into binary masks
            for i in np.unique(tracked_multilabel_masks):
                if i != 0:
                    binary_masks = tracked_multilabel_masks == i
                    geometry_type = label2id[i]["original_geometry"]
                    for j, mask in enumerate(binary_masks[1:]):
                        # check if mask is not empty
                        if np.any(mask):
                            if geometry_type == "polygon":
                                bitmap_geometry = sly.Bitmap(mask)
                                bitmap_obj_class = sly.ObjClass("bitmap", sly.Bitmap)
                                bitmap_label = sly.Label(bitmap_geometry, bitmap_obj_class)
                                polygon_obj_class = sly.ObjClass("polygon", sly.Polygon)
                                polygon_labels = bitmap_label.convert(polygon_obj_class)
                                geometries = [label.geometry for label in polygon_labels]
                            else:
                                geometries = [sly.Bitmap(mask)]
                            results[j].extend(
                                [
                                    {"type": geom.geometry_name(), "data": geom.to_json()}
                                    for geom in geometries
                                ]
                            )
        return results

    def _track_api_cached(self, request: Request, context: dict):
        sly.logger.info(f"Start tracking with settings: {context}.")
        video_id = context["video_id"]
        frame_indexes = list(
            range(context["frame_index"], context["frame_index"] + context["frames"] + 1)
        )
        geometries = map(self._deserialize_geometry, context["input_geometries"])
        frames = self.cache.get_frames_from_cache(video_id, frame_indexes)
        return self._inference(frames, geometries)

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
        return self._inference(frames, geometries)

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
            return self._track_api(request.state.api, request.state.context)

        @server.post("/track-api-files")
        def track_api_files(
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
                    # print("An error occured:")
                    # print(traceback.format_exc())
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

            self.video_interface = TrackerInterface(
                context=context,
                api=api,
                load_all_frames=True,
                notify_in_predict=True,
                per_point_polygon_tracking=False,
                frame_loader=self.cache.download_frame,
                frames_loader=self.cache.download_frames,
            )
            range_of_frames = [
                self.video_interface.frames_indexes[0],
                self.video_interface.frames_indexes[-1],
            ]

            if self.cache.is_persistent:
                # if cache is persistent, run cache task for whole video
                self.cache.run_cache_task_manually(
                    api,
                    None,
                    video_id=self.video_interface.video_id,
                )
            else:
                # if cache is not persistent, run cache task for range of frames
                self.cache.run_cache_task_manually(
                    api,
                    [range_of_frames],
                    video_id=self.video_interface.video_id,
                )

            api.logger.info("Starting tracking process")
            # load frames
            frames = self.video_interface.frames
            # combine several binary masks into one multilabel mask
            i = 0
            label2id = {}

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
                args=[upload_queue, stop_upload_event, self.video_interface],
                daemon=True,
            ).start()

            try:
                for (fig_id, geometry), obj_id in zip(
                    self.video_interface.geometries.items(),
                    self.video_interface.object_ids,
                ):
                    original_geometry = geometry.clone()
                    if not isinstance(geometry, sly.Bitmap) and not isinstance(
                        geometry, sly.Polygon
                    ):
                        stop_upload_event.set()
                        raise TypeError(
                            f"This app does not support {geometry.geometry_name()} tracking"
                        )
                    # convert polygon to bitmap
                    if isinstance(geometry, sly.Polygon):
                        polygon_obj_class = sly.ObjClass("polygon", sly.Polygon)
                        polygon_label = sly.Label(geometry, polygon_obj_class)
                        bitmap_obj_class = sly.ObjClass("bitmap", sly.Bitmap)
                        bitmap_label = polygon_label.convert(bitmap_obj_class)[0]
                        geometry = bitmap_label.geometry
                    if i == 0:
                        multilabel_mask = geometry.data.astype(int)
                        multilabel_mask = np.zeros(frames[0].shape, dtype=np.uint8)
                        geometry.draw(bitmap=multilabel_mask, color=[1, 1, 1])
                        i += 1
                    else:
                        i += 1
                        geometry.draw(bitmap=multilabel_mask, color=[i, i, i])
                    label2id[i] = {
                        "fig_id": fig_id,
                        "obj_id": obj_id,
                        "original_geometry": original_geometry.geometry_name(),
                    }
                # run tracker
                tracked_multilabel_masks = self.predict(
                    frames=frames, input_mask=multilabel_mask[:, :, 0]
                )
                tracked_multilabel_masks = np.array(tracked_multilabel_masks)
                # decompose multilabel masks into binary masks
                for i in np.unique(tracked_multilabel_masks):
                    if i != 0:
                        binary_masks = tracked_multilabel_masks == i
                        fig_id = label2id[i]["fig_id"]
                        obj_id = label2id[i]["obj_id"]
                        geometry_type = label2id[i]["original_geometry"]
                        for j, mask in enumerate(binary_masks[1:]):
                            # check if mask is not empty
                            if not np.any(mask):
                                api.logger.info(
                                    f"Skipping empty mask on frame {self.video_interface.frame_index + j + 1}"
                                )
                                # update progress bar anyway (otherwise it will not be finished)
                                self.video_interface._notify(task="add geometry on frame")
                            else:
                                if geometry_type == "polygon":
                                    bitmap_geometry = sly.Bitmap(mask)
                                    bitmap_obj_class = sly.ObjClass("bitmap", sly.Bitmap)
                                    bitmap_label = sly.Label(bitmap_geometry, bitmap_obj_class)
                                    polygon_obj_class = sly.ObjClass("polygon", sly.Polygon)
                                    polygon_labels = bitmap_label.convert(polygon_obj_class)
                                    geometries = [label.geometry for label in polygon_labels]
                                else:
                                    geometries = [sly.Bitmap(mask)]
                                for l, geometry in enumerate(geometries):
                                    if l == len(geometries) - 1:
                                        notify = True
                                    else:
                                        notify = False
                                    upload_queue.put(
                                        (
                                            geometry,
                                            obj_id,
                                            self.video_interface.frames_indexes[j + 1],
                                            notify,
                                        )
                                    )
                        if self.video_interface.global_stop_indicatior:
                            stop_upload_event.set()
                            return
                        api.logger.info(f"Figure with id {fig_id} was successfully tracked")
            except Exception:
                stop_upload_event.set()
                raise
            stop_upload_event.set()

        def _track_async(api: sly.Api, context, inference_request_uuid):
            inference_request = self._inference_requests[inference_request_uuid]

            session_id = context["sessionId"]
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

            # combine several binary masks into one multilabel mask
            i = 0
            label2id = {}

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
                                _add_to_inference_request(*item, figure_id)
                                if direct_progress:
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
            notify_thread = Thread(
                target=_upload_loop,
                args=[upload_queue, notify_queue, stop_upload_event],
                daemon=True,
            )
            notify_thread.start()
            upload_thread = Thread(
                target=_nofify_loop,
                args=[notify_queue, stop_upload_event],
                daemon=True,
            )
            upload_thread.start()

            # run tracker
            api.logger.info("Starting tracking process")
            try:
                frames = self.cache.download_frames(
                    api,
                    video_id,
                    list(range(range_of_frames[0], range_of_frames[1] + direction_n, direction_n)),
                )
                for figure in figures:
                    figure = api.video.figure._convert_json_info(figure)
                    fig_id = figure.id
                    obj_id = figure.object_id
                    geometry = sly.deserialize_geometry(figure.geometry_type, figure.geometry)
                    original_geometry = geometry.clone()
                    if not isinstance(geometry, sly.Bitmap) and not isinstance(
                        geometry, sly.Polygon
                    ):
                        stop_upload_event.set()
                        raise TypeError(
                            f"This app does not support {geometry.geometry_name()} tracking"
                        )
                    # convert polygon to bitmap
                    if isinstance(geometry, sly.Polygon):
                        polygon_obj_class = sly.ObjClass("polygon", sly.Polygon)
                        polygon_label = sly.Label(geometry, polygon_obj_class)
                        bitmap_obj_class = sly.ObjClass("bitmap", sly.Bitmap)
                        bitmap_label = polygon_label.convert(bitmap_obj_class)[0]
                        geometry = bitmap_label.geometry
                    if i == 0:
                        multilabel_mask = geometry.data.astype(int)
                        multilabel_mask = np.zeros(frames[0].shape, dtype=np.uint8)
                        geometry.draw(bitmap=multilabel_mask, color=[1, 1, 1])
                        i += 1
                    else:
                        i += 1
                        geometry.draw(bitmap=multilabel_mask, color=[i, i, i])
                    label2id[i] = {
                        "fig_id": fig_id,
                        "obj_id": obj_id,
                        "original_geometry": original_geometry.geometry_name(),
                    }
                tracked_multilabel_masks = self.predict(
                    frames=frames, input_mask=multilabel_mask[:, :, 0]
                )
                tracked_multilabel_masks = np.array(tracked_multilabel_masks)
                # decompose multilabel masks into binary masks
                for i in np.unique(tracked_multilabel_masks):
                    if i != 0:
                        binary_masks = tracked_multilabel_masks == i
                        fig_id = label2id[i]["fig_id"]
                        obj_id = label2id[i]["obj_id"]
                        geometry_type = label2id[i]["original_geometry"]
                        for j, mask in enumerate(binary_masks[1:]):
                            # check if mask is not empty
                            if not np.any(mask):
                                api.logger.info(
                                    f"Skipping empty mask on frame {frame_index + j + 1}"
                                )
                                # update progress bar anyway (otherwise it will not be finished)
                                progress.iter_done_report()
                            else:
                                if geometry_type == "polygon":
                                    bitmap_geometry = sly.Bitmap(mask)
                                    bitmap_obj_class = sly.ObjClass("bitmap", sly.Bitmap)
                                    bitmap_label = sly.Label(bitmap_geometry, bitmap_obj_class)
                                    polygon_obj_class = sly.ObjClass("polygon", sly.Polygon)
                                    polygon_labels = bitmap_label.convert(polygon_obj_class)
                                    geometries = [label.geometry for label in polygon_labels]
                                else:
                                    geometries = [sly.Bitmap(mask)]
                                for l, geometry in enumerate(geometries):
                                    upload_queue.put(
                                        (
                                            geometry,
                                            obj_id,
                                            frame_index + j + 1,
                                        )
                                    )
                        if global_stop_indicatior:
                            stop_upload_event.set()
                            return
                        api.logger.info(f"Figure with id {fig_id} was successfully tracked")
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
                _track_async,
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
            inference_request_uuid = request.state.state.get("inference_request_uuid")
            if inference_request_uuid is None:
                response.status_code = status.HTTP_400_BAD_REQUEST
                return {"message": "Error: 'inference_request_uuid' is required."}

            inference_request = self._inference_requests[inference_request_uuid]
            frame_range = request.state.state.get("frame_range", None)
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

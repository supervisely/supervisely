import functools
import json
import time
from pathlib import Path
from queue import Queue
from threading import Event, Thread
from typing import Any, Dict, List, Optional, Union

import numpy as np
from fastapi import BackgroundTasks, Form, Request, UploadFile

import supervisely as sly
import supervisely.nn.inference.tracking.functional as F
from supervisely.annotation.label import Geometry, Label
from supervisely.nn.inference import Inference
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

    def _track_api(self, api: sly.Api, context: dict):
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

        api.logger.info("Start tracking.")

        predictions = []
        for input_geom in input_bboxes:
            input_bbox = input_geom["data"]
            bbox = sly.Rectangle.from_json(input_bbox)
            predictions_for_object = []
            init = False
            for _ in video_interface.frames_loader_generator():
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
            predictions.append(predictions_for_object)

        # predictions must be NxK bboxes: N=number of frames, K=number of objects
        predictions = list(map(list, zip(*predictions)))
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
            sly.logger.info("Start tracking.")
            return self._track_api(request.state.api, request.state.context)

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

import time
import uuid
from pathlib import Path
from queue import Queue
from threading import Event, Thread
from typing import Any, BinaryIO, Dict, List, Optional

import numpy as np
from pydantic import ValidationError

from supervisely.annotation.annotation import Annotation
from supervisely.annotation.label import Geometry, Label
from supervisely.annotation.obj_class import ObjClass
from supervisely.api.api import Api
from supervisely.api.module_api import ApiField
from supervisely.api.video.video_figure_api import FigureInfo
from supervisely.geometry.helpers import deserialize_geometry
from supervisely.geometry.rectangle import Rectangle
from supervisely.imaging import image as sly_image
from supervisely.nn.inference.tracking.base_tracking import BaseTracking
from supervisely.nn.inference.tracking.tracker_interface import (
    TrackerInterface,
    TrackerInterfaceV2,
)
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

    def _track_async(self, api: Api, context: dict, inference_request_uuid: str = None):
        inference_request = self._inference_requests[inference_request_uuid]
        tracker_interface = TrackerInterfaceV2(api, context, self.cache)
        progress: Progress = inference_request["progress"]
        frames_count = tracker_interface.frames_count
        figures = tracker_interface.figures
        progress_total = frames_count * len(figures)
        progress.total = progress_total

        def _upload_f(items: List[FigureInfo]):
            with inference_request["lock"]:
                inference_request["pending_results"].extend(items)

        def _notify_f(items: List[FigureInfo]):
            items_by_object_id: Dict[int, List[FigureInfo]] = {}
            for item in items:
                items_by_object_id.setdefault(item.object_id, []).append(item)

            for object_id, object_items in items_by_object_id.items():
                frame_range = [
                    min(item.frame_index for item in object_items),
                    max(item.frame_index for item in object_items),
                ]
                progress.iters_done_report(len(object_items))
                tracker_interface.notify_progress(progress.current, progress.total, frame_range)

        api.logger.info("Start tracking.")
        try:
            with tracker_interface(_upload_f, _notify_f):
                for fig_i, figure in enumerate(figures, 1):
                    figure = api.video.figure._convert_json_info(figure)
                    if not figure.geometry_type == Rectangle.geometry_name():
                        raise TypeError(f"Tracking does not work with {figure.geometry_type}.")
                    api.logger.info("figure:", extra={"figure": figure._asdict()})
                    sly_geometry: Rectangle = deserialize_geometry(
                        figure.geometry_type, figure.geometry
                    )
                    init = False
                    for frame_i, (frame, next_frame) in enumerate(
                        tracker_interface.frames_loader_generator(), 1
                    ):
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
                            self.initialize(frame.image, target)
                            init = True

                        logger.debug("Start prediction")
                        t = time.time()
                        geometry = self.predict(
                            rgb_image=next_frame.image,
                            prev_rgb_image=frame.image,
                            target_bbox=target,
                            settings=self.custom_inference_settings_dict,
                        )
                        logger.debug("Prediction done. Time: %f sec", time.time() - t)
                        sly_geometry = self._to_sly_geometry(geometry)

                        figure_id = uuid.uuid5(
                            namespace=uuid.NAMESPACE_URL, name=f"{time.time()}"
                        ).hex
                        result_figure = api.video.figure._convert_json_info(
                            {
                                ApiField.ID: figure_id,
                                ApiField.OBJECT_ID: figure.object_id,
                                "meta": {"frame": next_frame.frame_index},
                                ApiField.GEOMETRY_TYPE: sly_geometry.geometry_name(),
                                ApiField.GEOMETRY: sly_geometry.to_json(),
                                ApiField.TRACK_ID: tracker_interface.track_id,
                            }
                        )

                        tracker_interface.add_prediction(result_figure)

                        logger.debug(
                            "Frame [%d / %d] processed.",
                            frame_i,
                            tracker_interface.frames_count,
                        )

                        if inference_request["cancel_inference"]:
                            return
                        if tracker_interface.is_stopped():
                            reason = tracker_interface.stop_reason()
                            if isinstance(reason, Exception):
                                raise reason
                            return

                    api.logger.info(
                        "Figure [%d, %d] tracked.",
                        fig_i,
                        len(figures),
                        extra={"figure_id": figure.id},
                    )
        except Exception:
            progress.message = "Error occured during tracking"
            raise
        else:
            progress.message = "Ready"
        finally:
            progress.set(current=0, total=1, report=True)

    def track(self, api: Api, state: Dict, context: Dict):
        fn = self.send_error_data(api, context)(self._track)
        self.schedule_task(fn, api, context, notify_annotation_tool=True)
        return {"message": "Track task started."}

    def track_api(self, api: Api, state: Dict, context: Dict):
        request_uuid = uuid.uuid5(namespace=uuid.NAMESPACE_URL, name=f"{time.time()}").hex
        result = self._track_api(api, context, request_uuid)
        logger.info("Track-api request processed.", extra={"request_uuid": request_uuid})
        return result

    def track_api_files(self, files: List[BinaryIO], settings: Dict):
        logger.info("Start tracking with settings:", extra={"settings": settings})
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
        fn = self.send_error_data(api, context)(self._track_async)
        self.schedule_task(fn, api, context, inference_request_uuid=inference_request_uuid)

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

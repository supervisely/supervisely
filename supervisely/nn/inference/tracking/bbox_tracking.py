import time
import uuid
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional

import numpy as np
from pydantic import ValidationError

from supervisely.annotation.annotation import Annotation
from supervisely.annotation.label import Geometry, Label, LabelingStatus
from supervisely.annotation.obj_class import ObjClass
from supervisely.api.api import Api
from supervisely.api.module_api import ApiField
from supervisely.api.video.video_figure_api import FigureInfo
from supervisely.geometry.helpers import deserialize_geometry
from supervisely.geometry.rectangle import Rectangle
from supervisely.imaging import image as sly_image
from supervisely.nn.inference.inference import Uploader
from supervisely.nn.inference.inference_request import InferenceRequest
from supervisely.nn.inference.tracking.base_tracking import BaseTracking
from supervisely.nn.inference.tracking.tracker_interface import (
    TrackerInterface,
    TrackerInterfaceV2,
)
from supervisely.nn.inference.uploader import Uploader
from supervisely.nn.prediction_dto import Prediction, PredictionBBox
from supervisely.sly_logger import logger


class BBoxTracking(BaseTracking):
    def _deserialize_geometry(self, data: dict):
        geometry_type_str = data["type"]
        geometry_json = data["data"]
        return deserialize_geometry(geometry_type_str, geometry_json)

    def _track(
        self,
        api: Api,
        context: dict,
        inference_request: InferenceRequest,
    ):
        video_interface = TrackerInterface(
            context=context,
            api=api,
            load_all_frames=False,
            frame_loader=self.cache.download_frame,
            frames_loader=self.cache.download_frames,
            should_notify=True,
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

        def _upload_f(items: List):
            video_interface.add_object_geometries_on_frames(*list(zip(*items)), notify=False)
            inference_request.done(len(items))

        def _notify_f(items: List):
            frame_range = [
                min(frame_index for (_, _, frame_index) in items),
                max(frame_index for (_, _, frame_index) in items),
            ]
            pos_inc = inference_request.progress.current - video_interface.global_pos

            video_interface._notify(
                pos_increment=pos_inc,
                fstart=frame_range[0],
                fend=frame_range[1],
                task=inference_request.stage,
            )

        def _exception_handler(exception: Exception):
            try:
                raise exception
            except Exception:
                api.logger.error(f"Error: {str(exception)}", exc_info=True)
            video_interface._notify(True, task="Stop tracking due to an error")
            raise exception

        api.logger.info("Start tracking.")
        total_progress = video_interface.frames_count * len(video_interface.figure_ids)
        inference_request.set_stage(InferenceRequest.Stage.INFERENCE, 0, total_progress)
        with Uploader(
            upload_f=_upload_f,
            notify_f=_notify_f,
            exception_handler=_exception_handler,
            logger=api.logger,
        ) as uploader:
            for fig_id, obj_id in zip(
                video_interface.geometries.keys(),
                video_interface.object_ids,
            ):
                init = False
                for _ in video_interface.frames_loader_generator():
                    geom = video_interface.geometries[fig_id]
                    if not isinstance(geom, Rectangle):
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

                    uploader.put([(sly_geometry, obj_id, video_interface._cur_frames_indexes[-1])])

                    if inference_request.is_stopped() or video_interface.global_stop_indicatior:
                        api.logger.info(
                            "Inference request stopped.",
                            extra={"inference_request_uuid": inference_request.uuid},
                        )
                        video_interface._notify(True, task="Stop tracking")
                        return
                    if uploader.has_exception():
                        exception = uploader.exception
                        if not isinstance(exception, Exception):
                            raise RuntimeError(
                                f"Uploader exception is not an instance of Exception: {str(exception)}"
                            )
                        raise uploader.exception

                api.logger.info(
                    f"Figure #{fig_id} tracked.",
                    extra={
                        "figure_id": fig_id,
                        "object_id": obj_id,
                        "inference_request_uuid": inference_request.uuid,
                    },
                )
            api.logger.info(
                "Finished tracking.", extra={"inference_request_uuid": inference_request.uuid}
            )
            video_interface._notify(True, task="Finished tracking")

    def _track_api(self, api: Api, context: dict, inference_request: InferenceRequest):
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

        inference_request.set_stage(InferenceRequest.Stage.INFERENCE, 0, frames_n * box_n)
        api.logger.info(
            "Start tracking.",
            extra={
                "video_id": video_interface.video_id,
                "frame_range": range_of_frames,
                "geometries_count": box_n,
                "frames_count": frames_n,
                "request_uuid": inference_request.uuid,
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
                inference_request.done()
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
                        "inference_request_uuid": inference_request.uuid,
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
                    "inference_request_uuid": inference_request.uuid,
                },
            )
            geom_t = time.monotonic()

        # predictions must be NxK bboxes: N=number of frames, K=number of objects
        predictions = list(map(list, zip(*predictions)))
        api.logger.info(
            "Tracking finished.",
            extra={
                "tracking_time": time.monotonic() - track_t,
                "inference_request_uuid": inference_request.uuid,
            },
        )
        inference_request.final_result = predictions

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

    def _track_async(self, api: Api, context: dict, inference_request: InferenceRequest):
        tracker_interface = TrackerInterfaceV2(api, context, self.cache)
        frames_count = tracker_interface.frames_count
        figures = tracker_interface.figures
        frame_range = [
            tracker_interface.frame_indexes[0],
            tracker_interface.frame_indexes[-1],
        ]
        frame_range_asc = [min(frame_range), max(frame_range)]
        progress_total = frames_count * len(figures)

        def _upload_f(items: List[FigureInfo]):
            inference_request.add_results(items)
            inference_request.done(len(items))

        def _notify_f(items: List[FigureInfo]):
            frame_range = [
                min(item.frame_index for item in items),
                max(item.frame_index for item in items),
            ]
            tracker_interface.notify_progress(
                inference_request.progress.current, inference_request.progress.total, frame_range
            )

        def _exception_handler(exception: Exception):
            api.logger.error(f"Error saving predictions: {str(exception)}", exc_info=True)
            tracker_interface.notify_progress(
                inference_request.progress.current,
                inference_request.progress.current,
                frame_range_asc,
            )
            tracker_interface.notify_error(exception)
            raise Exception

        api.logger.info("Start tracking.", extra={"inference_request_uuid": inference_request.uuid})
        inference_request.set_stage(InferenceRequest.Stage.INFERENCE, 0, progress_total)
        with Uploader(
            upload_f=_upload_f,
            notify_f=_notify_f,
            exception_handler=_exception_handler,
            logger=api.logger,
        ) as uploader:
            uploader.raise_from_notify
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

                    figure_id = uuid.uuid5(namespace=uuid.NAMESPACE_URL, name=f"{time.time()}").hex
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

                    uploader.put([result_figure])

                    logger.debug(
                        "Frame [%d / %d] processed.",
                        frame_i,
                        tracker_interface.frames_count,
                        extra={
                            "frame_index": frame_i,
                            "figure_index": fig_i,
                            "inference_request_uuid": inference_request.uuid,
                        },
                    )

                    if inference_request.is_stopped() or tracker_interface.is_stopped():
                        if isinstance(tracker_interface.stop_reason(), Exception):
                            raise tracker_interface.stop_reason()
                        api.logger.info(
                            "Inference request stopped.",
                            extra={"inference_request_uuid": inference_request.uuid},
                        )
                        tracker_interface.notify_progress(
                            inference_request.progress.current,
                            inference_request.progress.current,
                            frame_range_asc,
                        )
                        return
                    if uploader.has_exception():
                        raise uploader.exception

                api.logger.info(
                    "Figure [%d, %d] tracked.",
                    fig_i,
                    len(figures),
                    extra={
                        "figure_id": figure.id,
                        "figure_index": fig_i,
                        "inference_request_uuid": inference_request.uuid,
                    },
                )
            api.logger.info(
                "Finished tracking", extra={"inference_request_uuid": inference_request.uuid}
            )
            tracker_interface.notify_progress(
                inference_request.progress.current,
                inference_request.progress.current,
                frame_range_asc,
            )

    def track(self, api: Api, state: Dict, context: Dict):
        fn = self.send_error_data(api, context)(self._track)
        self.inference_requests_manager.schedule_task(fn, api, context)
        return {"message": "Track task started."}

    def track_api(self, api: Api, state: Dict, context: Dict):
        inference_request, future = self.inference_requests_manager.schedule_task(
            self._track_api, api, context
        )
        future.result()
        logger.info(
            "Track-api request processed.", extra={"inference_request_uuid": inference_request.uuid}
        )
        return inference_request.final_result

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

        fn = self.send_error_data(api, context)(self._track_async)
        inference_request, future = self.inference_requests_manager.schedule_task(fn, api, context)

        logger.debug(
            "Inference has scheduled from 'track_async' endpoint",
            extra={"inference_request_uuid": inference_request.uuid},
        )
        return {
            "message": "Inference has started.",
            "inference_request_uuid": inference_request.uuid,
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
                for lb in label:
                    lb.status = LabelingStatus.AUTO
                labels.extend(label)
                continue

            label.status = LabelingStatus.AUTO
            labels.append(label)

        # create annotation with correct image resolution
        ann = Annotation(img_size=image.shape[:2])
        ann = ann.add_labels(labels)
        return ann

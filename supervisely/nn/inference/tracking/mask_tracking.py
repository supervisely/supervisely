import inspect
import time
import uuid
from typing import BinaryIO, Dict, List, Tuple

import numpy as np
from pydantic import ValidationError

from supervisely._utils import find_value_by_keys, get_valid_kwargs
from supervisely.annotation.label import Geometry, Label
from supervisely.annotation.obj_class import ObjClass
from supervisely.api.api import Api
from supervisely.api.module_api import ApiField
from supervisely.api.video.video_figure_api import FigureInfo
from supervisely.geometry.bitmap import Bitmap
from supervisely.geometry.helpers import deserialize_geometry
from supervisely.geometry.polygon import Polygon
from supervisely.imaging import image as sly_image
from supervisely.nn.inference.inference_request import InferenceRequest
from supervisely.nn.inference.tracking.base_tracking import BaseTracking
from supervisely.nn.inference.tracking.tracker_interface import (
    TrackerInterface,
    TrackerInterfaceV2,
)
from supervisely.nn.inference.uploader import Uploader
from supervisely.sly_logger import logger


class MaskTracking(BaseTracking):
    def _deserialize_geometry(self, data: dict):
        geometry_type_str = data["type"]
        geometry_json = data["data"]
        return deserialize_geometry(geometry_type_str, geometry_json)

    def _inference(self, frames: List[np.ndarray], geometries: List[Geometry]):
        results = [[] for _ in range(len(frames) - 1)]
        for geometry in geometries:
            if not isinstance(geometry, Bitmap) and not isinstance(geometry, Polygon):
                raise TypeError(f"This app does not support {geometry.geometry_name()} tracking")

            # combine several binary masks into one multilabel mask
            i = 0
            label2id = {}

            original_geometry = geometry.clone()

            # convert polygon to bitmap
            if isinstance(geometry, Polygon):
                polygon_obj_class = ObjClass("polygon", Polygon)
                polygon_label = Label(geometry, polygon_obj_class)
                bitmap_obj_class = ObjClass("bitmap", Bitmap)
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
                                bitmap_geometry = Bitmap(mask)
                                bitmap_obj_class = ObjClass("bitmap", Bitmap)
                                bitmap_label = Label(bitmap_geometry, bitmap_obj_class)
                                polygon_obj_class = ObjClass("polygon", Polygon)
                                polygon_labels = bitmap_label.convert(polygon_obj_class)
                                geometries = [label.geometry for label in polygon_labels]
                            else:
                                geometries = [Bitmap(mask)]
                            results[j].extend(
                                [
                                    {"type": geom.geometry_name(), "data": geom.to_json()}
                                    for geom in geometries
                                ]
                            )
        return results

    def _track(self, api: Api, context: Dict, inference_request: InferenceRequest):
        video_interface = TrackerInterface(
            context=context,
            api=api,
            load_all_frames=False,
            notify_in_predict=True,
            per_point_polygon_tracking=False,
            frame_loader=self.cache.download_frame,
            frames_loader=self.cache.download_frames,
        )
        video_interface.stop += video_interface.frames_count + 1
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

        api.logger.debug("frames_count = %s", video_interface.frames_count)
        inference_request.set_stage("Downloading frames", 0, video_interface.frames_count)
        # load frames

        def _load_frames_cb(n: int = 1):
            inference_request.done(n)
            video_interface._notify(pos_increment=n, task="Downloading frames")

        frames = self.cache.download_frames(
            api,
            video_interface.video_id,
            video_interface.frames_indexes,
            progress_cb=_load_frames_cb,
        )

        # combine several binary masks into one multilabel mask
        i = 1
        label2id = {}
        multilabel_mask = np.zeros(frames[0].shape, dtype=np.uint8)
        for (fig_id, geometry), obj_id in zip(
            video_interface.geometries.items(),
            video_interface.object_ids,
        ):
            original_geometry = geometry.clone()
            if not isinstance(geometry, Bitmap) and not isinstance(geometry, Polygon):
                raise TypeError(f"This app does not support {geometry.geometry_name()} tracking")
            # convert polygon to bitmap
            if isinstance(geometry, Polygon):
                polygon_obj_class = ObjClass("polygon", Polygon)
                polygon_label = Label(geometry, polygon_obj_class)
                bitmap_obj_class = ObjClass("bitmap", Bitmap)
                bitmap_label = polygon_label.convert(bitmap_obj_class)[0]
                geometry = bitmap_label.geometry
            geometry.draw(bitmap=multilabel_mask, color=i)
            label2id[i] = {
                "fig_id": fig_id,
                "obj_id": obj_id,
                "original_geometry": original_geometry.geometry_name(),
            }
            i += 1

        unique_labels = np.unique(multilabel_mask)
        if 0 in unique_labels:
            unique_labels = unique_labels[1:]
        api.logger.debug("unique_labels = %s", unique_labels)
        total_progress = len(unique_labels) * video_interface.frames_count
        api.logger.info("Starting tracking process")
        api.logger.debug("total_progress = %s", total_progress)
        inference_request.set_stage(
            InferenceRequest.Stage.INFERENCE,
            0,
            total_progress,
        )

        def _upload_f(items: List):
            video_interface.add_object_geometries_on_frames(*list(zip(*items)))
            inference_request.done(sum(item[-1] for item in items))

        with Uploader(upload_f=_upload_f, logger=api.logger) as uploader:
            # run tracker
            tracked_multilabel_masks = self.predict(
                frames=frames, input_mask=multilabel_mask[:, :, 0]
            )
            for curframe_i, mask in enumerate(
                tracked_multilabel_masks, video_interface.frame_index
            ):
                if curframe_i == video_interface.frame_index:
                    continue
                for i in unique_labels:
                    binary_mask = mask == i
                    fig_id = label2id[i]["fig_id"]
                    obj_id = label2id[i]["obj_id"]
                    geometry_type = label2id[i]["original_geometry"]
                    if not np.any(binary_mask):
                        api.logger.info(f"Skipping empty mask on frame {curframe_i}")
                        inference_request.done()
                    else:
                        if geometry_type == "polygon":
                            bitmap_geometry = Bitmap(binary_mask)
                            bitmap_obj_class = ObjClass("bitmap", Bitmap)
                            bitmap_label = Label(bitmap_geometry, bitmap_obj_class)
                            polygon_obj_class = ObjClass("polygon", Polygon)
                            polygon_labels = bitmap_label.convert(polygon_obj_class)
                            geometries = [label.geometry for label in polygon_labels]
                        else:
                            geometries = [Bitmap(binary_mask)]
                        uploader.put(
                            [
                                (
                                    geometry,
                                    obj_id,
                                    curframe_i,
                                    True if g_idx == len(geometries) - 1 else False,
                                )
                                for g_idx, geometry in enumerate(geometries)
                            ]
                        )
                    if inference_request.is_stopped() or video_interface.global_stop_indicatior:
                        api.logger.info(
                            "Tracking stopped by user",
                            extra={"inference_request_uuid": inference_request.uuid},
                        )
                        video_interface._notify(True, task="Stop tracking")
                        return
                    if uploader.has_exception():
                        raise uploader.exception

                api.logger.info(f"Frame {curframe_i} was successfully tracked")

    def _track_async(self, api: Api, context: dict, inference_request: InferenceRequest):
        tracker_interface = TrackerInterfaceV2(api, context, self.cache)
        frames_count = tracker_interface.frames_count
        figures = tracker_interface.figures
        progress_total = frames_count * len(figures)
        frame_range = [
            tracker_interface.frame_indexes[0],
            tracker_interface.frame_indexes[-1],
        ]
        frame_range_asc = [min(frame_range), max(frame_range)]

        def _upload_f(items: List[Tuple[FigureInfo, bool]]):
            inference_request.add_results([item[0] for item in items])
            inference_request.done(sum(item[1] for item in items))

        def _notify_f(items: List[Tuple[FigureInfo, bool]]):
            frame_range = [
                min(item[0].frame_index for item in items),
                max(item[0].frame_index for item in items),
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

        def _maybe_stop():
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
                return True
            if uploader.has_exception():
                raise uploader.exception
            return False

        # run tracker
        frame_index = tracker_interface.frame_index
        direction_n = tracker_interface.direction_n
        api.logger.info("Start tracking.")
        inference_request.set_stage(InferenceRequest.Stage.INFERENCE, 0, progress_total)
        with Uploader(
            upload_f=_upload_f,
            notify_f=_notify_f,
            exception_handler=_exception_handler,
            logger=api.logger,
        ) as uploader:
            # combine several binary masks into one multilabel mask
            i = 0
            label2id = {}
            # load frames
            frames = tracker_interface.load_all_frames()
            frames = [frame.image for frame in frames]
            for figure in figures:
                figure = api.video.figure._convert_json_info(figure)
                fig_id = figure.id
                obj_id = figure.object_id
                geometry = deserialize_geometry(figure.geometry_type, figure.geometry)
                original_geometry = geometry.clone()
                if not isinstance(geometry, (Bitmap, Polygon)):
                    raise TypeError(
                        f"This app does not support {geometry.geometry_name()} tracking"
                    )
                # convert polygon to bitmap
                if isinstance(geometry, Polygon):
                    polygon_obj_class = ObjClass("polygon", Polygon)
                    polygon_label = Label(geometry, polygon_obj_class)
                    bitmap_obj_class = ObjClass("bitmap", Bitmap)
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

                if _maybe_stop():
                    return

            # predict
            tracked_multilabel_masks = self.predict(
                frames=frames, input_mask=multilabel_mask[:, :, 0]
            )
            tracked_multilabel_masks = np.array(tracked_multilabel_masks)

            # decompose multilabel masks into binary masks
            for i in np.unique(tracked_multilabel_masks):
                if _maybe_stop():
                    return
                if i != 0:
                    binary_masks = tracked_multilabel_masks == i
                    fig_id = label2id[i]["fig_id"]
                    obj_id = label2id[i]["obj_id"]
                    geometry_type = label2id[i]["original_geometry"]
                    for j, mask in enumerate(binary_masks[1:], 1):
                        if _maybe_stop():
                            return
                        this_figure_index = frame_index + j * direction_n
                        # check if mask is not empty
                        if not np.any(mask):
                            api.logger.info(f"Skipping empty mask on frame {this_figure_index}")
                            # update progress bar anyway (otherwise it will not be finished)
                            inference_request.done()
                        else:
                            if geometry_type == "polygon":
                                bitmap_geometry = Bitmap(mask)
                                bitmap_obj_class = ObjClass("bitmap", Bitmap)
                                bitmap_label = Label(bitmap_geometry, bitmap_obj_class)
                                polygon_obj_class = ObjClass("polygon", Polygon)
                                polygon_labels = bitmap_label.convert(polygon_obj_class)
                                geometries = [label.geometry for label in polygon_labels]
                            else:
                                geometries = [Bitmap(mask)]
                            for l, geometry in enumerate(geometries):
                                figure_id = uuid.uuid5(
                                    namespace=uuid.NAMESPACE_URL, name=f"{time.time()}"
                                ).hex
                                result_figure = api.video.figure._convert_json_info(
                                    {
                                        ApiField.ID: figure_id,
                                        ApiField.OBJECT_ID: obj_id,
                                        "meta": {"frame": this_figure_index},
                                        ApiField.GEOMETRY_TYPE: geometry.geometry_name(),
                                        ApiField.GEOMETRY: geometry.to_json(),
                                        ApiField.TRACK_ID: tracker_interface.track_id,
                                    }
                                )
                                should_notify = l == len(geometries) - 1
                                tracker_interface.add_prediction((result_figure, should_notify))
                    api.logger.info(
                        "Figure [%d, %d] tracked.",
                        i,
                        len(figures),
                        extra={"figure_id": figure.id},
                    )

    def _track_api(self, api: Api, context: Dict, inference_request: InferenceRequest):
        # unused fields:
        context["trackId"] = "auto"
        context["objectIds"] = []
        context["figureIds"] = []
        if "direction" not in context:
            context["direction"] = "forward"

        input_geometries: list = context["input_geometries"]

        video_interface = TrackerInterface(
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

        inference_request.set_stage("Downloading frames", 0, video_interface.frames_count)
        # load frames
        frames = self.cache.download_frames(
            api,
            video_interface.video_id,
            video_interface.frames_indexes,
            progress_cb=inference_request.done,
        )
        # combine several binary masks into one multilabel mask
        label2id = {}

        multilabel_mask = np.zeros(frames[0].shape, dtype=np.uint8)
        for i, input_geom in enumerate(input_geometries, 1):
            geometry = self._deserialize_geometry(input_geom)
            if not isinstance(geometry, Bitmap) and not isinstance(geometry, Polygon):
                raise TypeError(f"This app does not support {geometry.geometry_name()} tracking")
            # convert polygon to bitmap
            if isinstance(geometry, Polygon):
                polygon_obj_class = ObjClass("polygon", Polygon)
                polygon_label = Label(geometry, polygon_obj_class)
                bitmap_obj_class = ObjClass("bitmap", Bitmap)
                bitmap_label = polygon_label.convert(bitmap_obj_class)[0]
                geometry = bitmap_label.geometry
            geometry.draw(bitmap=multilabel_mask, color=i)
            label2id[i] = {
                "original_geometry": geometry.geometry_name(),
            }

        result_indexes = np.unique(multilabel_mask)
        progress_total = len(result_indexes)
        if 0 in result_indexes:
            progress_total -= 1
        progress_total = progress_total * video_interface.frames_count

        api.logger.info("Starting tracking process")
        inference_request.set_stage(
            InferenceRequest.Stage.INFERENCE,
            0,
            progress_total,
        )

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
                        geometry = Bitmap(mask)
                        predictions_for_label.append(
                            {"type": geometry.geometry_name(), "data": geometry.to_json()}
                        )
                predictions.append(predictions_for_label)

        # predictions must be NxK masks: N=number of frames, K=number of objects
        predictions = list(map(list, zip(*predictions)))
        inference_request.final_result = predictions
        return predictions

    # Implement the following methods in the derived class
    def track(self, api: Api, state: Dict, context: Dict):
        fn = self.send_error_data(api, context)(self._track)
        self.inference_requests_manager.schedule_task(fn, api, context)
        return {"message": "Tracking has started."}

    def track_api(self, api: Api, state: Dict, context: Dict):
        inference_request, future = self.inference_requests_manager.schedule_task(
            self._track_api, api, context
        )
        future.result()
        logger.info(
            "Track-api request processed.", extra={"inference_request_uuid": inference_request.uuid}
        )
        return inference_request.final_result

    def track_api_files(
        self,
        files: List[BinaryIO],
        settings: Dict,
    ):
        logger.info(f"Start tracking with settings:", extra={"settings": settings})
        frame_index = find_value_by_keys(settings, ["frame_index", "frameIndex"])
        frames_count = find_value_by_keys(settings, ["frames", "framesCount"])
        input_geometries = find_value_by_keys(settings, ["input_geometries", "inputGeometries"])
        direction = settings.get("direction", "forward")
        direction_n = 1 if direction == "forward" else -1
        frame_indexes = list(
            range(frame_index, frame_index + frames_count * direction_n + direction_n, direction_n)
        )
        geometries = map(self._deserialize_geometry, input_geometries)
        frames = []
        for file, frame_idx in zip(files, frame_indexes):
            img_bytes = file.read()
            frame = sly_image.read_bytes(img_bytes)
            frames.append(frame)
        logger.info("Start tracking.")
        return self._inference(frames, geometries)

    def track_async(self, api: Api, state: Dict, context: Dict):
        batch_size = context.get("batch_size", self.get_batch_size())
        if self.max_batch_size is not None and batch_size > self.max_batch_size:
            raise ValidationError(
                f"Batch size should be less than or equal to {self.max_batch_size} for this model."
            )

        fn = self.send_error_data(api, context)(self._track_async)
        inference_request, _ = self.inference_requests_manager.schedule_task(fn, api, context)

        logger.debug(
            "Inference has scheduled from 'track_async' endpoint",
            extra={"inference_request_uuid": inference_request.uuid},
        )
        return {
            "message": "Inference has started.",
            "inference_request_uuid": inference_request.uuid,
        }

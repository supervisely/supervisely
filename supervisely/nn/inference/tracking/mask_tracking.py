import time
import uuid
from queue import Queue
from threading import Event, Thread
from typing import BinaryIO, Dict, List, Tuple

import numpy as np
from pydantic import ValidationError

from supervisely._utils import find_value_by_keys
from supervisely.annotation.label import Geometry, Label
from supervisely.annotation.obj_class import ObjClass
from supervisely.api.api import Api
from supervisely.api.module_api import ApiField
from supervisely.api.video.video_figure_api import FigureInfo
from supervisely.geometry.bitmap import Bitmap
from supervisely.geometry.helpers import deserialize_geometry
from supervisely.geometry.polygon import Polygon
from supervisely.imaging import image as sly_image
from supervisely.nn.inference.tracking.base_tracking import BaseTracking
from supervisely.nn.inference.tracking.tracker_interface import (
    TrackerInterface,
    TrackerInterfaceV2,
)
from supervisely.sly_logger import logger
from supervisely.task.progress import Progress


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

    def _track(self, api: Api, context: Dict):
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
                if not isinstance(geometry, Bitmap) and not isinstance(geometry, Polygon):
                    stop_upload_event.set()
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
                                bitmap_geometry = Bitmap(mask)
                                bitmap_obj_class = ObjClass("bitmap", Bitmap)
                                bitmap_label = Label(bitmap_geometry, bitmap_obj_class)
                                polygon_obj_class = ObjClass("polygon", Polygon)
                                polygon_labels = bitmap_label.convert(polygon_obj_class)
                                geometries = [label.geometry for label in polygon_labels]
                            else:
                                geometries = [Bitmap(mask)]
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

    def _track_async(self, api: Api, context: dict, inference_request_uuid: str = None):
        inference_request = self._inference_requests[inference_request_uuid]
        tracker_interface = TrackerInterfaceV2(api, context, self.cache)
        progress: Progress = inference_request["progress"]
        frames_count = tracker_interface.frames_count
        figures = tracker_interface.figures
        progress_total = frames_count * len(figures)
        progress.total = progress_total

        def _upload_f(items: List[Tuple[FigureInfo, bool]]):
            with inference_request["lock"]:
                inference_request["pending_results"].extend([item[0] for item in items])

        def _notify_f(items: List[Tuple[FigureInfo, bool]]):
            items_by_object_id: Dict[int, List[Tuple[FigureInfo, bool]]] = {}
            for item in items:
                items_by_object_id.setdefault(item[0].object_id, []).append(item)

            for object_id, object_items in items_by_object_id.items():
                frame_range = [
                    min(item[0].frame_index for item in object_items),
                    max(item[0].frame_index for item in object_items),
                ]
                progress.iters_done_report(sum(1 for item in object_items if item[1]))
                tracker_interface.notify_progress(progress.current, progress.total, frame_range)

        # run tracker
        frame_index = tracker_interface.frame_index
        direction_n = tracker_interface.direction_n
        api.logger.info("Start tracking.")
        try:
            with tracker_interface(_upload_f, _notify_f):
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
                    if inference_request["cancel_inference"]:
                        return
                    if tracker_interface.is_stopped():
                        reason = tracker_interface.stop_reason()
                        if isinstance(reason, Exception):
                            raise reason
                        return

                # predict
                tracked_multilabel_masks = self.predict(
                    frames=frames, input_mask=multilabel_mask[:, :, 0]
                )
                tracked_multilabel_masks = np.array(tracked_multilabel_masks)

                # decompose multilabel masks into binary masks
                for i in np.unique(tracked_multilabel_masks):
                    if inference_request["cancel_inference"]:
                        return
                    if tracker_interface.is_stopped():
                        reason = tracker_interface.stop_reason()
                        if isinstance(reason, Exception):
                            raise reason
                        return
                    if i != 0:
                        binary_masks = tracked_multilabel_masks == i
                        fig_id = label2id[i]["fig_id"]
                        obj_id = label2id[i]["obj_id"]
                        geometry_type = label2id[i]["original_geometry"]
                        for j, mask in enumerate(binary_masks[1:], 1):
                            if inference_request["cancel_inference"]:
                                return
                            if tracker_interface.is_stopped():
                                reason = tracker_interface.stop_reason()
                                if isinstance(reason, Exception):
                                    raise reason
                                return
                            this_figure_index = frame_index + j * direction_n
                            # check if mask is not empty
                            if not np.any(mask):
                                api.logger.info(f"Skipping empty mask on frame {this_figure_index}")
                                # update progress bar anyway (otherwise it will not be finished)
                                progress.iter_done_report()
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
        except Exception:
            progress.message = "Error occured during tracking"
            raise
        else:
            progress.message = "Ready"
        finally:
            progress.set(current=0, total=1, report=True)

    # Implement the following methods in the derived class
    def track(self, api: Api, state: Dict, context: Dict):
        fn = self.send_error_data(api, context)(self._track)
        self.schedule_task(fn, api, context)
        return {"message": "Tracking has started."}

    def track_api(self, api: Api, state: Dict, context: Dict):
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
            if not isinstance(geometry, Bitmap) and not isinstance(geometry, Polygon):
                raise TypeError(f"This app does not support {geometry.geometry_name()} tracking")
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
                        geometry = Bitmap(mask)
                        predictions_for_label.append(
                            {"type": geometry.geometry_name(), "data": geometry.to_json()}
                        )
                predictions.append(predictions_for_label)

        # predictions must be NxK masks: N=number of frames, K=number of objects
        predictions = list(map(list, zip(*predictions)))
        return predictions

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

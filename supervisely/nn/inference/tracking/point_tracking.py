import time
import uuid
from pathlib import Path
from queue import Queue
from threading import Event, Thread
from typing import Any, BinaryIO, Dict, List, Union

import numpy as np
from pydantic import ValidationError

import supervisely.nn.inference.tracking.functional as F
from supervisely.annotation.annotation import Annotation
from supervisely.annotation.label import Geometry, Label, LabelingStatus
from supervisely.annotation.obj_class import ObjClass
from supervisely.api.api import Api
from supervisely.api.module_api import ApiField
from supervisely.api.video.video_figure_api import FigureInfo
from supervisely.geometry.graph import GraphNodes
from supervisely.geometry.helpers import deserialize_geometry
from supervisely.geometry.point import Point
from supervisely.geometry.polygon import Polygon
from supervisely.geometry.polyline import Polyline
from supervisely.geometry.rectangle import Rectangle
from supervisely.imaging import image as sly_image
from supervisely.nn.inference.tracking.base_tracking import BaseTracking
from supervisely.nn.inference.tracking.tracker_interface import (
    TrackerInterface,
    TrackerInterfaceV2,
)
from supervisely.nn.prediction_dto import Prediction, PredictionPoint
from supervisely.sly_logger import logger
from supervisely.task.progress import Progress


class PointTracking(BaseTracking):
    def _deserialize_geometry(self, data: dict):
        geometry_type_str = data["type"]
        geometry_json = data["data"]
        return deserialize_geometry(geometry_type_str, geometry_json)

    def _inference(self, frames: List[np.ndarray], geometries: List[Geometry], settings: dict):
        updated_settings = {
            **self.custom_inference_settings_dict,
            **settings,
        }
        results = [[] for _ in range(len(frames) - 1)]
        for geometry in geometries:
            if isinstance(geometry, Point):
                predictions = self._predict_point_geometries(geometry, frames, updated_settings)
            elif isinstance(geometry, Polygon):
                if len(geometry.interior) > 0:
                    raise ValueError("Can't track polygons with interior.")
                predictions = self._predict_polygon_geometries(
                    geometry,
                    frames,
                    updated_settings,
                )
            elif isinstance(geometry, GraphNodes):
                predictions = self._predict_graph_geometries(
                    geometry,
                    frames,
                    updated_settings,
                )
            elif isinstance(geometry, Polyline):
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

    def _track(self, api: Api, context: dict):
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
                    if isinstance(geom, Point):
                        geometries = self._predict_point_geometries(
                            geom,
                            video_interface.frames_with_notification,
                        )
                    elif isinstance(geom, Polygon):
                        if len(geom.interior) > 0:
                            stop_upload_event.set()
                            raise ValueError("Can't track polygons with interior.")
                        geometries = self._predict_polygon_geometries(
                            geom,
                            video_interface.frames_with_notification,
                        )
                    elif isinstance(geom, GraphNodes):
                        geometries = self._predict_graph_geometries(
                            geom,
                            video_interface.frames_with_notification,
                        )
                    elif isinstance(geom, Polyline):
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

    def _track_async(self, api: Api, context: dict, inference_request_uuid: str):
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

        frame_index = tracker_interface.frame_index
        direction_n = tracker_interface.direction_n
        api.logger.info("Start tracking.")
        try:
            with tracker_interface(_upload_f, _notify_f):
                frames = tracker_interface.load_all_frames()
                frames = [frame.image for frame in frames]
                for figure in figures:
                    figure = api.video.figure._convert_json_info(figure)
                    api.logger.info("geometry:", extra={"figure": figure._asdict()})
                    sly_geometry: Rectangle = deserialize_geometry(
                        figure.geometry_type, figure.geometry
                    )
                    api.logger.info("geometry:", extra={"geometry": type(sly_geometry)})
                    if isinstance(sly_geometry, Point):
                        geometries = self._predict_point_geometries(
                            sly_geometry,
                            frames,
                        )
                    elif isinstance(sly_geometry, Polygon):
                        if len(sly_geometry.interior) > 0:
                            raise ValueError("Can't track polygons with interior.")
                        geometries = self._predict_polygon_geometries(
                            sly_geometry,
                            frames,
                        )
                    elif isinstance(sly_geometry, GraphNodes):
                        geometries = self._predict_graph_geometries(
                            sly_geometry,
                            frames,
                        )
                    elif isinstance(sly_geometry, Polyline):
                        geometries = self._predict_polyline_geometries(
                            sly_geometry,
                            frames,
                        )
                    else:
                        raise TypeError(
                            f"Tracking does not work with {sly_geometry.geometry_name()}."
                        )

                    for i, geometry in enumerate(geometries, 1):
                        figure_id = uuid.uuid5(
                            namespace=uuid.NAMESPACE_URL, name=f"{time.time()}"
                        ).hex
                        result_figure = api.video.figure._convert_json_info(
                            {
                                ApiField.ID: figure_id,
                                ApiField.OBJECT_ID: figure.object_id,
                                "meta": {"frame": frame_index + i * direction_n},
                                ApiField.GEOMETRY_TYPE: geometry.geometry_name(),
                                ApiField.GEOMETRY: geometry.to_json(),
                                ApiField.TRACK_ID: tracker_interface.track_id,
                            }
                        )
                        tracker_interface.add_prediction(result_figure)
                    api.logger.info(f"Figure #{figure.id} tracked.")

                    if inference_request["cancel_inference"]:
                        return
                    if tracker_interface.is_stopped():
                        reason = tracker_interface.stop_reason()
                        if isinstance(reason, Exception):
                            raise reason
                        return
        except Exception as e:
            progress.message = "Error occured during tracking"
            raise
        else:
            progress.message = "Ready"
        finally:
            progress.set(current=0, total=1, report=True)

    def track(self, api: Api, state: Dict, context: Dict):
        fn = self.send_error_data(api, context)(self._track)
        self.schedule_task(fn, api, context)
        return {"message": "Track task started."}

    def track_api(self, api: Api, state: Dict, context: Dict):
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
                if isinstance(geom, Point):
                    geometries = self._predict_point_geometries(
                        geom,
                        video_interface.frames,
                    )
                elif isinstance(geom, Polygon):
                    if len(geom.interior) > 0:
                        raise ValueError("Can't track polygons with interior.")
                    geometries = self._predict_polygon_geometries(
                        geom,
                        video_interface.frames,
                    )
                elif isinstance(geom, GraphNodes):
                    geometries = self._predict_graph_geometries(
                        geom,
                        video_interface.frames,
                    )
                elif isinstance(geom, Polyline):
                    geometries = self._predict_polyline_geometries(
                        geom,
                        video_interface.frames,
                    )
                else:
                    raise TypeError(f"Tracking does not work with {geom.geometry_name()}.")

                if video_interface.global_stop_indicatior:
                    return

                geometries = [video_interface._crop_geometry(g) for g in geometries]
                geometries = [g for g in geometries if g is not None]
                geometries = [{"type": g.geometry_name(), "data": g.to_json()} for g in geometries]
                predictions.append(geometries)

        # predictions must be NxK figures: N=number of frames, K=number of objects
        predictions = list(map(list, zip(*predictions)))
        return predictions

    def track_api_files(
        self,
        files: List[BinaryIO],
        settings: Dict,
    ):
        logger.info(f"Start tracking with settings:", extra={"settings": settings})
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

    def _create_label(self, dto: PredictionPoint) -> Point:
        geometry = Point(row=dto.row, col=dto.col)
        return Label(geometry, ObjClass("", Point))

    def _get_obj_class_shape(self):
        return Point

    def _predict_point_geometries(
        self,
        geom: Point,
        frames: List[np.ndarray],
        settings: Dict[str, Any] = None,
    ) -> List[Point]:
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
        geom: Polygon,
        frames: List[np.ndarray],
        settings: Dict[str, Any] = None,
    ) -> List[Polygon]:
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
        geom: GraphNodes,
        frames: List[np.ndarray],
        settings: Dict[str, Any] = None,
    ) -> List[GraphNodes]:
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
        geom: Polyline,
        frames: List[np.ndarray],
        settings: Dict[str, Any] = None,
    ) -> List[Polyline]:
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

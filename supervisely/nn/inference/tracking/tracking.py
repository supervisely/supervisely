import numpy as np
import functools
from fastapi import Request, BackgroundTasks
from typing import Any, Dict, List, Literal, Optional, Union
from typing_extensions import Literal

import supervisely as sly
from supervisely.geometry.geometry import Geometry
from supervisely.nn.prediction_dto import Prediction, PredictionPoint
from supervisely.nn.inference.tracking.tracker_interface import TrackerInterface
from supervisely.nn.inference import Inference
import supervisely.nn.inference.tracking.functional as F


class Tracking(Inference):
    def __init__(
        self,
        model_dir: Optional[str] = None,
        custom_inference_settings: Optional[Union[Dict[str, Any], str]] = None,
    ):
        super().__init__(
            model_dir,
            custom_inference_settings,
            sliding_window_mode=None,
            use_gui=False,
        )

        try:
            self.load_on_device(model_dir, "cuda")
        except RuntimeError:
            self.load_on_device(model_dir, "cpu")

    def get_info(self):
        info = super().get_info()
        info["task type"] = "tracking"
        # recommended parameters:
        # info["model_name"] = ""
        # info["checkpoint_name"] = ""
        # info["pretrained_on_dataset"] = ""
        # info["device"] = ""
        return info

    def serve(self):
        super().serve()
        server = self._app.get_server()

        @server.post("/track")
        def start_track(request: Request, task: BackgroundTasks):
            task.add_task(track, request)
            return {"message": "Track task started."}

        def send_error_data(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                value = None
                try:
                    value = func(*args, **kwargs)
                except Exception as e:
                    request: Request = args[0]
                    context = request.state.context
                    api: sly.Api = request.state.api
                    track_id = context["trackId"]

                    api.post(
                        "videos.notify-annotation-tool",
                        data={
                            "type": "videos:tracking-error",
                            "data": {
                                "trackId": track_id,
                                "error": {"message": repr(e)},
                            },
                        },
                    )
                return value

            return wrapper

        @send_error_data
        def track(request: Request):
            context = request.state.context
            api: sly.Api = request.state.api

            video_interface = TrackerInterface(
                context=context,
                api=api,
            )

            api.logger.info("Start tracking.")

            for geom, obj_id in zip(
                video_interface.geometries, video_interface.object_ids
            ):
                if isinstance(geom, sly.Point):
                    pp_geom = PredictionPoint("point", col=geom.col, row=geom.row)
                    predicted: List[Prediction] = self.predict(
                        video_interface.frames,
                        self.custom_inference_settings_dict,
                        pp_geom,
                    )
                    geometries = F.dto_points_to_sly_points(predicted)
                elif isinstance(geom, sly.Polygon):
                    if len(geom.interior) > 0:
                        raise ValueError(f" Can't track polygons with iterior.")

                    polygon_points = F.numpy_to_dto_point(geom.exterior_np, "polygon")
                    exterior_per_time = [
                        [] for _ in range(video_interface.frames_count)
                    ]

                    for pp_geom in polygon_points:
                        points: List[Prediction] = self.predict(
                            video_interface.frames,
                            self.custom_inference_settings_dict,
                            pp_geom,
                        )
                        points_loc = F.dto_points_to_point_location(points)
                        for fi, point_loc in enumerate(points_loc):
                            exterior_per_time[fi].append(point_loc)

                    geometries = F.exteriors_to_sly_polygons(exterior_per_time)
                else:
                    raise TypeError(
                        f"Tracking does not work with {geom.geometry_name()}."
                    )

                video_interface.add_object_geometries(geometries, obj_id)
                api.logger.info(f"Object #{obj_id} tracked.")

    def predict(
        self,
        rgb_images: List[np.ndarray],
        settings: Dict[str, Any],
        start_object: PredictionPoint,
    ) -> List[PredictionPoint]:
        """
        Track point on given frames.

        :param rgb_images: RGB frames, `m` frames
        :type rgb_images: List[np.array]
        :param settings: model parameters
        :type settings: Dict[str, Any]
        :param start_objects: point to track on the initial frame
        :type start_objects: PredictionPoint
        :return: predicted points for frame range (0, m]; `m-1` prediction in total
        :rtype: List[PredictionPoint]
        """
        raise NotImplementedError

    def _get_obj_class_shape(self):
        return sly.Point

    def _predict_point_geometries(self) -> List[Geometry]:
        pass

    def _predict_polygon_geometries(self) -> List[Geometry]:
        pass

import numpy as np
from fastapi import Request
from typing import Any, Dict, List, Literal, Optional, Union
from typing_extensions import Literal

import supervisely as sly
from supervisely.nn.prediction_dto import Prediction, PredictionPoint
from supervisely.nn.inference.tracking.tracker_interface import TrackerInterface
from supervisely.nn.inference import Inference


class Tracking(Inference):
    def __init__(
        self,
        model_dir: Optional[str] = None,
        custom_inference_settings: Optional[Union[Dict[str, Any], str]] = None,
    ):
        super().__init__(
            model_dir, custom_inference_settings, sliding_window_mode=None, use_gui=False
        )

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
        def track(request: Request):
            context = request.state.context
            api: sly.Api = request.state.api

            video_interface = TrackerInterface(
                context=context,
                api=api,
            )

            api.logger.info("Start tracking.")

            for geom, obj_id in zip(video_interface.geometries, video_interface.object_ids):
                points: List[Prediction] = self.predict(
                    video_interface.frames,
                    self.custom_inference_settings_dict,
                    geom,
                )
                geometries = self._convert_to_sly_geometries(points)
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

        :param rgb_images: RGB frames, `M` frames
        :type rgb_images: List[np.array]
        :param settings: model parameters
        :type settings: Dict[str, Any]
        :param start_objects: tracking points on the initial frame; N points
        :type start_objects: List[PredictionPoint]
        :return: _description_
        :rtype: List[PredictionPoint]
        """
        raise NotImplementedError

    def _get_obj_class_shape(self):
        return sly.Point

    def _convert_to_sly_geometries(self, points: List[PredictionPoint]):
        obj_class = self._get_obj_class_shape()
        return [obj_class(row=p.row, col=p.col) for p in points]

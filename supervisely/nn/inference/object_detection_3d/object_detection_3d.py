from typing import Dict, List, Any
from supervisely.geometry.cuboid_3d import Cuboid3d
from supervisely.nn.prediction_dto import PredictionCuboid3d
from supervisely.annotation.label import Label
from supervisely.annotation.tag import Tag
from supervisely.nn.inference.inference import Inference
from fastapi import Response, Request, status
from supervisely.sly_logger import logger
import os
from supervisely import Api
from supervisely.io.fs import silent_remove
from supervisely._utils import rand_str
from supervisely.app.content import get_data_dir


class ObjectDetection3D(Inference):
    def get_info(self) -> dict:
        info = super().get_info()
        info["task type"] = "object detection 3d"
        info["videos_support"] = False
        info["async_video_inference_support"] = False
        info["tracking_on_videos_support"] = False
        info["async_image_inference_support"] = False
        
        # recommended parameters:
        # info["model_name"] = ""
        # info["checkpoint_name"] = ""
        # info["pretrained_on_dataset"] = ""
        # info["device"] = ""
        return info

    def _get_obj_class_shape(self):
        return Cuboid3d

    def _create_label(self, dto: PredictionCuboid3d):
        raise NotImplementedError()

    def predict(self, pcd_path: str, settings: Dict[str, Any]) -> List[PredictionCuboid3d]:
        raise NotImplementedError("Have to be implemented in child class")

    def predict_raw(self, pcd_path: str, settings: Dict[str, Any]) -> List[PredictionCuboid3d]:
        raise NotImplementedError(
            "Have to be implemented in child class If sliding_window_mode is 'advanced'."
        )
    
    def _inference_pointcloud_id(self, api: Api, pointcloud_id: int, settings: Dict[str, Any]):
        # 1. download pointcloud
        pcd_path = os.path.join(get_data_dir(), rand_str(10) + ".pcd")
        api.pointcloud.download_path(pointcloud_id, pcd_path)
        # 2. predict
        prediction = self.predict(pcd_path, settings)
        # 3. clean up
        silent_remove(pcd_path)
        return prediction

    def serve(self):
        super().serve()
        server = self._app.get_server()

        @server.post("/inference_pointcloud_id")
        def inference_pointcloud_id(response: Response, request: Request):
            logger.debug(
                f"inference_pointcloud_id: state=",
                extra={**request.state.state, "api_token": "***"},
            )
            state = request.state.state
            api : Api = request.state.api
            settings = self._get_inference_settings(state)
            prediction = self._inference_pointcloud_id(api, state.pointcloud_id, settings)
            return prediction

        @server.post("/inference_pointcloud_ids")
        def inference_pointcloud_ids(response: Response, request: Request):
            logger.debug(
                f"inference_pointcloud_ids: state=",
                extra={**request.state.state, "api_token": "***"},
            )
            state = request.state.state
            api : Api = request.state.api
            settings = self._get_inference_settings(state)
            predictions = []
            for pcd_id in state.pointcloud_ids:
                pred = self._inference_pointcloud_id(api, pcd_id, settings)
                predictions.append(pred)
            return predictions
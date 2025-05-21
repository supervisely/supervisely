import os
from typing import Any, Dict, List

from fastapi import Request, Response, status

from supervisely import Api, PointcloudAnnotation, PointcloudFigure, PointcloudObject
from supervisely._utils import rand_str
from supervisely.annotation.label import Label
from supervisely.annotation.tag import Tag
from supervisely.app.content import get_data_dir
from supervisely.geometry.cuboid_3d import Cuboid3d
from supervisely.io.fs import silent_remove
from supervisely.nn.inference.inference import Inference
from supervisely.nn.prediction_dto import PredictionCuboid3d
from supervisely.pointcloud_annotation.pointcloud_object_collection import (
    PointcloudObjectCollection,
)
from supervisely.sly_logger import logger


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

    def annotation_from_prediction(
        self,
        predictions: List[PredictionCuboid3d],
    ) -> PointcloudAnnotation:
        model_meta = self._model_meta
        objects = []
        figures = []
        for prediction in predictions:
            class_name = prediction.class_name
            geometry = prediction.cuboid_3d
            object = PointcloudObject(model_meta.get_obj_class(class_name))
            figure = PointcloudFigure(object, geometry)
            objects.append(object)
            figures.append(figure)
        objects = PointcloudObjectCollection(objects)
        annotation = PointcloudAnnotation(objects, figures)
        return annotation

    def raw_results_from_prediction(
        self, prediction: List[PredictionCuboid3d]
    ) -> List[Dict[str, Any]]:
        results = []
        for pred in prediction:
            detection_name = pred.class_name
            translation = list(pred.cuboid_3d.position.to_json().values())
            size = list(pred.cuboid_3d.dimensions.to_json().values())
            rotation_z = pred.cuboid_3d.rotation.z
            velocity = [0, 0]  # Is not supported now
            detection_score = pred.score
            results.append(
                {
                    "detection_name": detection_name,
                    "translation": translation,
                    "size": size,
                    "rotation": rotation_z,
                    "velocity": velocity,
                    "detection_score": detection_score,
                }
            )
        return results

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
            api: Api = request.state.api
            settings = self._get_inference_settings(state)
            prediction = self._inference_pointcloud_id(api, state["pointcloud_id"], settings)
            annotation = self.annotation_from_prediction(prediction)
            raw_results = self.raw_results_from_prediction(prediction)
            result = {
                "results": {
                    "annotation": annotation.to_json(),
                    "raw_results": raw_results,
                }
            }  # This format is used in "Apply 3D Detection to Pointcloud Project" app.
            return result

        @server.post("/inference_pointcloud_ids")
        def inference_pointcloud_ids(response: Response, request: Request):
            logger.debug(
                f"inference_pointcloud_ids: state=",
                extra={**request.state.state, "api_token": "***"},
            )
            state = request.state.state
            api: Api = request.state.api
            settings = self._get_inference_settings(state)
            annotations = []
            for pcd_id in state["pointcloud_ids"]:
                prediction = self._inference_pointcloud_id(api, pcd_id, settings)
                annotation = self.annotation_from_prediction(prediction)
                annotations.append(annotation.to_json())
            return annotations

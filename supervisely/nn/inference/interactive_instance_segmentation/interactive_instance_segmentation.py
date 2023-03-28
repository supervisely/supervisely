from typing import Dict, List, Any, Union
from supervisely.geometry.bitmap import Bitmap
from supervisely.nn.prediction_dto import PredictionMask
from supervisely.annotation.label import Label
from supervisely.sly_logger import logger
import numpy as np
import functools
from supervisely.geometry.rectangle import Rectangle
from supervisely.imaging import image as sly_image
from supervisely.decorators.inference import _scale_ann_to_original_size, _process_image_path
from supervisely.io.fs import silent_remove
from supervisely.decorators.inference import process_image_roi, process_image_sliding_window
from supervisely.nn.inference import InstanceSegmentation
from fastapi import Form, Response, Request, UploadFile, status


class InteractiveInstanceSegmentation(InstanceSegmentation):

    class Click:
        def __init__(self, x, y, is_positive) -> None:
            self.x = x
            self.y = y
            self.is_positive = is_positive
        
    def get_info(self) -> dict:
        info = super().get_info()
        info["task type"] = "interactive instance segmentation"
        return info

    def predict(self, image_path: str, clicks: List[Click], image_changed: bool, settings: Dict[str, Any]) -> List[PredictionMask]:
        raise NotImplementedError("Have to be implemented in child class")

    def on_image_changed(self, image_path: str):
        pass

    @process_image_sliding_window
    @process_image_roi
    def _inference_image_path(
        self,
        image_path: str,
        settings: Dict,
        data_to_return: Dict,  # for decorators
    ):
        inference_mode = settings.get("inference_mode", "full_image")
        logger.debug(
            "Inferring image_path:", extra={"inference_mode": inference_mode, "path": image_path}
        )

        interactive_info = settings['interactive_info']

        if self._is_image_changed():
            image_changed = True
            self.on_image_changed(image_path)

        # to local coords
        from itertools import chain
        crop_rect = interactive_info["crop"]
        for coords_xy in chain(interactive_info['positive'], interactive_info['negative']):
            coords_xy[0] -= crop_rect[0][0]
            coords_xy[1] -= crop_rect[0][1]

        clicks = [self.Click(x, y, is_positive=True) for x, y in interactive_info['positive']]
        clicks += [self.Click(x, y, is_positive=False) for x, y in interactive_info['negative']]

        if inference_mode == "sliding_window" and settings["sliding_window_mode"] == "advanced":
            predictions = self.predict_raw(image_path=image_path, clicks=clicks, image_changed=image_changed, settings=settings)
        else:
            predictions = self.predict(image_path=image_path, clicks=clicks, image_changed=image_changed, settings=settings)
        ann = self._predictions_to_annotation(image_path, predictions)

        logger.debug(
            f"Inferring image_path done. pred_annotation:",
            extra=dict(w=ann.img_size[1], h=ann.img_size[0], n_labels=len(ann.labels)),
        )
        return ann

    def _is_image_changed():
        raise

    def serve(self):
        super().serve()
        server = self._app.get_server()
        
        @server.post("/smart_segmentation")
        def smart_segmentation(response: Response, request: Request):
            logger.debug(f"smart_segmentation inference started")

            interactive_schema = ["figureId", "crop", "positive", "negative", "taskId"]
            # missing: click_list, image_id
            
            state = request.state.state

            # {"figureId":None,"crop":[[305,72],[1425,1091]],"positive":[[865,582]],"negative":[],"taskId":29920}
            if state.get("interactive_state") is None or list(state["interactive_state"].keys()) != interactive_schema:
                response.status_code = status.HTTP_400_BAD_REQUEST
                return {"message": "Error 400. Bad request.", "success": False}
            
            if "crop" in state:
                state["rectangle"] = state["crop"]

            assert "image_id" in state

            self._inference_image_id(request.state.api, state)

            return {"success": True}
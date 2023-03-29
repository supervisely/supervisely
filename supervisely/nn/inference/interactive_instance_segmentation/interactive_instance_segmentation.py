import functools
import os
import numpy as np
from fastapi import Form, Response, Request, UploadFile, status

from supervisely.geometry.bitmap import Bitmap
from supervisely.nn.prediction_dto import PredictionMask
from supervisely.annotation.label import Label
from supervisely.sly_logger import logger
from supervisely.geometry.rectangle import Rectangle
from supervisely.imaging import image as sly_image
from supervisely.decorators.inference import (
    _scale_ann_to_original_size,
    _process_image_path,
)
from supervisely.io.fs import silent_remove
from supervisely.decorators.inference import (
    process_image_roi,
    process_image_sliding_window,
)
from supervisely._utils import rand_str
from supervisely.app.content import get_data_dir

from supervisely.nn.inference import InstanceSegmentation
from . import utils

from typing import Dict, List, Any, Optional, Union
try:
    from typing import Literal
except ImportError:
    # for compatibility with python 3.7
    from typing_extensions import Literal


class InteractiveInstanceSegmentation(InstanceSegmentation):

    class Click:
        def __init__(self, x, y, is_positive):
            self.x = x
            self.y = y
            self.is_positive = is_positive

    def __init__(self, model_dir: Optional[str] = None, custom_inference_settings: Optional[Union[Dict[str, Any], str]] = None, sliding_window_mode: Optional[Literal["basic", "advanced", "none"]] = "basic", use_gui: Optional[bool] = False):
        super().__init__(model_dir, custom_inference_settings, sliding_window_mode, use_gui)
        self.current_smtool_state = None
        self.current_clicks = []
        self.current_image_path = None

    def get_info(self) -> dict:
        info = super().get_info()
        info["task type"] = "interactive instance segmentation"
        return info

    def predict(
        self,
        image_path: str,
        clicks: List[Click],
        image_changed: bool,
        settings: Dict[str, Any],
    ) -> PredictionMask:
        raise NotImplementedError("Have to be implemented in child class")

    def on_image_changed(self, image_path: str):
        pass

    def _check_image_changed(self, smtool_state):
        if self.current_smtool_state is not None:
            prev_state = self.current_smtool_state.copy()
            smtool_state = smtool_state.copy()
            smtool_state.pop('positive')
            smtool_state.pop('negative')
            prev_state.pop('positive')
            prev_state.pop('negative')
            return smtool_state != prev_state
        else:
            return True

    def _on_model_deployed(self):
        self._reset_current_state()

    def _reset_current_state(self):
        self.current_smtool_state = None
        self.current_clicks = []
        self.current_image_path = None

    def serve(self):
        super().serve()
        server = self._app.get_server()

        @server.post("/smart_segmentation")
        def smart_segmentation(response: Response, request: Request):
            logger.debug(f"smart_segmentation inference: context=", extra=request.state.context)

            try:
                state = request.state.state
                settings = self._get_inference_settings(state)
                smtool_state = request.state.context
                api = request.state.api
                crop = smtool_state['crop']
                positive_clicks, negative_clicks = smtool_state['positive'], smtool_state['negative']
            except Exception as exc:
                logger.warn("Error parsing request:" + str(exc), exc_info=True)
                response.status_code = status.HTTP_400_BAD_REQUEST
                return {"message": "Error 400. Bad request.", "success": False}

            is_image_changed = self._check_image_changed(smtool_state)
            if is_image_changed:
                if self.current_image_path is not None:
                    silent_remove(self.current_image_path)
                self._reset_current_state()
                image_np = utils.download_image_from_context(smtool_state, api, self.model_dir)
                image_np = utils.crop_image(crop, image_np)
                self.current_image_path = os.path.join(get_data_dir(), f"{rand_str(10)}.jpg")
                sly_image.write(self.current_image_path, image_np)
                self.on_image_changed(self.current_image_path)
                
            clicks = [{**click, "is_positive": True} for click in positive_clicks]
            clicks += [{**click, "is_positive": False} for click in negative_clicks]
            clicks = utils.transform_clicks_to_crop(crop, clicks)
            new_clicks = utils.get_new_clicks(self.current_clicks, clicks)
            if new_clicks is not None:
                print(f"Exactly one! {new_clicks=}")
                self.current_clicks += new_clicks
            else:
                new_clicks = clicks
                self.current_clicks = clicks
            self.current_smtool_state = smtool_state

            clicks_to_predict = [self.Click(c['x'], c['y'], c['is_positive']) for c in new_clicks]
            pred_mask = self.predict(self.current_image_path, clicks_to_predict, is_image_changed, settings)
            # ann = self._predictions_to_annotation(self.current_image_path, pred_mask)

            logger.debug(f"smart_segmentation inference done")

            bitmap = Bitmap(pred_mask.mask)
            bitmap_origin, bitmap_data = utils.format_bitmap(bitmap, crop)
            sly_image.write("pred.png", bitmap.data*255)

            response = {
                "origin": bitmap_origin,
                "bitmap": bitmap_data,
                "success": True,
                "error": None,
            }
            return response
